"""Procedural fruit-fly video simulator.

The lab video used in the reference project (Herbst, CS229 2018) is not public,
so we generate grayscale frames of one male and one female fly in a circular
well, together with exact ground-truth labels (sex, position, heading and the
two male wing angles). Everything is seeded and fully reproducible.

Conventions (image coordinates, y points down):
  * heading / angles are in degrees, measured from +x towards +y
  * wing angle 0 = wing folded along the body, 90 = wing perpendicular to it
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

IMG = 512          # frame size (pixels)
R = 236            # radius of the circular well
SH = 4             # sub-pixel shift used for anti-aliased drawing
K = 1 << SH

FEMALE_LEN = 46.0
MALE_LEN = 38.0


@dataclass
class Fly:
    sex: str                 # "M" or "F"
    x: float
    y: float
    heading: float           # degrees, direction the fly is facing
    length: float
    wing_r: float = 6.0      # fly's own right wing, 0-90 deg
    wing_l: float = 6.0

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------
def _ell(img, cx, cy, a, b, ang, color):
    cv2.ellipse(img, (int(round(cx * K)), int(round(cy * K))),
                (max(1, int(round(a * K))), max(1, int(round(b * K)))),
                float(ang), 0, 360, int(color), -1, cv2.LINE_AA, SH)


def _circ(img, cx, cy, r, color):
    cv2.circle(img, (int(round(cx * K)), int(round(cy * K))),
               max(1, int(round(r * K))), int(color), -1, cv2.LINE_AA, SH)


def _line(img, p, q, color):
    cv2.line(img, (int(round(p[0] * K)), int(round(p[1] * K))),
             (int(round(q[0] * K)), int(round(q[1] * K))),
             int(color), 1, cv2.LINE_AA, SH)


def _unit(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


_BG = None


def _background():
    global _BG
    if _BG is None:
        yy, xx = np.mgrid[:IMG, :IMG]
        r = np.hypot(xx - IMG / 2 + .5, yy - IMG / 2 + .5)
        bg = np.full((IMG, IMG), 255, np.float32)
        inside = r < R
        bg[inside] = 232 - 22 * (r[inside] / R) ** 2
        bg[(r >= R) & (r < R + 10)] = 150
        _BG = bg
    return _BG


def _draw_wings_legs(img, f, rng):
    L, t = f.length, f.heading
    c = np.array([f.x, f.y])
    h = _unit(t)
    perp_r = _unit(t + 90)            # fly's right-hand side
    # legs (thin, faint: removed again by the wing pre-processing)
    for side in (+1, -1):
        for off, spread in ((0.30, 55), (0.10, 85), (-0.08, 125)):
            base = c + off * L * h + side * 0.12 * L * perp_r
            a1 = t + side * (spread + rng.normal(0, 6))
            p1 = base + 0.30 * L * _unit(a1)
            p2 = p1 + 0.28 * L * _unit(a1 + side * rng.normal(35, 8))
            _line(img, base, p1, 150)
            _line(img, p1, p2, 150)
    # wings
    lw = 0.95 * L
    for side, ang in ((+1, f.wing_r), (-1, f.wing_l)):
        attach = c + 0.20 * L * h + side * 0.10 * L * perp_r
        wdir = t + 180 - ang if side > 0 else t + 180 + ang
        centre = attach + 0.5 * lw * _unit(wdir)
        _ell(img, centre[0], centre[1], 0.5 * lw, 0.14 * L, wdir, 172)


def _draw_body(img, f):
    L, t = f.length, f.heading
    c = np.array([f.x, f.y])
    h = _unit(t)
    perp = _unit(t + 90)
    ab = c - 0.18 * L * h
    th = c + 0.18 * L * h
    hd = c + 0.42 * L * h
    ab_col = 60 if f.sex == "F" else 52
    _ell(img, ab[0], ab[1], 0.34 * L, 0.21 * L, t, ab_col)
    stripes = (-0.22, -0.08, 0.06, 0.20) if f.sex == "F" else (-0.10, 0.10)
    for k in stripes:
        p = ab + k * L * h
        _ell(img, p[0], p[1], 0.19 * L, 0.022 * L, t + 90, 88)
    if f.sex == "M":                                    # dark abdomen tip
        p = ab - 0.24 * L * h
        _ell(img, p[0], p[1], 0.09 * L, 0.17 * L, t + 90, 24)
    _ell(img, th[0], th[1], 0.22 * L, 0.17 * L, t, 40)
    _circ(img, hd[0], hd[1], 0.11 * L, 46)
    for s in (+1, -1):                                  # eyes
        e = hd + s * 0.07 * L * perp + 0.02 * L * h
        _circ(img, e[0], e[1], 0.05 * L, 20)


def render(flies, rng, debris=True):
    """Render one grayscale (uint8) frame."""
    img = np.clip(_background() * rng.uniform(0.94, 1.04), 0, 255).astype(np.uint8)
    if debris:
        for _ in range(rng.poisson(1.3)):
            p = rng.uniform(-1, 1, 2) * (R - 30)
            while np.hypot(*p) > R - 25:
                p = rng.uniform(-1, 1, 2) * (R - 30)
            cx, cy = IMG / 2 + p[0], IMG / 2 + p[1]
            if rng.random() < 0.10:                      # hard negative blob
                _ell(img, cx, cy, rng.uniform(10, 15), rng.uniform(6, 9),
                     rng.uniform(0, 180), rng.uniform(50, 80))
            else:
                _circ(img, cx, cy, rng.uniform(2, 7), rng.uniform(50, 90))
    for f in flies:
        _draw_wings_legs(img, f, rng)
    for f in flies:
        _draw_body(img, f)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.7, 1.5))
    img = img.astype(np.float32) + rng.normal(0, rng.uniform(1.5, 3.5), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# random independent frames (the "dataset")
# --------------------------------------------------------------------------
def _wing_sample(rng):
    if rng.random() < 0.55:
        return float(np.clip(rng.normal(7, 3), 0, 25))
    return float(rng.uniform(15, 88))


def sample_scene(rng):
    """Random male+female pair. Roughly 78% separated, 10% close, 12% touching."""
    mode = rng.choice(["apart", "near", "touch"], p=[0.78, 0.10, 0.12])
    lim = R - 55
    while True:
        p1 = rng.uniform(-lim, lim, 2)
        if np.hypot(*p1) > lim:
            continue
        d = {"apart": rng.uniform(85, 300),
             "near": rng.uniform(52, 85),
             "touch": rng.uniform(24, 46)}[mode]
        p2 = p1 + d * _unit(rng.uniform(0, 360))
        if np.hypot(*p2) <= lim:
            break
    c = IMG / 2
    female = Fly("F", c + p1[0], c + p1[1], rng.uniform(0, 360),
                 rng.normal(FEMALE_LEN, 3.0), 5 + rng.normal(0, 1.5), 5 + rng.normal(0, 1.5))
    male = Fly("M", c + p2[0], c + p2[1], rng.uniform(0, 360),
               rng.normal(MALE_LEN, 2.5), _wing_sample(rng), _wing_sample(rng))
    flies = [female, male]
    rng.shuffle(flies)
    return flies


# --------------------------------------------------------------------------
# temporally coherent courtship video (held-out test video)
# --------------------------------------------------------------------------
def simulate_video(n_frames, seed, fps=30):
    rng = np.random.default_rng(seed)
    c = IMG / 2
    lim = R - 55
    pf = rng.uniform(-100, 100, 2)
    vf = np.zeros(2)
    pm = pf + np.array([150.0, 60.0])
    hf, hm = 0.0, 180.0
    # song bouts: (start, duration, frequency, wing, amplitude)
    bouts, t = [], 40
    while t < n_frames - 40:
        dur = int(rng.uniform(45, 90))
        bouts.append((t, dur, rng.uniform(1.6, 3.2), rng.choice(["r", "l"]), rng.uniform(40, 75)))
        t += dur + int(rng.uniform(40, 110))
    states = []
    for i in range(n_frames):
        vf = 0.96 * vf + rng.normal(0, 0.45, 2)
        pf = pf + vf
        if np.hypot(*pf) > lim:                          # bounce off the wall
            n = pf / np.hypot(*pf)
            vf = vf - 2 * np.dot(vf, n) * n
            pf = n * lim
        if np.hypot(*vf) > 0.6:
            tgt = np.degrees(np.arctan2(vf[1], vf[0]))
            hf += ((tgt - hf + 180) % 360 - 180) * 0.15
        to_f = pf - pm
        dist = np.hypot(*to_f)
        u = to_f / dist
        speed = 2.6 if dist > 95 else 0.4
        tang = np.array([-u[1], u[0]]) * 1.2
        pm = pm + speed * u * (1 if dist > 85 else -0.5) + (tang if dist < 140 else 0)
        dist = np.hypot(*(pf - pm))
        if dist < 75:                                   # keep bodies apart
            pm = pf - (pf - pm) / dist * 75
        if np.hypot(*pm) > lim:
            pm = pm / np.hypot(*pm) * lim
        tgt = np.degrees(np.arctan2(*(pf - pm)[::-1]))
        hm += ((tgt - hm + 180) % 360 - 180) * 0.2
        wr = wl = 7 + 2 * np.sin(i / 37.0)
        for (s, d, fq, side, amp) in bouts:
            if s <= i < s + d:
                gate = np.sin(np.pi * (i - s) / d) ** 0.5
                osc = 0.5 + 0.5 * np.sin(2 * np.pi * fq * (i - s) / fps * 3)
                if side == "r":
                    wr = 10 + gate * amp * osc
                    wl = 6 + gate * 8 * (1 - osc)
                else:
                    wl = 10 + gate * amp * osc
                    wr = 6 + gate * 8 * (1 - osc)
        states.append([Fly("F", c + pf[0], c + pf[1], hf % 360, FEMALE_LEN, 5.0, 5.0),
                       Fly("M", c + pm[0], c + pm[1], hm % 360, MALE_LEN,
                           float(np.clip(wr, 0, 90)), float(np.clip(wl, 0, 90)))])
    return states
