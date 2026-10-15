"""Drawing helpers: annotate a frame with what the pipeline found (used by demo.py and gui.py)."""
import cv2
import numpy as np

# colours are BGR (OpenCV order)
MALE_BGR = (200, 110, 30)        # blue
FEMALE_BGR = (50, 55, 200)       # red
WING_BGR = (0, 170, 255)         # orange
INK_BGR = (40, 30, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, color, scale=0.45):
    """Text with a light halo so it stays readable on any background."""
    cv2.putText(img, txt, org, FONT, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, 1, cv2.LINE_AA)


def _label_origin(img, txt, cx, cy, scale):
    (w, h), _ = cv2.getTextSize(txt, FONT, scale, 1)
    x = int(np.clip(cx - w / 2, 4, img.shape[1] - w - 4))
    y = int(np.clip(cy, h + 4, img.shape[0] - 6))
    return x, y


def draw_overlay(gray, flies, hud=None, arrows=True, wings=True, labels=True, contours=True):
    """Return a BGR image with the analysis drawn on top of the grayscale frame.

    flies: list of fly dicts from FlyPipeline.process (sex, x, y, heading, contour, wing_r, wing_l)
    hud:   optional text for the top-left corner (frame number, FPS, ...)
    """
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for f in flies:
        male = f["sex"] == "M"
        col = MALE_BGR if male else FEMALE_BGR
        p = np.array([f["x"], f["y"]])
        h = np.array([np.cos(np.deg2rad(f["heading"])), np.sin(np.deg2rad(f["heading"]))])
        if contours and "contour" in f:
            cv2.drawContours(img, [f["contour"]], -1, col, 1, cv2.LINE_AA)
        if arrows:
            cv2.arrowedLine(img, tuple(int(v) for v in p), tuple(int(v) for v in p + 36 * h),
                            col, 2, cv2.LINE_AA, tipLength=0.3)
        if male and wings and "wing_r" in f:
            perp = np.array([-h[1], h[0]])
            attach = p + 0.2 * 38 * h
            for side, a in ((+1, f["wing_r"]), (-1, f["wing_l"])):
                ang = f["heading"] + 180 - side * a
                d = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
                s = attach + side * 4 * perp
                cv2.line(img, tuple(int(v) for v in s), tuple(int(v) for v in s + 34 * d),
                         WING_BGR, 2, cv2.LINE_AA)
        if labels:
            if male and "wing_r" in f:
                lines = [f"MALE  heading {f['heading']:.0f} deg",
                         f"wings: right {f['wing_r']:.0f}  left {f['wing_l']:.0f} deg"]
            elif male:
                lines = [f"MALE  heading {f['heading']:.0f} deg"]
            else:
                lines = [f"FEMALE  heading {f['heading']:.0f} deg"]
            for i, t in enumerate(reversed(lines) if male else lines):
                # male label sits above the fly, female label below it, so close flies do not overlap
                y = p[1] - 34 - 16 * i if male else p[1] + 40 + 16 * i
                org = _label_origin(img, t, p[0], y, 0.42)
                _text(img, t, org, col, 0.42)
    if hud:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (255, 255, 255), -1)
        cv2.putText(img, hud, (10, 19), FONT, 0.5, INK_BGR, 1, cv2.LINE_AA)
    return img
