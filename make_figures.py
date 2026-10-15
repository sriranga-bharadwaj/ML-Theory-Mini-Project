"""Build the pipeline-overview and timing figures used in the write-up / slides.

    python make_figures.py        (run after train_evaluate.py and demo.py)
"""
import json
import os
import sys

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import features as F  # noqa: E402

BLUE, RED, GREY = "#1f5fbf", "#c8372d", "#555555"
plt.rcParams.update({"font.size": 9})

pipe = joblib.load("models/pipeline.joblib")
cap = cv2.VideoCapture("data/video.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
_, fr = cap.read()
gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
res = pipe.process(gray)
male = next(f for f in res["flies"] if f["sex"] == "M")
fem = next(f for f in res["flies"] if f["sex"] == "F")
cx, cy = int((male["x"] + fem["x"]) / 2), int((male["y"] + fem["y"]) / 2)
x0, y0 = max(cx - 110, 0), max(cy - 110, 0)
crop = lambda im: im[y0:y0 + 220, x0:x0 + 220]

# stage views
body = (gray < F.BODY_T).astype(np.uint8) * 255
cont = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(cont, [male["contour"]], -1, (200, 110, 30), 2)
cv2.drawContours(cont, [fem["contour"]], -1, (50, 55, 200), 2)
st_m = F.contour_stats(male["contour"])
st_f = F.contour_stats(fem["contour"])
p_m = F.orientation_patch(gray, male["contour"], st_m)
p_f = F.orientation_patch(gray, fem["contour"], st_f)
wr, wl = F.wing_patches(gray, male["x"], male["y"], male["heading"])

fig, ax = plt.subplots(1, 6, figsize=(10.5, 2.9), gridspec_kw={"width_ratios": [1.3, 1.3, 1.3, 0.55, 0.9, 1.3]})
ax[0].imshow(crop(gray), cmap="gray"); ax[0].set_title("1  input frame")
ax[1].imshow(crop(body), cmap="gray"); ax[1].set_title("2  threshold")
ax[2].imshow(crop(cv2.cvtColor(cont, cv2.COLOR_BGR2RGB))); ax[2].set_title("3  contours + sex")
ax[3].imshow(np.hstack([p_f, np.full((128, 4), 255, np.uint8), p_m]), cmap="gray")
ax[3].set_title("4  upright\n(F | M)")
ax[4].imshow(np.hstack([wl, np.full((96, 4), 255, np.uint8), wr]), cmap="gray")
ax[4].set_title("5  wing ROIs\n(L | R)")
ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for f in res["flies"]:
    col = (200, 110, 30) if f["sex"] == "M" else (50, 55, 200)
    p = np.array([f["x"], f["y"]]); h = np.array([np.cos(np.deg2rad(f["heading"])), np.sin(np.deg2rad(f["heading"]))])
    cv2.arrowedLine(ov, tuple(int(v) for v in p), tuple(int(v) for v in p + 40 * h), col, 2, cv2.LINE_AA, tipLength=0.3)
    cv2.drawContours(ov, [f["contour"]], -1, col, 1, cv2.LINE_AA)
ax[5].imshow(crop(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)))
ax[5].set_title(f"6  output\nwings R {male['wing_r']:.0f}° / L {male['wing_l']:.0f}°")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("results/pipeline_overview.png", dpi=220)
plt.close(fig)

# timing
s = json.load(open("results/demo_summary.json"))
order = ["reading_video_frame", "stage1_fly_counting", "stage2_sex_identification",
         "stage3_orientation", "stage4_wing_angles"]
labels = ["Reading video", "1 Fly counting", "2 Sex ID", "3 Orientation", "4 Wing angles"]
vals = [s["time_per_frame_milliseconds"][k] for k in order]
fig, ax = plt.subplots(figsize=(4.2, 2.3))
bars = ax.barh(labels[::-1], vals[::-1], color=BLUE)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f} ms", va="center", fontsize=8)
ax.set_xlabel("milliseconds per frame"); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(vals) * 1.25)
ax.set_title(f"{s['speed']['frames_per_second']:.0f} frames per second "
             f"({s['speed']['milliseconds_per_frame']:.1f} ms per frame)", fontsize=9)
fig.tight_layout()
fig.savefig("results/timing.png", dpi=220)
plt.close(fig)
print("figures written")

# ---------------------------------------------------------------- dataset samples
ann = json.load(open("data/annotations.json"))
frames_dir = "data/frames" if os.path.exists("data/frames/frame_0000.png") else "data/sample_frames"
picks = [(0, "Flies apart, one wing out"), (1, "Flies touching: one merged blob"), (5, "Both wings extended")]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
for a, (i, title) in zip(ax, picks):
    g = cv2.imread(f"{frames_dir}/{ann[i]['file']}", cv2.IMREAD_GRAYSCALE)
    fl = ann[i]["flies"]
    mx, my = np.mean([f["x"] for f in fl]), np.mean([f["y"] for f in fl])
    d = np.hypot(fl[0]["x"] - fl[1]["x"], fl[0]["y"] - fl[1]["y"])
    half = int(max(95, d / 2 + 70))
    x0, y0 = int(np.clip(mx - half, 0, 512 - 2 * half)), int(np.clip(my - half, 0, 512 - 2 * half))
    a.imshow(g[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap="gray", vmin=0, vmax=255)
    for f, other in ((fl[0], fl[1]), (fl[1], fl[0])):
        # put each label on the side facing away from the other fly so they never cover a body
        u = np.array([f["x"] - other["x"], f["y"] - other["y"]])
        u = u / max(np.hypot(*u), 1e-6)
        lx = np.clip(f["x"] - x0 + 46 * u[0], 25, 2 * half - 25)
        ly = np.clip(f["y"] - y0 + 46 * u[1], 18, 2 * half - 18)
        a.text(lx, ly, "male" if f["sex"] == "M" else "female", color=BLUE if f["sex"] == "M" else RED,
               ha="center", va="center", fontsize=9, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    a.set_title(title, fontsize=10)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(pad=1.2)
fig.savefig("results/dataset_samples.png", dpi=220)
plt.close(fig)
print("dataset sample figure written")

# ---------------------------------------------------------------- compact wing chart for the write-up
if os.path.exists("results/wing_series.json"):
    w = json.load(open("results/wing_series.json"))
    t = np.array(w["time_seconds"])
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 3.1), sharex=True)
    for a, side in zip(ax, ("right", "left")):
        a.plot(t, w[f"{side}_wing_true_deg"], c="#555555", lw=1.2, label="true")
        a.plot(t, w[f"{side}_wing_predicted_deg"], c=BLUE, lw=1.0, label="predicted")
        a.set_ylabel(f"{side} wing\n(degrees)", fontsize=9)
        a.tick_params(labelsize=9)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    ax[1].set_xlabel("time (seconds)", fontsize=9)
    fig.tight_layout(pad=0.6)
    fig.savefig("results/wing_timeseries_compact.png", dpi=250)
    plt.close(fig)
    print("compact wing chart written")

# ---------------------------------------------------------------- three frames of the annotated demo video
cap = cv2.VideoCapture("results/demo_output.avi")
if cap.isOpened():
    strip = []
    for k in range(300):
        ok, f = cap.read()
        if not ok:
            break
        if k in (60, 200, 290):
            strip.append(f[40:472, 40:472])
    if len(strip) == 3:
        cv2.imwrite("results/demo_frames.png", np.hstack(strip))
        print("demo frames strip written")
