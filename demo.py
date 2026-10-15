"""Run the trained pipeline on a video, show/save the annotated result and print a report.

    python demo.py                          # analyse data/video.avi, save results/demo_output.avi
    python demo.py --show                   # also open a live window (press q to quit)
    python demo.py --video my.avi --no-gt   # your own video (no ground truth available)
    python demo.py --max-frames 100

For every frame the pipeline finds the sex, position and heading (0-360 deg) of both flies and
the right/left wing angles (0-90 deg) of the male. At the end a plain-English report is printed
and saved to results/demo_summary.json. For a graphical view run `python gui.py` instead.
"""
import argparse
import json
import os
import sys
import time

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
import pipeline as P  # noqa: E402
from evaluation import build_summary, format_report, frame_errors, summarize_errors  # noqa: E402
from visualize import draw_overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fruit fly video analysis demo")
    ap.add_argument("--video", default=os.path.join(BASE, "data", "video.avi"))
    ap.add_argument("--gt", default=os.path.join(BASE, "data", "video_gt.json"),
                    help="ground-truth labels of the video (json)")
    ap.add_argument("--no-gt", action="store_true", help="skip the accuracy check (unlabelled video)")
    ap.add_argument("--model", default=os.path.join(BASE, "models", "pipeline.joblib"))
    ap.add_argument("--out", default=os.path.join(BASE, "results", "demo_output.avi"))
    ap.add_argument("--show", action="store_true", help="show a live window while processing")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("Trained model not found. Run `python train_evaluate.py` first.")
    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}\nRun `python generate_dataset.py` first.")
    pipe = joblib.load(args.model)
    TARGET_SIZE = (512, 512)

    def normalize_frame(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
        if g.shape == TARGET_SIZE:
            return g
        fh, fw = g.shape[:2]
        if fh == fw:
            return cv2.resize(g, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        min_dim = min(fh, fw)
        y0 = (fh - min_dim) // 2
        x0 = (fw - min_dim) // 2
        return cv2.resize(g[y0:y0 + min_dim, x0:x0 + min_dim], TARGET_SIZE, interpolation=cv2.INTER_AREA)

    gt = None
    if not args.no_gt:
        if os.path.abspath(args.video) == os.path.abspath(os.path.join(BASE, "data", "video.avi")) and os.path.exists(args.gt):
            gt = json.load(open(args.gt))
        else:
            cand = os.path.splitext(args.video)[0] + "_gt.json"
            if os.path.exists(cand):
                gt = json.load(open(cand))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res_dir = os.path.join(BASE, "results")
    os.makedirs(res_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n_total = min(n_total, args.max_frames)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read the video.")
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"MJPG"), source_fps, TARGET_SIZE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # warm-up so one-off start-up costs do not distort the timing
    g0 = normalize_frame(frame)
    for _ in range(3):
        pipe.process(g0)
    pipe.reset_timers()

    print(f"Analysing {n_total} frames of {os.path.basename(args.video)} ...")
    io_time, last, k = 0.0, [], 0
    status = {"ok": 0, "merged": 0, "unresolved": 0}
    rows = []
    series = {"t": [], "pred_r": [], "pred_l": [], "true_r": [], "true_l": []}
    while k < n_total:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        gray = normalize_frame(frame)
        io_time += time.perf_counter() - t0
        res = pipe.process(gray)
        status[res["status"]] += 1
        if res["status"] == "ok":
            last = res["flies"]
            m = next((f for f in res["flies"] if f["sex"] == "M" and "wing_r" in f), None)
            if m:
                series["t"].append(k / source_fps)
                series["pred_r"].append(m["wing_r"])
                series["pred_l"].append(m["wing_l"])
                if gt is not None and k < len(gt):
                    rows += frame_errors(res["flies"], gt[k])
                    gm = next((q for q in gt[k] if q["sex"] == "M"), None)
                    series["true_r"].append(gm["wing_r"] if gm else None)
                    series["true_l"].append(gm["wing_l"] if gm else None)
        elapsed = io_time + sum(pipe.times.values())
        fps = (k + 1) / elapsed if elapsed > 0 else 0
        hud = f"Frame {k + 1}/{n_total}    {fps:.0f} frames/s"
        if res["status"] != "ok":
            hud += f"    [{res['status']}: showing last result]"
        out = draw_overlay(gray, last, hud)
        writer.write(out)
        if args.show:
            cv2.imshow("Fruit fly analysis (press q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        k += 1
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    accuracy = summarize_errors(rows) if rows else None
    summary = build_summary(os.path.basename(args.video), k, source_fps, pipe.times, io_time, status, accuracy)
    print()
    print(format_report(summary))

    if series["t"]:
        t = np.array(series["t"])
        fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
        has_truth = any(v is not None for v in series["true_r"])
        for a, key, ttl in ((ax[0], "r", "Right wing"), (ax[1], "l", "Left wing")):
            if has_truth:
                a.plot(t, series["true_" + key], c="#555555", lw=1.0, label="true angle")
            a.plot(t, series["pred_" + key], c="#1f5fbf", lw=0.9, label="predicted angle")
            a.set_ylabel(f"{ttl} (deg)")
            a.spines[["top", "right"]].set_visible(False)
        ax[0].legend(frameon=False, ncol=2, loc="upper left")
        ax[1].set_xlabel("time (seconds)")
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, "wing_timeseries.png"), dpi=200)
        plt.close(fig)

    if series["t"]:
        data_to_save = {"time_seconds": series["t"],
                        "right_wing_predicted_deg": series["pred_r"],
                        "left_wing_predicted_deg": series["pred_l"]}
        if any(v is not None for v in series["true_r"]):
            data_to_save["right_wing_true_deg"] = series["true_r"]
            data_to_save["left_wing_true_deg"] = series["true_l"]
        json.dump(data_to_save, open(os.path.join(res_dir, "wing_series.json"), "w"))
    json.dump(summary, open(os.path.join(res_dir, "demo_summary.json"), "w"), indent=2)
    print(f"\nSaved: {os.path.relpath(args.out, BASE)}  (annotated video)")
    print(f"Saved: results/demo_summary.json  (all numbers above)")


if __name__ == "__main__":
    main()
