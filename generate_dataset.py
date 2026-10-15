"""Generate the synthetic dataset and the held-out courtship video.

    python generate_dataset.py                 # 900 labelled frames + 600-frame video
    python generate_dataset.py --n-frames 300  # smaller / faster

Outputs (under data/):
    frames/frame_0000.png ...   independent labelled frames (train/test pool)
    annotations.json            ground truth for every frame
    video.avi, video_gt.json    held-out test video (different seed, never trained on)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=900)
    ap.add_argument("--video-frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ann = []
    for i in range(args.n_frames):
        flies = synth.sample_scene(rng)
        img = synth.render(flies, rng)
        name = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(args.out, "frames", name), img)
        ann.append({"file": name, "flies": [f.to_dict() for f in flies]})
    with open(os.path.join(args.out, "annotations.json"), "w") as fh:
        json.dump(ann, fh)
    print(f"wrote {args.n_frames} frames")

    # held-out video (own seed so it never overlaps the training frames)
    states = synth.simulate_video(args.video_frames, seed=args.seed + 1000)
    vrng = np.random.default_rng(args.seed + 2000)
    writer = cv2.VideoWriter(os.path.join(args.out, "video.avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), 30, (synth.IMG, synth.IMG), False)
    gt = []
    for flies in states:
        writer.write(synth.render(flies, vrng))
        gt.append([f.to_dict() for f in flies])
    writer.release()
    with open(os.path.join(args.out, "video_gt.json"), "w") as fh:
        json.dump(gt, fh)
    print(f"wrote video with {args.video_frames} frames")


if __name__ == "__main__":
    main()
