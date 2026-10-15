"""Compare pipeline output with ground truth and print a plain-English report."""
import numpy as np

import features as F
import pipeline as P


def frame_errors(pred_flies, truth_flies):
    """Match each predicted fly to the nearest true fly and measure the errors."""
    rows = []
    for f in pred_flies:
        g = min(truth_flies, key=lambda q: np.hypot(q["x"] - f["x"], q["y"] - f["y"]))
        row = {"pred": f, "truth": g,
               "sex_correct": g["sex"] == f["sex"],
               "position_error_px": float(np.hypot(g["x"] - f["x"], g["y"] - f["y"])),
               "heading_error_deg": float(abs(F.ang_diff(f["heading"], g["heading"])))}
        if f["sex"] == "M" and g["sex"] == "M":
            row["wing_errors_deg"] = [f["wing_r"] - g["wing_r"], f["wing_l"] - g["wing_l"]]
        rows.append(row)
    return rows


def summarize_errors(rows):
    """Aggregate per-fly errors into named, unit-carrying numbers."""
    head = np.array([r["heading_error_deg"] for r in rows])
    wing = np.array([r["wing_errors_deg"] for r in rows if "wing_errors_deg" in r]).ravel()
    return {
        "flies_checked": len(rows),
        "sex_correct_percent": 100 * float(np.mean([r["sex_correct"] for r in rows])),
        "position_error_pixels_average": float(np.mean([r["position_error_px"] for r in rows])),
        "heading_error_degrees_average": float(head.mean()),
        "heading_error_degrees_median": float(np.median(head)),
        "heading_within_15_degrees_percent": 100 * float(np.mean(head < 15)),
        "wing_angle_error_degrees_average": float(np.abs(wing).mean()),
        "wing_angle_error_degrees_spread_std": float(wing.std()),
        "wing_angle_bias_degrees": float(wing.mean()),
    }


def build_summary(video, frames, source_fps, times_s, io_s, status, accuracy):
    """Assemble the JSON-able summary with self-explanatory key names."""
    n = max(frames, 1)
    ms = {P.STAGE_JSON_KEYS[s]: 1000 * times_s[s] / n for s in P.STAGES}
    ms = {"reading_video_frame": 1000 * io_s / n, **ms}
    total = sum(ms.values())
    out = {
        "video_file": video,
        "frames_processed": frames,
        "source_video_fps": source_fps,
        "speed": {"frames_per_second": 1000 / total, "milliseconds_per_frame": total,
                  "times_faster_than_source_video": (1000 / total) / source_fps},
        "time_per_frame_milliseconds": ms,
        "time_share_percent": {k: 100 * v / total for k, v in ms.items()},
        "frame_outcomes": {"analysed_successfully": status["ok"],
                           "skipped_flies_touching": status["merged"],
                           "skipped_could_not_resolve": status["unresolved"]},
    }
    if accuracy:
        out["accuracy_vs_ground_truth"] = accuracy
    return out


def format_report(s):
    """Human-readable text report for the terminal."""
    sp, ms, sh, fo = s["speed"], s["time_per_frame_milliseconds"], s["time_share_percent"], s["frame_outcomes"]
    bar = "=" * 66
    names = {"reading_video_frame": "Reading the video frame",
             "stage1_fly_counting": P.STAGE_LABELS["FlyCount"],
             "stage2_sex_identification": P.STAGE_LABELS["Sex"],
             "stage3_orientation": P.STAGE_LABELS["Orientation"],
             "stage4_wing_angles": P.STAGE_LABELS["WingAngle"]}
    L = [bar, "  FRUIT FLY VIDEO ANALYSIS - RESULTS", bar,
         f"  Video analysed        : {s['video_file']}  ({s['frames_processed']} frames)",
         "",
         "  SPEED",
         f"    Analysis speed      : {sp['frames_per_second']:.1f} frames per second "
         f"({sp['milliseconds_per_frame']:.2f} ms per frame)",
         f"    Compared to video   : {sp['times_faster_than_source_video']:.1f}x faster than the "
         f"{s['source_video_fps']} FPS source video  ->  real-time"
         if sp["frames_per_second"] > s["source_video_fps"] else
         f"    Compared to video   : {sp['times_faster_than_source_video']:.1f}x the "
         f"{s['source_video_fps']} FPS source video (NOT real-time)",
         "    Where the time goes (per frame):"]
    for k, v in ms.items():
        L.append(f"      {names[k]:<34}{v:6.2f} ms  ({sh[k]:4.1f} %)")
    L += ["",
          "  FRAME OUTCOMES",
          f"    Analysed successfully             : {fo['analysed_successfully']} of {s['frames_processed']}",
          f"    Skipped (flies touching)          : {fo['skipped_flies_touching']}",
          f"    Skipped (could not resolve flies) : {fo['skipped_could_not_resolve']}"]
    a = s.get("accuracy_vs_ground_truth")
    if a:
        L += ["",
              f"  ACCURACY vs GROUND TRUTH  ({a['flies_checked']} flies checked)",
              f"    Sex identified correctly            : {a['sex_correct_percent']:.2f} %",
              f"    Position error (average)            : {a['position_error_pixels_average']:.2f} pixels",
              f"    Heading error (average)             : {a['heading_error_degrees_average']:.2f} degrees",
              f"    Heading error (median)              : {a['heading_error_degrees_median']:.2f} degrees",
              f"    Headings within 15 degrees          : {a['heading_within_15_degrees_percent']:.2f} %",
              f"    Wing-angle error (average)          : {a['wing_angle_error_degrees_average']:.2f} degrees",
              f"    Wing-angle error (spread, std dev)  : {a['wing_angle_error_degrees_spread_std']:.2f} degrees",
              f"    Wing-angle bias (+ = over-estimates): {a['wing_angle_bias_degrees']:+.2f} degrees"]
    L.append(bar)
    return "\n".join(L)
