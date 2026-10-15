"""Train all four pipeline stages and evaluate them on a held-out third of the frames.

    python train_evaluate.py

Writes models/pipeline.joblib, results/metrics.json and the figures in results/.
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
import pipeline as P  # noqa: E402

DATA, RES, MOD = "data", "results", "models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
rng = np.random.default_rng(0)


def nearest_gt(st, flies):
    d = [np.hypot(st["cx"] - f["x"], st["cy"] - f["y"]) for f in flies]
    k = int(np.argmin(d))
    return flies[k] if d[k] < 25 else None


def count_in(c, flies):
    return sum(cv2.pointPolygonTest(c, (f["x"], f["y"]), False) >= 0 for f in flies)


# ---------------------------------------------------------------- build data
ann = json.load(open(os.path.join(DATA, "annotations.json")))
perm = rng.permutation(len(ann))
n_train = int(len(ann) * 2 / 3)
split = {int(i): ("train" if k < n_train else "test") for k, i in enumerate(perm)}

D = {s: {"count_X": [], "count_y": [], "sex_X": [], "sex_y": [],
         "or_X": {"M": [], "F": []}, "or_y": {"M": [], "F": []}, "or_phi": {"M": [], "F": []},
         "or_true": {"M": [], "F": []}, "or_patch": {"M": [], "F": []},
         "wing_X": [], "wing_y": [], "wing_patch": []} for s in ("train", "test")}

for i, a in enumerate(ann):
    gray = cv2.imread(os.path.join(DATA, "frames", a["file"]), cv2.IMREAD_GRAYSCALE)
    d = D[split[i]]
    flies = a["flies"]
    cnts = F.get_contours(gray)
    stats = [F.contour_stats(c) for c in cnts]
    singles = []
    for c, st in zip(cnts, stats):
        if st is None:
            continue
        n = count_in(c, flies)
        d["count_X"].append([st["area"]])
        d["count_y"].append(n)
        if n == 1:
            g = nearest_gt(st, flies)
            if g is not None:
                singles.append((c, st, g))
    if len(singles) == 2 and {s[2]["sex"] for s in singles} == {"M", "F"}:
        (c1, s1, g1), (c2, s2, g2) = singles
        for (sa, ga), (sb, gb) in (((s1, g1), (s2, g2)), ((s2, g2), (s1, g1))):   # swap augmentation
            d["sex_X"].append([sa["area"], sa["aspect"], sb["area"], sb["aspect"]])
            d["sex_y"].append(int(ga["sex"] == "M"))
    for c, st, g in singles:                                  # orientation
        patch = F.orientation_patch(gray, c, st)
        flip = int(np.cos(np.deg2rad(F.ang_diff(g["heading"], st["phi"]))) < 0)
        d["or_X"][g["sex"]].append(F.orientation_feature(patch))
        d["or_y"][g["sex"]].append(flip)
        d["or_phi"][g["sex"]].append(st["phi"])
        d["or_true"][g["sex"]].append(g["heading"])
        d["or_patch"][g["sex"]].append(patch)
        if g["sex"] == "M" and flip is not None:              # wings: trained on true heading
            r, l = F.wing_patches(gray, st["cx"], st["cy"], g["heading"])
            d["wing_X"] += [F.wing_feature(r), F.wing_feature(l)]
            d["wing_y"] += [g["wing_r"], g["wing_l"]]
            d["wing_patch"] += [r, l]

for s in D:
    for k in ("count_X", "count_y", "sex_X", "sex_y", "wing_X", "wing_y"):
        D[s][k] = np.array(D[s][k])
    for sx in "MF":
        for k in ("or_X", "or_y", "or_phi", "or_true"):
            D[s][k][sx] = np.array(D[s][k][sx])

tr, te = D["train"], D["test"]
pct = lambda x: 100.0 * float(x)           # fraction -> percent
metrics = {"dataset": {"total_frames": len(ann), "training_frames": n_train,
                       "test_frames": len(ann) - n_train}}

# ---------------------------------------------------------------- stage 1
count = P.make_count_model().fit(tr["count_X"], tr["count_y"])
metrics["stage1_fly_counting"] = {
    "model": "Decision tree",
    "training_error_percent": pct(1 - count.score(tr["count_X"], tr["count_y"])),
    "test_error_percent": pct(1 - count.score(te["count_X"], te["count_y"])),
    "training_samples": len(tr["count_y"]), "test_samples": len(te["count_y"])}

# ---------------------------------------------------------------- stage 2
sex = P.make_sex_model().fit(tr["sex_X"], tr["sex_y"])
metrics["stage2_sex_identification"] = {
    "model": "Logistic regression",
    "training_error_percent": pct(1 - sex.score(tr["sex_X"], tr["sex_y"])),
    "test_error_percent": pct(1 - sex.score(te["sex_X"], te["sex_y"])),
    "training_samples": len(tr["sex_y"]), "test_samples": len(te["sex_y"])}
sex_nofeat = P.make_sex_model().fit(tr["sex_X"][:, [0, 2]], tr["sex_y"])
metrics["stage2_sex_identification"]["test_error_percent_if_area_only"] = pct(
    1 - sex_nofeat.score(te["sex_X"][:, [0, 2]], te["sex_y"]))

# ---------------------------------------------------------------- stage 3
orient = {}
for sx in "MF":
    # augmentation: rotate every patch by 180 deg and invert its label
    Xa = np.vstack([tr["or_X"][sx]] + [F.orientation_feature(np.ascontiguousarray(p[::-1, ::-1]))[None]
                                        for p in tr["or_patch"][sx]])
    ya = np.concatenate([tr["or_y"][sx], 1 - tr["or_y"][sx]])
    m = P.make_orient_model(15).fit(Xa, ya)
    orient[sx] = m
    pred = m.predict(te["or_X"][sx])
    head = (te["or_phi"][sx] + 180 * pred) % 360
    err = np.abs(F.ang_diff(head, te["or_true"][sx]))
    metrics["stage3_orientation_" + ("male" if sx == "M" else "female")] = {
        "model": "HOG + PCA + logistic regression",
        "training_error_percent": pct(1 - m.score(tr["or_X"][sx], tr["or_y"][sx])),
        "test_error_percent": pct(np.mean(pred != te["or_y"][sx])),
        "heading_error_degrees_average": float(err.mean()),
        "heading_error_degrees_median": float(np.median(err)),
        "heading_within_10_degrees_percent": pct(np.mean(err < 10)),
        "training_samples": len(tr["or_y"][sx]), "test_samples": len(te["or_y"][sx])}

# ---------------------------------------------------------------- stage 4
wing = P.make_wing_model(40).fit(tr["wing_X"], tr["wing_y"])
res_tr = wing.predict(tr["wing_X"]) - tr["wing_y"]
pw = np.clip(wing.predict(te["wing_X"]), 0, 90)
res_te = pw - te["wing_y"]
metrics["stage4_wing_angle"] = {
    "model": "HOG + PCA + linear regression",
    "training_error_degrees_spread_std": float(res_tr.std()),
    "test_error_degrees_spread_std": float(res_te.std()),
    "test_error_degrees_average": float(np.abs(res_te).mean()),
    "test_bias_degrees": float(res_te.mean()),
    "true_wing_angle_spread_std_degrees": float(te["wing_y"].std()),
    "r_squared": float(1 - (res_te ** 2).sum() / ((te["wing_y"] - te["wing_y"].mean()) ** 2).sum()),
    "training_samples": len(tr["wing_y"]), "test_samples": len(te["wing_y"])}

# ---------------------------------------------------------------- study: number of PCA components
ncs = [1, 2, 5, 10, 15, 25, 40, 60]
abl = {"n_components": ncs, "orient_M": [], "orient_F": [], "wing_std": []}
for nc in ncs:
    for sx in "MF":
        Xa = np.vstack([tr["or_X"][sx]] + [F.orientation_feature(np.ascontiguousarray(p[::-1, ::-1]))[None]
                                            for p in tr["or_patch"][sx]])
        ya = np.concatenate([tr["or_y"][sx], 1 - tr["or_y"][sx]])
        m = P.make_orient_model(nc).fit(Xa, ya)
        abl[f"orient_{sx}"].append(float(np.mean(m.predict(te["or_X"][sx]) != te["or_y"][sx])))
    m = P.make_wing_model(min(nc, len(tr["wing_y"]) - 1)).fit(tr["wing_X"], tr["wing_y"])
    abl["wing_std"].append(float((np.clip(m.predict(te["wing_X"]), 0, 90) - te["wing_y"]).std()))
metrics["pca_size_study"] = {
    "number_of_components": ncs,
    "orientation_male_test_error_percent": [pct(v) for v in abl["orient_M"]],
    "orientation_female_test_error_percent": [pct(v) for v in abl["orient_F"]],
    "wing_angle_test_error_degrees_spread_std": abl["wing_std"]}

# ---------------------------------------------------------------- save
pipe = P.FlyPipeline(count, sex, orient["M"], orient["F"], wing)
joblib.dump(pipe, os.path.join(MOD, "pipeline.joblib"))
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

# ---------------------------------------------------------------- figures
BLUE, RED, GREY = "#1f5fbf", "#c8372d", "#555555"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# PCA scatter (female orientation)
pca = orient["F"].steps[0][1]
Z = pca.transform(te["or_X"]["F"])
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for lab, col, mk, nm in ((0, BLUE, "o", "head up"), (1, RED, "x", "head down")):
    s = te["or_y"]["F"] == lab
    ax.scatter(Z[s, 0], Z[s, 1], c=col, marker=mk, s=14, label=nm)
ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.legend(frameon=False)
ax.set_title("Orientation: HOG projected on 2 PCs (female)")
fig.tight_layout(); fig.savefig(os.path.join(RES, "orientation_pca.png"), dpi=200); plt.close(fig)

# wing angle vs PC1 and predicted vs true
pca_w = wing.steps[0][1]
Zw = pca_w.transform(te["wing_X"])
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax[0].scatter(Zw[:, 0], te["wing_y"], s=8, c=BLUE)
ax[0].set_xlabel("PCA component 1"); ax[0].set_ylabel("true wing angle (deg)")
ax[0].set_title("Wing angle vs first PC")
ax[1].scatter(te["wing_y"], pw, s=8, c=BLUE)
ax[1].plot([0, 90], [0, 90], c=GREY, lw=1)
ax[1].set_xlabel("true wing angle (deg)"); ax[1].set_ylabel("predicted (deg)")
ax[1].set_title(f"Test: error spread {metrics['stage4_wing_angle']['test_error_degrees_spread_std']:.2f} deg")
fig.tight_layout(); fig.savefig(os.path.join(RES, "wing_angle_fit.png"), dpi=200); plt.close(fig)

# ablation
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax[0].plot(ncs, np.array(abl["orient_M"]) * 100, "-o", c=BLUE, label="male")
ax[0].plot(ncs, np.array(abl["orient_F"]) * 100, "-s", c=RED, label="female")
ax[0].set_xscale("log"); ax[0].set_xlabel("PCA components"); ax[0].set_ylabel("flip test error (%)")
ax[0].set_ylim(-0.5, 5); ax[0].legend(frameon=False); ax[0].set_title("Orientation stage")
ax[1].plot(ncs, abl["wing_std"], "-o", c=BLUE)
ax[1].set_xscale("log"); ax[1].set_xlabel("PCA components"); ax[1].set_ylabel("wing error std (deg)")
ax[1].set_title("Wing-angle stage")
fig.tight_layout(); fig.savefig(os.path.join(RES, "pca_ablation.png"), dpi=200); plt.close(fig)

# explained variance
fig, ax = plt.subplots(figsize=(3.4, 2.8))
full = P.make_wing_model(100).fit(tr["wing_X"], tr["wing_y"]).steps[0][1]
ax.plot(np.cumsum(full.explained_variance_ratio_), c=BLUE)
ax.set_xlabel("PCA components"); ax.set_ylabel("explained variance ratio")
ax.set_title("Wing HOG: cumulative variance")
fig.tight_layout(); fig.savefig(os.path.join(RES, "wing_variance.png"), dpi=200); plt.close(fig)

d = metrics["dataset"]
m1, m2 = metrics["stage1_fly_counting"], metrics["stage2_sex_identification"]
m3m, m3f = metrics["stage3_orientation_male"], metrics["stage3_orientation_female"]
m4 = metrics["stage4_wing_angle"]
bar = "=" * 78
print(f"\n{bar}\n  TRAINING RESULTS  ({d['training_frames']} training frames, {d['test_frames']} unseen test frames)\n{bar}")
print(f"  {'Stage':<34}{'Model':<32}{'Training':>10}{'Test':>10}")
print(f"  {'-' * 76}")
for name, mm, model in (("Stage 1 - Fly counting", m1, "Decision tree"),
                        ("Stage 2 - Sex identification", m2, "Logistic regression"),
                        ("Stage 3 - Orientation (male)", m3m, "HOG+PCA+Logistic regr."),
                        ("Stage 3 - Orientation (female)", m3f, "HOG+PCA+Logistic regr.")):
    print(f"  {name:<34}{model:<32}{mm['training_error_percent']:>9.2f}%{mm['test_error_percent']:>9.2f}%")
print(f"  {'Stage 4 - Wing angles':<34}{'HOG+PCA+Linear regr.':<32}"
      f"{m4['training_error_degrees_spread_std']:>4.2f} deg".rjust(10) + f"{m4['test_error_degrees_spread_std']:>4.2f} deg".rjust(10))
print(f"\n  Columns show the ERROR on training / test data (lower is better).")
print(f"  Stages 1-3: % of wrong decisions.  Stage 4: spread (std dev) of the wing-angle error in degrees.")
print(f"  Wing-angle test error: average {m4['test_error_degrees_average']:.2f} deg, "
      f"R-squared {m4['r_squared']:.2f}")
print(f"  Heading error on test frames: male {m3m['heading_error_degrees_average']:.2f} deg, "
      f"female {m3f['heading_error_degrees_average']:.2f} deg (average)")
print(f"\n  Saved: models/pipeline.joblib, results/metrics.json and figures in results/\n{bar}")
