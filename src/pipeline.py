"""Four-stage real-time analysis pipeline.

    Stage 1  FlyCount      decision tree on contour area        -> 0 / 1 / 2 flies
    Stage 2  Sex           standardise + logistic regression    -> which contour is male
    Stage 3  Orientation   moments + HOG + PCA + log. reg.      -> heading 0-360 deg
    Stage 4  WingAngle     HOG + PCA + linear regression        -> wing angles 0-90 deg
"""
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import features as F

STAGES = ["FlyCount", "Sex", "Orientation", "WingAngle"]

# Plain-English names used in every printed report, JSON file and the GUI
STAGE_LABELS = {
    "FlyCount": "Stage 1 - Fly counting",
    "Sex": "Stage 2 - Sex identification",
    "Orientation": "Stage 3 - Orientation (heading)",
    "WingAngle": "Stage 4 - Wing angles",
}
STAGE_JSON_KEYS = {
    "FlyCount": "stage1_fly_counting",
    "Sex": "stage2_sex_identification",
    "Orientation": "stage3_orientation",
    "WingAngle": "stage4_wing_angles",
}


def make_count_model():
    return DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=5, random_state=0)


def make_sex_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))


def make_orient_model(n_components=15):
    return make_pipeline(PCA(n_components, random_state=0), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000))


def make_wing_model(n_components=40):
    return make_pipeline(PCA(n_components, random_state=0), LinearRegression())


class FlyPipeline:
    def __init__(self, count=None, sex=None, orient_m=None, orient_f=None, wing=None):
        self.count, self.sex, self.wing = count, sex, wing
        self.orient = {"M": orient_m, "F": orient_f}
        self.times = {s: 0.0 for s in STAGES}
        self.n_frames = 0

    def reset_timers(self):
        self.times = {s: 0.0 for s in STAGES}
        self.n_frames = 0

    # ------------------------------------------------------------------
    def process(self, gray):
        """Analyse one grayscale frame.

        Returns {"flies": [...], "status": "ok" | "merged" | "unresolved"}.
        Each fly dict has sex, x, y, heading and (males) wing_r / wing_l.
        """
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.shape != (512, 512):
            h, w = gray.shape[:2]
            if h == w:
                gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
            else:
                min_dim = min(h, w)
                y0 = (h - min_dim) // 2
                x0 = (w - min_dim) // 2
                gray = cv2.resize(gray[y0:y0 + min_dim, x0:x0 + min_dim], (512, 512), interpolation=cv2.INTER_AREA)

        T = self.times
        t0 = time.perf_counter()
        cnts = F.get_contours(gray)
        stats = [F.contour_stats(c) for c in cnts]
        keep = [i for i, s in enumerate(stats) if s is not None]
        cnts, stats = [cnts[i] for i in keep], [stats[i] for i in keep]
        cls = self.count.predict(np.array([[s["area"]] for s in stats])) if stats else []
        T["FlyCount"] += time.perf_counter() - t0
        self.n_frames += 1

        ones = [i for i, k in enumerate(cls) if k == 1]
        if any(k == 2 for k in cls):
            return {"flies": [], "status": "merged"}
        if len(ones) > 2:                                # stray debris: keep two largest
            ones = sorted(ones, key=lambda i: -stats[i]["area"])[:2]
        if len(ones) != 2:
            return {"flies": [], "status": "unresolved"}

        t0 = time.perf_counter()
        a, b = ones
        x = np.array([[stats[a]["area"], stats[a]["aspect"], stats[b]["area"], stats[b]["aspect"]]])
        male_first = self.sex.predict(x)[0] == 1
        idx = {"M": a, "F": b} if male_first else {"M": b, "F": a}
        T["Sex"] += time.perf_counter() - t0

        flies = []
        for sex in ("F", "M"):
            t0 = time.perf_counter()
            i = idx[sex]
            st = stats[i]
            patch = F.orientation_patch(gray, cnts[i], st)
            flip = self.orient[sex].predict(F.orientation_feature(patch)[None])[0]
            heading = (st["phi"] + 180.0 * flip) % 360.0
            T["Orientation"] += time.perf_counter() - t0
            fly = {"sex": sex, "x": st["cx"], "y": st["cy"], "heading": heading, "contour": cnts[i]}
            if sex == "M":
                t0 = time.perf_counter()
                right, left = F.wing_patches(gray, st["cx"], st["cy"], heading)
                X = np.stack([F.wing_feature(right), F.wing_feature(left)])
                w = np.clip(self.wing.predict(X), 0, 90)
                fly["wing_r"], fly["wing_l"] = float(w[0]), float(w[1])
                T["WingAngle"] += time.perf_counter() - t0
            flies.append(fly)
        return {"flies": flies, "status": "ok"}
