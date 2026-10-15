# Team Contributions

| Member | SRN |
|---|---|
| Sriranga Bhardwaj | PES1UG24AM286 |
| Sumuk D Hegde | PES1UG24AM293 |
| Shraddha | PES1UG25AM812 |

---

## Contributions by Area

| Area | Done by |
|---|---|
| Problem understanding, reading the reference paper (S. Herbst, Stanford CS229 2018) | Sriranga Bhardwaj, Sumuk D Hegde, Shraddha |
| Data simulator (`src/synth.py`) — fly renderer, courtship trajectories, noise/blur/debris | Sumuk D Hegde |
| Dataset generation script (`generate_dataset.py`) — 900 labelled frames + 600-frame test video | Sumuk D Hegde |
| Feature extraction (`src/features.py`) — thresholding, contours, image moments, HOG, rotation | Sriranga Bhardwaj |
| Stage 1 model — fly counting via decision tree on blob area | Sriranga Bhardwaj |
| Stage 2 model — sex identification via logistic regression on area & aspect ratio | Sriranga Bhardwaj |
| Stage 3 model — orientation (HOG → PCA → logistic regression for head vs. tail) | Sumuk D Hegde |
| Stage 4 model — wing-angle regression (HOG → PCA → linear regression, mirrored) | Shraddha |
| Pipeline integration (`src/pipeline.py`) — `FlyPipeline.process()` end-to-end | Sriranga Bhardwaj |
| Model training & evaluation (`train_evaluate.py`, `src/evaluation.py`) | Sumuk D Hegde, Shraddha |
| PCA ablation study and performance figures (`make_figures.py`) | Shraddha |
| Demo script (`demo.py`) — CLI video annotator, plain-English report | Sriranga Bhardwaj |
| Graphical interface (`gui.py`) — live player, analysis tabs, wing-angle plot | Sriranga Bhardwaj, Sumuk D Hegde |
| Visualisation overlay (`src/visualize.py`) — outlines, arrows, wing lines | Shraddha |
| PDF write-up generation (`make_report.py`, `Fruit_Fly_Video_Analysis_Writeup.pdf`) | Shraddha, Sumuk D Hegde |
| README documentation and repo setup | Sriranga Bhardwaj |

---

## Individual Summaries

**Sriranga Bhardwaj (PES1UG24AM286)**
Led the overall pipeline design and integration. Implemented feature extraction (`src/features.py`), Stages 1 and 2 (fly counting and sex identification), the end-to-end `FlyPipeline` class, the CLI demo script, the GUI, and project documentation.

**Sumuk D Hegde (PES1UG24AM293)**
Built the fly simulator and data generation pipeline (`src/synth.py`, `generate_dataset.py`), implemented Stage 3 (orientation model using HOG + PCA + logistic regression), contributed to the GUI and evaluation pipeline, and co-authored the write-up.

**Shraddha (PES1UG25AM812)**
Implemented Stage 4 (wing-angle estimation using HOG + PCA + linear regression), the visualisation overlay (`src/visualize.py`), the PCA ablation study and result figures, and led the PDF write-up (`make_report.py`).
