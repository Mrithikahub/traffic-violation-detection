# Helmet Detection — Evaluation and Decision Not to Ship

**Status: NOT IMPLEMENTED — documented limitation.**

Helmet / no-helmet detection was evaluated as a possible violation layer using
existing pretrained models (no training was attempted, by design). Two candidate
models were tested against our own footage and **neither performed well enough to
ship**. This document records what was tested and why the feature was dropped, so
the decision is auditable rather than a silent omission.

- **Date:** 2026-08-18
- **Test footage:** `teammate_video.mp4` — 1280×624, 30 fps, 413 frames, Indian
  urban arterial, fixed elevated CCTV
- **Tracking source:** `detect_track.py` output (fine-tuned yolov8s, 84 bike
  vehicles tracked)

---

## 1. Candidate models

Most publicly available "helmet detection" models target **construction hard
hats**, not motorcycle helmets. After filtering for motorcycle context and a
licence usable in an academic project, two candidates remained. Both are
Apache-2.0 and downloadable without an account.

| Model | Classes | Size | Training evidence |
|---|---|---|---|
| `JarvanLee/yolov8-helmet-violation-detection` | `helmet`, `head`, `person` | yolov8m, 25.9M params | Full training artifacts: **P 0.959, R 0.614, mAP50 0.647, mAP50-95 0.436** (20 epochs, imgsz 640) |
| `aneesarom/Helmet-Violation-Detection` | `with helmet`, `without helmet`, `rider`, `number plate` | 43.6M params | **120 training images**, no metrics published |

**Neither documents its training-data provenance**, so it could not be confirmed
that either was trained on Indian or Asian traffic. `JarvanLee`'s class scheme
(`helmet` / `head` / `person`) is the signature of the construction hard-hat
datasets (SHWD, "Hard Hat Workers"), which suggests its strong published metrics
were earned in a different domain from ours.

Only `aneesarom` has a true motorcycle no-helmet class.

---

## 2. Test method

Second-pass detection on crops, not whole frames — the intended production
design. For each tracked `bike`, its largest bounding box was cropped with
padding (45% above, 25% each side) so the rider's head was included, and each
crop was passed to both models at `conf=0.15`.

- 84 bike vehicles tracked
- **50** had a box ≥45 px tall (smaller ones cannot contain a resolvable head)
- Median crop height: **110 px**; rider head region ≈ **15–30 px**

---

## 3. Results — both models unusable on this footage

| Model | No detection at all | Helmet class | **No-helmet class** |
|---|---|---|---|
| `JarvanLee` | **37/50 (74%)** | `helmet` ×10 (conf 0.45) | `head` ×8 (conf 0.43) |
| `aneesarom` | **35/50 (70%)** | `with helmet` ×7 (conf 0.51) | **`without helmet` ×1 (conf 0.17)** |

The decisive result: the only model with a genuine motorcycle no-helmet class
produced **one** such detection across 50 riders, at 0.17 confidence — below any
threshold that would be deployed.

A violation layer built on this would flag approximately zero riders, and
**"no violations" would be indistinguishable from "the class never fires"**.
That failure mode is worse than not having the feature, because it looks like a
working detector reporting compliance.

### Upscaling does not fix it

Crops were re-tested at 3× and 6× to check whether the input was simply too
small for the network:

| Model | ×1 | ×3 | ×6 |
|---|---|---|---|
| `JarvanLee` no-detection | 37/50 | 36/50 | 36/50 |
| `aneesarom` no-detection | 35/50 | 33/50 | 35/50 |
| `aneesarom` `without helmet` | 1 | **0** | **0** |

Detection rate barely moves, and the no-helmet class disappears entirely.

**Interpretation, stated carefully:** interpolation cannot add detail that the
sensor never captured, so this test rules out "the input tensor is simply too
small" — it does **not** by itself isolate the cause. The likely explanation is
a combination: at this camera distance the head region carries too little real
detail to classify, and these models were not trained on riders at this scale or
this overhead viewing angle. Distinguishing those two causes would require
testing the same models on closer footage of the same scene.

---

## 4. Decision

**Helmet detection is not implemented.** It requires footage where the rider's
head spans roughly **60 px or more** — i.e. a camera closer to the traffic, a
higher-resolution sensor, or a tighter field of view than this deployment has.

This is a limitation of the available footage and of off-the-shelf models, not
of the pipeline architecture. The second-pass-on-crops design was validated end
to end; only the classifier underperformed.

### If revisited

The layer would slot in beside `speed_violation.py`, `wrong_lane.py` and
`lane_change.py` using the same decoupled pattern — read the tracks CSV, crop
per tracked bike, run a second-pass model, write a `no_helmet` flag plus its own
metadata file. What is missing is a model that works at this scale, which needs
either:

- footage with a larger rider head region, or
- a model trained on overhead CCTV of Indian two-wheelers (Roboflow Universe
  hosts candidates, but downloading trained weights requires an account), or
- fine-tuning on a helmet-labelled dataset — explicitly out of scope here.

The evaluated weights are retained in `models/` (`helmet_jarvanlee_yolov8m.pt`,
`helmet_aneesarom.pt`) so this evaluation can be reproduced.
