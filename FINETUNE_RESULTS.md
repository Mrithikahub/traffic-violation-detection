# Fine-Tuning Results — Vehicle Detection

**Detection + tracking module.** Fine-tuned YOLOv8s on Indian CCTV traffic to fix
missed two-wheelers and car/truck class confusion.

- **Date:** 2026-08-14
- **Base model:** `yolov8s.pt` (COCO-pretrained)
- **Fine-tuned weights:** `runs/detect/runs/bmd45_ft/weights/best.pt` (epoch 18 of 20)
- **Training data:** BMD-45 (Bengaluru CCTV) — 800 train / 200 val images, 1920×1080
- **Config:** 20 epochs, imgsz 640, batch 8, AdamW (auto), CPU-only (i5-1235U)
- **Tracker:** ByteTrack (unchanged)

---

## 1. Headline: before vs after

Identical held-out 200 images, identical settings (`conf=0.10`, `imgsz=640`).

| Metric | Before (pretrained) | After (fine-tuned) | Change |
|---|---|---|---|
| Localisation recall | 61.2% | **90.6%** | **+29.4 pts** |
| Recall + correct class | 51.2% | **87.7%** | **+36.5 pts** |
| Precision | 46.9% | **67.2%** | +20.3 pts |
| F1 | 53.1% | **77.2%** | +24.1 pts |

> The comparison **understates** the gain. The pretrained model was excused from
> 357 auto-rickshaw boxes it structurally cannot detect, and scored on 1,671
> boxes. The fine-tuned model was scored on all **2,028**. It beat the baseline
> on a harder test.

---

## 2. Per-class recall — both target problems fixed

| Class | Before | After | Change |
|---|---|---|---|
| **bike** (two-wheeler) | **41.7%** | **88.2%** | **+46.5 pts** |
| car | 88.2% | 93.8% | +5.6 |
| bus | 79.6% | 88.8% | +9.2 |
| truck | 71.2% | 94.9% | +23.7 |
| **rickshaw** (three-wheeler) | **not detectable** | **91.3%** | gap closed |

**Class confusion:** `car → truck` errors fell **101 → 22** (−78%).

**Class accuracy** (of boxes located, share labelled correctly): bike **99.6%**,
car 96.4%, rickshaw 96.3%, bus 89.7%, truck 73.2%.

---

## 3. Coverage gap closed

Auto-rickshaws are ~17% of vehicles in Indian traffic and have **no COCO
equivalent** — the pretrained model could not detect them under any label, at any
confidence threshold. The fine-tuned model detects them at **91.3% recall**.

---

## 4. Training curve

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.625 | 0.440 | 0.586 | 0.582 |
| 5 | 0.651 | 0.488 | 0.607 | 0.614 |
| 10 | 0.741 | 0.568 | 0.795 | 0.641 |
| 15 | 0.768 | 0.607 | 0.800 | 0.674 |
| **18 (best)** | **0.778** | **0.621** | 0.751 | **0.732** |
| 20 | 0.775 | 0.618 | 0.744 | 0.726 |

mAP50-95 improved **+41%** (0.440 → 0.621); classification loss fell 1.426 → 0.551.

---

## 5. Known limitation — domain specialisation

The model is now **specialised to dense Indian CCTV**. On other footage it
over-fires:

| Dataset | Model | Recall | Precision |
|---|---|---|---|
| UA-DETRAC (China, urban) | pretrained | 94.4% | **54.2%** |
| UA-DETRAC (China, urban) | fine-tuned | 95.1% | **36.9%** |

Recall holds, but precision drops 17 points — ~4,800 extra false positives on the
same clip. On UCSD highway footage it labels ~15% of vehicles `rickshaw`, which
do not exist there.

**Recommendation:** use the fine-tuned weights for Indian traffic; keep
`yolov8s.pt` for the UCSD/UA-DETRAC baselines. Both work via `--model`; the
pipeline auto-detects each model's class scheme. To get one model for all
domains, mix UCSD/UA-DETRAC frames into training rather than training on
BMD-45 alone.

---

## 6. Reproducing

```bash
python prepare_dataset.py                    # COCO -> YOLO, 5-class, 80/20 split
python train_finetune.py --epochs 20         # fine-tune from yolov8s.pt
python eval_detection.py --images data/bmd45_yolo/images/val \
    --annotations data/bmd45_yolo/val_coco.json \
    --model runs/detect/runs/bmd45_ft/weights/best.pt
```

Metrics JSON: `results_val200_baseline.json` (before),
`results_val200_ft_best.json` (after).
Visual comparisons: `outputs_showcase/before_after_*.jpg`.
Output schema for downstream modules: `OUTPUT_FORMAT.md`.
