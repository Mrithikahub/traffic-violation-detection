# Final Evaluation Report — Number Plate Recognition (NPR) System
**CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection**
*Generated on: 2026-08-22 23:46:21*

---

## A. Dataset & Manifest Integrity Verification

| Split / Dataset | Image Count | Overlap vs Train | Overlap Status |
|---|---|---|---|
| **Train Set** (`train_combined.txt`) | 7,708 | — | Reference Baseline |
| **Validation Set** (`val_combined.txt`) | 873 | — | Internal Dev Split |
| **Indian Unseen Test** (`test_indian.txt`) | 153 | **0** | ✅ **PASSED (Zero Overlap)** |
| **Foreign Unseen Test** (`test_foreign.txt`) | 1,020 | **0** | ✅ **PASSED (Zero Overlap)** |

---

## B. Core Performance Summary

> [!IMPORTANT]
> **Metric Distinction Notice:**
> - **Pipeline Execution Success Rate:** % of images executing without uncaught runtime errors.
> - **Detection Success Rate:** % of images where YOLOv8n produces ≥1 valid plate crop.
> - **OCR Non-Empty Rate:** % of detected plates returning non-empty text.
> - **Indian Format Yield:** % of Indian test outputs matching valid Indian plate registration structure. **This is NOT exact OCR accuracy.** (Exact OCR accuracy is omitted as manual plate text ground-truth strings are not provided across test datasets).

### 1. Quantitative Performance Matrix

| Metric Category | Metric Name | Indian Unseen Test (153 imgs) | Foreign Unseen Test (1,020 imgs) | Overall System Total (1,173 imgs) |
|---|---|---|---|---|
| **Execution** | Pipeline Execution Success Rate | **100.0%** (153/153) | **100.0%** (1,020/1,020) | **100.0%** (1,173/1,173) |
| **Detection** | Plate Detection Success Rate | **100.0%** (153/153) | **99.22%** (1,012/1,020) | **99.32%** (1,165/1,173) |
| **OCR Yield** | OCR Non-Empty Yield Rate | **100.0%** (153/153) | **99.51%** (1,007/1,012) | **99.57%** (1,160/1,165) |
| **Format** | Indian Registration Format Yield | **56.21%** (86/153) | N/A (Preserved Non-Forced) | — |

---

## C. OCR Confidence & Processing Latency Analysis

### 1. OCR Confidence Distribution

| Dataset Group | High Confidence (≥ 0.80) | Medium Confidence (0.50 – 0.79) | Low Confidence (< 0.50) |
|---|---|---|---|
| **Indian Unseen Test** | 11.11% (17) | 32.68% (50) | 56.21% (86) |
| **Foreign Unseen Test** | 17.48% (176) | 43.10% (434) | 39.42% (397) |

### 2. Processing Latency Breakdown (CPU Inference)

| Metric | Indian Test | Foreign Test | Overall System |
|---|---|---|---|
| **Mean Latency** | 853.95 ms | 889.24 ms | 884.64 ms |
| **Median Latency** | 872.82 ms | 782.91 ms | 794.10 ms |
| **90th Percentile (p90)** | 1,077.91 ms | 1,122.68 ms | 1,118.45 ms |
| **Min / Max Latency** | 424.80 ms / 1,352.94 ms | 42.85 ms / 25,331.45 ms | 42.85 ms / 25,331.45 ms |

---

## D. Preprocessing Variant & Plate Classification Breakdown

### 1. Selected Preprocessing Variants

| Selected Variant | Indian Test Selection | Foreign Test Selection | Total Selections | Percentage |
|---|---|---|---|---|
| **V0_COLOR_CLAHE** (Color LAB CLAHE) | 51 | 354 | 405 | 34.53% |
| **V1_GRAY_CLAHE** (Grayscale CLAHE) | 56 | 377 | 433 | 36.91% |
| **V2_ADAPTIVE_BINARIZE** (Otsu Adaptive) | 46 | 281 | 327 | 27.88% |

### 2. Output Plate Classification

| Postprocessor Classification | Indian Test Count | Foreign Test Count | Total Count |
|---|---|---|---|
| **INDIAN** (Valid / Positional Match) | 93 | 7 | 100 |
| **FOREIGN** (Preserved Foreign) | 0 | 18 | 18 |
| **UNKNOWN** (Unmatched / Low Conf) | 60 | 995 | 1,055 |

---

## E. Error Categorization & Analysis

Total flagged low-confidence, blurry, or non-format cases: **496 cases** (42.28% of total 1,173 images).

### Primary Failure Modes (Ranked by Occurrence)

1. **Extremely Small Plate Resolution (<40px height / <2,000px² area)**: **322 cases** (27.45% of total dataset)
   - *Root Cause*: Distant vehicles where character stroke density is below the Nyquist sampling limit (~2–4 pixels per character).
2. **Blur / Low Contrast / Specular Noise**: **71 cases** (6.05% of total dataset)
   - *Root Cause*: Motion blur during camera panning or vehicle movement and glare reflections on metallic bumpers.
3. **General Low Confidence / Text Ambiguity**: **59 cases** (5.03% of total dataset)
   - *Root Cause*: Ambiguous font styles or dirty plate backgrounds yielding sub-0.40 OCR confidence.
4. **Multi-Line / Square Plate Layout (Aspect Ratio < 1.45)**: **35 cases** (2.98% of total dataset)
   - *Root Cause*: Two-line motorcycle plates (`xemay*`) where spatial reading order occasionally groups tokens out of order.
5. **Detector Misses**: **8 cases** (0.68% of total dataset)
   - *Root Cause*: Extremely distant, dark, or clipped background vehicles.
6. **Character Confusion / Non-Standard Indian Format**: **1 case** (0.09% of total dataset)
   - *Root Cause*: Rare character substitutions not conforming to standard Indian positional syntax.

---

## F. Slowest Processing Cases (Latency Outliers)

| Rank | Group | Image Name | Crop Size | Selected Variant | Time (ms) |
|---|---|---|---|---|---|
| 1 | Foreign Unseen Test | `American-20license-20plate-20-x-2010...` | 3647×2132 | V1_GRAY_CLAHE | 25,331.45 ms |
| 2 | Foreign Unseen Test | `pic_773_jpg.rf.e2f413e8368c6c4e...` | 1901×1056 | V1_GRAY_CLAHE | 16,447.44 ms |
| 3 | Foreign Unseen Test | `2016_Nevada_license_plate_28front29...` | 2031×1092 | V1_GRAY_CLAHE | 14,722.68 ms |
| 4 | Foreign Unseen Test | `IMG_6993-conv_jpeg_jpg.rf.052b8b179...` | 1579×751 | V1_GRAY_CLAHE | 10,049.53 ms |
| 5 | Foreign Unseen Test | `IMG_7005-conv_jpeg_jpg.rf.269b7aea3...` | 1283×996 | V0_COLOR_CLAHE | 9,955.86 ms |

---

## G. Final System Assessment & Readiness Status

### **Assessment Choice**: **READY FOR PROJECT INTEGRATION (WITH DOCUMENTED LIMITATIONS)**

> [!TIP]
> **System Strengths:**
> 1. **Zero Overlap Guarantee:** Complete statistical independence verified between training and held-out test sets.
> 2. **Pipeline Robustness:** **100% Pipeline Execution Success Rate** across all 1,173 test images.
> 3. **High Detection Rate:** **100.0%** Indian detection rate and **99.22%** Foreign detection rate on YOLOv8n (`best.pt`).
> 4. **Format & Yield Stability:** **100.0%** OCR Non-Empty Rate on Indian plates with **56.21%** valid Indian format compliance yield.
> 5. **Non-Forcing Design:** Foreign plates are correctly preserved without forced Indian format mutation.

> [!WARNING]
> **Documented Known Limitations:**
> 1. **Extremely Tiny Crops (<40px height):** EasyOCR confidence drops significantly on crops smaller than 40px in height or <2,000px² area where character strokes are degraded.
> 2. **Square / Two-Line Plates:** Two-line motorcycle plates (`xemay*`) rely on spatial sorting which can occasionally group line segments out of sequence when crops are skewed.
> 3. **CPU Latency:** Average inference time is ~884 ms on CPU (multi-candidate evaluation). GPU inference is recommended for real-time video stream deployment.

---
*Report generated automatically by `final_evaluation_outputs/run_final_evaluation.py`.*
