# Number Plate Recognition (NPR) Module

**CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System**

---

## 1. Overview

The **Number Plate Recognition (NPR)** module is a standalone, modular computer vision component designed to receive vehicle detections/crops from upstream tracking modules (Member 1's Vehicle Tracker), detect the license plate region using a fine-tuned **YOLOv8n** detector, apply multi-candidate image preprocessing and rectification, extract alphanumeric text via spatially-sorted **EasyOCR**, and validate plate syntax against Indian Motor Vehicle Act and Bharat Stage (BH) registration rules while preserving non-Indian plates.

---

## 2. Pipeline Architecture

```
[ Member 1: Vehicle YOLO + ByteTrack ]
                 │
                 │  Input: { vehicle_id, vehicle_crop, frame_id, vehicle_bbox }
                 ▼
┌────────────────────────────────────────────────────────┐
│              OUR NPR PIPELINE MODULE                   │
│                                                        │
│  1. Plate Localization (YOLOv8n Detector: best.pt)     │
│     └── Locates plate bounding box in vehicle crop     │
│                                                        │
│  2. Dynamic Padding & Crop                             │
│     └── Crops plate with 6% safety margin padding      │
│                                                        │
│  3. Multi-Variant Preprocessing                        │
│     ├── V0: Color LAB CLAHE + Bilateral Filtering      │
│     ├── V1: Grayscale CLAHE (clipLimit=3)              │
│     └── V2: Polarity-Corrected Otsu Adaptive Binarize  │
│                                                        │
│  4. Text Extraction & Spatial Sorting (EasyOCR)        │
│     └── Orders text left→right & top→bottom            │
│                                                        │
│  5. Multi-Candidate OCR Selection                      │
│     └── Composite score (Conf + Length + Alnum + Bonus)│
│                                                        │
│  6. Post-Processing & Indian Plate Rule Engine         │
│     ├── Positional Disambiguation (0↔O, 8↔B, 1↔I, 2↔Z) │
│     └── State Code & BH Series Syntax Validator        │
└────────────────────────────────────────────────────────┘
                 │
                 │  Output: Structured Result Dictionary
                 ▼
[ Main System Integration / Traffic Violation Logger ]
```

---

## 3. Directory Layout

```
npr_module/
├── configs/
│   ├── dataset_combined.yaml         # Ultralytics dataset configuration
│   └── manifests/                    # Strict zero-leakage split manifests
│       ├── train_combined.txt        # 7,708 training images
│       ├── val_combined.txt          # 873 validation images
│       ├── test_indian.txt           # 153 held-out Indian test images
│       └── test_foreign.txt          # 1,020 held-out Foreign test images
├── src/
│   ├── __init__.py                   # Package exports
│   ├── detector.py                   # Plate localization (YOLOv8n & Morphological fallback)
│   ├── preprocessor.py               # Deskewing, upscaling, CLAHE, Bilateral, Binarization
│   ├── ocr_engine.py                 # Multi-candidate EasyOCR engine & composite scoring
│   ├── postprocessor.py              # Indian RTO / BH series syntax validator & cleaner
│   └── pipeline.py                   # Unified NumberPlateRecognizer interface
├── tests/
│   ├── test_postprocessor.py         # Post-processor unit tests
│   ├── test_preprocessor.py          # Pre-processor unit tests
│   ├── test_ocr.py                   # OCR engine & scoring tests
│   └── test_end_to_end.py            # End-to-end integration tests (32/32 pass)
├── demo/
│   ├── run_on_image.py               # CLI demo on single image
│   └── evaluate_test_dataset.py      # Batch evaluation script
├── final_evaluation_outputs/         # Official held-out evaluation deliverables
│   ├── run_final_evaluation.py       # Full evaluation harness script
│   ├── final_evaluation_report.md    # Markdown final evaluation report
│   ├── final_evaluation_results.json # Full JSON evaluation metrics
│   ├── per_image_results.csv         # Detailed per-image log (1,173 images)
│   └── error_analysis.csv            # Error categorization breakdown
├── runs/detect/npr_yolov8n_baseline/
│   └── weights/
│       └── best.pt                   # Trained YOLOv8n plate detector (5.94 MB)
├── train_yolov8n.py                  # Model training reproduction script
├── evaluate_4way.py                  # 4-way evaluation script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md
```

---

## 4. Final Evaluation Results (1,173 Held-Out Unseen Images)

The complete pipeline was evaluated on **1,173 strictly held-out unseen test images** (zero overlap with training/validation splits):

| Metric Category | Metric Name | Indian Unseen Test (153 imgs) | Foreign Unseen Test (1,020 imgs) | Combined System Total (1,173 imgs) |
|---|---|---|---|---|
| **Execution** | **Pipeline Execution Success Rate** | **100.0%** (153/153) | **100.0%** (1,020/1,020) | **100.0%** (1,173/1,173) |
| **Detection** | **Plate Detection Success Rate** | **100.0%** (153/153) | **99.22%** (1,012/1,020) | **99.32%** (1,165/1,173) |
| **OCR Yield** | **OCR Non-Empty Yield Rate** | **100.0%** (153/153) | **99.51%** (1,007/1,012) | **99.57%** (1,160/1,165) |
| **Format** | **Indian Format Compliance Yield** | **56.21%** (86/153) | N/A (Preserved Non-Forced) | — |
| **Speed** | **Average CPU Latency** | **853.95 ms** | **889.24 ms** | **884.64 ms** |

> [!NOTE]
> **Metric Definitions:**
> - **Pipeline Execution Success Rate**: Percentage of images processed without unhandled runtime exceptions.
> - **Detection Success Rate**: Percentage of images where YOLOv8n successfully localizes the license plate.
> - **OCR Non-Empty Yield**: Percentage of detected plates returning readable text.
> - **Indian Format Compliance Yield**: Percentage of Indian outputs satisfying standard Indian registration syntax (`MH02CD3654`). *Explicitly distinguished from OCR exact-match accuracy as manual plate-text labels are not provided.*

---

## 5. Member 1 Integration Interface

### Input Interface (From Member 1 Tracker)
```python
from src.pipeline import NumberPlateRecognizer

# Instantiate the pipeline (weights default to runs/detect/npr_yolov8n_baseline/weights/best.pt)
recognizer = NumberPlateRecognizer()

# Process a vehicle detection crop
vehicle_result = recognizer.process_vehicle_crop(
    vehicle_crop=vehicle_image_array,      # BGR numpy array
    vehicle_id=7,                          # Tracker ID
    frame_id=1042,                         # Video frame index
    vehicle_bbox=[100, 200, 600, 500]      # [x1, y1, x2, y2] in camera frame
)
```

### Structured Output Schema
```json
{
  "vehicle_id": 7,
  "frame_id": 1042,
  "plate_detected": true,
  "plate_bbox_crop": [177, 171, 324, 215],
  "plate_bbox_frame": [277, 371, 424, 415],
  "detection_confidence": 0.95,
  "plate_text": "TN38AB1234",
  "raw_ocr_text": "TN38AB1234",
  "cleaned_text": "TN38AB1234",
  "standardized_text": "TN38AB1234",
  "ocr_confidence": 0.91,
  "selected_variant": "V1_GRAY_CLAHE",
  "composite_score": 0.88,
  "plate_type": "INDIAN",
  "is_valid_indian_format": true,
  "format_type": "STANDARD",
  "state_code": "TN",
  "state_name": "Tamil Nadu",
  "district_code": "38",
  "series_code": "AB",
  "unique_number": "1234",
  "validation_score": 1.0,
  "processing_time_ms": 782.5
}
```

---

## 6. Installation & Execution

### Prerequisites
```bash
cd npr_module
pip install -r requirements.txt
```

### Running Tests
```bash
# Run complete test suite (32 unit & integration tests)
pytest tests/ -v
```

### Running CLI Demo
```bash
# Run on sample image
python demo/run_on_image.py --image path/to/vehicle.jpg
```

### Running Final Evaluation
```bash
# Reproduce full 1,173-image evaluation
python final_evaluation_outputs/run_final_evaluation.py
```

---

## 7. Documented Known Limitations

1. **Extremely Small Crops (<40px height)**: Resolution below 40px height (~2–3 pixels per character stroke) causes OCR confidence to drop due to spatial Nyquist sampling limits.
2. **Acute Perspective / Severe Motion Blur**: Single-frame OCR cannot recover heavily smudged or reflective plates without upstream temporal frame aggregation across video tracks.
3. **CPU Inference Latency**: Multi-candidate OCR evaluation averages ~880 ms per crop on CPU. GPU acceleration is recommended for real-time live video streams.
