"""
STRICT IMPLEMENTATION AUDIT
Runs the NPR pipeline on 5 real test dataset images.
Prints every intermediate value so we can verify whether outputs are
genuinely computed or are hardcoded/mocked.
"""

import os, sys, time, json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import NumberPlateDetector
from src.preprocessor import PlatePreprocessor
from src.ocr_engine import PlateOCREngine
from src.postprocessor import IndianPlatePostProcessor
from src.pipeline import NumberPlateRecognizer

# ─── Pick 5 real test images ────────────────────────────────────────────────
TEST_IMG_DIR  = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"
TEST_LBL_DIR  = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\labels"
OUTPUT_DIR    = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\audit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".jpg")])

# Pick images spread across the sorted list (not just the first few)
indices = [0, 200, 400, 600, 850]
chosen  = [all_files[i] for i in indices if i < len(all_files)]

# ─── Initialise components ───────────────────────────────────────────────────
print("Initialising pipeline components...")
detector     = NumberPlateDetector()
preprocessor = PlatePreprocessor(target_height=140, pad_percent=0.06)
ocr_engine   = PlateOCREngine(use_paddle=True)
postprocessor= IndianPlatePostProcessor()
recognizer   = NumberPlateRecognizer()

print(f"Detector type   : {detector.detector_type}")
print(f"OCR engine type : {ocr_engine.engine_type}")
print()

audit_rows = []

for fname in chosen:
    stem     = os.path.splitext(fname)[0]
    img_path = os.path.join(TEST_IMG_DIR, fname)
    lbl_path = os.path.join(TEST_LBL_DIR,  stem + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Could not read {fname}")
        continue

    h, w = img.shape[:2]

    # ── Ground truth boxes ────────────────────────────────────────────────
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append([
                        int((xc - bw/2)*w), int((yc - bh/2)*h),
                        int((xc + bw/2)*w), int((yc + bh/2)*h)
                    ])

    print("="*70)
    print(f"IMAGE  : {fname}")
    print(f"SIZE   : {w}x{h}")
    print(f"GT BOXES ({len(gt_boxes)}) : {gt_boxes}")

    # ── Step 1: Detector ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    detections = detector.detect(img)
    det_time_ms = (time.perf_counter() - t0) * 1000

    print(f"\n--- STEP 1: Detector ({detector.detector_type}) ---")
    print(f"  Detections returned       : {len(detections)}")
    for d in detections[:5]:
        print(f"    bbox={d['bbox']}  confidence={d['confidence']}")
    print(f"  Detection time            : {det_time_ms:.2f} ms")

    # ── Step 2: Crop best candidate ──────────────────────────────────────
    if not detections:
        best_box = [0, 0, w, h]
        det_conf = 0.50   # NOTE: This is the HARDCODED FALLBACK VALUE in pipeline.py line 95
        print("  [WARN] No detection — fallback to full image crop (conf=0.50 HARDCODED)")
    else:
        best = max(detections, key=lambda d: d["confidence"])
        best_box = best["bbox"]
        det_conf = best["confidence"]

    plate_crop = preprocessor.crop_with_padding(img, best_box)
    print(f"\n--- STEP 2: Crop ---")
    print(f"  Plate box (crop coords)   : {best_box}")
    print(f"  detection_confidence      : {det_conf}  <- {'REAL computed' if detections else 'HARDCODED FALLBACK (0.50)'}")
    print(f"  Plate crop shape          : {plate_crop.shape}")

    # ── Step 3: Preprocessing ────────────────────────────────────────────
    t1 = time.perf_counter()
    preprocessed = preprocessor.preprocess_pipeline(plate_crop)
    pre_time_ms = (time.perf_counter() - t1) * 1000
    print(f"\n--- STEP 3: Preprocessing ---")
    print(f"  Deskewed shape            : {preprocessed['deskewed'].shape}")
    print(f"  Upscaled shape            : {preprocessed['upscaled'].shape}")
    print(f"  Enhanced color shape      : {preprocessed['enhanced_color'].shape}")
    print(f"  Binarized shape           : {preprocessed['binarized'].shape}")
    print(f"  Preprocessing time        : {pre_time_ms:.2f} ms")

    # ── Step 4: OCR Engine ───────────────────────────────────────────────
    t2 = time.perf_counter()
    ocr_result = ocr_engine.recognize_plate(
        enhanced_color=preprocessed["enhanced_color"],
        binarized=preprocessed["binarized"]
    )
    ocr_time_ms = (time.perf_counter() - t2) * 1000

    raw_ocr_text = ocr_result.get("raw_text", "")
    ocr_conf = ocr_result.get("confidence", 0.0)
    engine_used = ocr_result.get("engine_used", "")

    print(f"\n--- STEP 4: OCR Engine ({engine_used}) ---")
    print(f"  raw_text                  : '{raw_ocr_text}'")
    print(f"  ocr_confidence            : {ocr_conf}")
    print(f"  Engine used               : {engine_used}")
    if "character_blobs_found" in ocr_result:
        print(f"  character_blobs_found     : {ocr_result['character_blobs_found']}  <- OCR returned empty; blob count is a fallback heuristic")
    print(f"  OCR time                  : {ocr_time_ms:.2f} ms")

    # ── Step 5: Postprocessor ────────────────────────────────────────────
    parsed = postprocessor.validate_and_parse(raw_ocr_text)
    print(f"\n--- STEP 5: Postprocessor ---")
    print(f"  standardized_text         : '{parsed['standardized_text']}'")
    print(f"  is_valid                  : {parsed['is_valid']}")
    print(f"  format_type               : {parsed['format_type']}")
    print(f"  state_name                : {parsed['state_name']}")

    # ── Step 6: Full pipeline result (for timing) ─────────────────────────
    t_full = time.perf_counter()
    full_result = recognizer.process_frame(img, frame_id=0)
    total_ms = (time.perf_counter() - t_full) * 1000
    plate_text = full_result[0]["plate_text"]
    reported_det_conf = full_result[0]["detection_confidence"]
    reported_ocr_conf = full_result[0]["ocr_confidence"]
    reported_proc_ms  = full_result[0]["processing_time_ms"]

    print(f"\n--- FULL PIPELINE OUTPUT ---")
    print(f"  plate_text                : '{plate_text}'")
    print(f"  detection_confidence      : {reported_det_conf}")
    print(f"  ocr_confidence            : {reported_ocr_conf}")
    print(f"  processing_time_ms        : {reported_proc_ms} (measured via time.perf_counter)")

    # ── Annotate and save output image ────────────────────────────────────
    annotated = img.copy()
    # Draw GT boxes in green
    for gb in gt_boxes:
        cv2.rectangle(annotated, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
        cv2.putText(annotated, "GT", (gb[0], gb[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Draw predicted box in blue
    bx = best_box
    cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (255, 100, 0), 2)
    label_text = f"OCR: '{plate_text}' Det:{reported_det_conf:.2f} OCR:{reported_ocr_conf:.2f}"
    cv2.putText(annotated, label_text, (bx[0], max(bx[1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{stem}.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\n  Annotated image saved     : {out_path}")

    audit_rows.append({
        "filename": fname,
        "gt_boxes": len(gt_boxes),
        "plate_text": plate_text,
        "raw_ocr": raw_ocr_text,
        "det_conf": reported_det_conf,
        "ocr_conf": reported_ocr_conf,
        "proc_ms": reported_proc_ms,
        "engine": engine_used
    })
    print()

# ─── Summary Table ────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("AUDIT SUMMARY TABLE")
print("="*90)
print(f"{'Image':<50} {'GT':>3} {'OCR Output':>16} {'Det Conf':>10} {'OCR Conf':>10} {'ms':>8} {'Engine':<12}")
print("-"*90)
for row in audit_rows:
    short_name = row["filename"][:48]
    print(f"{short_name:<50} {row['gt_boxes']:>3} {repr(row['plate_text']):>16} "
          f"{row['det_conf']:>10.3f} {row['ocr_conf']:>10.3f} {row['proc_ms']:>8.1f} {row['engine']:<12}")

print("\n" + "="*90)
print("CLASSIFICATION OF OUTPUT FIELDS")
print("="*90)
classifications = {
    "plate_detected":         "A - Real: True if detector.detect() or fallback runs without error",
    "plate_bbox_crop":        "A - Real: Computed bounding box from morphological detector",
    "plate_bbox_frame":       "A - Real: Derived by adding vehicle_bbox offset (None if no vehicle_bbox given)",
    "detection_confidence":   "A/E - Real if morphological detector finds candidates (std-contrast heuristic). E (Hardcoded=0.50) if no candidates found (pipeline.py line 95)",
    "raw_ocr_text":           "E - Fallback EMPTY string. PaddleOCR and EasyOCR both missing. Built-in fallback returns '' (blank)",
    "plate_text":             "E - Derived from empty raw_ocr_text; currently returns '' on all real images",
    "ocr_confidence":         "E - Fallback: 0.50 if >=6 blobs found, else 0.0. NOT from a real recognition model",
    "is_valid_indian_format": "A - Real: Regex match on plate_text (but plate_text is empty so always False/NON_STANDARD)",
    "format_type":            "A - Real: Regex-derived, but returns NON_STANDARD because text is empty",
    "state_code/state_name":  "A - Real: Only populated if regex matches (impossible with empty text)",
    "validation_score":       "A - Real: Set by postprocessor based on regex/heuristic, currently 0.0",
    "processing_time_ms":     "A - Real: Measured using time.perf_counter() in pipeline.py lines 66 and 134",
    "100% localization":      "MISLEADING: pipeline.py line 90-98 forces plate_detected=True even when detector finds NOTHING, by creating a hardcoded fallback detection with confidence=0.50",
    "63 FPS throughput":      "A - Real measured throughput, but includes the hardcoded-fallback path, NOT a real detection at that speed",
    "avg IoU=0.090":          "A - Real computed from actual GT boxes vs predicted boxes. Low because morphological detector is weak",
}
for field, verdict in classifications.items():
    print(f"  {field:<30} : {verdict}")

print("\n" + "="*90)
any_text = any(r["plate_text"] for r in audit_rows)
print(f"\n>>> Is this NPR module genuinely producing plate_text from image inference: {'YES' if any_text else 'NO'}")
print(">>> Reason: PaddleOCR and EasyOCR are not installed. The built-in fallback engine")
print("    detects character blob COUNT but does NOT return text. raw_text is always '' empty.")
print("    The 100% localization hit rate was inflated by a hardcoded confidence=0.50 fallback")
print("    that fires even when the detector returns zero candidates.")
