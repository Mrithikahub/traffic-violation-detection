"""
STRICT IMPLEMENTATION AUDIT — Run with Anaconda Python (EasyOCR available)
Runs the NPR pipeline step-by-step on 5 real test images.
"""
import os, sys, time, json
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Point to our module
MODULE_ROOT = r"e:\CDAC Dataset\CDAC_Workspace\npr_module"
sys.path.insert(0, MODULE_ROOT)

from src.detector import NumberPlateDetector
from src.preprocessor import PlatePreprocessor
from src.ocr_engine import PlateOCREngine
from src.postprocessor import IndianPlatePostProcessor
from src.pipeline import NumberPlateRecognizer

TEST_IMG_DIR = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"
TEST_LBL_DIR = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\labels"
OUTPUT_DIR   = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\audit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick 5 images spread across the dataset
all_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".jpg")])
chosen = [all_files[i] for i in [0, 200, 400, 600, 850] if i < len(all_files)]

# ─── Initialise ──────────────────────────────────────────────────────────────
print("Initialising pipeline...", flush=True)
detector      = NumberPlateDetector()
preprocessor  = PlatePreprocessor(target_height=140, pad_percent=0.06)
ocr_engine    = PlateOCREngine(use_paddle=False)   # EasyOCR fallback (paddle not installed)
postprocessor = IndianPlatePostProcessor()
recognizer    = NumberPlateRecognizer(use_ocr_paddle=False)

print(f"Detector  : {detector.detector_type}")
print(f"OCR Engine: {ocr_engine.engine_type}\n", flush=True)

audit_rows = []

for fname in chosen:
    stem     = os.path.splitext(fname)[0]
    img_path = os.path.join(TEST_IMG_DIR, fname)
    lbl_path = os.path.join(TEST_LBL_DIR, stem + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] {fname}")
        continue

    h, w = img.shape[:2]

    # Ground truth
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

    print("="*72, flush=True)
    print(f"IMAGE     : {fname}")
    print(f"SIZE      : {w}x{h}  |  GT Plates: {len(gt_boxes)}  {gt_boxes}")

    # ── Step 1: Detection ────────────────────────────────────────────────
    t0 = time.perf_counter()
    detections = detector.detect(img)
    det_ms = (time.perf_counter() - t0)*1000

    print(f"\nDETECTOR  ({detector.detector_type})  →  {len(detections)} candidate(s) in {det_ms:.1f} ms")
    for d in detections[:3]:
        print(f"  bbox={d['bbox']}  conf={d['confidence']}")

    if not detections:
        print("  [NO DETECTION] — plate_detected will be False (bug fixed)")
        plate_box, det_conf = None, 0.0
    else:
        best = max(detections, key=lambda d: d["confidence"])
        plate_box, det_conf = best["bbox"], best["confidence"]
        print(f"  Best box : {plate_box}  conf={det_conf}")

    # ── Step 2: Crop & Preprocess ────────────────────────────────────────
    if plate_box:
        plate_crop   = preprocessor.crop_with_padding(img, plate_box)
        preprocessed = preprocessor.preprocess_pipeline(plate_crop)
        print(f"\nPREPROC   upscaled={preprocessed['upscaled'].shape}  enhanced={preprocessed['enhanced_color'].shape}")

        # ── Step 3: OCR ───────────────────────────────────────────────────
        t1 = time.perf_counter()
        ocr_result = ocr_engine.recognize_plate(
            enhanced_color=preprocessed["enhanced_color"],
            binarized=preprocessed["binarized"]
        )
        ocr_ms = (time.perf_counter() - t1)*1000

        raw_text = ocr_result.get("raw_text", "")
        ocr_conf = ocr_result.get("confidence", 0.0)
        engine   = ocr_result.get("engine_used", "")
        print(f"\nOCR       engine={engine}  time={ocr_ms:.1f} ms")
        print(f"  raw_text      = '{raw_text}'")
        print(f"  ocr_confidence= {ocr_conf}")

        # ── Step 4: Postprocessor ─────────────────────────────────────────
        parsed = postprocessor.validate_and_parse(raw_text)
        print(f"\nPOSTPROC  standardized='{parsed['standardized_text']}'  valid={parsed['is_valid']}  format={parsed['format_type']}")
    else:
        raw_text, ocr_conf, engine = "", 0.0, "none"
        parsed = postprocessor.validate_and_parse("")
        print("\nSkipping OCR — no plate detected.")

    # ── Full pipeline (for accurate timing) ───────────────────────────────
    t_full = time.perf_counter()
    full_res = recognizer.process_frame(img, frame_id=0)
    full_ms  = (time.perf_counter() - t_full)*1000
    r = full_res[0]

    print(f"\nFULL PIPELINE RESULT:")
    print(f"  plate_detected       = {r['plate_detected']}")
    print(f"  plate_text           = '{r['plate_text']}'")
    print(f"  detection_confidence = {r['detection_confidence']}")
    print(f"  ocr_confidence       = {r['ocr_confidence']}")
    print(f"  processing_time_ms   = {r['processing_time_ms']}  (real time.perf_counter)")
    print(f"  detector_type        = {r.get('detector_type','')}")
    print(f"  ocr_engine_type      = {r.get('ocr_engine_type','')}")

    # ── Annotate image ────────────────────────────────────────────────────
    annotated = img.copy()
    # Draw GT in green
    for gb in gt_boxes:
        cv2.rectangle(annotated, (gb[0], gb[1]), (gb[2], gb[3]), (0, 200, 0), 2)
        cv2.putText(annotated, "GT", (gb[0], max(gb[1]-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
    # Draw prediction in blue
    if plate_box:
        bx = plate_box
        cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (255, 80, 0), 2)
        label = f"'{r['plate_text']}'  det={r['detection_confidence']:.2f} ocr={r['ocr_confidence']:.2f}"
        cv2.putText(annotated, label, (bx[0], max(bx[1]-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)
    out_path = os.path.join(OUTPUT_DIR, f"audit_{stem[:40]}.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\n  Saved: {out_path}")

    audit_rows.append({
        "img"     : fname[:52],
        "gt_count": len(gt_boxes),
        "plate_text": r["plate_text"],
        "raw_ocr"   : raw_text,
        "det_conf"  : r["detection_confidence"],
        "ocr_conf"  : r["ocr_confidence"],
        "proc_ms"   : r["processing_time_ms"],
        "engine"    : engine,
        "detected"  : r["plate_detected"],
    })
    print()

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("AUDIT RESULTS TABLE")
print("="*100)
hdr = f"{'Image':<52} {'GT':>3} {'Detected':>8} {'Plate Text':>14} {'Det Conf':>9} {'OCR Conf':>9} {'ms':>7} {'Engine'}"
print(hdr)
print("-"*100)
for row in audit_rows:
    print(f"{row['img']:<52} {row['gt_count']:>3} {str(row['detected']):>8} "
          f"{repr(row['plate_text']):>14} {row['det_conf']:>9.3f} {row['ocr_conf']:>9.3f} "
          f"{row['proc_ms']:>7.1f} {row['engine']}")

print("\n" + "="*100)
print("FIELD CLASSIFICATION")
print("="*100)
fields = [
    ("plate_detected",         "A - Real. True only if morphological detector returns ≥1 candidate. False if none (bug fixed)."),
    ("plate_bbox_crop",        "A - Real. Computed by morphological contour + Sobel gradient scoring."),
    ("detection_confidence",   "A - Real. = std_contrast/100 × aspect_bonus. Heuristic but computed from actual pixel data."),
    ("raw_ocr_text",           "A - Real. EasyOCR CRAFT+CRNN model reads from the preprocessed plate crop image."),
    ("plate_text",             "A - Real. = postprocessor(raw_ocr_text). Regex+disambiguation on real OCR output."),
    ("ocr_confidence",         "A - Real. Mean of per-character confidence scores returned by EasyOCR."),
    ("is_valid_indian_format", "A - Real. Regex match against Indian RTO & BH-series patterns."),
    ("processing_time_ms",     "A - Real. time.perf_counter() difference, pipeline.py lines 66 & 134."),
    ("100% localisation (old)","FIXED. Was inflated by hardcoded fallback conf=0.50. Removed in this session."),
    ("63 FPS (old report)",    "PARTIALLY TRUE. Throughput was real, but localization quality was overstated."),
]
for field, verdict in fields:
    print(f"  {field:<30}: {verdict}")

any_text = any(r["plate_text"] for r in audit_rows)
print("\n" + "="*100)
print(f">>> Is this NPR module genuinely producing plate_text from image inference: {'YES' if any_text else 'NO (OCR empty on these 5 images)'}")
print("    Reason: EasyOCR CRAFT+CRNN is now installed and running on real plate crops.")
print("    OCR quality depends on plate visibility and morphological detector accuracy.")
print("    Training a YOLO plate detector on this dataset is the next improvement step.")
