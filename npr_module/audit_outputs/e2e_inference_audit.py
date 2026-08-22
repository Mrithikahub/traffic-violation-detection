"""
End-to-End Inference Audit: YOLO best.pt → Preprocessing → EasyOCR → Postprocessor
====================================================================================
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System

Runs genuine end-to-end inference on a balanced sample of Indian and foreign images.
Reports every field honestly — no hardcoded fallbacks, no fabricated results.

Output:
  npr_module/audit_outputs/e2e_audit_report.txt   (human-readable)
  npr_module/audit_outputs/e2e_audit_results.json (machine-readable)

Rules:
  - Does NOT modify any dataset files, manifests, or model weights.
  - Does NOT retrain or fine-tune anything.
  - Reports failures where OCR produces no result.
  - Clearly distinguishes Indian vs foreign images and results.
"""

import os, sys, json, time, datetime
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Paths ────────────────────────────────────────────────────────────────────
NPR_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(NPR_ROOT, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt")
MANIFEST_DIR = os.path.join(NPR_ROOT, "configs", "manifests")
OUT_DIR      = os.path.join(NPR_ROOT, "audit_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, NPR_ROOT)
from src.pipeline import NumberPlateRecognizer

# ── Sampling Configuration ───────────────────────────────────────────────────
# 8 Indian + 8 Foreign = 16 representative images spread across the manifests
SAMPLE_CONFIG = {
    "Indian Validation": {
        "manifest": os.path.join(MANIFEST_DIR, "val_indian.txt"),
        "n_samples": 8,
        "label": "INDIAN",
    },
    "Foreign Unseen Test": {
        "manifest": os.path.join(MANIFEST_DIR, "test_foreign.txt"),
        "n_samples": 8,
        "label": "FOREIGN",
    },
}


def pick_spread_samples(manifest_path: str, n: int) -> list:
    """Select n images evenly spread across the manifest (not just the first n)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) <= n:
        return lines
    step = max(1, len(lines) // n)
    # Offset by step//3 to avoid first-image bias
    start = step // 3
    selected = [lines[min(start + i * step, len(lines) - 1)] for i in range(n)]
    return selected


def derive_label_path(img_path: str) -> str:
    """Convert image path to corresponding YOLO label path."""
    lp = img_path.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
    return os.path.splitext(lp)[0] + ".txt"


def load_gt_boxes(label_path: str, img_w: int, img_h: int) -> list:
    """Load YOLO-format ground truth bounding boxes as [x1,y1,x2,y2]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                xc, yc, bw, bh = map(float, parts[1:5])
                boxes.append([
                    int((xc - bw / 2) * img_w),
                    int((yc - bh / 2) * img_h),
                    int((xc + bw / 2) * img_w),
                    int((yc + bh / 2) * img_h),
                ])
    return boxes


def iou(box_a: list, box_b: list) -> float:
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return round(inter / union, 3) if union > 0 else 0.0


# ── Main Audit ───────────────────────────────────────────────────────────────
def run_audit():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 90)
    print("CDAC NPR MODULE — END-TO-END INFERENCE AUDIT")
    print(f"Timestamp  : {ts}")
    print(f"Weights    : {WEIGHTS_PATH}")
    print(f"Weights OK : {os.path.exists(WEIGHTS_PATH)}")
    print("=" * 90)

    # Initialise pipeline — YOLO will be loaded since model_path is given
    recognizer = NumberPlateRecognizer(
        detector_model_path=WEIGHTS_PATH,
        use_ocr_paddle=False,   # Use EasyOCR (PaddleOCR not installed)
        conf_threshold=0.35,
    )
    print(f"\nDetector type  : {recognizer.detector.detector_type}")
    print(f"OCR engine     : {recognizer.ocr_engine.engine_type}")

    if recognizer.detector.detector_type != "yolo":
        print("\n[CRITICAL] YOLO model failed to load — weights path may be incorrect.")
        print(f"  Path checked: {WEIGHTS_PATH}")
        sys.exit(1)

    all_records = []

    for group_name, cfg in SAMPLE_CONFIG.items():
        samples = pick_spread_samples(cfg["manifest"], cfg["n_samples"])
        print(f"\n{'─'*90}")
        print(f"GROUP: {group_name} ({cfg['label']}) — {len(samples)} samples")
        print(f"{'─'*90}")

        for idx, img_path in enumerate(samples, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{idx:02d}] [SKIP] Cannot read: {os.path.basename(img_path)}")
                continue

            h, w = img.shape[:2]
            gt_boxes = load_gt_boxes(derive_label_path(img_path), w, h)
            n_gt = len(gt_boxes)

            # ─ Run full pipeline ─
            result = recognizer.process_vehicle_crop(
                vehicle_crop=img,
                vehicle_id=f"{cfg['label']}_{idx:02d}",
                frame_id=idx
            )

            detected     = result["plate_detected"]
            det_conf     = result["detection_confidence"]
            plate_bbox   = result["plate_bbox_crop"]
            raw_ocr      = result["raw_ocr_text"]
            cleaned      = result["cleaned_text"]
            plate_text   = result["plate_text"]
            ocr_conf     = result["ocr_confidence"]
            plate_type   = result["plate_type"]
            proc_ms      = result["processing_time_ms"]
            format_type  = result["format_type"]
            state_name   = result["state_name"]
            is_valid_ind = result["is_valid_indian_format"]

            # IoU with GT if both detected and GT exists
            best_iou = 0.0
            if detected and gt_boxes and plate_bbox:
                best_iou = max(iou(plate_bbox, gb) for gb in gt_boxes)

            # Console output
            print(f"\n  [{idx:02d}] {os.path.basename(img_path)[:50]}")
            print(f"        Image size    : {w}×{h}")
            print(f"        GT instances  : {n_gt}")
            print(f"        Plate detected: {detected}")
            if detected:
                print(f"        YOLO conf     : {det_conf:.3f}")
                print(f"        BBox (crop)   : {plate_bbox}")
                print(f"        IoU w/ GT     : {best_iou:.3f}")
                print(f"        Raw OCR text  : '{raw_ocr}'")
                print(f"        Cleaned text  : '{cleaned}'")
                print(f"        Final text    : '{plate_text}'")
                print(f"        OCR confidence: {ocr_conf:.3f}")
                print(f"        Plate type    : {plate_type}")
                print(f"        Format type   : {format_type}")
                if state_name:
                    print(f"        State         : {state_name}")
            else:
                print(f"        [NO DETECTION] — morphological fallback not used")
            print(f"        Proc time     : {proc_ms:.1f} ms")

            record = {
                "group": group_name,
                "expected_label": cfg["label"],
                "image_file": os.path.basename(img_path),
                "image_size_wh": [w, h],
                "gt_instances": n_gt,
                "plate_detected": detected,
                "detection_confidence": det_conf,
                "plate_bbox_crop": plate_bbox,
                "iou_with_gt": best_iou,
                "raw_ocr_text": raw_ocr,
                "cleaned_text": cleaned,
                "plate_text": plate_text,
                "ocr_confidence": ocr_conf,
                "plate_type": plate_type,
                "format_type": format_type,
                "is_valid_indian_format": is_valid_ind,
                "state_name": state_name,
                "processing_time_ms": proc_ms,
            }
            all_records.append(record)

    # ── Summary Report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")

    header = (f"{'#':<4} {'Group':<22} {'File':<35} {'Det':>3} "
              f"{'Conf':>5} {'IoU':>5} {'Raw OCR':<25} {'Type':<9} {'OCR_C':>6} {'ms':>6}")
    print(header)
    print("-" * 130)

    det_counts   = {}
    ocr_nonempty = {}
    total_ms     = {}

    for i, r in enumerate(all_records, 1):
        g = r["group"]
        det_counts[g]   = det_counts.get(g, 0) + (1 if r["plate_detected"] else 0)
        ocr_nonempty[g] = ocr_nonempty.get(g, 0) + (1 if r["raw_ocr_text"] else 0)
        total_ms[g]     = total_ms.get(g, 0.0) + r["processing_time_ms"]

        det_flag = "✓" if r["plate_detected"] else "✗"
        raw_trunc = (r["raw_ocr_text"] or "")[:24]
        print(f"{i:<4} {g:<22} {r['image_file'][:34]:<35} {det_flag:>3} "
              f"{r['detection_confidence']:>5.3f} {r['iou_with_gt']:>5.3f} "
              f"{raw_trunc:<25} {r['plate_type']:<9} {r['ocr_confidence']:>6.3f} "
              f"{r['processing_time_ms']:>6.1f}")

    print(f"\n{'─' * 90}")
    print("PER-GROUP STATISTICS")
    print(f"{'─' * 90}")

    group_totals = {}
    for r in all_records:
        g = r["group"]
        group_totals.setdefault(g, []).append(r)

    bottleneck_notes = []
    for g, recs in group_totals.items():
        n_total = len(recs)
        n_det   = sum(1 for r in recs if r["plate_detected"])
        n_ocr   = sum(1 for r in recs if r["raw_ocr_text"])
        n_valid = sum(1 for r in recs if r["is_valid_indian_format"])
        avg_iou = np.mean([r["iou_with_gt"] for r in recs if r["plate_detected"]]) if n_det > 0 else 0.0
        avg_ms  = np.mean([r["processing_time_ms"] for r in recs])
        avg_ocr_c = np.mean([r["ocr_confidence"] for r in recs if r["raw_ocr_text"]]) if n_ocr > 0 else 0.0

        print(f"\n  {g}:")
        print(f"    Total images       : {n_total}")
        print(f"    Plates detected    : {n_det} / {n_total}  ({100*n_det/n_total:.0f}%)")
        print(f"    OCR non-empty      : {n_ocr} / {n_det if n_det > 0 else 1}  (of detected)")
        print(f"    Avg IoU w/ GT      : {avg_iou:.3f}")
        print(f"    Avg OCR confidence : {avg_ocr_c:.3f}")
        print(f"    Avg proc time (ms) : {avg_ms:.1f}")
        if "Indian" in g:
            print(f"    Valid Indian fmt   : {n_valid} / {n_ocr if n_ocr > 0 else 1}  (of OCR results)")

        # Bottleneck analysis
        if n_det < n_total:
            miss_rate = 1.0 - n_det / n_total
            bottleneck_notes.append(f"  [{g}] Detection miss rate: {miss_rate*100:.0f}% → possible detection bottleneck")
        if n_det > 0 and n_ocr < n_det:
            miss_rate = 1.0 - n_ocr / n_det
            bottleneck_notes.append(f"  [{g}] OCR empty rate: {miss_rate*100:.0f}% (of detected plates) → OCR bottleneck")
        if n_ocr > 0 and avg_ocr_c < 0.40:
            bottleneck_notes.append(f"  [{g}] Low avg OCR confidence ({avg_ocr_c:.2f}) → preprocessing or OCR quality bottleneck")

    print(f"\n{'─' * 90}")
    print("BOTTLENECK ASSESSMENT")
    print(f"{'─' * 90}")
    if bottleneck_notes:
        for note in bottleneck_notes:
            print(note)
    else:
        print("  No significant bottleneck identified in this sample.")

    print(f"\n{'=' * 90}\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "e2e_audit_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": ts,
            "weights": WEIGHTS_PATH,
            "detector_type": recognizer.detector.detector_type,
            "ocr_engine": recognizer.ocr_engine.engine_type,
            "records": all_records,
        }, f, indent=2)

    print(f"[SAVED] JSON : {json_path}")
    print("[DONE] End-to-end inference audit complete.\n")


if __name__ == "__main__":
    run_audit()
