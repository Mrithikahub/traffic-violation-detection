"""
Multi-Candidate OCR Pipeline — Comparative Audit
=================================================
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System

Runs the same 16 representative images (8 Indian validation + 8 foreign unseen
test) through the NEW multi-candidate pipeline and compares against:
  - Original baseline (stored from the first e2e audit)
  - Current (two-pass gray_clahe/adaptive_binarize) integrated pipeline results

Outputs:
  audit_outputs/comparative_audit_report.txt
  audit_outputs/comparative_audit_results.json

Rules:
  - Does NOT modify any dataset files, manifests, or model weights.
  - Does NOT retrain anything.
  - Reports failures honestly — no fabricated results.
  - Clearly distinguishes Indian vs foreign images and results.
  - "Improved" means the new text is structurally better or the plate type
    changed from UNKNOWN→INDIAN.  It is NOT called improved if only confidence
    rose while text became worse.
"""

import os, sys, json, time, datetime
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPR_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(NPR_ROOT, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt")
OUT_DIR      = os.path.join(NPR_ROOT, "audit_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, NPR_ROOT)
from src.pipeline import NumberPlateRecognizer

# ── Hard-coded baselines from original two audit runs ────────────────────────
# Format: {filename: {raw: str, conf: float, type: str, valid: bool}}
ORIGINAL_BASELINE = {
    "video5_130.jpg":      {"raw": "MH.03.CS.6266",    "conf": 0.866, "type": "INDIAN",  "valid": True},
    "video5_500.jpg":      {"raw": "MH03C5.6266",       "conf": 0.433, "type": "INDIAN",  "valid": True},
    "video6_1140.jpg":     {"raw": "9365MHO2 DS",       "conf": 0.940, "type": "UNKNOWN", "valid": False},
    "video6_910.jpg":      {"raw": "MH02 DS 9365",      "conf": 0.755, "type": "INDIAN",  "valid": True},
    "video8_1350.jpg":     {"raw": "Mh0186716",         "conf": 0.493, "type": "INDIAN",  "valid": True},
    "video8_20.jpg":       {"raw": "'CbMhO2C1,3654]",   "conf": 0.169, "type": "UNKNOWN", "valid": False},
    "video8_790.jpg":      {"raw": "HH 12SF3212",       "conf": 0.978, "type": "UNKNOWN", "valid": False},
    "video9_260.jpg":      {"raw": '"Wh02FN2783',       "conf": 0.117, "type": "UNKNOWN", "valid": False},
    # Foreign
    "42-1280px-Alberta_1997_license_plate_-_SNX-000_jpg.rf.d10e9c1a44ea0413a2e40c1a2b06bd36.jpg":
                           {"raw": "DrBB VS OarOf JUN Abea Abe", "conf": 0.468, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen1231_jpg.rf.5262da9a0ae856b18adc04275f0e2b15.jpg":
                           {"raw": "61E.07973",         "conf": 0.404, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen2310_jpg.rf.e2e02c43d0fd1aa9eea2f7b8548c35e0.jpg":
                           {"raw": "614.897 14",        "conf": 0.392, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen3407_jpg.rf.a6e5af41538fdfa730f4820d551c1186.jpg":
                           {"raw": "1",                 "conf": 0.155, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen998_jpg.rf.08124bcec86fbde5a451e8aee0ca92c4.jpg":
                           {"raw": "srriu4",            "conf": 0.057, "type": "UNKNOWN", "valid": False},
    "pic_259_jpg.rf.bdeedbfb55112ba4c5d924fe8e736fb0.jpg":
                           {"raw": "M8 7227 nz9nnartua?", "conf": 0.250, "type": "UNKNOWN", "valid": False},
    "xemay1573_jpg.rf.6b2d6461a87d8babbb107ee90edba7cd.jpg":
                           {"raw": "59-B712338,",       "conf": 0.757, "type": "UNKNOWN", "valid": False},
    "xemay424_jpg.rf.6ee2887c5e4e8dd45f69501451592b57.jpg":
                           {"raw": "59-C1 399.1 6",     "conf": 0.482, "type": "UNKNOWN", "valid": False},
}

# Previous (two-pass gray_clahe primary) integrated pipeline results
PREV_INTEGRATED = {
    "video5_130.jpg":      {"raw": "MH.03.CS.6266",    "conf": 0.849, "type": "INDIAN",  "valid": True},
    "video5_500.jpg":      {"raw": "MH03C3.6266",       "conf": 0.439, "type": "INDIAN",  "valid": True},
    "video6_1140.jpg":     {"raw": "MH02 DS 9365",      "conf": 0.717, "type": "INDIAN",  "valid": True},
    "video6_910.jpg":      {"raw": "MH02 DS 9365",      "conf": 0.676, "type": "INDIAN",  "valid": True},
    "video8_1350.jpg":     {"raw": "MFOiBG 716",        "conf": 0.451, "type": "UNKNOWN", "valid": False},  # REGRESSION
    "video8_20.jpg":       {"raw": "Hho2C LCd,J654",    "conf": 0.250, "type": "UNKNOWN", "valid": False},
    "video8_790.jpg":      {"raw": "HH 12 SF 3212",     "conf": 0.980, "type": "UNKNOWN", "valid": False},
    "video9_260.jpg":      {"raw": "MHOZFN2783",        "conf": 0.622, "type": "INDIAN",  "valid": True},
    "42-1280px-Alberta_1997_license_plate_-_SNX-000_jpg.rf.d10e9c1a44ea0413a2e40c1a2b06bd36.jpg":
                           {"raw": "DrEes US O4r 0 JUN bba A", "conf": 0.565, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen1231_jpg.rf.5262da9a0ae856b18adc04275f0e2b15.jpg":
                           {"raw": "51F.07973",         "conf": 0.550, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen2310_jpg.rf.e2e02c43d0fd1aa9eea2f7b8548c35e0.jpg":
                           {"raw": "51A .897.1L",       "conf": 0.286, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen3407_jpg.rf.a6e5af41538fdfa730f4820d551c1186.jpg":
                           {"raw": "614 691,72",        "conf": 0.597, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen998_jpg.rf.08124bcec86fbde5a451e8aee0ca92c4.jpg":
                           {"raw": "1672w9]",           "conf": 0.071, "type": "UNKNOWN", "valid": False},
    "pic_259_jpg.rf.bdeedbfb55112ba4c5d924fe8e736fb0.jpg":
                           {"raw": "MU 7227 nzIMwNMtVa?", "conf": 0.313, "type": "UNKNOWN", "valid": False},
    "xemay1573_jpg.rf.6b2d6461a87d8babbb107ee90edba7cd.jpg":
                           {"raw": "59-B1 338.72",      "conf": 0.507, "type": "UNKNOWN", "valid": False},
    "xemay424_jpg.rf.6ee2887c5e4e8dd45f69501451592b57.jpg":
                           {"raw": "59-C1 39916",       "conf": 0.742, "type": "UNKNOWN", "valid": False},
}

SAMPLE_CONFIG = {
    "Indian Validation": {
        "manifest": os.path.join(NPR_ROOT, "configs", "manifests", "val_indian.txt"),
        "n_samples": 8,
        "label":    "INDIAN",
    },
    "Foreign Unseen Test": {
        "manifest": os.path.join(NPR_ROOT, "configs", "manifests", "test_foreign.txt"),
        "n_samples": 8,
        "label":    "FOREIGN",
    },
}


def pick_spread_samples(manifest_path: str, n: int) -> list:
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) <= n:
        return lines
    step  = max(1, len(lines) // n)
    start = step // 3
    return [lines[min(start + i * step, len(lines) - 1)] for i in range(n)]


def _vs_label(new_raw, new_type, new_valid, ref_raw, ref_type, ref_valid):
    """Determine IMPROVED / SAME / REGRESSED comparing new to a reference."""
    # Gained valid Indian: strong improvement
    if new_valid and not ref_valid and new_type == "INDIAN":
        return "IMPROVED (gained valid Indian)"
    # Lost valid Indian: regression
    if not new_valid and ref_valid and ref_type == "INDIAN":
        return "REGRESSED (lost valid Indian)"
    # Fixied reversed text to correct order
    if new_type == "INDIAN" and ref_type == "UNKNOWN":
        return "IMPROVED (UNKNOWN→INDIAN)"
    if new_type == "UNKNOWN" and ref_type == "INDIAN":
        return "REGRESSED (INDIAN→UNKNOWN)"
    return "SAME"


def run_audit():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 110

    print(sep)
    print("CDAC NPR MODULE — MULTI-CANDIDATE OCR COMPARATIVE AUDIT")
    print(f"Timestamp  : {ts}")
    print(f"Weights    : {WEIGHTS_PATH}")
    print(f"Weights OK : {os.path.exists(WEIGHTS_PATH)}")
    print(sep)

    recognizer = NumberPlateRecognizer(
        detector_model_path=WEIGHTS_PATH,
        use_ocr_paddle=False,
        conf_threshold=0.35,
    )
    print(f"\nDetector type  : {recognizer.detector.detector_type}")
    print(f"OCR engine     : {recognizer.ocr_engine.engine_type}\n")

    if recognizer.detector.detector_type != "yolo":
        print("[CRITICAL] YOLO model not loaded — check weights path.")
        sys.exit(1)

    records  = []
    img_counter = 0

    for group_name, cfg in SAMPLE_CONFIG.items():
        samples = pick_spread_samples(cfg["manifest"], cfg["n_samples"])
        print(f"\n{'─'*110}")
        print(f"GROUP: {group_name} — {len(samples)} images")
        print(f"{'─'*110}")

        for idx, img_path in enumerate(samples, 1):
            fname = os.path.basename(img_path)
            img   = cv2.imread(img_path)
            if img is None:
                print(f"  [{idx:02d}] [SKIP] {fname}")
                continue

            h, w = img.shape[:2]
            img_counter += 1

            result = recognizer.process_vehicle_crop(
                vehicle_crop=img,
                vehicle_id=f"{cfg['label']}_{idx:02d}",
                frame_id=idx
            )

            detected      = result["plate_detected"]
            raw_ocr       = result.get("raw_ocr_text", "")
            ocr_conf      = result.get("ocr_confidence", 0.0)
            plate_type    = result.get("plate_type", "UNKNOWN")
            is_valid      = result.get("is_valid_indian_format", False)
            plate_text    = result.get("plate_text", "")
            proc_ms       = result.get("processing_time_ms", 0.0)
            variant       = result.get("selected_variant", "—")
            comp_score    = result.get("composite_score", 0.0)
            candidates    = result.get("ocr_candidates", {})

            # Crop dimensions for reporting
            plate_bbox = result.get("plate_bbox_crop")
            crop_h = crop_w = 0
            if plate_bbox:
                crop_w = plate_bbox[2] - plate_bbox[0]
                crop_h = plate_bbox[3] - plate_bbox[1]

            # Reference lookups
            base  = ORIGINAL_BASELINE.get(fname, {})
            prev  = PREV_INTEGRATED.get(fname, {})

            vs_base = _vs_label(raw_ocr, plate_type, is_valid,
                                base.get("raw",""), base.get("type","UNKNOWN"), base.get("valid", False))
            vs_prev = _vs_label(raw_ocr, plate_type, is_valid,
                                prev.get("raw",""), prev.get("type","UNKNOWN"), prev.get("valid", False))

            # Console output
            print(f"\n  [{idx:02d}] {fname[:60]}")
            print(f"        Image size       : {w}×{h}   Crop: {crop_w}×{crop_h}")
            print(f"        Plate detected   : {detected}")
            if detected:
                print(f"        YOLO conf        : {result.get('detection_confidence', 0):.3f}")
                print(f"        Selected variant : {variant}")
                print(f"        Composite score  : {comp_score:.4f}")
                print(f"        Raw OCR          : '{raw_ocr}'")
                print(f"        Plate text       : '{plate_text}'")
                print(f"        OCR conf         : {ocr_conf:.3f}")
                print(f"        Plate type       : {plate_type}  (valid_indian={is_valid})")

                # Candidate comparison
                for vname, vinfo in candidates.items():
                    ran_flag = "" if vinfo.get("ran", True) else " [SKIPPED]"
                    print(f"          {vname:<24} : text='{vinfo.get('text',''):25s}' conf={vinfo.get('conf',0):.3f} score={vinfo.get('score',0):.4f}{ran_flag}")

                print(f"        Baseline OCR     : '{base.get('raw','—')}'  type={base.get('type','?')}  valid={base.get('valid','?')}")
                print(f"        Prev pipeline OCR: '{prev.get('raw','—')}'  type={prev.get('type','?')}  valid={prev.get('valid','?')}")
                print(f"        vs Baseline      : {vs_base}")
                print(f"        vs Prev Pipeline : {vs_prev}")
            else:
                print(f"        [NO PLATE DETECTED]")
            print(f"        Proc time        : {proc_ms:.1f} ms")

            records.append({
                "group":           group_name,
                "image_file":      fname,
                "image_size":      [w, h],
                "crop_size":       [crop_w, crop_h],
                "plate_detected":  detected,
                "detection_conf":  result.get("detection_confidence", 0.0),
                "selected_variant": variant,
                "composite_score": comp_score,
                "raw_ocr":         raw_ocr,
                "plate_text":      plate_text,
                "ocr_conf":        ocr_conf,
                "plate_type":      plate_type,
                "is_valid_indian": is_valid,
                "proc_ms":         proc_ms,
                "baseline_raw":    base.get("raw", "—"),
                "baseline_type":   base.get("type", "?"),
                "baseline_valid":  base.get("valid", False),
                "prev_raw":        prev.get("raw", "—"),
                "prev_type":       prev.get("type", "?"),
                "prev_valid":      prev.get("valid", False),
                "vs_baseline":     vs_base,
                "vs_prev":         vs_prev,
                "candidates":      {k: {"text": v.get("text",""), "conf": v.get("conf",0), "score": v.get("score",0), "ran": v.get("ran", True)} for k,v in candidates.items()},
            })

    # ── Summary Table ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY TABLE — NEW MULTI-CANDIDATE PIPELINE")
    print(sep)
    print(f"{'#':<4} {'Group':<22} {'File':<38} {'New OCR':<25} {'Type':<9} {'Variant':<24} {'vs Baseline':<38} {'vs Prev':<38}")
    print("─" * 170)

    improved_base, same_base, regressed_base = 0, 0, 0
    improved_prev, same_prev, regressed_prev = 0, 0, 0

    for i, r in enumerate(records, 1):
        vb = r["vs_baseline"]
        vp = r["vs_prev"]

        if   "IMPROVED"  in vb: improved_base  += 1
        elif "REGRESSED" in vb: regressed_base += 1
        else:                   same_base      += 1

        if   "IMPROVED"  in vp: improved_prev  += 1
        elif "REGRESSED" in vp: regressed_prev += 1
        else:                   same_prev      += 1

        raw_trunc = (r["raw_ocr"] or "")[:24]
        print(f"{i:<4} {r['group']:<22} {r['image_file'][:37]:<38} {raw_trunc:<25} {r['plate_type']:<9} {r['selected_variant']:<24} {vb:<38} {vp:<38}")

    print(f"\n{'─'*110}")
    print(f"vs Original Baseline : IMPROVED={improved_base}  SAME={same_base}  REGRESSED={regressed_base}")
    print(f"vs Prev Pipeline     : IMPROVED={improved_prev}  SAME={same_prev}  REGRESSED={regressed_prev}")

    # ── Per-group stats ───────────────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print("PER-GROUP AGGREGATE METRICS")
    print(f"{'─'*110}")

    for group in SAMPLE_CONFIG:
        grp_recs    = [r for r in records if r["group"] == group]
        n_total     = len(grp_recs)
        n_det       = sum(1 for r in grp_recs if r["plate_detected"])
        n_ocr       = sum(1 for r in grp_recs if r["raw_ocr"])
        n_valid_ind = sum(1 for r in grp_recs if r["is_valid_indian"])
        latencies   = [r["proc_ms"] for r in grp_recs]
        confs       = [r["ocr_conf"] for r in grp_recs if r["raw_ocr"]]
        avg_conf    = float(np.mean(confs)) if confs else 0.0
        avg_ms      = float(np.mean(latencies)) if latencies else 0.0

        n_valid_base = sum(1 for r in grp_recs if r["baseline_valid"])
        n_valid_prev = sum(1 for r in grp_recs if r["prev_valid"])

        print(f"\n  {group}:")
        print(f"    Total images          : {n_total}")
        print(f"    Plates detected       : {n_det}/{n_total}")
        print(f"    OCR non-empty         : {n_ocr}/{n_det if n_det else 1}")
        if "Indian" in group:
            print(f"    Valid Indian (baseline): {n_valid_base}/{n_total}")
            print(f"    Valid Indian (prev)     : {n_valid_prev}/{n_total}")
            print(f"    Valid Indian (NEW)      : {n_valid_ind}/{n_total}   ← target: ≥ prev")
        print(f"    Avg OCR confidence    : {avg_conf:.3f}")
        print(f"    Avg proc time (ms)    : {avg_ms:.1f}")

    # ── Regression analysis ───────────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print("REGRESSION ANALYSIS")
    print(f"{'─'*110}")
    regressions = [r for r in records if "REGRESSED" in r["vs_baseline"]]
    if regressions:
        print(f"  REGRESSIONS vs baseline ({len(regressions)}):")
        for r in regressions:
            print(f"    [{r['image_file']}]  new='{r['raw_ocr']}'  baseline='{r['baseline_raw']}'  vs={r['vs_baseline']}")
    else:
        print("  ✅ Zero regressions vs original baseline.")

    regressions_prev = [r for r in records if "REGRESSED" in r["vs_prev"]]
    if regressions_prev:
        print(f"\n  REGRESSIONS vs prev pipeline ({len(regressions_prev)}):")
        for r in regressions_prev:
            print(f"    [{r['image_file']}]  new='{r['raw_ocr']}'  prev='{r['prev_raw']}'  vs={r['vs_prev']}")
    else:
        print("  ✅ Zero regressions vs previous pipeline.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "comparative_audit_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": ts, "records": records}, f, indent=2)

    print(f"\n{sep}")
    print(f"[SAVED] {json_path}")
    print("[DONE] Comparative audit complete.\n")


if __name__ == "__main__":
    run_audit()
