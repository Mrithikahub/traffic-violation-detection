"""
OCR Candidate Selection Experiment
====================================
Isolated audit script — zero production source changes.

For every detected plate crop in the 16 reference audit images:
  1. Generate 4 preprocessing variants independently.
  2. Run EasyOCR with spatial bounding-box sorting on each variant.
  3. Compute a transparent, weighted composite quality score per candidate.
  4. Select the winner using the composite score.
  5. Compare against (a) original baseline and (b) current integrated pipeline.

Preprocessing Variants:
  V0_COLOR_CLAHE       : Original baseline  — color CLAHE → bilateral filter
  V1_GRAY_CLAHE        : Current primary    — grayscale + CLAHE + bilateral
  V2_ADAPTIVE_BINARIZE : Polarity-corrected Otsu binarization
  V3_SHARP_GRAY        : Sharpened grayscale (unsharp mask, only used when genuinely distinct)

Composite Score:
  For Indian-expected images:
    score = 0.40 * ocr_confidence
          + 0.20 * length_score          (min(1, len(alnum_only)/8))
          + 0.15 * alnum_ratio           (fraction of non-garbage chars)
          + 0.25 * indian_bonus          (1.0=valid, 0.5=plausible, 0.0=no)

  For Foreign-expected images:
    score = 0.50 * ocr_confidence
          + 0.30 * length_score
          + 0.20 * alnum_ratio
          + 0.00 * indian_bonus          (never forced)

Goal: zero unnecessary regressions — specifically recover video8_1350 while
retaining video6_1140, video9_260, and CarLongPlateGen3407.
"""

import os, sys, time, json, re, datetime
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(NPR_ROOT, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt")
OUT_DIR = os.path.join(NPR_ROOT, "audit_outputs")

sys.path.insert(0, NPR_ROOT)
from src.detector import NumberPlateDetector
from src.preprocessor import PlatePreprocessor
from src.postprocessor import IndianPlatePostProcessor

import easyocr


# ── Hard-coded reference results from recorded audits ────────────────────────
# Baseline = original pipeline (color CLAHE + arbitrary EasyOCR ordering, NO spatial sort)
# New pipeline = current integration (gray_clahe primary + adaptive_binarize fallback + spatial sort)

BASELINE = {
    "video5_130.jpg":      {"raw": "MH.03.CS.6266",           "conf": 0.866, "type": "INDIAN",  "valid": True},
    "video5_500.jpg":      {"raw": "MH03C5.6266",             "conf": 0.433, "type": "INDIAN",  "valid": True},
    "video6_1140.jpg":     {"raw": "9365MHO2 DS",             "conf": 0.940, "type": "UNKNOWN", "valid": False},
    "video6_910.jpg":      {"raw": "MH02 DS 9365",            "conf": 0.755, "type": "INDIAN",  "valid": True},
    "video8_1350.jpg":     {"raw": "Mh0186716",               "conf": 0.493, "type": "INDIAN",  "valid": True},
    "video8_20.jpg":       {"raw": "'CbMhO2C1,3654]",         "conf": 0.169, "type": "UNKNOWN", "valid": False},
    "video8_790.jpg":      {"raw": "HH 12SF3212",             "conf": 0.978, "type": "UNKNOWN", "valid": False},
    "video9_260.jpg":      {"raw": '"Wh02FN2783',             "conf": 0.117, "type": "UNKNOWN", "valid": False},
    "42-1280px-Alberta_1997_license_plate_-_SNX-000_jpg.rf.5f5ae71c2ea5e7557309703b93ffe152.jpg": {"raw": "DrBB VS OarOf JUN Abea Aber", "conf": 0.468, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen1231_jpg.rf.5262da9a0ae856b18adc04275cf1ee9f.jpg": {"raw": "61E.07973",     "conf": 0.404, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen2310_jpg.rf.e2e02c43d0fd1aa9eea2f7b85015acdc.jpg": {"raw": "614.897 14",   "conf": 0.392, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen3407_jpg.rf.a6e5af41538fdfa730f4820d589970cc.jpg": {"raw": "1",             "conf": 0.155, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen998_jpg.rf.08124bcec86fbde5a451e8aee0946197.jpg":  {"raw": "srriu4",        "conf": 0.057, "type": "UNKNOWN", "valid": False},
    "pic_259_jpg.rf.bdeedbfb55112ba4c5d924fe8e736fb0.jpg":             {"raw": "M8 7227 nz9nnartua?", "conf": 0.250, "type": "UNKNOWN", "valid": False},
    "xemay1573_jpg.rf.6b2d6461a87d8babbb107ee90edba7cd.jpg":           {"raw": "59-B712338,",  "conf": 0.757, "type": "UNKNOWN", "valid": False},
    "xemay424_jpg.rf.6ee2887c5e4e8dd45f69501451592b57.jpg":            {"raw": "59-C1 399.1 6","conf": 0.482, "type": "UNKNOWN", "valid": False},
}

NEW_PIPELINE = {
    "video5_130.jpg":      {"raw": "MH.03.CS.6266",           "conf": 0.849, "type": "INDIAN",  "valid": True},
    "video5_500.jpg":      {"raw": "MH03C3.6266",             "conf": 0.439, "type": "INDIAN",  "valid": True},
    "video6_1140.jpg":     {"raw": "MH02 DS 9365",            "conf": 0.717, "type": "INDIAN",  "valid": True},
    "video6_910.jpg":      {"raw": "MH02 DS 9365",            "conf": 0.676, "type": "INDIAN",  "valid": True},
    "video8_1350.jpg":     {"raw": "MFOiBG 716",              "conf": 0.451, "type": "UNKNOWN", "valid": False},
    "video8_20.jpg":       {"raw": "Hho2C LCd,J654",          "conf": 0.250, "type": "UNKNOWN", "valid": False},
    "video8_790.jpg":      {"raw": "HH 12 SF 3212",           "conf": 0.980, "type": "UNKNOWN", "valid": False},
    "video9_260.jpg":      {"raw": "MHOZFN2783",              "conf": 0.622, "type": "INDIAN",  "valid": True},
    "42-1280px-Alberta_1997_license_plate_-_SNX-000_jpg.rf.5f5ae71c2ea5e7557309703b93ffe152.jpg": {"raw": "DrEes US O4r 0 JUN bba A", "conf": 0.565, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen1231_jpg.rf.5262da9a0ae856b18adc04275cf1ee9f.jpg": {"raw": "51F.07973",     "conf": 0.550, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen2310_jpg.rf.e2e02c43d0fd1aa9eea2f7b85015acdc.jpg": {"raw": "51A .897.1L",  "conf": 0.286, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen3407_jpg.rf.a6e5af41538fdfa730f4820d589970cc.jpg": {"raw": "614 691,72",   "conf": 0.597, "type": "UNKNOWN", "valid": False},
    "CarLongPlateGen998_jpg.rf.08124bcec86fbde5a451e8aee0946197.jpg":  {"raw": "1672w9]",      "conf": 0.071, "type": "UNKNOWN", "valid": False},
    "pic_259_jpg.rf.bdeedbfb55112ba4c5d924fe8e736fb0.jpg":             {"raw": "MU 7227 nzIMwNMtVa?","conf": 0.313,"type": "UNKNOWN","valid": False},
    "xemay1573_jpg.rf.6b2d6461a87d8babbb107ee90edba7cd.jpg":           {"raw": "59-B1 338.72", "conf": 0.507, "type": "UNKNOWN", "valid": False},
    "xemay424_jpg.rf.6ee2887c5e4e8dd45f69501451592b57.jpg":            {"raw": "59-C1 39916",  "conf": 0.742, "type": "UNKNOWN", "valid": False},
}


# ── Preprocessing Variants (isolated from production) ────────────────────────

def v0_color_clahe(crop: np.ndarray, prep: PlatePreprocessor) -> np.ndarray:
    """V0: Original baseline — color CLAHE → bilateral filter."""
    result = prep.preprocess_pipeline(crop)
    return result["enhanced_color"]


def v1_gray_clahe(crop: np.ndarray, prep: PlatePreprocessor) -> np.ndarray:
    """V1: Grayscale + CLAHE + bilateral (current primary path)."""
    return prep.preprocess_grayscale_clahe(crop)


def v2_adaptive_binarize(crop: np.ndarray, prep: PlatePreprocessor) -> np.ndarray:
    """V2: Polarity-corrected Otsu binarization."""
    return prep.preprocess_adaptive_binarize(crop)


def v3_sharp_gray(crop: np.ndarray) -> np.ndarray:
    """V3: Sharpened grayscale — upscale, CLAHE, then unsharp-mask sharpening."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    effective_target = 180
    scale = max(1.0, effective_target / float(h))
    interp = cv2.INTER_LANCZOS4 if scale > 2.0 else cv2.INTER_CUBIC
    upscaled = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=interp)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled.copy()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def _is_distinct_from(text_a: str, text_b: str, threshold: int = 3) -> bool:
    """Returns True if edit distance between cleaned texts exceeds threshold."""
    a = re.sub(r'[^A-Za-z0-9]', '', text_a).upper()
    b = re.sub(r'[^A-Za-z0-9]', '', text_b).upper()
    if not a and not b:
        return False
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n] >= threshold


# ── EasyOCR with Spatial Bounding-Box Sorting (isolated copy) ─────────────────

def sort_easyocr_spatially(results: list) -> tuple:
    """Sort raw EasyOCR boxes top-to-bottom, left-to-right within each line cluster."""
    if not results:
        return [], []
    boxes = []
    for res in results:
        bbox, text, conf = res[0], res[1].strip(), float(res[2])
        if not text:
            continue
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        min_y, max_y = min(ys), max(ys)
        boxes.append({
            "text": text, "conf": conf,
            "min_x": min(xs),
            "cy": (min_y + max_y) / 2.0,
            "box_h": max(1.0, max_y - min_y),
        })
    if not boxes:
        return [], []
    boxes.sort(key=lambda b: b["cy"])
    mean_box_h = float(np.mean([b["box_h"] for b in boxes]))
    cluster_threshold = mean_box_h * 0.40
    lines, current_line = [], [boxes[0]]
    for box in boxes[1:]:
        if abs(box["cy"] - current_line[-1]["cy"]) <= cluster_threshold:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b["min_x"]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda b: b["min_x"]))
    texts, confs = [], []
    for line in lines:
        for b in line:
            texts.append(b["text"])
            confs.append(b["conf"])
    full_text = " ".join(texts)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, avg_conf


def run_easyocr(reader: easyocr.Reader, image: np.ndarray) -> tuple:
    """Run EasyOCR with spatial sorting. Returns (raw_text, confidence)."""
    try:
        results = reader.readtext(image)
        raw_text, conf = sort_easyocr_spatially(results)
        return raw_text, conf
    except Exception as e:
        return "", 0.0


# ── Composite Quality Scoring ─────────────────────────────────────────────────

def _alnum_only(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', text) if text else ""


def indian_plausibility_bonus(raw_text: str, postproc: IndianPlatePostProcessor) -> float:
    """
    0.0 = text has no resemblance to Indian format
    0.5 = text plausibly starts with a known (or near-miss) state code
    1.0 = text passes full STANDARD_REGEX or BH-series validation
    """
    try:
        parsed = postproc.validate_and_parse(raw_text)
        if parsed.get("is_valid", False):
            return 1.0
        cleaned = postproc.clean_raw_text(raw_text)
        if postproc._looks_like_indian(cleaned):
            return 0.5
    except Exception:
        pass
    return 0.0


def compute_score(raw_text: str, conf: float, is_indian_group: bool,
                  postproc: IndianPlatePostProcessor) -> dict:
    """
    Compute composite quality score with explicit per-component breakdown.

    Indian-expected:  0.40*conf + 0.20*length + 0.15*alnum_ratio + 0.25*indian_bonus
    Foreign-expected: 0.50*conf + 0.30*length + 0.20*alnum_ratio + 0.00*indian_bonus

    Returns a dict with score + all components.
    """
    text = raw_text.strip() if raw_text else ""
    alnum = _alnum_only(text)
    length_score = min(1.0, len(alnum) / 8.0)
    alnum_ratio = len(alnum) / max(1, len(text)) if text else 0.0
    bonus = indian_plausibility_bonus(raw_text, postproc) if is_indian_group else 0.0

    if is_indian_group:
        score = 0.40 * conf + 0.20 * length_score + 0.15 * alnum_ratio + 0.25 * bonus
    else:
        score = 0.50 * conf + 0.30 * length_score + 0.20 * alnum_ratio

    return {
        "composite_score": round(score, 4),
        "conf_component":  round((0.40 if is_indian_group else 0.50) * conf, 4),
        "length_component": round((0.20 if is_indian_group else 0.30) * length_score, 4),
        "alnum_component": round((0.15 if is_indian_group else 0.20) * alnum_ratio, 4),
        "indian_bonus_component": round(0.25 * bonus if is_indian_group else 0.0, 4),
        "raw_length_score": round(length_score, 4),
        "raw_alnum_ratio": round(alnum_ratio, 4),
        "raw_indian_bonus": round(bonus, 4),
    }


def outcome_vs_baseline(winner_score: float, winner_valid: bool,
                         baseline_raw: str, baseline_conf: float,
                         baseline_valid: bool, is_indian_group: bool,
                         postproc: IndianPlatePostProcessor) -> str:
    """
    Classify whether the selected winner is:
      IMPROVED  - valid Indian gained, or composite score clearly higher
      SAME      - no material difference
      REGRESSED - valid Indian lost, or composite score clearly lower
    """
    # Compute score for the baseline result using the same formula
    bscore = compute_score(baseline_raw, baseline_conf, is_indian_group, postproc)["composite_score"]
    delta = winner_score - bscore

    if is_indian_group:
        if winner_valid and not baseline_valid:
            return "IMPROVED (gained valid Indian)"
        if not winner_valid and baseline_valid:
            return "REGRESSED (lost valid Indian)"

    if delta >= 0.04:
        return f"IMPROVED (score +{delta:.3f})"
    elif delta <= -0.04:
        return f"REGRESSED (score {delta:.3f})"
    else:
        return f"SAME (score Δ={delta:+.3f})"


# ── Main Experiment ───────────────────────────────────────────────────────────

# 16 audit images with group and path info
AUDIT_IMAGES = [
    # Indian Validation
    {"file": "video5_130.jpg",  "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video5_500.jpg",  "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video6_1140.jpg", "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video6_910.jpg",  "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video8_1350.jpg", "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video8_20.jpg",   "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video8_790.jpg",  "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    {"file": "video9_260.jpg",  "group": "Indian Validation", "expected": "INDIAN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val"},
    # Foreign Unseen Test
    {"file": "42-1280px-Alberta_1997_license_plate_-_SNX-000_jpg.rf.5f5ae71c2ea5e7557309703b93ffe152.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "CarLongPlateGen1231_jpg.rf.5262da9a0ae856b18adc04275cf1ee9f.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "CarLongPlateGen2310_jpg.rf.e2e02c43d0fd1aa9eea2f7b85015acdc.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "CarLongPlateGen3407_jpg.rf.a6e5af41538fdfa730f4820d589970cc.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "CarLongPlateGen998_jpg.rf.08124bcec86fbde5a451e8aee0946197.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "pic_259_jpg.rf.bdeedbfb55112ba4c5d924fe8e736fb0.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "xemay1573_jpg.rf.6b2d6461a87d8babbb107ee90edba7cd.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
    {"file": "xemay424_jpg.rf.6ee2887c5e4e8dd45f69501451592b57.jpg",
     "group": "Foreign Unseen Test", "expected": "FOREIGN",
     "path": r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"},
]


def run_experiment():
    print("=" * 120)
    print("OCR CANDIDATE SELECTION EXPERIMENT")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    detector = NumberPlateDetector(model_path=WEIGHTS_PATH, conf_threshold=0.35)
    prep     = PlatePreprocessor(target_height=140, pad_percent=0.06)
    postproc = IndianPlatePostProcessor()
    reader   = easyocr.Reader(['en'], gpu=False)

    all_results = []
    variant_names = ["V0_COLOR_CLAHE", "V1_GRAY_CLAHE", "V2_ADAPTIVE_BINARIZE", "V3_SHARP_GRAY"]

    for idx, entry in enumerate(AUDIT_IMAGES, 1):
        fname  = entry["file"]
        group  = entry["group"]
        is_indian = entry["expected"] == "INDIAN"
        img_path  = os.path.join(entry["path"], fname)

        print(f"\n[{idx:02d}] {fname[:52]} | {group}")

        if not os.path.exists(img_path):
            print(f"       !! FILE NOT FOUND: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print("       !! imread returned None")
            continue

        # Detect plate
        detections = detector.detect(img)
        if not detections:
            print("       !! No plate detected — skipping")
            continue
        best_det = max(detections, key=lambda d: d["confidence"])
        raw_crop = prep.crop_with_padding(img, best_det["bbox"])
        ch, cw   = raw_crop.shape[:2]
        crop_ar  = round(cw / float(ch), 2) if ch > 0 else 1.0

        print(f"       Crop: {cw}x{ch}  AR: {crop_ar}  YOLO conf: {best_det['confidence']:.3f}")

        # Generate all 4 variants
        t_start = time.perf_counter()
        try:
            img_v0 = v0_color_clahe(raw_crop, prep)
        except Exception as e:
            img_v0 = raw_crop; print(f"       V0 prep error: {e}")
        try:
            img_v1 = v1_gray_clahe(raw_crop, prep)
        except Exception as e:
            img_v1 = raw_crop; print(f"       V1 prep error: {e}")
        try:
            img_v2 = v2_adaptive_binarize(raw_crop, prep)
        except Exception as e:
            img_v2 = raw_crop; print(f"       V2 prep error: {e}")
        try:
            img_v3 = v3_sharp_gray(raw_crop)
        except Exception as e:
            img_v3 = raw_crop; print(f"       V3 prep error: {e}")

        variants_images = {
            "V0_COLOR_CLAHE":      img_v0,
            "V1_GRAY_CLAHE":       img_v1,
            "V2_ADAPTIVE_BINARIZE": img_v2,
            "V3_SHARP_GRAY":       img_v3,
        }

        # Run OCR on each variant
        candidates = {}
        for vname, vimg in variants_images.items():
            t0 = time.perf_counter()
            raw_text, conf = run_easyocr(reader, vimg)
            ocr_ms = (time.perf_counter() - t0) * 1000.0

            parsed = postproc.validate_and_parse(raw_text)
            score_info = compute_score(raw_text, conf, is_indian, postproc)

            candidates[vname] = {
                "raw_text":      raw_text,
                "conf":          round(conf, 3),
                "cleaned_text":  parsed.get("cleaned_text", ""),
                "plate_type":    parsed.get("plate_type", "UNKNOWN"),
                "is_valid_indian": parsed.get("is_valid", False),
                "ocr_ms":        round(ocr_ms, 1),
                **score_info,
            }

        # Determine V3 distinctiveness (compare vs V1)
        v3_is_distinct = _is_distinct_from(
            candidates["V3_SHARP_GRAY"]["raw_text"],
            candidates["V1_GRAY_CLAHE"]["raw_text"],
            threshold=3
        )
        candidates["V3_SHARP_GRAY"]["is_distinct_from_v1"] = v3_is_distinct

        # Select winner: if V3 is not genuinely distinct, exclude it from selection
        # to avoid adding noise from an uninformative duplicate
        eligible = {k: v for k, v in candidates.items()
                    if k != "V3_SHARP_GRAY" or v["is_distinct_from_v1"]}
        if not eligible:
            eligible = candidates  # fallback: use all

        winner_name = max(eligible, key=lambda k: eligible[k]["composite_score"])
        winner = eligible[winner_name]

        # Cross-check: if winner is Indian-valid and V0 is also Indian-valid,
        # prefer higher composite score (the bonus already handles this)
        # Edge case: if winner changes valid→invalid vs baseline, flag it
        baseline_rec = BASELINE.get(fname, {})
        new_pipe_rec = NEW_PIPELINE.get(fname, {})

        outcome_vs_bl  = outcome_vs_baseline(
            winner["composite_score"], winner["is_valid_indian"],
            baseline_rec.get("raw", ""), baseline_rec.get("conf", 0.0),
            baseline_rec.get("valid", False), is_indian, postproc
        )
        outcome_vs_new = outcome_vs_baseline(
            winner["composite_score"], winner["is_valid_indian"],
            new_pipe_rec.get("raw", ""), new_pipe_rec.get("conf", 0.0),
            new_pipe_rec.get("valid", False), is_indian, postproc
        )

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Print candidate table
        print(f"       {'Variant':<22} | {'Raw OCR':<26} | Conf  | Score  | Type    | Valid | ms")
        print(f"       {'-'*100}")
        for vname in variant_names:
            c = candidates[vname]
            distinct_tag = "  [D]" if vname == "V3_SHARP_GRAY" and c["is_distinct_from_v1"] else ""
            winner_tag   = " <<" if vname == winner_name else ""
            print(f"       {vname:<22} | {c['raw_text'][:26]:<26} | {c['conf']:.3f} | {c['composite_score']:.4f} | {c['plate_type']:<7} | {'YES' if c['is_valid_indian'] else 'NO':<5} | {c['ocr_ms']:.0f}{distinct_tag}{winner_tag}")

        bline_score = compute_score(baseline_rec.get("raw",""), baseline_rec.get("conf",0.0), is_indian, postproc)["composite_score"]
        npipe_score = compute_score(new_pipe_rec.get("raw",""), new_pipe_rec.get("conf",0.0), is_indian, postproc)["composite_score"]
        print(f"       {'[BASELINE]':<22} | {baseline_rec.get('raw','')[:26]:<26} | {baseline_rec.get('conf',0):.3f} | {bline_score:.4f} | {baseline_rec.get('type',''):<7} | {'YES' if baseline_rec.get('valid') else 'NO':<5} |")
        print(f"       {'[NEW_PIPELINE]':<22} | {new_pipe_rec.get('raw','')[:26]:<26} | {new_pipe_rec.get('conf',0):.3f} | {npipe_score:.4f} | {new_pipe_rec.get('type',''):<7} | {'YES' if new_pipe_rec.get('valid') else 'NO':<5} |")
        print(f"       Winner: {winner_name}  |  vs Baseline: {outcome_vs_bl}  |  vs New Pipeline: {outcome_vs_new}")

        all_results.append({
            "index": idx,
            "file":  fname,
            "group": group,
            "expected_label": entry["expected"],
            "crop_dimensions_wh": [cw, ch],
            "crop_aspect_ratio": crop_ar,
            "yolo_conf": best_det["confidence"],
            "candidates": candidates,
            "winner_variant": winner_name,
            "winner": winner,
            "baseline": baseline_rec,
            "new_pipeline": new_pipe_rec,
            "outcome_vs_baseline":    outcome_vs_bl,
            "outcome_vs_new_pipeline": outcome_vs_new,
            "total_experiment_ms":    round(total_ms, 1),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY: CANDIDATE SELECTION OUTCOME")
    print("=" * 120)
    print(f"{'#':<4} {'File':<35} {'Winner':<22} {'Winner Text':<22} {'Valid':<6} {'vs Baseline':<38} {'vs New Pipeline'}")
    print("-" * 160)

    improved_bl, same_bl, regressed_bl   = 0, 0, 0
    improved_np, same_np, regressed_np   = 0, 0, 0

    for r in all_results:
        bl_tag  = r["outcome_vs_baseline"]
        np_tag  = r["outcome_vs_new_pipeline"]
        if "IMPROVED" in bl_tag:  improved_bl  += 1
        elif "SAME" in bl_tag:    same_bl      += 1
        else:                     regressed_bl += 1
        if "IMPROVED" in np_tag:  improved_np  += 1
        elif "SAME" in np_tag:    same_np      += 1
        else:                     regressed_np += 1

        print(f"{r['index']:<4} {r['file'][:34]:<35} {r['winner_variant']:<22} "
              f"{r['winner']['raw_text'][:21]:<22} {'YES' if r['winner']['is_valid_indian'] else 'NO':<6} "
              f"{bl_tag[:37]:<38} {np_tag[:37]}")

    print("-" * 160)
    print(f"\nvs ORIGINAL BASELINE : IMPROVED={improved_bl}  SAME={same_bl}  REGRESSED={regressed_bl}")
    print(f"vs NEW PIPELINE      : IMPROVED={improved_np}  SAME={same_np}  REGRESSED={regressed_np}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "candidate_selection_experiment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scoring": {
                "indian": "0.40*conf + 0.20*length + 0.15*alnum_ratio + 0.25*indian_bonus",
                "foreign": "0.50*conf + 0.30*length + 0.20*alnum_ratio",
            },
            "summary": {
                "vs_baseline":     {"improved": improved_bl, "same": same_bl, "regressed": regressed_bl},
                "vs_new_pipeline": {"improved": improved_np, "same": same_np, "regressed": regressed_np},
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n[SAVED] {json_path}")


if __name__ == "__main__":
    run_experiment()
