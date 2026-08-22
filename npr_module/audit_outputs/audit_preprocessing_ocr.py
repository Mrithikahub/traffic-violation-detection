"""
Comprehensive Preprocessing & OCR Improvement Audit
===================================================
Evaluates multiple preprocessing variants and OCR recognition strategies on
the 16 representative images from the E2E inference audit (8 Indian + 8 Foreign).

Variants Evaluated:
-------------------
1. V0_Baseline: Current pipeline (deskew -> upscale(140) -> CLAHE -> Bilateral)
2. V1_HighRes_Upscale: High-resolution upscaling (target_height=220, Lanczos/Cubic)
3. V2_Grayscale_CLAHE: Grayscale + CLAHE (clipLimit=3.0, grid=(8,8))
4. V3_Adaptive_Binarize: Adaptive Gaussian / Otsu thresholding with background polarity check
5. V4_Sharpen_Unsharp: Unsharp mask sharpening + Bilateral edge preservation
6. V5_LineAware_Adaptive: Aspect-ratio aware routing (1-line long vs 2-line square) + spatial box sorting

Also evaluates:
- EasyOCR spatial bounding-box layout sorting (top-to-bottom, left-to-right)
- Indian Postprocessor recovery of character confusions (0/O, 1/I, 2/Z, 5/S, 8/B)

Outputs:
- npr_module/audit_outputs/preprocessing_ocr_audit.json
"""

import os, sys, time, json, datetime
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(NPR_ROOT, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt")
MANIFEST_DIR = os.path.join(NPR_ROOT, "configs", "manifests")
OUT_DIR = os.path.join(NPR_ROOT, "audit_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, NPR_ROOT)
from src.detector import NumberPlateDetector
from src.preprocessor import PlatePreprocessor
from src.postprocessor import IndianPlatePostProcessor

import easyocr

# ── 1. Helper Preprocessing Functions ────────────────────────────────────────

def preprocess_v0_baseline(crop: np.ndarray, prep: PlatePreprocessor) -> np.ndarray:
    """Current baseline pipeline: deskew -> upscale(140) -> CLAHE -> Bilateral"""
    res = prep.preprocess_pipeline(crop)
    return res["enhanced_color"]

def preprocess_v1_highres(crop: np.ndarray, target_height: int = 220) -> np.ndarray:
    """High-resolution upscaling with Lanczos/Cubic interpolation + CLAHE + Bilateral"""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    scale = max(1.0, target_height / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if scale > 1.5 else cv2.INTER_CUBIC)
    
    # CLAHE
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

def preprocess_v2_gray_clahe(crop: np.ndarray, target_height: int = 160) -> np.ndarray:
    """Grayscale + CLAHE + Bilateral Denoising"""
    h, w = crop.shape[:2]
    scale = max(1.0, target_height / float(h)) if h > 0 else 1.0
    resized = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced_gray, d=5, sigmaColor=40, sigmaSpace=40)
    return denoised

def preprocess_v3_binarize(crop: np.ndarray, target_height: int = 160) -> np.ndarray:
    """Otsu / Adaptive Thresholding with background polarity check"""
    gray_prep = preprocess_v2_gray_clahe(crop, target_height)
    # Otsu thresholding
    _, otsu = cv2.threshold(gray_prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Check if plate background is dark vs light
    white_px = cv2.countNonZero(otsu)
    total_px = otsu.shape[0] * otsu.shape[1]
    if white_px < total_px * 0.45:
        otsu = cv2.bitwise_not(otsu)
    return otsu

def preprocess_v4_sharpen(crop: np.ndarray, target_height: int = 180) -> np.ndarray:
    """Unsharp Mask Sharpening + Edge-Preserving Contrast"""
    highres = preprocess_v1_highres(crop, target_height)
    gaussian = cv2.GaussianBlur(highres, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(highres, 1.6, gaussian, -0.6, 0)
    return sharpened

def preprocess_v5_line_aware(crop: np.ndarray) -> tuple:
    """
    Aspect-Ratio Aware Routing:
    - If aspect ratio < 2.3: 2-line / Square plate -> high vertical scaling & vertical padding
    - If aspect ratio >= 2.3: 1-line / Long plate -> moderate scaling & horizontal contrast boost
    """
    h, w = crop.shape[:2]
    ar = w / float(h) if h > 0 else 1.0
    if ar < 2.3:
        layout = "2-LINE/SQUARE"
        # 2-line plates need taller vertical resolution for stacked characters
        upscaled = preprocess_v4_sharpen(crop, target_height=220)
    else:
        layout = "1-LINE/LONG"
        upscaled = preprocess_v4_sharpen(crop, target_height=150)
    return upscaled, layout


# ── 2. OCR Extraction with Spatial Layout Sorting ───────────────────────────

def run_easyocr_with_spatial_sorting(reader: easyocr.Reader, image: np.ndarray) -> tuple:
    """
    Runs EasyOCR and sorts detected bounding boxes spatially:
    1. Clusters boxes by vertical Y-line
    2. Sorts within each line horizontally by X-coordinate
    Ensures correct reading order for 2-line and multi-box plates.
    """
    results = reader.readtext(image)
    if not results:
        return "", 0.0, []

    # Parse bounding boxes and text
    parsed_boxes = []
    for res in results:
        bbox, text, conf = res[0], res[1].strip(), float(res[2])
        if not text:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cy = (min_y + max_y) / 2.0
        h_box = max(1, max_y - min_y)
        parsed_boxes.append({
            "text": text,
            "conf": conf,
            "min_x": min_x,
            "min_y": min_y,
            "cy": cy,
            "h": h_box
        })

    if not parsed_boxes:
        return "", 0.0, []

    # Sort boxes: determine if multi-line based on Y centers
    parsed_boxes = sorted(parsed_boxes, key=lambda b: b["cy"])
    
    lines = []
    current_line = [parsed_boxes[0]]
    for box in parsed_boxes[1:]:
        # If vertical separation between box and line is > 45% of average box height -> new line
        avg_h = np.mean([b["h"] for b in current_line])
        if abs(box["cy"] - current_line[-1]["cy"]) > avg_h * 0.45:
            lines.append(sorted(current_line, key=lambda b: b["min_x"]))
            current_line = [box]
        else:
            current_line.append(box)
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b["min_x"]))

    # Join lines top-to-bottom, left-to-right
    ordered_texts = []
    confs = []
    for line in lines:
        line_str = " ".join(b["text"] for b in line)
        ordered_texts.append(line_str)
        confs.extend(b["conf"] for b in line)

    full_text = " ".join(ordered_texts)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, avg_conf, ordered_texts


# ── 3. Main Evaluation Runner ────────────────────────────────────────────────

def run_preprocessing_audit():
    print("=" * 100)
    print("PREPROCESSING & OCR VARIANT COMPARISON AUDIT")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    detector = NumberPlateDetector(model_path=WEIGHTS_PATH, conf_threshold=0.35)
    baseline_prep = PlatePreprocessor(target_height=140, pad_percent=0.06)
    postproc = IndianPlatePostProcessor()
    reader = easyocr.Reader(['en'], gpu=False)

    with open(os.path.join(OUT_DIR, "e2e_audit_results.json"), "r") as f:
        e2e_data = json.load(f)
    
    records = e2e_data["records"]
    print(f"Loaded {len(records)} test images from E2E audit record.\n")

    variants = [
        "V0_Baseline",
        "V1_HighRes_Upscale",
        "V2_Grayscale_CLAHE",
        "V3_Adaptive_Binarize",
        "V4_Sharpen_Unsharp",
        "V5_LineAware_Adaptive"
    ]

    all_audit_results = []

    for idx, rec in enumerate(records, 1):
        img_name = rec["image_file"]
        group = rec["group"]
        expected_label = rec["expected_label"]
        
        if expected_label == "INDIAN":
            img_path = os.path.join(r"e:\CDAC Dataset\CDAC_Workspace\indian_no_plates_B\images\val", img_name)
        else:
            img_path = os.path.join(r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images", img_name)

        if not os.path.exists(img_path):
            print(f"[{idx:02d}] Missing file: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        detections = detector.detect(img)
        if not detections:
            print(f"[{idx:02d}] {img_name}: No plate detected.")
            continue

        best_det = max(detections, key=lambda d: d["confidence"])
        plate_box = best_det["bbox"]
        raw_crop = baseline_prep.crop_with_padding(img, plate_box)
        orig_h, orig_w = raw_crop.shape[:2]
        crop_ar = round(orig_w / float(orig_h), 2) if orig_h > 0 else 1.0

        print(f"\n[{idx:02d}] Image: {img_name} ({group}) | Crop: {orig_w}x{orig_h} (AR={crop_ar})")

        img_results = {
            "image_file": img_name,
            "group": group,
            "expected_label": expected_label,
            "crop_dimensions_wh": [orig_w, orig_h],
            "crop_aspect_ratio": crop_ar,
            "variant_results": {}
        }

        for var_name in variants:
            t0 = time.perf_counter()

            if var_name == "V0_Baseline":
                proc_img = preprocess_v0_baseline(raw_crop, baseline_prep)
            elif var_name == "V1_HighRes_Upscale":
                proc_img = preprocess_v1_highres(raw_crop, target_height=220)
            elif var_name == "V2_Grayscale_CLAHE":
                proc_img = preprocess_v2_gray_clahe(raw_crop, target_height=160)
            elif var_name == "V3_Adaptive_Binarize":
                proc_img = preprocess_v3_binarize(raw_crop, target_height=160)
            elif var_name == "V4_Sharpen_Unsharp":
                proc_img = preprocess_v4_sharpen(raw_crop, target_height=180)
            elif var_name == "V5_LineAware_Adaptive":
                proc_img, layout_tag = preprocess_v5_line_aware(raw_crop)

            raw_text, ocr_conf, lines = run_easyocr_with_spatial_sorting(reader, proc_img)
            proc_time_ms = (time.perf_counter() - t0) * 1000.0

            parsed = postproc.validate_and_parse(raw_text)

            img_results["variant_results"][var_name] = {
                "raw_ocr_text": raw_text,
                "ocr_confidence": round(ocr_conf, 3),
                "cleaned_text": parsed["cleaned_text"],
                "standardized_text": parsed["standardized_text"],
                "plate_type": parsed["plate_type"],
                "is_valid_indian": parsed["is_valid"],
                "format_type": parsed["format_type"],
                "processing_time_ms": round(proc_time_ms, 1)
            }

            valid_tag = " [VALID_IN]" if parsed["is_valid"] else ""
            print(f"   {var_name:<24}: OCR='{raw_text[:25]:<25}' Conf={ocr_conf:.3f} Type={parsed['plate_type']:<7}{valid_tag} ({proc_time_ms:.1f}ms)")

        all_audit_results.append(img_results)

    json_path = os.path.join(OUT_DIR, "preprocessing_ocr_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weights": WEIGHTS_PATH,
            "images_evaluated": len(all_audit_results),
            "variants_tested": variants,
            "results": all_audit_results
        }, f, indent=2)

    print(f"\n[SAVED] JSON data: {json_path}")


if __name__ == "__main__":
    run_preprocessing_audit()
