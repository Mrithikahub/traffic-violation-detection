"""
Optical Character Recognition (OCR) Engine Module
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System

Multi-Candidate OCR Selection Strategy
========================================
Three preprocessing candidates are evaluated for every plate crop:

  V0_COLOR_CLAHE       : CLAHE on LAB luminance → bilateral (original baseline)
  V1_GRAY_CLAHE        : Grayscale → CLAHE clipLimit=3 → bilateral
  V2_ADAPTIVE_BINARIZE : Polarity-corrected Otsu binarization on V1 output

Candidate execution is lazy and guarded:
  - V0 always runs.
  - V1 always runs.
  - V2 only runs when:
      (a) V0 and V1 both produce weak results, OR
      (b) crop area is small (height ≤ 60px) — binarization helps small crops more.
      (c) Large crops (area > 400×200px) skip V2 unless V0 and V1 are both weak,
          to avoid tripling latency on frames with already-large plate captures.

Final candidate is selected using a composite quality score:
  Indian-expected weighting  : 0.40*conf + 0.20*length + 0.15*alnum_ratio + 0.25*indian_bonus
  Foreign/unknown weighting  : 0.50*conf + 0.30*length + 0.20*alnum_ratio

The indian_bonus is 0.0 / 0.5 / 1.0 and is evaluated using the plate's cleaned
alphanumeric string against known state codes without importing postprocessor
(to keep the engine independent — the postprocessor still runs downstream).
"""

import re
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# ── Indian state-code knowledge embedded directly (mirrors postprocessor) ─────
_VALID_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL","MP","MH",
    "MN","ML","MZ","NL","OD","OR","PB","RJ","SK","TN","TS","TR","UP","UK",
    "UA","WB","AN","CH","DN","DD","DH","DL","JK","LA","LD","PY",
}
_STANDARD_REGEX   = re.compile(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$')
_BH_SERIES_REGEX  = re.compile(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$')
_TYPO_STATE_MAP   = {"0L":"DL","0D":"OD","1N":"TN","1S":"TS","K4":"KA","M8":"MH","U9":"UP"}

# Large-crop pixel area threshold: above this, V2 is skipped unless both V0 & V1 are weak
_LARGE_CROP_AREA_THRESHOLD = 400 * 200   # 80 000 px²
# Small-crop height threshold: at or below this, V2 always runs (binarize helps small crops)
_SMALL_CROP_HEIGHT_THRESHOLD = 60        # pixels


def _alnum_only(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', text) if text else ""


_RELAXED_INDIAN_REGEX = re.compile(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{3,4})$')

# Minimal positional maps (mirror postprocessor without importing it)
_L2D = {'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','E':'3','A':'4',
         'S':'5','G':'6','b':'6','T':'7','B':'8','g':'9','q':'9','P':'9'}
_D2L = {'0':'O','1':'I','2':'Z','3':'J','4':'A','5':'S',
         '6':'G','7':'T','8':'B','9':'P'}


def _positional_correct(cleaned: str) -> str:
    """
    Apply lightweight Indian plate positional correction without postprocessor.
    Positions 0-1: must be letters (digits → DIGIT_TO_LETTER)
    Positions 2-3: must be digits (letters → LETTER_TO_DIGIT)
    Last 4 positions: must be digits (letters → LETTER_TO_DIGIT)
    Middle positions (series): must be letters (digits → DIGIT_TO_LETTER)
    """
    n = len(cleaned)
    if n < 6:
        return cleaned
    chars = list(cleaned)
    # State code positions: letters
    for i in range(min(2, n)):
        if chars[i].isdigit():
            chars[i] = _D2L.get(chars[i], chars[i])
    # District positions: digits
    for i in range(2, min(4, n)):
        chars[i] = _L2D.get(chars[i].upper(), chars[i])
    # Last 4: digits
    for i in range(max(0, n - 4), n):
        chars[i] = _L2D.get(chars[i].upper(), chars[i])
    # Middle (series): letters
    for i in range(4, max(0, n - 4)):
        if chars[i].isdigit():
            chars[i] = _D2L.get(chars[i], chars[i])
    return ''.join(chars)


def _indian_bonus(text: str) -> float:
    """
    Returns 0.0 / 0.5 / 1.0 indicating how strongly the OCR text resembles
    a valid Indian registration plate.

    0.0 : No resemblance to Indian format.
    0.5 : Starts with a valid or near-valid state code and has plausible length,
          but doesn't fully satisfy the strict format.
    1.0 : Matches Standard Indian or BH-Series regex (strict or relaxed),
          optionally after lightweight positional correction.

    Relaxed match (3-digit terminal group) handles the common OCR error of
    dropping one terminal digit. Positional pre-correction handles digit↔letter
    confusion in fixed positions (e.g. '8G' series → 'BG').

    No postprocessor dependency — uses a self-contained lightweight heuristic.
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper()) if text else ""
    n = len(cleaned)
    if n < 6 or n > 12:
        return 0.0

    # 1. Full strict regex match → maximum bonus
    if _STANDARD_REGEX.match(cleaned) and cleaned[:2] in _VALID_STATE_CODES:
        return 1.0
    if _BH_SERIES_REGEX.match(cleaned):
        return 1.0

    # 2. Relaxed match: allows 3-digit terminal group (one OCR-dropped digit)
    relax = _RELAXED_INDIAN_REGEX.match(cleaned)
    if relax and relax.group(1) in _VALID_STATE_CODES:
        return 1.0

    # 3. Try after lightweight positional correction
    corrected = _positional_correct(cleaned)
    if corrected != cleaned:
        if _STANDARD_REGEX.match(corrected) and corrected[:2] in _VALID_STATE_CODES:
            return 1.0
        relax2 = _RELAXED_INDIAN_REGEX.match(corrected)
        if relax2 and relax2.group(1) in _VALID_STATE_CODES:
            return 1.0

    # 4. Typo state-code map correction
    prefix2 = cleaned[:2]
    corrected_prefix = _TYPO_STATE_MAP.get(prefix2, prefix2)
    if corrected_prefix in _VALID_STATE_CODES:
        # Must have at least a digit or plausible OCR digit after the state code
        if n >= 7 and (cleaned[2].isdigit() or cleaned[2] in "OIZASB"):
            return 0.5

    # 5. Check corrected prefix for relaxed regex
    if prefix2 in _TYPO_STATE_MAP:
        remapped = _TYPO_STATE_MAP[prefix2] + cleaned[2:]
        if _RELAXED_INDIAN_REGEX.match(remapped):
            return 1.0
        # Also try with positional correction on remapped
        remapped_corr = _positional_correct(remapped)
        if _RELAXED_INDIAN_REGEX.match(remapped_corr) or (_STANDARD_REGEX.match(remapped_corr) and remapped_corr[:2] in _VALID_STATE_CODES):
            return 1.0

    return 0.0



def _composite_score(raw_text: str, conf: float, crop_h: int, crop_w: int) -> dict:
    """
    Compute the multi-signal composite quality score for a single OCR candidate.

    Score breakdown:
      Indian-likely (indian_bonus > 0):
        0.40 * ocr_confidence
        + 0.20 * length_score       (min(1, alnum_len / 8))
        + 0.15 * alnum_ratio        (alnum chars / total raw chars)
        + 0.25 * indian_bonus       (0 / 0.5 / 1.0)

      Foreign/unknown (indian_bonus == 0):
        0.50 * ocr_confidence
        + 0.30 * length_score
        + 0.20 * alnum_ratio

    Returns a dict with the total score and all component values for transparency.
    """
    text = (raw_text or "").strip()
    alnum = _alnum_only(text)
    length_score = min(1.0, len(alnum) / 8.0)
    alnum_ratio  = len(alnum) / max(1, len(text)) if text else 0.0
    bonus        = _indian_bonus(text)

    if bonus > 0.0:
        score = (0.40 * conf
                 + 0.20 * length_score
                 + 0.15 * alnum_ratio
                 + 0.25 * bonus)
    else:
        score = (0.50 * conf
                 + 0.30 * length_score
                 + 0.20 * alnum_ratio)

    return {
        "composite_score":    round(score, 4),
        "conf_component":     round(conf, 3),
        "length_score":       round(length_score, 4),
        "alnum_ratio":        round(alnum_ratio, 4),
        "indian_bonus":       round(bonus, 4),
    }


def _is_weak(raw_text: str, conf: float, crop_h: int = 0) -> bool:
    """
    Returns True when an OCR result is weak enough to warrant trying additional
    preprocessing variants.

    Criteria (any one is sufficient):
      - Text is empty or fewer than 4 characters.
      - OCR confidence is below 0.32.
      - Alnum-only text has fewer than 4 characters (filters punctuation-only results).
      - For small crops (height ≤ SMALL threshold): confidence below 0.45.
    """
    text  = (raw_text or "").strip()
    alnum = _alnum_only(text)
    if len(text) < 4 or len(alnum) < 4:
        return True
    if conf < 0.32:
        return True
    if crop_h > 0 and crop_h <= _SMALL_CROP_HEIGHT_THRESHOLD and conf < 0.45:
        return True
    return False


class PlateOCREngine:
    """
    Robust OCR Engine for license plates.

    Supports:
    - PaddleOCR (when installed)
    - EasyOCR (primary fallback on this system)
    - Built-in contour-based character segmentation (last resort)

    Multi-Candidate Selection:
      Three preprocessing variants (V0/V1/V2) are run selectively.
      The best candidate is chosen via a transparent composite quality score
      that weights OCR confidence, text length, character cleanliness, and
      Indian plate structural validity.
    """

    def __init__(self, use_paddle: bool = True, use_gpu: bool = False):
        self.paddle_ocr   = None
        self.easy_ocr     = None
        self.engine_type  = "built_in"

        if use_paddle:
            try:
                from paddleocr import PaddleOCR
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True, lang='en',
                    show_log=False, use_gpu=use_gpu
                )
                self.engine_type = "paddleocr"
            except Exception:
                self.paddle_ocr = None

        if self.paddle_ocr is None:
            try:
                import easyocr
                self.easy_ocr = easyocr.Reader(['en'], gpu=use_gpu)
                self.engine_type = "easyocr"
            except Exception:
                self.easy_ocr = None

    # ── Contour-based segmentation (last-resort built-in path) ────────────────

    def segment_character_contours(self, binarized_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments potential alphanumeric character blobs using contour analysis,
        aspect ratio filters, and bounding box sorting.
        """
        h, w = binarized_img.shape[:2]
        contours, _ = cv2.findContours(
            cv2.bitwise_not(binarized_img),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        char_candidates = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            rel_h = ch / float(h)
            rel_w = cw / float(w)
            aspect_ratio = cw / float(ch) if ch > 0 else 0
            if 0.20 <= rel_h <= 0.90 and 0.02 <= rel_w <= 0.35 and 0.15 <= aspect_ratio <= 1.2:
                char_candidates.append({
                    "bbox":   (x, y, cw, ch),
                    "center": (x + cw / 2.0, y + ch / 2.0),
                    "area":   cw * ch,
                    "crop":   binarized_img[y:y+ch, x:x+cw]
                })
        return char_candidates

    def analyze_layout_and_sort_characters(
        self,
        char_candidates: List[Dict[str, Any]],
        img_height: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Groups characters into 1-line (standard) or 2-line (stacked) clusters
        and sorts each cluster horizontally left-to-right.
        """
        if not char_candidates:
            return []
        y_centers = [c["center"][1] for c in char_candidates]
        min_y, max_y = min(y_centers), max(y_centers)
        if (max_y - min_y) > img_height * 0.28:
            mid_y   = (min_y + max_y) / 2.0
            line1   = sorted([c for c in char_candidates if c["center"][1] <  mid_y], key=lambda c: c["bbox"][0])
            line2   = sorted([c for c in char_candidates if c["center"][1] >= mid_y], key=lambda c: c["bbox"][0])
            return [l for l in [line1, line2] if l]
        return [sorted(char_candidates, key=lambda c: c["bbox"][0])]

    # ── EasyOCR with spatial sorting ──────────────────────────────────────────

    @staticmethod
    def _sort_easyocr_results_spatially(
        results: List[Any]
    ) -> Tuple[List[str], List[float]]:
        """
        Spatially reorders raw EasyOCR detection results for correct reading order:
          - One-line plate:  left → right
          - Two-line plate:  top line first; within each line left → right

        Algorithm:
          1. Extract (cy, min_x) for each detected box.
          2. Sort by cy (vertical center).
          3. Group boxes within 40% of mean box height into the same line cluster.
          4. Sort each cluster by min_x.
          5. Concatenate clusters top-to-bottom.
        """
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
                "text":  text,
                "conf":  conf,
                "min_x": min(xs),
                "cy":    (min_y + max_y) / 2.0,
                "box_h": max(1.0, max_y - min_y),
            })

        if not boxes:
            return [], []

        boxes.sort(key=lambda b: b["cy"])
        mean_box_h        = float(np.mean([b["box_h"] for b in boxes]))
        cluster_threshold = mean_box_h * 0.40

        lines: List[List[dict]] = []
        current_line = [boxes[0]]
        for box in boxes[1:]:
            if abs(box["cy"] - current_line[-1]["cy"]) <= cluster_threshold:
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b["min_x"]))
                current_line = [box]
        lines.append(sorted(current_line, key=lambda b: b["min_x"]))

        ordered_texts: List[str]  = []
        ordered_confs: List[float] = []
        for line in lines:
            for b in line:
                ordered_texts.append(b["text"])
                ordered_confs.append(b["conf"])

        return ordered_texts, ordered_confs

    def read_with_paddle(self, image: np.ndarray) -> Tuple[str, float, List[str]]:
        """Reads text using PaddleOCR."""
        result = self.paddle_ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return "", 0.0, []
        detected_lines, confs = [], []
        for line in result[0]:
            text, conf = line[1][0], line[1][1]
            detected_lines.append(text)
            confs.append(conf)
        full_text = "".join(detected_lines)
        avg_conf  = float(np.mean(confs)) if confs else 0.0
        return full_text, avg_conf, detected_lines

    def read_with_easyocr(self, image: np.ndarray) -> Tuple[str, float, List[str]]:
        """
        Reads text using EasyOCR with spatial bounding-box sorting.
        Ensures correct reading order for one-line and two-line plates.
        """
        results = self.easy_ocr.readtext(image)
        if not results:
            return "", 0.0, []
        ordered_texts, ordered_confs = self._sort_easyocr_results_spatially(results)
        if not ordered_texts:
            return "", 0.0, []
        full_text = " ".join(ordered_texts)
        avg_conf  = float(np.mean(ordered_confs)) if ordered_confs else 0.0
        return full_text, avg_conf, ordered_texts

    # ── Multi-candidate OCR selection (main path) ─────────────────────────────

    def select_best_ocr_candidate(
        self,
        color_clahe:    np.ndarray,
        gray_clahe:     np.ndarray,
        adaptive_bin:   np.ndarray,
        crop_h:         int = 0,
        crop_w:         int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluates up to three preprocessing candidates and selects the best OCR
        result using the composite quality score.

        Candidates:
          V0_COLOR_CLAHE       : `color_clahe`   — original baseline path
          V1_GRAY_CLAHE        : `gray_clahe`    — grayscale + CLAHE path
          V2_ADAPTIVE_BINARIZE : `adaptive_bin`  — polarity-corrected Otsu

        Execution policy (lazy + guarded):
          V0 and V1 always run.
          V2 runs when:
            - Crop is small (height ≤ _SMALL_CROP_HEIGHT_THRESHOLD), OR
            - At least one of V0/V1 is weak (per _is_weak()), AND
              crop area is below _LARGE_CROP_AREA_THRESHOLD.

        Selection:
          Winner = argmax(composite_score) across all executed candidates.
          Indian structural validity contributes 25% of the score, preventing
          a high-confidence garbled result from beating a valid Indian reading.

        Returns structured dict compatible with the existing pipeline output schema.
        """
        crop_area   = crop_h * crop_w
        is_large    = crop_area > _LARGE_CROP_AREA_THRESHOLD
        is_small    = crop_h > 0 and crop_h <= _SMALL_CROP_HEIGHT_THRESHOLD

        # ── Run V0 (color CLAHE) ───────────────────────────────────────────
        v0_text, v0_conf, v0_lines = "", 0.0, []
        try:
            v0_text, v0_conf, v0_lines = self.read_with_easyocr(color_clahe)
        except Exception:
            pass
        v0_score_info = _composite_score(v0_text, v0_conf, crop_h, crop_w)
        v0_weak = _is_weak(v0_text, v0_conf, crop_h)

        # ── Run V1 (gray CLAHE) ────────────────────────────────────────────
        v1_text, v1_conf, v1_lines = "", 0.0, []
        try:
            v1_text, v1_conf, v1_lines = self.read_with_easyocr(gray_clahe)
        except Exception:
            pass
        v1_score_info = _composite_score(v1_text, v1_conf, crop_h, crop_w)
        v1_weak = _is_weak(v1_text, v1_conf, crop_h)

        # ── Decide whether to run V2 ───────────────────────────────────────
        # V2 runs when:
        #   (a) Crop is small (height ≤ threshold) — binarize helps small crops, OR
        #   (b) Both V0 and V1 are weak — regardless of crop size (do not skip V2
        #       just because the crop is large if the only available results are poor)
        #   (c) At least one is weak AND crop is not large
        both_weak = v0_weak and v1_weak
        run_v2 = is_small or both_weak or (not is_large and (v0_weak or v1_weak))
        v2_text, v2_conf, v2_lines = "", 0.0, []
        v2_score_info = _composite_score("", 0.0, crop_h, crop_w)
        if run_v2 and adaptive_bin is not None:
            try:
                v2_text, v2_conf, v2_lines = self.read_with_easyocr(adaptive_bin)
            except Exception:
                pass
            v2_score_info = _composite_score(v2_text, v2_conf, crop_h, crop_w)

        # ── Select winner ──────────────────────────────────────────────────
        candidates = [
            ("V0_COLOR_CLAHE",       v0_text, v0_conf, v0_lines, v0_score_info),
            ("V1_GRAY_CLAHE",        v1_text, v1_conf, v1_lines, v1_score_info),
        ]
        if run_v2:
            candidates.append(
                ("V2_ADAPTIVE_BINARIZE", v2_text, v2_conf, v2_lines, v2_score_info)
            )

        best_variant, best_text, best_conf, best_lines, best_score_info = max(
            candidates, key=lambda c: c[4]["composite_score"]
        )

        return {
            "raw_text":       best_text,
            "confidence":     round(best_conf, 3),
            "lines":          best_lines,
            "engine_used":    "easyocr",
            "ocr_pass":       best_variant,
            "selected_variant": best_variant,
            "composite_score":  best_score_info["composite_score"],
            # Per-variant metadata for audit/debugging
            "candidates": {
                "V0_COLOR_CLAHE":       {"text": v0_text, "conf": round(v0_conf, 3), "score": v0_score_info["composite_score"]},
                "V1_GRAY_CLAHE":        {"text": v1_text, "conf": round(v1_conf, 3), "score": v1_score_info["composite_score"]},
                "V2_ADAPTIVE_BINARIZE": {"text": v2_text, "conf": round(v2_conf, 3), "score": v2_score_info["composite_score"], "ran": run_v2},
            },
        }

    # ── Backward-compatible recognize_plate() ─────────────────────────────────

    def recognize_plate(
        self,
        enhanced_color: np.ndarray,
        binarized:      np.ndarray,
        color_clahe:    Optional[np.ndarray] = None,
        gray_clahe:     Optional[np.ndarray] = None,
        crop_h:         int = 0,
        crop_w:         int = 0,
    ) -> Dict[str, Any]:
        """
        Backward-compatible entry point for single-crop OCR recognition.

        Preferred call (from updated pipeline.py):
          recognize_plate(color_clahe=..., gray_clahe=...,
                          binarized=..., crop_h=..., crop_w=...)

        Legacy call (two-argument form — preserved for backward compatibility):
          recognize_plate(enhanced_color=<gray_clahe_img>, binarized=<adaptive_bin>)
          In this form, color_clahe falls back to enhanced_color.

        Delegates to select_best_ocr_candidate() when EasyOCR is available.
        Falls back to built-in contour segmentation otherwise.
        """
        # ── PaddleOCR path (if installed) ──────────────────────────────────
        if self.paddle_ocr is not None:
            try:
                raw_text, conf, lines = self.read_with_paddle(enhanced_color)
                if raw_text:
                    return {
                        "raw_text":   raw_text,
                        "confidence": round(conf, 3),
                        "lines":      lines,
                        "engine_used": "paddleocr",
                        "ocr_pass":   "primary",
                    }
            except Exception:
                pass

        # ── EasyOCR multi-candidate path ───────────────────────────────────
        if self.easy_ocr is not None:
            # Resolve which images to use for V0 and V1
            # When called from the updated pipeline.py, color_clahe and gray_clahe
            # are supplied explicitly. When called in legacy form, fall back to
            # enhanced_color for both V0 and V1 (matches old two-pass behaviour).
            _color   = color_clahe if color_clahe is not None else enhanced_color
            _gray    = gray_clahe  if gray_clahe  is not None else enhanced_color
            _binary  = binarized

            return self.select_best_ocr_candidate(
                color_clahe=_color,
                gray_clahe=_gray,
                adaptive_bin=_binary,
                crop_h=crop_h,
                crop_w=crop_w,
            )

        # ── Built-in contour segmentation fallback ─────────────────────────
        char_candidates = self.segment_character_contours(binarized)
        sorted_lines    = self.analyze_layout_and_sort_characters(
            char_candidates, binarized.shape[0]
        )
        num_chars_found = sum(len(line) for line in sorted_lines)
        is_two_line     = len(sorted_lines) == 2

        return {
            "raw_text":             "",
            "confidence":           0.50 if num_chars_found >= 6 else 0.0,
            "lines":                [],
            "character_blobs_found": num_chars_found,
            "is_two_line":          is_two_line,
            "engine_used":          self.engine_type,
            "ocr_pass":             "builtin_fallback",
        }

    # ── Legacy quality helpers (kept for backward compat with existing tests) ──

    @staticmethod
    def _ocr_quality_score(raw_text: str, confidence: float) -> float:
        """
        Legacy two-signal quality score (confidence + length).
        Retained for backward compatibility with existing tests only.
        Use _composite_score() for new code.
        """
        text = (raw_text or "").strip()
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / 8.0)
        return round(confidence * 0.65 + length_score * 0.35, 4)

    @staticmethod
    def _is_weak_ocr_result(raw_text: str, confidence: float) -> bool:
        """
        Legacy weak-result detector.
        Retained for backward compatibility with existing tests only.
        Use _is_weak() for new code.
        """
        text = (raw_text or "").strip()
        if len(text) < 4:
            return True
        if confidence < 0.30:
            return True
        if PlateOCREngine._ocr_quality_score(text, confidence) < 0.25:
            return True
        return False
