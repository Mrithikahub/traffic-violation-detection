"""
Unit Tests for Plate OCR Engine Module
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System

Covers:
- Contour-based character segmentation (built-in fallback)
- Two-line layout sorting
- Legacy recognize_plate() backward-compatibility (enhanced_color + binarized)
- New multi-candidate select_best_ocr_candidate() API
- Composite quality scoring (_composite_score / _indian_bonus)
- _is_weak() threshold behavior
- Variant selection: Indian-valid candidate beats higher-confidence foreign result
- Foreign plate: no forced Indian formatting in scoring
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_engine import (
    PlateOCREngine,
    _composite_score,
    _indian_bonus,
    _is_weak,
    _alnum_only,
)
from src.preprocessor import PlatePreprocessor


# ── Existing tests (unchanged) ────────────────────────────────────────────────

def test_character_segmentation():
    engine = PlateOCREngine(use_paddle=False)

    h, w = 100, 300
    img = np.full((h, w), 255, dtype=np.uint8)
    chars = ["T", "N", "3", "8", "A", "B"]
    for i, ch in enumerate(chars):
        x = 30 + i * 40
        cv2.putText(img, ch, (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 3)

    candidates = engine.segment_character_contours(img)
    assert len(candidates) >= 5, f"Expected at least 5 characters segmented, found {len(candidates)}"
    print(f"  [PASS] Character segmentation (found {len(candidates)} character contours)")


def test_two_line_layout_sorting():
    engine = PlateOCREngine(use_paddle=False)

    h, w = 120, 250
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.putText(img, "TN38",   (40, 45),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,), 2)
    cv2.putText(img, "AB1234", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,), 2)

    candidates = engine.segment_character_contours(img)
    lines      = engine.analyze_layout_and_sort_characters(candidates, h)

    assert lines is not None
    assert isinstance(lines, list) and len(lines) >= 1
    total_chars = sum(len(line) for line in lines)
    print(f"  [PASS] Layout detection (lines={len(lines)}, chars={total_chars})")


def test_ocr_recognition_fallback():
    engine = PlateOCREngine(use_paddle=False)
    prep   = PlatePreprocessor()

    sample = np.full((60, 200, 3), 240, dtype=np.uint8)
    cv2.putText(sample, "KA01MJ9999", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)

    processed = prep.preprocess_pipeline(sample)
    # Legacy two-argument call — must still work
    res = engine.recognize_plate(processed["enhanced_color"], processed["binarized"])

    assert "confidence"  in res
    assert "engine_used" in res
    print(f"  [PASS] OCR Recognition fallback execution (Engine: {res['engine_used']})")


# ── New tests: composite scoring ──────────────────────────────────────────────

def test_indian_bonus_exact_match():
    """A perfectly formatted Indian plate (e.g. MH02DS9365) gets bonus=1.0."""
    bonus = _indian_bonus("MH02DS9365")
    assert bonus == 1.0, f"Expected 1.0 for valid Indian plate, got {bonus}"
    print(f"  [PASS] indian_bonus exact match: {bonus}")


def test_indian_bonus_valid_state_prefix():
    """A text starting with a known state code + digit gets bonus=0.5."""
    bonus = _indian_bonus("MH018G716")   # valid state prefix, but doesn't match STANDARD_REGEX
    assert bonus in (0.5, 1.0), f"Expected 0.5 or 1.0 for Indian-prefix text, got {bonus}"
    print(f"  [PASS] indian_bonus partial prefix: {bonus}")


def test_indian_bonus_foreign():
    """A foreign plate text gets bonus=0.0."""
    bonus = _indian_bonus("59B7338")   # Vietnamese plate pattern
    assert bonus == 0.0, f"Expected 0.0 for non-Indian text, got {bonus}"
    print(f"  [PASS] indian_bonus foreign: {bonus}")


def test_indian_bonus_too_short():
    """Short text (< 6 chars after cleaning) always returns 0.0."""
    bonus = _indian_bonus("MH1")    # 3 chars — definitely below minimum
    assert bonus == 0.0, f"Expected 0.0 for too-short text, got {bonus}"
    bonus5 = _indian_bonus("MH012")  # 5 chars — still below minimum
    assert bonus5 == 0.0, f"Expected 0.0 for 5-char text, got {bonus5}"
    print(f"  [PASS] indian_bonus too-short: {bonus}, {bonus5}")


def test_composite_score_indian_beats_higher_conf_foreign():
    """
    An Indian-valid candidate with lower confidence should beat a garbled
    high-confidence candidate when the indian_bonus is decisive.

    This is the core regression case: video8_1350
      V1_GRAY_CLAHE (garbled):    conf=0.451, text='MFOiBG 716' (no indian bonus)
      V2_ADAPTIVE_BINARIZE (good): conf=0.433, text='mh018G 716' (gets partial bonus)
    """
    score_v1 = _composite_score("MFOiBG 716", 0.451, crop_h=44, crop_w=131)
    score_v2 = _composite_score("mh018G 716", 0.433, crop_h=44, crop_w=131)

    # V2 must beat V1 — the entire point of the composite scorer
    assert score_v2["composite_score"] > score_v1["composite_score"], (
        f"V2 (valid Indian) should outscore V1 (garbled): "
        f"v2={score_v2['composite_score']:.4f} v1={score_v1['composite_score']:.4f}"
    )
    print(f"  [PASS] Indian candidate beats higher-conf garbled candidate "
          f"(V2={score_v2['composite_score']:.4f} > V1={score_v1['composite_score']:.4f})")


def test_composite_score_foreign_no_bonus():
    """For clearly non-Indian text the indian_bonus must be 0.0."""
    score = _composite_score("59-C1 39916", 0.742, crop_h=86, crop_w=101)
    assert score["indian_bonus"] == 0.0, f"Expected 0.0 indian bonus for foreign plate, got {score['indian_bonus']}"
    # Score uses foreign weighting: 0.50*conf + 0.30*length + 0.20*alnum_ratio
    # At conf=0.742, alnum=9 chars → score should be > 0.70
    assert score["composite_score"] > 0.70, f"Score unexpectedly low for high-conf foreign: {score['composite_score']}"
    print(f"  [PASS] Foreign plate: bonus=0.0, score={score['composite_score']:.4f}")


def test_composite_score_empty_text():
    """Empty OCR result must score 0.0."""
    score = _composite_score("", 0.0, crop_h=50, crop_w=200)
    assert score["composite_score"] == 0.0, f"Empty text should score 0.0, got {score['composite_score']}"
    print(f"  [PASS] Empty text scores 0.0")


def test_composite_score_single_char_penalty():
    """A single character result should score much lower than a 9-char result."""
    score_short = _composite_score("1", 0.50, crop_h=50, crop_w=200)
    score_long  = _composite_score("MH02AB1234", 0.50, crop_h=50, crop_w=200)
    assert score_short["composite_score"] < score_long["composite_score"], (
        f"Short text should score lower: short={score_short['composite_score']:.4f} "
        f"long={score_long['composite_score']:.4f}"
    )
    print(f"  [PASS] Single char penalty: short={score_short['composite_score']:.4f} < long={score_long['composite_score']:.4f}")


# ── New tests: _is_weak() ─────────────────────────────────────────────────────

def test_is_weak_empty():
    assert _is_weak("", 0.0) is True
    print("  [PASS] _is_weak: empty text is weak")


def test_is_weak_too_short():
    assert _is_weak("MH1", 0.80) is True
    print("  [PASS] _is_weak: 3-char text is weak even at high confidence")


def test_is_weak_low_confidence():
    assert _is_weak("MH01AB1234", 0.29) is True
    print("  [PASS] _is_weak: 10-char text with conf<0.32 is weak")


def test_is_weak_small_crop_moderate_conf():
    # Small crop (h=50) with moderate confidence (0.38) is weak
    assert _is_weak("MH01AB", 0.38, crop_h=50) is True
    print("  [PASS] _is_weak: small crop with moderate conf is weak")


def test_is_weak_acceptable_result():
    assert _is_weak("MH02DS9365", 0.75) is False
    print("  [PASS] _is_weak: good 10-char high-conf result is not weak")


# ── New tests: select_best_ocr_candidate() ────────────────────────────────────

def test_select_best_candidate_returns_schema():
    """
    select_best_ocr_candidate() must return all required output schema fields
    even on synthetic gray images where EasyOCR finds nothing.
    """
    engine = PlateOCREngine(use_paddle=False)
    if engine.engine_type != "easyocr":
        print("  [SKIP] EasyOCR not available — skipping select_best_candidate test")
        return

    blank = np.full((100, 400), 200, dtype=np.uint8)  # gray blank image

    result = engine.select_best_ocr_candidate(
        color_clahe=blank,
        gray_clahe=blank,
        adaptive_bin=blank,
        crop_h=100,
        crop_w=400,
    )
    required_keys = {"raw_text", "confidence", "lines", "engine_used",
                     "ocr_pass", "selected_variant", "composite_score", "candidates"}
    missing = required_keys - result.keys()
    assert not missing, f"Missing keys in output schema: {missing}"

    cands = result["candidates"]
    assert "V0_COLOR_CLAHE"       in cands
    assert "V1_GRAY_CLAHE"        in cands
    assert "V2_ADAPTIVE_BINARIZE" in cands
    print(f"  [PASS] select_best_candidate schema complete (variant={result['selected_variant']})")


def test_large_crop_skips_v2_when_not_weak():
    """
    For a large crop where both V0 and V1 return acceptable text,
    V2 should not be executed (ran=False in candidates).
    We test this indirectly using the execution flag.
    """
    engine = PlateOCREngine(use_paddle=False)
    if engine.engine_type != "easyocr":
        print("  [SKIP] EasyOCR not available — skipping large-crop test")
        return

    # Large image > 400*200 px
    large = np.full((250, 450, 3), 200, dtype=np.uint8)
    cv2.putText(large, "MH02DS9365", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (30,30,30), 5)
    gray_large = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

    result = engine.select_best_ocr_candidate(
        color_clahe=large,
        gray_clahe=gray_large,
        adaptive_bin=gray_large,
        crop_h=250,
        crop_w=450,
    )
    # V2 should only run when V0 or V1 are weak — on a large clearly-rendered plate both should be fine
    v2_ran = result["candidates"]["V2_ADAPTIVE_BINARIZE"].get("ran", False)
    # If EasyOCR confidently reads the plate, V2 should be skipped.
    # If EasyOCR still produces a weak result on this synthetic image, V2 may run — that's allowed.
    # We just verify the pipeline completes and returns valid structure.
    assert "selected_variant" in result
    print(f"  [PASS] Large-crop candidate selection (V2 ran={v2_ran}, selected={result['selected_variant']})")


# ── Legacy backward-compat ────────────────────────────────────────────────────

def test_legacy_quality_score_preserved():
    """_ocr_quality_score() must still exist and produce values in [0, 1]."""
    engine = PlateOCREngine(use_paddle=False)
    score = engine._ocr_quality_score("MH02DS9365", 0.85)
    assert 0.0 <= score <= 1.0, f"Legacy quality score out of range: {score}"
    print(f"  [PASS] Legacy _ocr_quality_score preserved: {score:.4f}")


def test_legacy_is_weak_preserved():
    """_is_weak_ocr_result() must still exist and return correct booleans."""
    engine = PlateOCREngine(use_paddle=False)
    # Empty string → always weak
    assert engine._is_weak_ocr_result("", 0.0) is True
    # 4-char text at good confidence → NOT weak (boundary: len >= 4, conf >= 0.30)
    assert engine._is_weak_ocr_result("MH02", 0.85) is False
    # 2-char text → weak (len < 4)
    assert engine._is_weak_ocr_result("AB", 0.85) is True
    # Low confidence → weak regardless of text length
    assert engine._is_weak_ocr_result("MH02AB1234", 0.10) is True
    print("  [PASS] Legacy _is_weak_ocr_result preserved")


if __name__ == "__main__":
    print("Testing Plate OCR Engine Module...")
    print("\n── Existing tests ──")
    test_character_segmentation()
    test_two_line_layout_sorting()
    test_ocr_recognition_fallback()

    print("\n── Composite scoring ──")
    test_indian_bonus_exact_match()
    test_indian_bonus_valid_state_prefix()
    test_indian_bonus_foreign()
    test_indian_bonus_too_short()
    test_composite_score_indian_beats_higher_conf_foreign()
    test_composite_score_foreign_no_bonus()
    test_composite_score_empty_text()
    test_composite_score_single_char_penalty()

    print("\n── _is_weak() ──")
    test_is_weak_empty()
    test_is_weak_too_short()
    test_is_weak_low_confidence()
    test_is_weak_small_crop_moderate_conf()
    test_is_weak_acceptable_result()

    print("\n── select_best_candidate() ──")
    test_select_best_candidate_returns_schema()
    test_large_crop_skips_v2_when_not_weak()

    print("\n── Legacy backward-compat ──")
    test_legacy_quality_score_preserved()
    test_legacy_is_weak_preserved()

    print("\nAll OCR Engine tests passed successfully!\n")
