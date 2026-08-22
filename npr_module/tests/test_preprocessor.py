"""
Unit Tests for Plate Preprocessor Module
"""

import os
import sys
import cv2
import numpy as np

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessor import PlatePreprocessor


def test_crop_with_padding():
    prep = PlatePreprocessor(pad_percent=0.10)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    bbox = (50, 50, 150, 100)  # w=100, h=50 -> pad_x=10, pad_y=5
    
    crop = prep.crop_with_padding(img, bbox)
    # Expected: x1=40, y1=45, x2=160, y2=105 -> shape = (60, 120, 3)
    assert crop.shape == (60, 120, 3), f"Expected (60, 120, 3), got {crop.shape}"
    print("  [PASS] Crop with safety padding")


def test_dynamic_upscaling():
    prep = PlatePreprocessor(target_height=140)
    small_crop = np.zeros((35, 100, 3), dtype=np.uint8)  # height=35
    
    upscaled = prep.upscale_to_target_height(small_crop)
    assert upscaled.shape[0] == 140, f"Expected height 140, got {upscaled.shape[0]}"
    assert upscaled.shape[1] == 400, f"Expected width 400, got {upscaled.shape[1]}"
    print("  [PASS] Dynamic bicubic upscaling")


def test_deskewing():
    prep = PlatePreprocessor()
    # Create a synthetic rotated white rectangle on black background
    canvas = np.zeros((200, 400, 3), dtype=np.uint8)
    pts = np.array([[100, 80], [300, 50], [310, 120], [110, 150]], np.int32)
    cv2.fillPoly(canvas, [pts], (255, 255, 255))
    
    deskewed = prep.deskew_plate(canvas)
    assert deskewed.shape == canvas.shape
    print("  [PASS] Tilt angle detection and deskewing")


def test_complete_pipeline():
    prep = PlatePreprocessor(target_height=140)
    # Create a sample synthetic plate image with low contrast
    sample = np.full((40, 120, 3), 100, dtype=np.uint8)
    cv2.putText(sample, "MH12DE1433", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    
    res = prep.preprocess_pipeline(sample)
    assert "deskewed" in res
    assert "upscaled" in res
    assert "enhanced_color" in res
    assert "gray" in res
    assert "binarized" in res
    assert res["upscaled"].shape[0] == 140
    print("  [PASS] Complete Preprocessing Pipeline outputs verified")


if __name__ == "__main__":
    print("Testing Plate Preprocessor Module...")
    test_crop_with_padding()
    test_dynamic_upscaling()
    test_deskewing()
    test_complete_pipeline()
    print("All Preprocessor tests passed successfully!\n")
