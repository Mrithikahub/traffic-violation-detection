"""
End-to-End Pipeline Integration Tests (Dual-Mode: Indian + Foreign)
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import sys
import cv2
import numpy as np

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import NumberPlateRecognizer


def test_member1_vehicle_crop_interface():
    recognizer = NumberPlateRecognizer()

    # 1. Test Synthetic Indian Vehicle Crop
    v_height, v_width = 300, 500
    vehicle_crop = np.full((v_height, v_width, 3), 180, dtype=np.uint8)
    cv2.rectangle(vehicle_crop, (50, 50), (450, 260), (120, 120, 120), -1)
    plate_x1, plate_y1, plate_x2, plate_y2 = 180, 170, 320, 215
    cv2.rectangle(vehicle_crop, (plate_x1, plate_y1), (plate_x2, plate_y2), (250, 250, 250), -1)
    cv2.rectangle(vehicle_crop, (plate_x1, plate_y1), (plate_x2, plate_y2), (10, 10, 10), 2)
    cv2.putText(vehicle_crop, "TN38AB1234", (plate_x1 + 10, plate_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    input_vehicle_id = 7
    input_frame_id = 1042
    input_vehicle_bbox = [100, 200, 600, 500]

    print("Executing End-to-End Test (Indian Vehicle Crop)...")
    output = recognizer.process_vehicle_crop(
        vehicle_crop=vehicle_crop,
        vehicle_id=input_vehicle_id,
        frame_id=input_frame_id,
        vehicle_bbox=input_vehicle_bbox
    )

    print("\n--- Indian Vehicle Crop NPR Output ---")
    for k, v in output.items():
        print(f"  {k:24s}: {v}")

    assert output["vehicle_id"] == 7
    assert output["frame_id"] == 1042
    assert output["plate_detected"] is True
    assert output["plate_bbox_crop"] is not None
    assert output["plate_bbox_frame"] is not None
    assert "plate_type" in output, "plate_type missing from output schema"
    assert output["plate_type"] in ["INDIAN", "FOREIGN", "UNKNOWN"]
    assert isinstance(output["processing_time_ms"], float)
    # YOLO detector + EasyOCR on CPU can take several seconds on first call.
    # 15 seconds is a generous ceiling that would only fail on a hung process.
    assert output["processing_time_ms"] < 15000.0, (
        f"Processing took {output['processing_time_ms']:.1f}ms — pipeline may be hung"
    )

    print("  [PASS] Indian Vehicle Crop Processed Successfully!")


def test_foreign_vehicle_crop_interface():
    recognizer = NumberPlateRecognizer()

    # 2. Test Synthetic Foreign Vehicle Crop (UK plate format BD21SMX)
    v_height, v_width = 300, 500
    foreign_crop = np.full((v_height, v_width, 3), 160, dtype=np.uint8)
    cv2.rectangle(foreign_crop, (50, 50), (450, 260), (90, 90, 90), -1)
    plate_x1, plate_y1, plate_x2, plate_y2 = 170, 170, 330, 215
    cv2.rectangle(foreign_crop, (plate_x1, plate_y1), (plate_x2, plate_y2), (240, 240, 50), -1)  # Yellow UK rear plate
    cv2.rectangle(foreign_crop, (plate_x1, plate_y1), (plate_x2, plate_y2), (10, 10, 10), 2)
    cv2.putText(foreign_crop, "BD21SMX", (plate_x1 + 10, plate_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    print("\nExecuting End-to-End Test (Foreign Vehicle Crop)...")
    output = recognizer.process_vehicle_crop(
        vehicle_crop=foreign_crop,
        vehicle_id=15,
        frame_id=2001,
        vehicle_bbox=[50, 100, 550, 400]
    )

    print("\n--- Foreign Vehicle Crop NPR Output ---")
    for k, v in output.items():
        print(f"  {k:24s}: {v}")

    assert output["plate_detected"] is True
    assert "plate_type" in output
    assert output["is_valid_indian_format"] is False, "Foreign plate must not be valid Indian"
    print("  [PASS] Foreign Vehicle Crop Processed Successfully!")


def test_real_test_dataset_image():
    recognizer = NumberPlateRecognizer()
    test_img_dir = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"
    
    if not os.path.exists(test_img_dir):
        print("Test image directory not found, skipping real image test.")
        return

    test_files = [f for f in os.listdir(test_img_dir) if f.endswith(".jpg")]
    if not test_files:
        return

    sample_img_path = os.path.join(test_img_dir, test_files[0])
    img = cv2.imread(sample_img_path)
    assert img is not None, f"Failed to load {sample_img_path}"

    print(f"\nTesting on actual dataset image: {test_files[0]} ({img.shape[1]}x{img.shape[0]})...")
    res = recognizer.process_frame(img, frame_id=1)
    assert len(res) > 0, "No results returned for real image"
    plate_res = res[0]
    print(f"  Plate Detected : {plate_res['plate_detected']}")
    print(f"  Plate BBox     : {plate_res['plate_bbox_crop']}")
    print(f"  Plate Type     : {plate_res['plate_type']}")
    print(f"  Confidence     : {plate_res['detection_confidence']}")
    print(f"  Latency        : {plate_res['processing_time_ms']} ms")
    print("  [PASS] Real dataset image processed successfully")


if __name__ == "__main__":
    test_member1_vehicle_crop_interface()
    test_foreign_vehicle_crop_interface()
    test_real_test_dataset_image()
    print("\nAll End-to-End Tests Passed Successfully!\n")

