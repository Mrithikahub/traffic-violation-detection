"""
Demo: Run Number Plate Recognition on a Single Image
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import sys
import argparse
import json
import cv2

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import NumberPlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="Run NPR Module on an image")
    parser.add_argument("--image", type=str, default=None, help="Path to vehicle or scene image")
    parser.add_argument("--output", type=str, default="output_annotated.jpg", help="Path to save annotated image")
    parser.add_argument("--detector_weights", type=str, default=None, help="Path to YOLO detector weights (optional)")
    args = parser.parse_args()

    # Default to first test image if none provided
    if args.image is None:
        default_dir = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"
        if os.path.exists(default_dir):
            files = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith(".jpg")]
            if files:
                args.image = files[0]
        if args.image is None:
            print("Error: No image path specified and no test images found.")
            return

    print(f"Loading Image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at {args.image}")
        return

    recognizer = NumberPlateRecognizer(detector_model_path=args.detector_weights)
    
    print("Processing image through NPR Pipeline...")
    results = recognizer.process_frame(img)

    for idx, res in enumerate(results):
        print(f"\n--- Plate Result #{idx + 1} ---")
        print(json.dumps(res, indent=2))

        # Annotate image
        box = res.get("plate_bbox_crop")
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{res['plate_text']} ({res['detection_confidence']:.2f})" if res['plate_text'] else f"Plate ({res['detection_confidence']:.2f})"
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.output, img)
    print(f"\nAnnotated result saved to: {args.output}")


if __name__ == "__main__":
    main()
