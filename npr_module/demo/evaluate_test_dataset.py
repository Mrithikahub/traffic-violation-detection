"""
Batch Evaluation on Test Dataset
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import List, Tuple

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import NumberPlateRecognizer


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """Calculates Intersection over Union between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    
    return inter_area / float(union_area) if union_area > 0 else 0.0


def evaluate_test_split(max_samples: int = 50):
    test_img_dir = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\images"
    test_lbl_dir = r"e:\CDAC Dataset\CDAC_Workspace\no_plates_test_C\labels"

    if not os.path.exists(test_img_dir) or not os.path.exists(test_lbl_dir):
        print("Test dataset directories not found.")
        return

    img_files = [f for f in os.listdir(test_img_dir) if f.endswith(".jpg")][:max_samples]
    recognizer = NumberPlateRecognizer()

    total_images = len(img_files)
    total_ground_truth_plates = 0
    detected_count = 0
    matched_gt_count = 0
    ious = []
    latencies = []

    print(f"Starting Evaluation on {total_images} Test Images...")

    for idx, fname in enumerate(img_files):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(test_img_dir, fname)
        lbl_path = os.path.join(test_lbl_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Read Ground Truth Bounding Boxes
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        gx1 = int((xc - bw / 2) * w)
                        gy1 = int((yc - bh / 2) * h)
                        gx2 = int((xc + bw / 2) * w)
                        gy2 = int((yc + bh / 2) * h)
                        gt_boxes.append([gx1, gy1, gx2, gy2])

        total_ground_truth_plates += len(gt_boxes)

        # Run NPR Pipeline
        t0 = time.perf_counter()
        results = recognizer.process_frame(img, frame_id=idx + 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if results and results[0]["plate_detected"]:
            detected_count += 1
            pred_box = results[0]["plate_bbox_crop"]
            # Find best matching GT box
            if gt_boxes:
                best_iou = max([calculate_iou(pred_box, gb) for gb in gt_boxes])
                ious.append(best_iou)
                if best_iou >= 0.25:
                    matched_gt_count += 1

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_iou = float(np.mean(ious)) if ious else 0.0

    print("\n=======================================================")
    print("           NPR MODULE EVALUATION REPORT                ")
    print("=======================================================")
    print(f"Total Test Images Evaluated   : {total_images}")
    print(f"Total Ground Truth Plates     : {total_ground_truth_plates}")
    print(f"Plates Localized by Detector  : {detected_count} ({detected_count/total_images*100:.1f}%)")
    print(f"Plates Matched to GT (IoU>0.25): {matched_gt_count}")
    print(f"Average IoU (Localization)   : {avg_iou:.3f}")
    print(f"Average Processing Latency    : {avg_latency:.2f} ms per frame")
    print(f"Throughput (FPS on CPU)       : {1000.0/avg_latency if avg_latency > 0 else 0:.1f} FPS")
    print("=======================================================\n")


if __name__ == "__main__":
    evaluate_test_split(max_samples=50)
