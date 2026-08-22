import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any

# Ensure OpenMP runtime compatibility on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import NumberPlateRecognizer


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """Calculates IoU between two [x1, y1, x2, y2] boxes."""
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


def load_ground_truth(label_path: str, img_w: int, img_h: int) -> List[List[int]]:
    """Loads YOLO format ground truth boxes converted to [x1, y1, x2, y2] pixel coordinates."""
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gx1 = int((xc - bw / 2) * img_w)
                    gy1 = int((yc - bh / 2) * img_h)
                    gx2 = int((xc + bw / 2) * img_w)
                    gy2 = int((yc + bh / 2) * img_h)
                    gt_boxes.append([gx1, gy1, gx2, gy2])
    return gt_boxes


def evaluate_split(
    split_name: str,
    manifest_path: str,
    recognizer: NumberPlateRecognizer,
    max_samples: int = None
) -> Dict[str, Any]:
    """Evaluates the pipeline on a single manifest split."""
    import cv2

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        img_paths = [l.strip() for l in f if l.strip()]

    if max_samples and max_samples > 0:
        img_paths = img_paths[:max_samples]

    total_images = len(img_paths)
    total_gt_plates = 0
    total_detected_plates = 0
    tp_50 = 0  # True positives at IoU >= 0.50
    ious = []
    latencies = []
    plate_type_counts = {"INDIAN": 0, "FOREIGN": 0, "UNKNOWN": 0}

    for idx, img_path in enumerate(img_paths):
        # Derive label path (standard YOLO convention: replace /images/ with /labels/ and extension to .txt)
        lbl_path = img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = load_ground_truth(lbl_path, w, h)
        total_gt_plates += len(gt_boxes)

        t0 = time.perf_counter()
        results = recognizer.process_frame(img, frame_id=idx + 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if results and results[0]["plate_detected"]:
            total_detected_plates += 1
            pred_box = results[0]["plate_bbox_crop"]
            ptype = results[0].get("plate_type", "UNKNOWN")
            plate_type_counts[ptype] = plate_type_counts.get(ptype, 0) + 1

            if gt_boxes:
                best_iou = max([calculate_iou(pred_box, gb) for gb in gt_boxes])
                ious.append(best_iou)
                if best_iou >= 0.50:
                    tp_50 += 1

    precision_50 = tp_50 / float(total_detected_plates) if total_detected_plates > 0 else 0.0
    recall_50 = tp_50 / float(total_gt_plates) if total_gt_plates > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "split_name": split_name,
        "total_images": total_images,
        "total_gt_plates": total_gt_plates,
        "detected_plates": total_detected_plates,
        "tp_50": tp_50,
        "precision_50": round(precision_50, 4),
        "recall_50": round(recall_50, 4),
        "mean_iou": round(mean_iou, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "plate_types": plate_type_counts
    }


def run_4way_evaluation(weights_path: str = None, max_samples: int = None):
    manifests_dir = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\configs\manifests"
    
    splits = [
        ("A. Indian Validation Set", os.path.join(manifests_dir, "val_indian.txt")),
        ("B. Foreign Validation Set", os.path.join(manifests_dir, "val_foreign.txt")),
        ("C. Indian Unseen Test Set", os.path.join(manifests_dir, "test_indian.txt")),
        ("D. Foreign Unseen Test Set", os.path.join(manifests_dir, "test_foreign.txt")),
    ]

    print("=" * 100)
    print("CDAC NPR MODULE - 4-WAY DISAGGREGATED EVALUATION REPORT")
    print(f"Model Weights: {weights_path if weights_path else 'Default / Morphological'}")
    print("=" * 100)

    recognizer = NumberPlateRecognizer(detector_model_path=weights_path)
    all_reports = []

    for name, mpath in splits:
        print(f"\nEvaluating: {name} ({mpath})...")
        rep = evaluate_split(name, mpath, recognizer, max_samples=max_samples)
        if rep:
            all_reports.append(rep)
            print(f"  Images: {rep['total_images']:4d} | GT Plates: {rep['total_gt_plates']:4d} | Detected: {rep['detected_plates']:4d}")
            print(f"  Precision@50: {rep['precision_50']:.4f} | Recall@50: {rep['recall_50']:.4f} | Mean IoU: {rep['mean_iou']:.4f}")
            print(f"  Latency: {rep['avg_latency_ms']:.2f} ms | Plate Types: {rep['plate_types']}")

    # Print Summary Table
    print("\n" + "=" * 100)
    print(f"{'Split Name':<30} {'Images':>7} {'GT':>6} {'Det':>6} {'Prec@50':>9} {'Rec@50':>9} {'IoU':>7} {'Latency':>9}")
    print("-" * 100)
    for r in all_reports:
        print(f"{r['split_name']:<30} {r['total_images']:>7d} {r['total_gt_plates']:>6d} {r['detected_plates']:>6d} "
              f"{r['precision_50']:>9.4f} {r['recall_50']:>9.4f} {r['mean_iou']:>7.4f} {r['avg_latency_ms']:>7.1f}ms")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-way evaluation on NPR module")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained YOLO detector weights")
    parser.add_argument("--max_samples", type=int, default=None, help="Max sample images per split (for fast audit)")
    args = parser.parse_args()

    run_4way_evaluation(weights_path=args.weights, max_samples=args.max_samples)
