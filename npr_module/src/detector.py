"""
Number Plate Detector Module
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class NumberPlateDetector:
    """
    Modular Number Plate Detector supporting:
    1. YOLO (Ultralytics YOLOv8/v11) when model weights are provided
    2. Edge-Gradient & Morphological Localization Engine (zero-dependency fallback)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.detector_type = "morphological"

        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.detector_type = "yolo"
            except Exception:
                self.model = None

    def detect_plates_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLO detection on the input image."""
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append({
                    "bbox": coords,  # [x1, y1, x2, y2]
                    "confidence": round(conf, 3),
                    "class_id": cls_id,
                    "class_name": "License_Plate"
                })
        return detections

    def detect_plates_morphological(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects license plate candidates in a vehicle crop or scene using
        morphological filtering, vertical gradient energy, and rectangularity heuristics.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. Morphological Black-Hat and Top-Hat to emphasize high-contrast alphanumeric regions
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(gray, top_hat)
        enhanced = cv2.subtract(enhanced, black_hat)

        # 2. Sobel Vertical Gradient (plates have dense vertical character edges)
        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val + 1e-6))
        grad_x = grad_x.astype(np.uint8)

        # 3. Gaussian Blur + Morphological Close to connect characters into a single plate blob
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, close_kernel)

        # 4. Otsu Thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean small artifacts
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, clean_kernel, iterations=1)
        thresh = cv2.dilate(thresh, clean_kernel, iterations=2)

        # 5. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh <= 0 or bw <= 0:
                continue

            ar = bw / float(bh)
            area = bw * bh
            rel_area = area / float(w * h)

            # Filter for plausible license plate geometry
            # Aspect ratio usually between 1.0 and 6.0, relative area between 0.3% and 50%
            if 1.0 <= ar <= 6.5 and 0.003 <= rel_area <= 0.50 and bw >= 25 and bh >= 10:
                # Score candidate by edge density and contrast
                roi = gray[y:y+bh, x:x+bw]
                std_contrast = float(np.std(roi)) if roi.size > 0 else 0
                score = min(1.0, (std_contrast / 100.0) * (0.8 + 0.2 * (ar >= 2.0)))

                candidates.append({
                    "bbox": [x, y, x + bw, y + bh],
                    "confidence": round(score, 3),
                    "class_id": 0,
                    "class_name": "License_Plate"
                })

        # Non-Maximum Suppression (NMS)
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda d: d["confidence"], reverse=True)
        keep = []
        for cand in candidates:
            box_a = cand["bbox"]
            overlap = False
            for k in keep:
                box_b = k["bbox"]
                # Compute IoU
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
                iou = inter_area / float(union_area) if union_area > 0 else 0
                if iou > self.iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(cand)

        return keep

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Dispatches detection to YOLO if loaded, otherwise uses Morphological Localization.
        """
        if image is None or image.size == 0:
            return []

        if self.model is not None:
            return self.detect_plates_yolo(image)
        else:
            return self.detect_plates_morphological(image)
