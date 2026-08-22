"""
End-to-End Number Plate Recognizer Pipeline
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .detector import NumberPlateDetector
from .preprocessor import PlatePreprocessor
from .ocr_engine import PlateOCREngine
from .postprocessor import IndianPlatePostProcessor

# Default YOLO weights path — trained baseline model (best.pt from 50-epoch run)
_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS_PATH = os.path.join(
    _MODULE_DIR, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt"
)


class NumberPlateRecognizer:
    """
    Unified Number Plate Recognition pipeline that integrates:
    1. Plate Localization (Detector)
    2. Dynamic Padding & Crop (Preprocessor)
    3. Deskewing, CLAHE, Bilateral Denoising & Rescaling (Preprocessor)
    4. Text Extraction & Multi-Line Layout Analysis (OCR Engine)
    5. Indian RTO & BH-Series Syntax Validation & Disambiguation (Post-Processor)
    """

    def __init__(
        self,
        detector_model_path: Optional[str] = YOLO_WEIGHTS_PATH,
        use_ocr_paddle: bool = True,
        conf_threshold: float = 0.35,
        target_plate_height: int = 140
    ):
        self.detector = NumberPlateDetector(
            model_path=detector_model_path,
            conf_threshold=conf_threshold
        )
        self.preprocessor = PlatePreprocessor(
            target_height=target_plate_height,
            pad_percent=0.06
        )
        self.ocr_engine = PlateOCREngine(
            use_paddle=use_ocr_paddle
        )
        self.post_processor = IndianPlatePostProcessor()

    def process_vehicle_crop(
        self,
        vehicle_crop: np.ndarray,
        vehicle_id: Optional[Any] = None,
        frame_id: Optional[int] = None,
        vehicle_bbox: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Primary Interface with Member 1's Vehicle Detection Module.
        
        Args:
            vehicle_crop: BGR image array of the detected vehicle
            vehicle_id: Unique tracking ID assigned by Member 1
            frame_id: Video frame sequence index
            vehicle_bbox: [x1, y1, x2, y2] relative to camera frame
            
        Returns:
            Structured dictionary with plate text, confidences, coordinates, and validity.
        """
        start_time = time.perf_counter()

        if vehicle_crop is None or vehicle_crop.size == 0:
            return {
                "vehicle_id": vehicle_id,
                "frame_id": frame_id,
                "plate_detected": False,
                "plate_bbox_crop": None,
                "plate_bbox_frame": None,
                "detection_confidence": 0.0,
                "plate_text": "",
                "cleaned_text": "",
                "standardized_text": "",
                "ocr_confidence": 0.0,
                "is_valid_indian_format": False,
                "format_type": "NONE",
                "state_name": None,
                "processing_time_ms": 0.0,
                "error": "Empty or invalid vehicle crop"
            }

        # 1. Detect Number Plate within the vehicle crop
        detections = self.detector.detect(vehicle_crop)

        if not detections:
            # Detector found nothing — return an honest no-detection result
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "vehicle_id": vehicle_id,
                "frame_id": frame_id,
                "plate_detected": False,
                "plate_bbox_crop": None,
                "plate_bbox_frame": None,
                "detection_confidence": 0.0,
                "raw_ocr_text": "",
                "cleaned_text": "",
                "standardized_text": "",
                "plate_text": "",
                "ocr_confidence": 0.0,
                "ocr_pass": "none",
                "plate_type": "UNKNOWN",
                "is_valid_indian_format": False,
                "format_type": "NONE",
                "state_code": None,
                "state_name": None,
                "district_code": None,
                "series_code": None,
                "unique_number": None,
                "validation_score": 0.0,
                "processing_time_ms": round(elapsed_ms, 2),
                "detector_type": self.detector.detector_type,
                "ocr_engine_type": self.ocr_engine.engine_type
            }

        # Pick detection with highest confidence
        best_det = max(detections, key=lambda d: d["confidence"])
        plate_box = best_det["bbox"]
        det_conf = best_det["confidence"]

        # Calculate coordinates in camera frame if vehicle_bbox is provided
        plate_bbox_frame = None
        if vehicle_bbox is not None and len(vehicle_bbox) == 4:
            vx1, vy1 = vehicle_bbox[0], vehicle_bbox[1]
            plate_bbox_frame = [
                vx1 + plate_box[0],
                vy1 + plate_box[1],
                vx1 + plate_box[2],
                vy1 + plate_box[3]
            ]

        # 2. Crop with margin padding
        plate_crop = self.preprocessor.crop_with_padding(vehicle_crop, plate_box)

        # 3. Preprocess (Deskew, CLAHE, Rescale, Bilateral filter)
        preprocessed = self.preprocessor.preprocess_pipeline(plate_crop)

        # 4. OCR Recognition — multi-candidate strategy:
        #    V0 (color CLAHE), V1 (gray CLAHE), V2 (adaptive binarize) are evaluated.
        #    V2 runs only for small crops or when V0/V1 are both weak on non-large crops.
        #    Selection uses a composite score weighting OCR confidence, text length,
        #    alphanumeric ratio, and Indian plate structural validity.
        plate_h = plate_crop.shape[0]
        plate_w = plate_crop.shape[1]
        ocr_result = self.ocr_engine.recognize_plate(
            enhanced_color=preprocessed["enhanced_color"],   # legacy-compat param
            binarized=preprocessed["adaptive_binarized"],
            color_clahe=preprocessed["enhanced_color"],
            gray_clahe=preprocessed["gray_clahe"],
            crop_h=plate_h,
            crop_w=plate_w,
        )

        raw_ocr_text = ocr_result.get("raw_text", "")
        ocr_conf     = ocr_result.get("confidence", 0.0)

        # 5. Post-Processing & Multi-Format Validation
        parsed = self.post_processor.validate_and_parse(raw_ocr_text)

        # Select plate_text: use standardized for Indian, raw/clean for foreign/unknown
        if parsed.get("plate_type") == "INDIAN" and parsed.get("standardized_text"):
            final_plate_text = parsed["standardized_text"]
        elif parsed.get("cleaned_text"):
            final_plate_text = parsed["cleaned_text"]
        else:
            final_plate_text = raw_ocr_text

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "vehicle_id": vehicle_id,
            "frame_id": frame_id,
            "plate_detected": True,
            "plate_bbox_crop": plate_box,
            "plate_bbox_frame": plate_bbox_frame,
            "detection_confidence": det_conf,
            "raw_ocr_text": raw_ocr_text,
            "cleaned_text": parsed["cleaned_text"],
            "standardized_text": parsed["standardized_text"],
            "plate_text": final_plate_text,
            "ocr_confidence": ocr_conf,
            "ocr_pass": ocr_result.get("ocr_pass", "unknown"),
            "selected_variant": ocr_result.get("selected_variant", "unknown"),
            "composite_score": ocr_result.get("composite_score", 0.0),
            "ocr_candidates": ocr_result.get("candidates", {}),
            "plate_type": parsed.get("plate_type", "UNKNOWN"),
            "is_valid_indian_format": parsed["is_valid"],
            "format_type": parsed["format_type"],
            "state_code": parsed["state_code"],
            "state_name": parsed["state_name"],
            "district_code": parsed["district_code"],
            "series_code": parsed["series_code"],
            "unique_number": parsed["unique_number"],
            "validation_score": parsed["validation_score"],
            "processing_time_ms": round(elapsed_ms, 2),
            "detector_type": self.detector.detector_type,
            "ocr_engine_type": self.ocr_engine.engine_type
        }

    def process_frame(
        self,
        frame: np.ndarray,
        vehicle_detections: Optional[List[Dict[str, Any]]] = None,
        frame_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a full camera frame.
        If vehicle_detections is provided (from Member 1's YOLO tracker), crops each vehicle and runs NPR.
        Otherwise, runs NPR directly on the frame.
        """
        results = []
        if vehicle_detections:
            for v_det in vehicle_detections:
                v_id = v_det.get("vehicle_id")
                v_box = v_det.get("bbox")  # [x1, y1, x2, y2]
                if v_box:
                    x1, y1, x2, y2 = v_box
                    v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    res = self.process_vehicle_crop(
                        vehicle_crop=v_crop,
                        vehicle_id=v_id,
                        frame_id=frame_id,
                        vehicle_bbox=v_box
                    )
                    results.append(res)
        else:
            # Direct frame detection
            res = self.process_vehicle_crop(
                vehicle_crop=frame,
                vehicle_id=None,
                frame_id=frame_id
            )
            results.append(res)

        return results
