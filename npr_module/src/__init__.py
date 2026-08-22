"""
Number Plate Recognition (NPR) Module
AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System (CDAC)
"""

from .detector import NumberPlateDetector
from .preprocessor import PlatePreprocessor
from .ocr_engine import PlateOCREngine
from .postprocessor import IndianPlatePostProcessor
from .pipeline import NumberPlateRecognizer

__all__ = [
    "NumberPlateDetector",
    "PlatePreprocessor",
    "PlateOCREngine",
    "IndianPlatePostProcessor",
    "NumberPlateRecognizer"
]
