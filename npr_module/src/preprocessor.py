"""
Plate Image Preprocessing & Rectification Module
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class PlatePreprocessor:
    """
    Implements robust image preprocessing for license plates:
    - Safe bounding box cropping with margin padding
    - Perspective deskewing and horizontal rectification
    - Bilateral edge-preserving filtering
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Dynamic bicubic upscaling to target OCR resolution
    - Adaptive multi-thresholding
    """

    def __init__(
        self,
        target_height: int = 140,
        pad_percent: float = 0.06,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.target_height = target_height
        self.pad_percent = pad_percent
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )

    def crop_with_padding(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crops the bounding box from the image with a safety margin padding.
        bbox: (x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        pad_x = int(box_w * self.pad_percent)
        pad_y = int(box_h * self.pad_percent)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)

        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        if cropped.size == 0:
            return image
        return cropped

    def deskew_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Detects tilt angle using minimum area rotated rectangle / Hough lines
        and horizontally rectifies the plate image.
        """
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, apertureSize=3)

        # Find non-zero points or contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Find largest contour candidate representing the plate boundary
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return image

        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        # OpenCV minAreaRect angle adjustment
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is slight / reasonable (between -35 and 35 degrees)
        if 0.5 < abs(angle) < 35.0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(
                image, rot_mat, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return deskewed

        return image

    def upscale_to_target_height(self, image: np.ndarray) -> np.ndarray:
        """
        Dynamically rescales the plate image so its height reaches self.target_height
        while maintaining aspect ratio.

        Small crop handling:
          - Crops with height < 30px (very small) are upscaled to a minimum of 90px
            using Lanczos interpolation, then further scaled to target_height if needed.
          - Crops with height < 20px cannot reliably be recovered by upscaling alone;
            the method still attempts best-effort upscaling but results will be limited
            by the available pixel information.
          - Crops already at or above target_height are returned unchanged.
        """
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image

        # Determine effective target: enforce a minimum of 90px for tiny crops
        effective_target = self.target_height
        if h < 30:
            effective_target = max(90, self.target_height)

        if h >= effective_target:
            return image

        scale = effective_target / float(h)
        new_w = int(w * scale)
        # Use Lanczos for high upscale factors (>2x), cubic otherwise
        interp = cv2.INTER_LANCZOS4 if scale > 2.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (new_w, effective_target), interpolation=interp)

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances local contrast using CLAHE on the luminance channel.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            enhanced_lab = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    def denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Bilateral Filter to suppress background noise while preserving sharp character edges.
        """
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    def preprocess_grayscale_clahe(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Grayscale + CLAHE + Bilateral Denoising preprocessing path.

        This is the primary preprocessing path for the two-pass OCR strategy.
        It strips colour information (reducing noise from plate background colour
        variation), then boosts local contrast with tighter CLAHE settings
        (clipLimit=3.0) before bilateral denoising to preserve character edges.

        Args:
            plate_crop: Raw plate crop in BGR colour format.

        Returns:
            Single-channel (grayscale) enhanced image ready for EasyOCR.
        """
        if plate_crop is None or plate_crop.size == 0:
            raise ValueError("Input plate crop is empty or invalid.")

        upscaled = self.upscale_to_target_height(plate_crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=40)
        return denoised

    def preprocess_adaptive_binarize(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Polarity-corrected Otsu adaptive binarization preprocessing path.

        This is the fallback preprocessing path for the two-pass OCR strategy.
        It applies Otsu thresholding on an enhanced grayscale image and then
        checks whether the majority of pixels are background (white) or
        foreground (dark). If the polarity is inverted (dark background, which
        is common on night / dark-coloured plates), the image is automatically
        inverted so that EasyOCR always sees dark characters on a white background.

        Args:
            plate_crop: Raw plate crop in BGR colour format.

        Returns:
            Single-channel binary image ready for EasyOCR.
        """
        if plate_crop is None or plate_crop.size == 0:
            raise ValueError("Input plate crop is empty or invalid.")

        gray_enhanced = self.preprocess_grayscale_clahe(plate_crop)
        _, binary = cv2.threshold(
            gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Polarity check: if white pixels are in the minority, the image is
        # inverted (dark background). Flip so characters are dark on white.
        white_px = cv2.countNonZero(binary)
        total_px = binary.shape[0] * binary.shape[1]
        if white_px < total_px * 0.45:
            binary = cv2.bitwise_not(binary)
        return binary

    def binarize_adaptive(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Applies Otsu and Gaussian adaptive thresholding.
        """
        # Otsu thresholding
        _, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if plate has dark characters on light background (standard)
        # If white pixels are minority, invert so background is bright and characters are dark
        white_count = cv2.countNonZero(otsu)
        total_pixels = otsu.shape[0] * otsu.shape[1]
        if white_count < total_pixels * 0.45:
            otsu = cv2.bitwise_not(otsu)

        return otsu

    def preprocess_pipeline(
        self,
        plate_crop: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Executes complete preprocessing pipeline.
        
        Returns dictionary of processed representations:
        - "original_crop": raw input crop
        - "deskewed": tilt-rectified crop
        - "upscaled": resolution-enhanced color image
        - "enhanced_color": CLAHE + Bilateral filtered color image (best for modern OCR)
        - "gray": enhanced grayscale image
        - "binarized": high-contrast binarized image (for fallback segmentation)
        """
        if plate_crop is None or plate_crop.size == 0:
            raise ValueError("Input plate crop is empty or invalid.")

        deskewed = self.deskew_plate(plate_crop)
        upscaled = self.upscale_to_target_height(deskewed)
        enhanced_color = self.enhance_contrast_clahe(upscaled)
        denoised_color = self.denoise_bilateral(enhanced_color)

        gray = cv2.cvtColor(denoised_color, cv2.COLOR_BGR2GRAY)
        binarized = self.binarize_adaptive(gray)

        return {
            "original_crop": plate_crop,
            "deskewed": deskewed,
            "upscaled": upscaled,
            "enhanced_color": denoised_color,
            "gray": gray,
            "binarized": binarized,
            # Two-pass OCR paths (added without removing existing keys)
            "gray_clahe": self.preprocess_grayscale_clahe(plate_crop),
            "adaptive_binarized": self.preprocess_adaptive_binarize(plate_crop),
        }
