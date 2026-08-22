"""
Pre-download EasyOCR models so first run doesn't fail.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import easyocr
print("Initializing EasyOCR Reader (downloading models if needed)...")
r = easyocr.Reader(['en'], gpu=False, verbose=True)
print("EasyOCR models ready.")
import numpy as np, cv2
img = np.full((80, 300, 3), 240, dtype=np.uint8)
cv2.putText(img, 'MH12DE1433', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10,10,10), 3)
result = r.readtext(img)
print("Smoke test OCR result:", result)
