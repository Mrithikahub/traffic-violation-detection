"""
Pre-compute the plate recognition samples shown in the demo app.

Live plate recognition (YOLOv8n + EasyOCR) does not fit Streamlit Community
Cloud's memory floor, so the app shows results computed here instead. This
script runs the real NumberPlateRecognizer on each photo and writes what it
returned: the detection box, plate crop, raw and cleaned OCR text and both
confidences. Nothing is edited by hand.

Each photo is passed with the plate text a person read off it. A result is
written only if the OCR output matches that text exactly, so a misread can
never end up in the demo.

Usage:
    python webapp/precompute_plates.py \
        "photos/a.jpeg=MH01CP2655" "photos/b.jpeg=MH03CS6266"
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "assets" / "plates"
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("items", nargs="+", metavar="PHOTO=PLATE_TEXT",
                   help="photo path and the plate text as read by a person")
    p.add_argument("--max-width", type=int, default=900,
                   help="downscale the saved context photo to this width")
    args = p.parse_args()

    from npr_module.src.pipeline import NumberPlateRecognizer
    recognizer = NumberPlateRecognizer()

    OUT.mkdir(parents=True, exist_ok=True)
    samples, rejected = [], 0
    for n, item in enumerate(args.items, 1):
        path, _, truth = item.rpartition("=")
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"cannot read {path}")
        h, w = img.shape[:2]
        r = recognizer.process_vehicle_crop(img, vehicle_id=n, frame_id=0,
                                            vehicle_bbox=[0, 0, w, h])
        text = r.get("plate_text") or ""
        if not r.get("plate_detected") or text != truth:
            print(f"REJECT {path}: OCR {text!r} != verified {truth!r}")
            rejected += 1
            continue

        x1, y1, x2, y2 = r["plate_bbox_crop"]
        cv2.imwrite(str(OUT / f"plate_{n}_crop.png"), img[y1:y2, x1:x2])
        ctx = img.copy()
        cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 200, 0), max(2, w // 300))
        if w > args.max_width:
            ctx = cv2.resize(ctx, (args.max_width, int(h * args.max_width / w)),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"plate_{n}_photo.jpg"), ctx,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        samples.append({
            "crop": f"plate_{n}_crop.png",
            "photo": f"plate_{n}_photo.jpg",
            "verified_text": truth,
            "plate_text": text,
            "raw_ocr_text": r.get("raw_ocr_text"),
            "ocr_confidence": r.get("ocr_confidence"),
            "detection_confidence": r.get("detection_confidence"),
            "is_valid_indian_format": r.get("is_valid_indian_format"),
            "state_name": r.get("state_name"),
            "ocr_engine": r.get("ocr_engine_type"),
            "detector": r.get("detector_type"),
        })
        print(f"OK     {path}: {text} (OCR conf {r.get('ocr_confidence')})")

    (OUT / "results.json").write_text(json.dumps(
        {"generated": date.today().isoformat(), "samples": samples}, indent=2),
        encoding="utf-8")
    print(f"wrote {len(samples)} samples to {OUT}, rejected {rejected}")


if __name__ == "__main__":
    main()
