"""
Live number plate reading for the demo app.

Runs as its own process after detection and tracking have exited, so the plate
models (YOLOv8n + EasyOCR, ~1.5 GB resident) are never loaded alongside the
vehicle detector. Reuses run_pipeline.run_npr_stage, which crops each tracked
vehicle at its best frame and passes the crop to NumberPlateRecognizer.

Writes <out-dir>/<stem>_plates.json plus one PNG crop per detected plate.

Each plate is labelled for display:
  "reading"   OCR text in a valid Indian registration format with OCR
              confidence >= MIN_OCR_CONF. Shown as an unverified reading.
  "illegible" a plate was found but the text failed either check, or the
              plate touches the left or right edge of its vehicle box and is
              probably cut off, which drops characters: OCR then reads a
              truncated plate that can still look like a valid registration.
              Shown as a crop with no text. Top and bottom edges are not
              checked, since a close-up vehicle box often ends at the bumper,
              right at the bottom of a complete plate.

When two vehicle boxes in the same frame overlap, the same plate can be read
twice; only the better of the two is kept.

Why both checks: on 31 hand-checked phone photos, every correct reading
scored >= 0.567, while format validity alone let through 9 wrong readings
(one at 0.073). A 0.5 floor kept all 18 correct readings and dropped 6 of those
9. It does not make readings trustworthy, only fewer wrong ones, and it was
chosen on that same small set.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "run_pipeline.py").exists() else _HERE.parent
sys.path.insert(0, str(ROOT))

MIN_OCR_CONF = 0.5
EDGE_PX = 2  # plate this close to its vehicle box's side counts as cut off


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--vehicles", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from run_pipeline import run_npr_stage

    video, out = Path(args.video), Path(args.out_dir)
    results = run_npr_stage(video, Path(args.tracks), Path(args.vehicles))

    boxes = {}
    with open(args.tracks, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            boxes[(int(row["frame_id"]), int(row["vehicle_id"]))] = row

    cap = cv2.VideoCapture(str(video))
    plates, checked = [], len(results)
    for vid, r in sorted(results.items()):
        box = r.get("plate_bbox_frame")
        if not r.get("plate_detected") or not box:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_id"]))
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        name = f"{video.stem}_plate_{vid}.png"
        cv2.imwrite(str(out / name), crop)

        conf = float(r.get("ocr_confidence") or 0.0)
        vb = boxes.get((int(r["frame_id"]), vid))
        cut_off = vb is not None and (x1 - int(float(vb["x1"])) <= EDGE_PX
                                      or int(float(vb["x2"])) - x2 <= EDGE_PX)
        legible = (bool(r.get("is_valid_indian_format")) and conf >= MIN_OCR_CONF
                   and not cut_off)
        plates.append({
            "vehicle_id": vid,
            "frame_id": int(r["frame_id"]),
            "crop": name,
            "crop_width_px": x2 - x1,
            "box": [x1, y1, x2, y2],
            "cut_off": cut_off,
            "status": "reading" if legible else "illegible",
            "plate_text": r.get("plate_text") if legible else None,
            "raw_ocr_text": r.get("raw_ocr_text"),
            "ocr_confidence": conf,
            "detection_confidence": r.get("detection_confidence"),
            "state_name": r.get("state_name") if legible else None,
        })
    cap.release()

    # Same plate seen through two overlapping vehicle boxes: keep the better one.
    rank = sorted(plates, key=lambda p: (p["status"] != "reading", -p["ocr_confidence"]))
    kept = []
    for p in rank:
        if not any(k["frame_id"] == p["frame_id"] and _iou(k["box"], p["box"]) > 0.3
                   for k in kept):
            kept.append(p)
    plates = sorted(kept, key=lambda p: p["vehicle_id"])

    (out / f"{video.stem}_plates.json").write_text(json.dumps({
        "vehicles_checked": checked,
        "min_ocr_conf": MIN_OCR_CONF,
        "plates": plates,
    }, indent=2), encoding="utf-8")
    print(f"{checked} vehicles checked, {len(plates)} plates found, "
          f"{sum(p['status'] == 'reading' for p in plates)} readable")


if __name__ == "__main__":
    main()
