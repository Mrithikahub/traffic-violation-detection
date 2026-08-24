"""
Vehicle detection + tracking module.

Runs a YOLO detector (Ultralytics) per frame on a video, assigns persistent
vehicle IDs with ByteTrack, and writes one row per detection per frame to
CSV / JSON. Defaults to YOLOv11; --model selects any Ultralytics weights.

This module is detection + tracking only. Downstream modules (violation rules,
plate recognition) consume the CSV/JSON described in OUTPUT_FORMAT.md.

Usage:
    python detect_track.py --video data/video/cctv052x2004080516x01638.avi
    python detect_track.py --video <path> --model yolo11s.pt --save-video

To run the whole dataset, use batch_run.py, which imports process_video() from
here so both paths share one implementation.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml

# --------------------------------------------------------------------------
# Class mapping
# --------------------------------------------------------------------------
# COCO class ids emitted by the pretrained YOLOv8 weights, folded into the four
# canonical classes this project uses. Anything not listed here is dropped.
COCO_TO_CANONICAL = {
    1: "bike",    # bicycle
    2: "car",
    3: "bike",    # motorcycle
    5: "bus",
    7: "truck",
}
# "rickshaw" (three-wheeler / auto-rickshaw) has NO COCO equivalent, so the
# pretrained weights can never emit it -- it only appears from a model
# fine-tuned on a taxonomy that includes it. Downstream code must therefore
# treat it as optional: present with our fine-tuned weights, absent with stock
# COCO weights. See OUTPUT_FORMAT.md.
CANONICAL_CLASSES = ["car", "bike", "bus", "truck", "rickshaw"]


def build_class_map(model):
    """Map a model's own class indices to canonical names.

    Pretrained COCO weights and a model fine-tuned on our 4-class scheme use
    *different, overlapping* index spaces, so a hardcoded COCO table silently
    mislabels a fine-tuned model: its index 2 is 'bus' but COCO's index 2 is
    'car', and its index 3 is 'truck' where COCO's is 'motorcycle'. Every box
    would still be emitted, just wrong. Decide from the model's own names.
    """
    names = getattr(getattr(model, "model", None), "names", None) or {}
    canon = set(CANONICAL_CLASSES)
    if names and all(str(v) in canon for v in names.values()):
        return {int(k): str(v) for k, v in names.items()}
    return dict(COCO_TO_CANONICAL)


# Stable BGR colour per canonical class, used for the annotated video.
CLASS_COLORS = {
    "car": (0, 200, 255),
    "bike": (255, 120, 0),
    "bus": (0, 255, 120),
    "truck": (200, 0, 255),
    "rickshaw": (60, 220, 220),
}

# CSV column order. Downstream modules depend on this — see OUTPUT_FORMAT.md.
CSV_COLUMNS = [
    "frame_id",
    "vehicle_id",
    "class",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
]

# Below this, a track's majority-voted class is not trustworthy.
AGREEMENT_FLOOR = 0.6


def add_pipeline_args(p):
    """Arguments shared by the single-video and batch entry points."""
    p.add_argument("--model", default="yolo11s.pt",
                   help="Ultralytics model weights. The 's' size is the default because "
                        "on this footage yolov8s found ~50%% more vehicles than yolov8n "
                        "for roughly 1.5x the time; yolo11s is the same size class. Pass "
                        "yolov8s.pt to reproduce the original v8 baseline.")
    p.add_argument("--conf", type=float, default=0.10,
                   help="Confidence threshold, also used to set ByteTrack's gates (see "
                        "build_tracker_cfg). Defaults low because vehicles in 320x240 CCTV "
                        "score poorly; at the usual 0.25 most of the scene is missed.")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference size. Frames are letterboxed to this. Note that raising "
                        "it on 320x240 footage LOSES detections - there is no extra detail "
                        "to recover and enlarged blurry vehicles leave the trained size "
                        "distribution.")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Base tracker config (bytetrack.yaml or botsort.yaml). Its "
                        "confidence gates are overridden from --conf unless "
                        "--raw-tracker-cfg is set.")
    p.add_argument("--track-buffer", type=int, default=30,
                   help="Frames a lost track is kept alive before its ID is retired. "
                        "This is in FRAMES, so at 10 fps the default is 3 seconds.")
    p.add_argument("--match-thresh", type=float, default=0.8,
                   help="IoU threshold for associating a detection to an existing track.")
    p.add_argument("--raw-tracker-cfg", action="store_true",
                   help="Use --tracker as-is instead of deriving its thresholds from "
                        "--conf. Set this if you are supplying your own tuned YAML.")
    p.add_argument("--start-frame", type=int, default=0,
                   help="0-based index of the first frame to process. The UCSD TrafficDB "
                        "clips have a corrupted first frame, so use 1 for those.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many processed frames (debugging).")
    p.add_argument("--device", default="cpu", help="'cpu', '0' for CUDA device 0, etc.")
    p.add_argument("--fps", type=float, default=None,
                   help="Frame rate to record when the source is a directory of numbered "
                        "frames (stills carry no timing). Defaults to 25 for sequences, "
                        "which is UA-DETRAC's rate. Ignored for video files, which report "
                        "their own.")
    p.add_argument("--annot-scale", type=float, default=3.0,
                   help="Upscale factor for the annotated video only (does not affect "
                        "detection). Small CCTV frames are unreadable at native size.")
    p.add_argument("--format", choices=["csv", "json", "both"], default="both",
                   help="Which result files to write.")
    return p


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO + ByteTrack vehicle detection and tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video file.")
    p.add_argument("--out-dir", default="outputs", help="Directory for output files.")
    p.add_argument("--save-video", action="store_true",
                   help="Write an annotated .mp4 with boxes + track IDs.")
    add_pipeline_args(p)
    return p.parse_args()


def build_tracker_cfg(args, out_dir):
    """Return a path to the tracker YAML to hand Ultralytics.

    ByteTrack applies its own confidence gates on top of the detector's --conf:
    a detection below `track_high_thresh` is only ever used as a weak second-pass
    association candidate, and one below `new_track_thresh` can never start a new
    track. Both default to 0.25, so lowering --conf alone changes nothing about
    the tracked output -- the extra detections are found and then dropped. We
    therefore rewrite those gates to follow --conf.

    `track_low_thresh` is kept at half of --conf so ByteTrack retains its
    low-confidence second association pass, which is the whole point of the
    algorithm and what recovers partially occluded vehicles.
    """
    if args.raw_tracker_cfg:
        return args.tracker

    # check_yaml resolves a bare name like "bytetrack.yaml" to the file shipped
    # inside the ultralytics package.
    with open(check_yaml(args.tracker), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base.update({
        "track_high_thresh": args.conf,
        "new_track_thresh": args.conf,
        "track_low_thresh": round(args.conf / 2, 4),
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
    })
    path = Path(out_dir) / f"_tracker_{Path(args.tracker).stem}_conf{args.conf}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return str(path)


def reset_tracker(model):
    """Clear tracker state so the next video starts from a clean slate.

    Necessary when one model instance processes several videos. Because
    Model.track() binds `persist` from the first call and we always pass True,
    Ultralytics' own between-video reset never fires -- without this, track IDs
    would continue across clip boundaries and a vehicle in clip 2 could inherit
    an ID (and Kalman state) from clip 1. BYTETracker.reset() also resets the
    shared ID counter, so every clip's IDs start at 1.
    """
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        tracker.reset()


# Frame-sequence datasets (UA-DETRAC and friends) ship numbered stills rather
# than a video file. OpenCV reads those natively given a printf-style pattern,
# so a directory of frames needs no separate decode path -- only a pattern and
# an externally supplied frame rate, since stills carry no timing.
SEQUENCE_PATTERNS = [
    ("img{:05d}.jpg", "img%05d.jpg"),      # UA-DETRAC
    ("{:05d}.jpg", "%05d.jpg"),
    ("{:06d}.jpg", "%06d.jpg"),            # MOT-style
    ("{:06d}.png", "%06d.png"),
    ("frame{:04d}.jpg", "frame%04d.jpg"),
]


def resolve_source(path, fps_hint=None):
    """Return (opencv_source, frame_rate_override) for a video file or a
    directory of numbered frames.

    Raises with the directory listing if no known naming pattern matches, rather
    than letting VideoCapture fail with an empty error later.
    """
    p = Path(path)
    if p.is_file():
        return str(p), fps_hint
    if not p.is_dir():
        raise RuntimeError(f"source not found: {p}")

    for probe, pattern in SEQUENCE_PATTERNS:
        # Sequences are 1-based in UA-DETRAC; accept 0-based too.
        for first in (1, 0):
            if (p / probe.format(first)).exists():
                src = str(p / pattern)
                if first == 0:
                    # VideoCapture starts at the lowest index it finds, so a
                    # 0-based sequence needs no special handling beyond this.
                    pass
                return src, (fps_hint or 25.0)

    sample = sorted(x.name for x in p.iterdir())[:6]
    raise RuntimeError(
        f"{p} is a directory but no known frame-naming pattern matched.\n"
        f"  tried: {', '.join(x[1] for x in SEQUENCE_PATTERNS)}\n"
        f"  found: {sample}\n"
        f"  Add the pattern to SEQUENCE_PATTERNS in detect_track.py.")


def open_video(path, fps_hint=None):
    src, fps_override = resolve_source(path, fps_hint)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    # An image sequence reports no usable frame rate; every downstream
    # timestamp (frame_id / fps) depends on this, so take the override.
    if fps_override and (fps <= 0 or Path(path).is_dir()):
        fps = float(fps_override)
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "source": src,
    }
    return cap, meta


def draw_annotations(frame, rows, scale):
    """Draw boxes + labels for one frame's rows onto an upscaled copy."""
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    for r in rows:
        color = CLASS_COLORS.get(r["class"], (255, 255, 255))
        x1, y1 = int(r["x1"] * scale), int(r["y1"] * scale)
        x2, y2 = int(r["x2"] * scale), int(r["y2"] * scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        vid = r["vehicle_id"]
        label = f"{'?' if vid == -1 else f'#{vid}'} {r['class']} {r['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Keep the label inside the frame when the box is near the top edge.
        ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def process_video(video_path, args, out_dir, model, tracker_cfg,
                  save_video=False):
    """Detect + track one video and write its output files.

    Returns a dict with the parsed rows, the meta block, the per-vehicle
    summary, and the quality stats. The caller owns the model so that batch runs
    load the weights once; call reset_tracker(model) between videos.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap, vmeta = open_video(video_path, getattr(args, "fps", None))
    class_map = build_class_map(model)
    keep_ids = sorted(class_map)

    writer = None
    rows = []
    frames_seen = 0        # frames actually fed to the model
    frames_empty = 0
    t0 = time.time()

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < args.start_frame:
            continue
        if args.max_frames is not None and frames_seen >= args.max_frames:
            break
        frames_seen += 1

        # persist MUST be True on every call, including the first.
        #
        # Model.track() only calls register_tracker() when the predictor has no
        # trackers yet, and register_tracker binds `persist` into the callback
        # with functools.partial. So the value from the FIRST call is the one
        # used forever; passing persist=False once poisons every later frame.
        # When that happens the reset check in on_predict_postprocess_end fires
        # each frame (predictor.save_dir increments per call, so the vid_path
        # comparison never matches), the tracker is wiped, and IDs restart at 1
        # every frame -- i.e. vehicle_id silently degrades into "rank of this
        # detection by confidence", which still looks like plausible output.
        #
        # Between videos, state is cleared explicitly via reset_tracker().
        res = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=keep_ids,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        frame_rows = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            # ByteTrack withholds an ID from a box until the track is confirmed,
            # so .id can be None for the whole frame or absent for some boxes.
            ids = (boxes.id.cpu().numpy().astype(int)
                   if boxes.id is not None else np.full(len(boxes), -1))

            for (x1, y1, x2, y2), cf, cl, tid in zip(xyxy, confs, clss, ids):
                name = class_map.get(int(cl))
                if name is None:
                    continue
                frame_rows.append({
                    "frame_id": frame_idx,
                    "vehicle_id": int(tid),
                    "class": name,
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                    "confidence": round(float(cf), 4),
                })

        if not frame_rows:
            frames_empty += 1
        rows.extend(frame_rows)

        if save_video:
            annotated = draw_annotations(frame, frame_rows, args.annot_scale)
            if writer is None:
                h, w = annotated.shape[:2]
                fps = vmeta["fps"] if vmeta["fps"] > 0 else 10.0
                writer = cv2.VideoWriter(
                    str(out_dir / f"{stem}_annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    meta = {
        "source_video": str(video_path),
        "width": vmeta["width"],
        "height": vmeta["height"],
        "fps": vmeta["fps"],
        "frames_processed": frames_seen,
        "start_frame": args.start_frame,
        "model": args.model,
        "tracker": args.tracker,
        "tracker_cfg_used": str(tracker_cfg),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "classes": CANONICAL_CLASSES,
        "total_detections": len(rows),
        "unique_vehicle_ids": len({r["vehicle_id"] for r in rows if r["vehicle_id"] != -1}),
        "processing_seconds": round(elapsed, 2),
    }

    written = []
    if args.format in ("csv", "both"):
        csv_path = out_dir / f"{stem}_tracks.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        written.append((csv_path, f"{len(rows)} rows"))

    if args.format in ("json", "both"):
        json_path = out_dir / f"{stem}_tracks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "detections": rows}, f, indent=2)
        written.append((json_path, ""))

    # Per-frame class is noisy on low-resolution footage (a box can be correct
    # while its label flips car<->truck between frames), so also emit one
    # majority-voted row per vehicle. Downstream modules that need "what kind of
    # vehicle is this" should read this file, not a single frame's label.
    summary = build_track_summary(rows, vmeta["fps"],
                                  vmeta["width"], vmeta["height"])
    if summary:
        sum_path = out_dir / f"{stem}_vehicles.csv"
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        written.append((sum_path, f"{len(summary)} vehicles"))

    if save_video:
        written.append((out_dir / f"{stem}_annotated.mp4", ""))

    stats = compute_stats(video_path.stem, rows, summary, meta,
                          frames_seen, frames_empty)
    return {"rows": rows, "meta": meta, "summary": summary,
            "stats": stats, "written": written}


def pick_best_frame(recs, width=None, height=None, edge_margin=2):
    """The frame of a track most likely to yield a readable number plate.

    Plate legibility is driven by how many pixels the plate spans, which scales
    with the *linear* size of the vehicle box -- so the score uses sqrt(area),
    not area, keeping it proportional to plate height rather than to its square.
    Detector confidence multiplies it as a proxy for a clean, unoccluded view:
    between two boxes of equal size, the one the detector was surer about is the
    less obstructed one.

    Boxes touching the frame border are discarded first. A vehicle halfway out
    of shot is often the largest box in its track while showing no plate at all,
    so size alone would reliably pick the worst frame. If every box in the track
    is truncated the filter is dropped rather than returning nothing.

    Returns the chosen record. `width`/`height` are optional: without them the
    truncation filter is skipped and selection falls back to size x confidence.
    """
    def truncated(r):
        if width is None or height is None:
            return False
        return (r["x1"] <= edge_margin or r["y1"] <= edge_margin
                or r["x2"] >= width - edge_margin
                or r["y2"] >= height - edge_margin)

    def score(r):
        area = max((r["x2"] - r["x1"]) * (r["y2"] - r["y1"]), 0.0)
        return (area ** 0.5) * r["confidence"]

    pool = [r for r in recs if not truncated(r)] or recs
    return max(pool, key=score)


def build_track_summary(rows, fps, width=None, height=None):
    """One row per tracked vehicle, with a majority-voted class.

    Ties are broken by summed confidence, so a class the detector was
    consistently sure about beats one it guessed weakly the same number of times.

    Also names the single best frame per vehicle for downstream plate OCR, so
    that module does not have to re-derive the selection from the per-frame CSV.
    See pick_best_frame() for how it is chosen.
    """
    tracks = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            tracks[r["vehicle_id"]].append(r)

    out = []
    for tid in sorted(tracks):
        recs = sorted(tracks[tid], key=lambda r: r["frame_id"])
        votes = Counter(r["class"] for r in recs)
        weight = defaultdict(float)
        for r in recs:
            weight[r["class"]] += r["confidence"]
        best = max(votes, key=lambda c: (votes[c], weight[c]))
        confs = [r["confidence"] for r in recs]
        bf = pick_best_frame(recs, width, height)
        bf_area = (bf["x2"] - bf["x1"]) * (bf["y2"] - bf["y1"])
        out.append({
            "vehicle_id": tid,
            "class": best,
            "class_agreement": round(votes[best] / len(recs), 3),
            "n_frames": len(recs),
            "first_frame": recs[0]["frame_id"],
            "last_frame": recs[-1]["frame_id"],
            "duration_sec": round(len(recs) / fps, 2) if fps else None,
            "mean_confidence": round(float(np.mean(confs)), 4),
            "max_confidence": round(float(max(confs)), 4),
            # Appended after max_confidence so the existing column order is
            # unchanged for anything already reading this file.
            "best_frame_id": bf["frame_id"],
            "best_frame_area": round(float(bf_area), 1),
            "best_frame_confidence": round(float(bf["confidence"]), 4),
        })
    return out


def compute_stats(name, rows, summary, meta, frames_seen, frames_empty):
    """Quality numbers for one clip, as a plain dict so batch runs can pool them.

    Everything here is computed from the model's own output. There are no ground
    truth boxes for this footage, so none of it measures recall -- a vehicle the
    model never detects is invisible to every number below.
    """
    s = {
        "clip": name,
        "frames_processed": frames_seen,
        "frames_empty": frames_empty,
        "total_detections": len(rows),
        "processing_seconds": meta["processing_seconds"],
    }
    if not rows:
        s.update({"dets_per_frame_mean": 0.0, "n_tracks": 0})
        return s

    per_frame = Counter(r["frame_id"] for r in rows)
    counts = [per_frame.get(f, 0) for f in
              range(meta["start_frame"], meta["start_frame"] + frames_seen)]
    confs = np.array([r["confidence"] for r in rows])

    s.update({
        "dets_per_frame_mean": round(float(np.mean(counts)), 2),
        "dets_per_frame_min": int(min(counts)),
        "dets_per_frame_max": int(max(counts)),
        "conf_mean": round(float(confs.mean()), 4),
        "conf_median": round(float(np.median(confs)), 4),
        "conf_pct_below_040": round(float(100 * (confs < 0.4).mean()), 2),
        "class_counts": {c: sum(1 for r in rows if r["class"] == c)
                         for c in CANONICAL_CLASSES},
        "untracked_detections": sum(1 for r in rows if r["vehicle_id"] == -1),
    })

    tracks = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            tracks[r["vehicle_id"]].append(r)
    s["n_tracks"] = len(tracks)

    if tracks:
        lengths = np.array([len(v) for v in tracks.values()])
        flipping = {tid: Counter(r["class"] for r in v)
                    for tid, v in tracks.items()
                    if len({r["class"] for r in v}) > 1}
        pairs = Counter()
        for c in flipping.values():
            for a, b in zip(sorted(c)[:-1], sorted(c)[1:]):
                pairs[f"{a} <-> {b}"] += 1
        s.update({
            "track_len_mean": round(float(lengths.mean()), 2),
            "track_len_median": int(np.median(lengths)),
            "track_len_max": int(lengths.max()),
            "tracks_le_2_frames": int((lengths <= 2).sum()),
            "tracks_changing_class": len(flipping),
            "class_flip_pairs": dict(pairs),
            # Kept so a batch run can pool exact percentiles instead of
            # averaging per-clip medians, which is not a real statistic.
            "track_lengths": [int(x) for x in lengths],
        })
    if summary:
        agree = np.array([v["class_agreement"] for v in summary])
        s.update({
            "n_vehicles": len(summary),
            "agreement_mean": round(float(agree.mean()), 4),
            "agreement_unanimous": int((agree >= 0.999).sum()),
            "agreement_below_floor": int((agree < AGREEMENT_FLOOR).sum()),
            "vehicle_class_counts": {c: sum(1 for v in summary if v["class"] == c)
                                     for c in CANONICAL_CLASSES},
            "agreements": [float(x) for x in agree],
        })
    return s


def report_quality(s):
    """Print the per-clip quality report from a compute_stats() dict."""
    print("\n" + "=" * 62)
    print("DETECTION / TRACKING QUALITY REPORT")
    print("=" * 62)

    frames_seen = s["frames_processed"]
    secs = s["processing_seconds"]
    print(f"Frames processed      : {frames_seen}  "
          f"({secs}s, {frames_seen / secs if secs else 0:.1f} fps)")

    if not s["total_detections"]:
        print("\n!! No detections at all. The model found nothing in this clip.")
        return

    print(f"Total detections      : {s['total_detections']}")
    print(f"Detections per frame  : mean {s['dets_per_frame_mean']}, "
          f"min {s['dets_per_frame_min']}, max {s['dets_per_frame_max']}")
    print(f"Frames with 0 dets    : {s['frames_empty']} "
          f"({100 * s['frames_empty'] / frames_seen:.1f}%)")
    print(f"Confidence            : mean {s['conf_mean']:.3f}, "
          f"median {s['conf_median']:.3f}, "
          f"{s['conf_pct_below_040']:.1f}% below 0.40")

    print("\nClass distribution (detections):")
    for c in CANONICAL_CLASSES:
        n = s["class_counts"].get(c, 0)
        print(f"  {c:<6}: {n:>5}  ({100 * n / s['total_detections']:>5.1f}%)")

    n_tracks = s["n_tracks"]
    print(f"\nUnique vehicle IDs    : {n_tracks}")
    print(f"Detections w/o an ID  : {s['untracked_detections']} "
          f"({100 * s['untracked_detections'] / s['total_detections']:.1f}%)"
          f"  [written as vehicle_id=-1]")

    if n_tracks:
        print(f"Track length (frames) : mean {s['track_len_mean']:.1f}, "
              f"median {s['track_len_median']}, max {s['track_len_max']}")
        print(f"Tracks lasting <=2 fr : {s['tracks_le_2_frames']}/{n_tracks} "
              f"({100 * s['tracks_le_2_frames'] / n_tracks:.1f}%)"
              f"  [fragmentation / ID churn]")
        print(f"Tracks changing class : {s['tracks_changing_class']}/{n_tracks} "
              f"({100 * s['tracks_changing_class'] / n_tracks:.1f}%)")
        if s.get("class_flip_pairs"):
            print("  most common confusions:")
            for pair, n in Counter(s["class_flip_pairs"]).most_common(5):
                print(f"    {pair:<18} {n} tracks")

    warnings = []
    if s["frames_empty"] / frames_seen > 0.15:
        warnings.append(f"{100 * s['frames_empty'] / frames_seen:.0f}% of frames have zero "
                        f"detections - likely missed vehicles, try --imgsz 960 --conf 0.15")
    if s["conf_pct_below_040"] > 50:
        warnings.append("over half the detections are below 0.40 confidence - the model "
                        "is unsure; consider a larger model (e.g. yolo11m) or fine-tuning")
    if n_tracks:
        if s["tracks_le_2_frames"] / n_tracks > 0.4:
            warnings.append(f"{100 * s['tracks_le_2_frames'] / n_tracks:.0f}% of tracks last "
                            f"<=2 frames - IDs are not persisting; likely too few "
                            f"detections per frame")
        if s["tracks_changing_class"] / n_tracks > 0.2:
            warnings.append(f"{100 * s['tracks_changing_class'] / n_tracks:.0f}% of tracks "
                            f"change class mid-track - real class confusion, fine-tuning "
                            f"would help")
    if s["untracked_detections"] / s["total_detections"] > 0.2:
        warnings.append(f"{100 * s['untracked_detections'] / s['total_detections']:.0f}% of "
                        f"detections never got a track ID")

    if warnings:
        print("\n" + "-" * 62)
        print("WARNINGS")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("\nNo quality warnings triggered.")

    print("\nNote: there are no ground-truth boxes for this footage, so these numbers\n"
          "cannot measure recall. A vehicle the model never detects is invisible to\n"
          "every statistic above. Check the annotated video before trusting them.")
    print("=" * 62)


def run(args):
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"ERROR: source not found: {video_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, vmeta = open_video(video_path, args.fps)
    cap.release()
    print(f"Video   : {video_path.name}")
    print(f"          {vmeta['width']}x{vmeta['height']} @ {vmeta['fps']:.1f} fps, "
          f"{vmeta['frame_count']} frames")
    print(f"Model   : {args.model}  (device={args.device}, imgsz={args.imgsz}, "
          f"conf={args.conf})")

    model = YOLO(args.model)
    tracker_cfg = build_tracker_cfg(args, out_dir)
    print(f"Tracker : {args.tracker}"
          + ("" if args.raw_tracker_cfg else
             f" (gates set from --conf -> {Path(tracker_cfg).name})") + "\n")

    result = process_video(video_path, args, out_dir, model, tracker_cfg,
                           save_video=args.save_video)
    for path, note in result["written"]:
        print(f"Wrote {path}" + (f"  ({note})" if note else ""))
    report_quality(result["stats"])
    return result


if __name__ == "__main__":
    run(parse_args())
