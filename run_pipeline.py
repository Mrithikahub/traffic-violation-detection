"""
Master Pipeline Orchestrator — End-to-End Traffic Intelligence System
======================================================================
Coordinates the complete multi-stage pipeline:
  Stage 1: Vehicle Detection & Tracking (detect_track.py)
  Stage 2: 4-Way Traffic Violation Analysis (speed, wrong-side, lane-change, tailgating)
  Stage 3: Multi-Violation Video Rendering (combine_violations.py)
  Stage 4: License Plate Recognition / NPR (npr_module)
  Stage 5: Fine Estimation Engine (fine_estimation)
  Stage 6: Tagged Evidence Crop Generation & Unified Vehicle Record Merger (dashboard_data_contract.md)

Usage:
    python run_pipeline.py --video data/video/teammate_video.mp4 --out-dir outputs/ --save-video
    python run_pipeline.py --video teammate_video.mp4 --out-dir outputs_calib --reuse-tracks
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_estimation.src import FineCalculator, RuleEngine

CLASS_COLORS = {
    "car": (0, 200, 255),       # Yellow-Orange
    "bike": (255, 120, 0),      # Blue-Cyan
    "bus": (0, 255, 120),       # Green
    "truck": (200, 0, 255),     # Pink
    "rickshaw": (60, 220, 220), # Pale Yellow
    "unknown": (180, 180, 180), # Grey
}


# Fallback plausibility ceiling for the speed layer when neither the command
# line nor a site_calibration.json supplies one. Readings above this are
# rejected as projection/tracking artifacts. This is NOT a speed limit.
DEFAULT_MAX_PLAUSIBLE_KMH = 100.0


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def resolve_calibration_dir(args) -> Path:
    """Directory holding the site calibration bundle used by this run.

    The homography defines which camera view we are calibrated for, so the
    guards that belong to that view live beside it.
    """
    if args.homography and Path(args.homography).exists():
        return Path(args.homography).parent
    return PROJECT_ROOT / "outputs_demo"


def load_site_calibration(cal_dir: Path) -> Dict[str, Any]:
    """Site-specific speed guards, if this camera view has any recorded.

    Absent file is normal -- footage from an uncalibrated camera simply runs
    without the site guards. A malformed one is worth a warning, because
    silently dropping the guards is how implausible speeds reach the fines.
    """
    p = cal_dir / "site_calibration.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as ex:
        log(f"  WARNING: {p} is unreadable ({ex}). Speed guards NOT applied; "
            f"far-field readings will not be filtered.")
        return {}


def apply_speed_guards(cmd: List[str], args) -> None:
    """Append the calibration guards to the speed_violation.py command.

    Without these the layer reports speeds extrapolated far outside the region
    the homography was fitted over. On the reference clip that produced 37
    speeding vehicles of which 18 exceeded 100 km/h, topping out at 148.5 --
    and the fine engine charged every one of them.

    Precedence: command line > site_calibration.json > built-in fallback.
    """
    site = load_site_calibration(resolve_calibration_dir(args))
    applied = []

    min_ref_y = args.min_ref_y if args.min_ref_y is not None else site.get("min_ref_y")
    if min_ref_y is not None:
        cmd.extend(["--min-ref-y", str(min_ref_y)])
        applied.append(f"min-ref-y={min_ref_y}")

    max_kmh = args.max_plausible_kmh
    if max_kmh is None:
        max_kmh = site.get("max_plausible_kmh", DEFAULT_MAX_PLAUSIBLE_KMH)
    cmd.extend(["--max-plausible-kmh", str(max_kmh)])
    applied.append(f"max-plausible-kmh={max_kmh}")

    regions = args.ignore_region or site.get("ignore_regions") or []
    for r in regions:
        spec = r if isinstance(r, str) else ",".join(str(v) for v in r)
        cmd.extend(["--ignore-region", spec])
        applied.append(f"ignore-region={spec}")

    if min_ref_y is None:
        log("  Notice: no min-ref-y for this camera view. Speeds will be "
            "reported outside the calibrated band and may be extrapolated.")
    log(f"  Speed guards: {', '.join(applied)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run end-to-end Traffic Monitoring and Violation Detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video file or video name (e.g. data/video/teammate_video.mp4)")
    p.add_argument("--out-dir", default="outputs", help="Directory where all intermediate and final outputs are saved")
    p.add_argument("--model", default="yolov8n.pt", help="Vehicle detection model weights (YOLO)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config name or path")
    p.add_argument("--speed-limit", type=float, default=60.0, help="Road speed limit in km/h")
    p.add_argument("--min-ref-y", type=float, default=None,
                   help="Report speed only where the box bottom-centre is at or below this "
                        "image row, i.e. inside the band the homography was fitted over. "
                        "Defaults to the value in site_calibration.json beside the homography.")
    p.add_argument("--max-plausible-kmh", type=float, default=None,
                   help="Reject speed readings above this as artifacts. NOT a speed limit. "
                        f"Falls back to site_calibration.json, then {DEFAULT_MAX_PLAUSIBLE_KMH}.")
    p.add_argument("--ignore-region", action="append", default=None, metavar="x1,y1,x2,y2",
                   help="Exclude boxes whose centre falls in this rectangle from the speed "
                        "layer. Repeatable. Defaults to site_calibration.json.")
    p.add_argument("--homography", default=None, help="Path to homography.npy matrix for road calibration")
    p.add_argument("--zones", default=None, help="Path to zones JSON for wrong-lane detection")
    p.add_argument("--lanes", default=None, help="Path to lanes JSON for lane-change detection")
    p.add_argument("--rules-config", default=None, help="Path to custom fine_rules.json config")
    p.add_argument("--reuse-tracks", action="store_true", default=False, help="Reuse existing _tracks.csv and _vehicles.csv in out-dir if present")
    p.add_argument("--skip-npr", action="store_true", default=False, help="Skip license plate OCR stage")
    p.add_argument("--skip-crops", action="store_true", default=False, help="Skip generating tagged evidence crop images")
    p.add_argument("--save-video", action="store_true", default=False, help="Render multi-violation annotated video")
    p.add_argument("--conf", type=float, default=0.10, help="Vehicle detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    p.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    p.add_argument("--device", default="cpu", help="Compute device for YOLO models ('cpu' or '0')")
    return p.parse_args()


def run_command_stage(cmd: List[str], stage_name: str) -> bool:
    """Executes an existing pipeline script as a subprocess."""
    log(f"Starting {stage_name}...")
    cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    log(f"  Command: {cmd_str}")
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        log(f"WARNING in {stage_name} (exited with {res.returncode}):")
        if res.stderr:
            print(res.stderr.strip(), file=sys.stderr)
        return False
    log(f"Finished {stage_name} in {dt:.2f}s")
    return True


def run_npr_stage(
    video_path: Optional[Path],
    tracks_csv: Path,
    vehicles_csv: Path,
    min_area: float = 3000.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs Number Plate Recognition on each vehicle's best_frame_id crop.
    """
    log("Starting Stage 4: Number Plate Recognition (NPR Module)...")
    t0 = time.time()

    if not video_path or not video_path.exists():
        log("  Notice: Raw video file not directly accessible for pixel cropping. Preserving existing plate records if available.")
        return {}

    try:
        from npr_module.src.pipeline import NumberPlateRecognizer
        recognizer = NumberPlateRecognizer()
    except Exception as e:
        log(f"  Notice: Could not load live NumberPlateRecognizer ({e}). Continuing with downstream pipeline.")
        return {}

    # 1. Load vehicles to find best_frame_id
    vehicles: Dict[int, Dict[str, str]] = {}
    with open(vehicles_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = int(r["vehicle_id"])
            if vid > 0:
                vehicles[vid] = r

    # 2. Load bounding boxes for (best_frame_id, vehicle_id)
    needed_pairs = set()
    for vid, v in vehicles.items():
        if v.get("best_frame_id"):
            needed_pairs.add((int(v["best_frame_id"]), vid))

    boxes: Dict[Tuple[int, int], Dict[str, str]] = {}
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["vehicle_id"]))
            if key in needed_pairs:
                boxes[key] = r

    # 3. Group by frame_id to minimize video seek overhead
    frame_to_vids = defaultdict(list)
    for vid, v in vehicles.items():
        if v.get("best_frame_id"):
            fid = int(v["best_frame_id"])
            area = float(v.get("best_frame_area", 0.0) or 0.0)
            if (fid, vid) in boxes and area >= min_area:
                frame_to_vids[fid].append((vid, boxes[(fid, vid)]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"  Notice: Cannot open video for NPR at {video_path}")
        return {}

    results: Dict[int, Dict[str, Any]] = {}
    plates_found = 0
    total_crops_tested = 0

    sorted_frames = sorted(frame_to_vids.keys())
    for fid in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        for vid, b in frame_to_vids[fid]:
            x1 = max(0, int(float(b["x1"])))
            y1 = max(0, int(float(b["y1"])))
            x2 = min(w, int(float(b["x2"])))
            y2 = min(h, int(float(b["y2"])))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            total_crops_tested += 1
            try:
                res = recognizer.process_vehicle_crop(
                    vehicle_crop=crop,
                    vehicle_id=vid,
                    frame_id=fid,
                    vehicle_bbox=[x1, y1, x2, y2],
                )
                results[vid] = res
                if res.get("plate_detected") and res.get("plate_text"):
                    plates_found += 1
            except Exception as ex:
                results[vid] = {
                    "vehicle_id": vid,
                    "plate_detected": False,
                    "plate_text": None,
                    "error": str(ex),
                }

    cap.release()
    dt = time.time() - t0
    log(f"Finished NPR Stage in {dt:.2f}s: Processed {total_crops_tested} vehicle crops, localized {plates_found} plates.")
    return results


def generate_tagged_evidence_crops(
    video_path: Path,
    tracks_csv: Path,
    unified_records: List[Dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> None:
    """
    Generates high-resolution tagged vehicle evidence crops with telemetry banners
    and saves them to out_dir / "evidence_crops".
    """
    if not video_path.exists():
        return

    crops_dir = out_dir / "evidence_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build lookup for vehicle records
    rec_by_vid = {r["vehicle_id"]: r for r in unified_records}

    # 2. Collect boxes for target frames
    needed_pairs = set()
    for r in unified_records:
        fid = r["evidence"].get("best_frame_id")
        if fid is not None:
            needed_pairs.add((fid, r["vehicle_id"]))

    boxes: Dict[Tuple[int, int], Dict[str, str]] = {}
    if tracks_csv.exists():
        with open(tracks_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (int(r["frame_id"]), int(r["vehicle_id"]))
                if key in needed_pairs:
                    boxes[key] = r

    # 3. Group by frame_id
    frame_to_vids = defaultdict(list)
    for vid, r in rec_by_vid.items():
        fid = r["evidence"].get("best_frame_id")
        if fid is not None and (fid, vid) in boxes:
            frame_to_vids[fid].append(vid)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    generated_count = 0
    for fid in sorted(frame_to_vids.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue

        H, W = frame.shape[:2]
        for vid in frame_to_vids[fid]:
            b = boxes[(fid, vid)]
            u_rec = rec_by_vid[vid]

            # Bounding box with 10% context margin
            x1 = float(b["x1"])
            y1 = float(b["y1"])
            x2 = float(b["x2"])
            y2 = float(b["y2"])
            bw, bh = x2 - x1, y2 - y1

            pad_x = bw * 0.12
            pad_y = bh * 0.12

            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(W, int(x2 + pad_x))
            cy2 = min(H, int(y2 + pad_y))

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = frame[cy1:cy2, cx1:cx2].copy()
            ch, cw = crop.shape[:2]

            # Scale up small crops for crisp text rendering
            target_w = max(420, cw)
            if target_w > cw:
                scale = target_w / cw
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
                ch, cw = crop.shape[:2]

            # Add the banners as EXTRA canvas above and below the crop rather
            # than painting them onto it. Drawing filled rectangles straight on
            # the image overwrote 76px of vehicle -- measured across this clip
            # that obliterated a median 17% of the output image, covered vehicle
            # pixels on 99 of 164 crops, and cost one vehicle 57% of its body.
            # The bottom bar in particular sat exactly where the number plate is,
            # which is the one thing an evidence image exists to show.
            top_bar_h = 32
            bot_bar_h = 44
            crop = cv2.copyMakeBorder(crop, top_bar_h, bot_bar_h, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(15, 23, 42))
            ch, cw = crop.shape[:2]

            v_cls = u_rec["class"].upper()
            cls_col = CLASS_COLORS.get(u_rec["class"].lower(), (200, 200, 200))
            cv2.putText(crop, f"#{vid} [{v_cls}]", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cls_col, 2, cv2.LINE_AA)

            sec_txt = f"f:{fid} ({fid/30.0:.2f}s)"
            (tw, _), _ = cv2.getTextSize(sec_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(crop, sec_txt, (cw - tw - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Bottom telemetry panel. The canvas for it already exists (added
            # above), so nothing here overwrites vehicle pixels.

            # Line 1: Plate & Fine
            p_text = u_rec["plate"]["plate_text"] or "PLATE UNRESOLVED"
            p_col = (16, 185, 129) if u_rec["plate"]["plate_detected"] else (148, 163, 184)  # Emerald vs Grey
            cv2.putText(crop, f"PLATE: {p_text}", (8, ch - bot_bar_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_col, 1, cv2.LINE_AA)

            fine_txt = f"FINE: INR {u_rec['fine']['total_fine']:,.1f}"
            fine_col = (239, 68, 68) if u_rec["fine"]["total_fine"] > 0 else (148, 163, 184)
            (ftw, _), _ = cv2.getTextSize(fine_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(crop, fine_txt, (cw - ftw - 8, ch - bot_bar_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fine_col, 1, cv2.LINE_AA)

            # Line 2: Violations
            viols = u_rec["violations"]
            if viols:
                v_str = " | ".join(v.upper() for v in viols)
                if u_rec["speed"]["max_speed_kmh"]:
                    v_str += f" ({u_rec['speed']['max_speed_kmh']:.0f}km/h)"
                cv2.putText(crop, f"VIOLATIONS: {v_str}", (8, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (248, 113, 113), 1, cv2.LINE_AA)
            else:
                cv2.putText(crop, "STATUS: COMPLIANT (0 VIOLATIONS)", (8, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

            # Save tagged image
            img_rel_path = f"evidence_crops/{stem}_{vid}_evidence.jpg"
            img_abs_path = out_dir / img_rel_path
            cv2.imwrite(str(img_abs_path), crop)
            u_rec["evidence"]["evidence_image_path"] = img_rel_path
            generated_count += 1

    cap.release()
    log(f"Generated {generated_count} tagged vehicle evidence crop images in {crops_dir.name}/")


def main():
    args = parse_args()
    video_raw = Path(args.video)
    stem = video_raw.stem

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"TRAFFIC MONITORING & VIOLATION DETECTION PIPELINE")
    log(f"Video Identifier : {stem}")
    log(f"Output Directory : {out_dir}")
    log(f"Speed Limit      : {args.speed_limit} km/h")
    log(f"Detection Model  : {args.model}")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Vehicle Detection & Tracking
    # -------------------------------------------------------------------------
    tracks_csv = out_dir / f"{stem}_tracks.csv"
    vehicles_csv = out_dir / f"{stem}_vehicles.csv"

    # Also check outputs_demo as fallback for pre-extracted tracks
    if not tracks_csv.exists() and (PROJECT_ROOT / "outputs_demo" / f"{stem}_tracks.csv").exists():
        tracks_csv = PROJECT_ROOT / "outputs_demo" / f"{stem}_tracks.csv"
    if not vehicles_csv.exists() and (PROJECT_ROOT / "outputs_demo" / f"{stem}_vehicles.csv").exists():
        vehicles_csv = PROJECT_ROOT / "outputs_demo" / f"{stem}_vehicles.csv"

    if args.reuse_tracks and tracks_csv.exists() and vehicles_csv.exists():
        log(f"Stage 1: Reusing existing tracks from {tracks_csv}")
    else:
        if not video_raw.exists():
            log(f"ERROR: Input video not found at {video_raw}. Cannot run detection.")
            sys.exit(1)
        cmd_track = [
            "detect_track.py",
            "--video", str(video_raw),
            "--out-dir", str(out_dir),
            "--model", str(args.model),
            "--tracker", str(args.tracker),
            "--conf", str(args.conf),
            "--imgsz", str(args.imgsz),
            "--start-frame", str(args.start_frame),
        ]
        if not run_command_stage(cmd_track, "Stage 1: Vehicle Detection & Tracking"):
            log("Stage 1 failed. Aborting pipeline.")
            sys.exit(1)
        tracks_csv = out_dir / f"{stem}_tracks.csv"
        vehicles_csv = out_dir / f"{stem}_vehicles.csv"

    # -------------------------------------------------------------------------
    # STAGE 2: 4-Way Traffic Violation Analysis
    # -------------------------------------------------------------------------
    speed_violations_csv = out_dir / f"{stem}_violations.csv"
    speed_tracks_csv = out_dir / f"{stem}_tracks_speed.csv"
    speed_meta_json = out_dir / f"{stem}_speed_meta.json"

    if video_raw.exists():
        # 2A: Speed Estimation
        cmd_speed = [
            "speed_violation.py",
            "--video", str(video_raw),
            "--tracks", str(tracks_csv),
            "--out-dir", str(out_dir),
            "--speed-limit", str(args.speed_limit),
        ]
        apply_speed_guards(cmd_speed, args)
        if args.homography and Path(args.homography).exists():
            cmd_speed.extend(["--homography", str(args.homography)])
        elif (PROJECT_ROOT / "outputs_demo" / "homography.npy").exists():
            cmd_speed.extend(["--homography", str(PROJECT_ROOT / "outputs_demo" / "homography.npy")])
        run_command_stage(cmd_speed, "Stage 2A: Speed Estimation & Violation")

        # 2B: Wrong-Side Driving
        wrong_lane_csv = out_dir / f"{stem}_wrong_lane.csv"
        wrong_lane_violations_csv = out_dir / f"{stem}_wrong_lane_violations.csv"
        cmd_wrong = [
            "wrong_lane.py",
            "--video", str(video_raw),
            "--tracks", str(tracks_csv),
            "--out-dir", str(out_dir),
        ]
        if args.zones and Path(args.zones).exists():
            cmd_wrong.extend(["--zones", str(args.zones)])
        elif (PROJECT_ROOT / "outputs_demo" / "zones.json").exists():
            cmd_wrong.extend(["--zones", str(PROJECT_ROOT / "outputs_demo" / "zones.json")])
        run_command_stage(cmd_wrong, "Stage 2B: Wrong-Side Driving Analysis")

        # 2C: Lane Change Detection
        lane_change_csv = out_dir / f"{stem}_lane_change.csv"
        lane_change_events_csv = out_dir / f"{stem}_lane_change_events.csv"
        cmd_lane = [
            "lane_change.py",
            "--video", str(video_raw),
            "--tracks", str(tracks_csv),
            "--out-dir", str(out_dir),
        ]
        if args.lanes and Path(args.lanes).exists():
            cmd_lane.extend(["--lanes", str(args.lanes)])
        elif (PROJECT_ROOT / "outputs_demo" / "lanes.json").exists():
            cmd_lane.extend(["--lanes", str(PROJECT_ROOT / "outputs_demo" / "lanes.json")])
        if args.homography and Path(args.homography).exists():
            cmd_lane.extend(["--homography", str(args.homography)])
        elif (PROJECT_ROOT / "outputs_demo" / "homography.npy").exists():
            cmd_lane.extend(["--homography", str(PROJECT_ROOT / "outputs_demo" / "homography.npy")])
        run_command_stage(cmd_lane, "Stage 2C: Lane Change Event Detection")

        # 2D: Tailgating Detection
        tailgating_csv = out_dir / f"{stem}_tailgating.csv"
        tailgating_events_csv = out_dir / f"{stem}_tailgating_events.csv"
        cmd_tail = [
            "tailgating.py",
            "--video", str(video_raw),
            "--tracks", str(tracks_csv),
            "--out-dir", str(out_dir),
        ]
        run_command_stage(cmd_tail, "Stage 2D: Tailgating Violation Analysis")
    else:
        log("Notice: Video file not found directly on disk; using existing summary violation files in out-dir if present.")

    # Fallback to check if summary files exist in out-dir or outputs_calib
    def resolve_file(name: str) -> Optional[Path]:
        p1 = out_dir / name
        if p1.exists():
            return p1
        p2 = PROJECT_ROOT / "outputs_calib" / name
        if p2.exists():
            return p2
        return None

    speed_violations_csv = resolve_file(f"{stem}_violations.csv")
    wrong_lane_violations_csv = resolve_file(f"{stem}_wrong_lane_violations.csv")
    lane_change_events_csv = resolve_file(f"{stem}_lane_change_events.csv")
    tailgating_events_csv = resolve_file(f"{stem}_tailgating_events.csv")
    speed_meta_json = resolve_file(f"{stem}_speed_meta.json")

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Violation Video Rendering (Optional)
    # -------------------------------------------------------------------------
    annotated_video = out_dir / f"{stem}_all_violations.mp4"
    if args.save_video and video_raw.exists():
        cmd_render = [
            "combine_violations.py",
            "--video", str(video_raw),
            "--tracks", str(tracks_csv),
            "--out-dir", str(out_dir),
        ]
        sp_csv = resolve_file(f"{stem}_tracks_speed.csv")
        wl_csv = resolve_file(f"{stem}_wrong_lane.csv")
        lc_csv = resolve_file(f"{stem}_lane_change.csv")
        tg_csv = resolve_file(f"{stem}_tailgating.csv")
        if sp_csv:
            cmd_render.extend(["--speed-csv", str(sp_csv)])
        if wl_csv:
            cmd_render.extend(["--wrong-lane-csv", str(wl_csv)])
        if lc_csv:
            cmd_render.extend(["--lane-change-csv", str(lc_csv)])
        if tg_csv:
            cmd_render.extend(["--tailgating-csv", str(tg_csv)])
        run_command_stage(cmd_render, "Stage 3: Multi-Violation Video Rendering")

    # -------------------------------------------------------------------------
    # STAGE 4: License Plate Recognition (NPR Module)
    # -------------------------------------------------------------------------
    npr_results: Dict[int, Dict[str, Any]] = {}
    if not args.skip_npr and video_raw.exists():
        npr_results = run_npr_stage(video_raw, tracks_csv, vehicles_csv)

    # -------------------------------------------------------------------------
    # STAGE 5: Fine Estimation Engine
    # -------------------------------------------------------------------------
    log("Starting Stage 5: Fine Estimation Engine...")
    fine_engine = RuleEngine(config_path=args.rules_config)
    fine_calc = FineCalculator(rule_engine=fine_engine)

    fine_records = fine_calc.process_records(
        video_id=stem,
        vehicles_csv=vehicles_csv,
        speed_violations_csv=speed_violations_csv,
        wrong_lane_csv=wrong_lane_violations_csv,
        lane_change_csv=lane_change_events_csv,
        tailgating_csv=tailgating_events_csv,
        speed_meta_json=speed_meta_json,
        speed_limit_kmh=args.speed_limit,
    )
    fine_calc.save_outputs(records=fine_records, out_dir=out_dir, basename=stem)
    log(f"Finished Stage 5: Evaluated fines for {len(fine_records)} vehicles.")

    # -------------------------------------------------------------------------
    # STAGE 6: Unified Vehicle Record Merger & Evidence Tagging
    # -------------------------------------------------------------------------
    log("Starting Stage 6: Unified Record Aggregation & Evidence Image Tagging...")
    unified_records: List[Dict[str, Any]] = []

    fine_by_vid = {r["vehicle_id"]: r for r in fine_records}

    veh_details = {}
    if vehicles_csv and vehicles_csv.exists():
        with open(vehicles_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = int(r["vehicle_id"])
                if vid > 0:
                    veh_details[vid] = r

    for vid in sorted(fine_by_vid.keys()):
        f_rec = fine_by_vid[vid]
        v_det = veh_details.get(vid, {})
        npr_rec = npr_results.get(vid, {})

        plate_obj = {
            "plate_detected": bool(npr_rec.get("plate_detected", False)),
            "plate_text": npr_rec.get("plate_text"),
            "raw_ocr_text": npr_rec.get("raw_ocr_text"),
            "ocr_confidence": npr_rec.get("ocr_confidence"),
            "is_valid_indian_format": npr_rec.get("is_valid_indian_format"),
            "format_type": npr_rec.get("format_type"),
            "state_code": npr_rec.get("state_code"),
            "state_name": npr_rec.get("state_name"),
            "district_code": npr_rec.get("district_code"),
            "series_code": npr_rec.get("series_code"),
            "unique_number": npr_rec.get("unique_number"),
        }

        max_spd = f_rec["max_speed_kmh"]
        avg_spd = f_rec["avg_speed_kmh"]
        spd_lim = f_rec["speed_limit_kmh"]
        spd_delta = round(max_spd - spd_lim, 2) if (max_spd is not None and spd_lim is not None and max_spd > spd_lim) else None

        speed_obj = {
            "speed_limit_kmh": spd_lim,
            "max_speed_kmh": max_spd,
            "avg_speed_kmh": avg_spd,
            "speed_delta_kmh": spd_delta,
            "is_speed_measured": max_spd is not None,
        }

        tracking_obj = {
            "n_frames": int(v_det.get("n_frames")) if v_det.get("n_frames") else None,
            "first_frame": int(v_det.get("first_frame")) if v_det.get("first_frame") else None,
            "last_frame": int(v_det.get("last_frame")) if v_det.get("last_frame") else None,
            "duration_sec": float(v_det.get("duration_sec")) if v_det.get("duration_sec") else None,
            "mean_confidence": float(v_det.get("mean_confidence")) if v_det.get("mean_confidence") else None,
        }

        best_area = float(v_det.get("best_frame_area", 0.0) or 0.0)
        evidence_obj = {
            "best_frame_id": int(v_det.get("best_frame_id")) if v_det.get("best_frame_id") else None,
            "best_frame_area": best_area if best_area > 0 else None,
            "best_frame_confidence": float(v_det.get("best_frame_confidence")) if v_det.get("best_frame_confidence") else None,
            "has_resolvable_plate_budget": best_area >= 5000.0,
            "evidence_image_path": None,
        }

        fine_obj = {
            "total_fine": f_rec["total_fine"],
            "currency": f_rec["currency"],
            "compounding_policy": f_rec["compounding_policy"],
            "fine_breakdown": f_rec["fine_breakdown"],
        }

        unified_rec = {
            "video_id": stem,
            "vehicle_id": vid,
            "composite_id": f"{stem}_{vid}",
            "class": f_rec["class"],
            "class_agreement": f_rec["class_agreement"],
            "tracking": tracking_obj,
            "speed": speed_obj,
            "plate": plate_obj,
            "violations": f_rec["violations"],
            "violation_count": f_rec["violation_count"],
            "first_violation_frame": f_rec["first_violation_frame"],
            "first_violation_sec": f_rec["first_violation_sec"],
            "timing_by_violation": f_rec["timing_by_violation"],
            "fine": fine_obj,
            "evidence": evidence_obj,
            "disclaimer": f_rec["disclaimer"],
        }
        unified_records.append(unified_rec)

    # Generate Tagged Evidence Crop Images (if video available and not skipped)
    if not args.skip_crops and video_raw.exists():
        generate_tagged_evidence_crops(video_raw, tracks_csv, unified_records, out_dir, stem)

    # Save Unified JSON (Full Archive)
    unified_json_path = out_dir / f"{stem}_unified_records.json"
    with open(unified_json_path, "w", encoding="utf-8") as f:
        json.dump(unified_records, f, indent=2)

    # Save Lightweight Dashboard Violators JSON (Only violating vehicles, clean presentation fields)
    dashboard_violators = []
    for r in unified_records:
        if r["violation_count"] > 0:
            dashboard_violators.append({
                "track_id": r["vehicle_id"],
                "vehicle_class": r["class"],
                "plate_number": r["plate"]["plate_text"] or "UNRESOLVED",
                "is_plate_detected": r["plate"]["plate_detected"],
                "violations": r["violations"],
                "violation_count": r["violation_count"],
                "speed_kmh": r["speed"]["max_speed_kmh"],
                "speed_limit_kmh": r["speed"]["speed_limit_kmh"],
                "fine_amount": r["fine"]["total_fine"],
                "currency": r["fine"]["currency"],
                "time_sec": r["first_violation_sec"],
                "evidence_image": r["evidence"]["evidence_image_path"],
            })

    violators_json_path = out_dir / f"{stem}_dashboard_violators.json"
    with open(violators_json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_violators, f, indent=2)

    # Save Unified CSV
    unified_csv_path = out_dir / f"{stem}_unified_records.csv"
    csv_headers = [
        "video_id", "vehicle_id", "composite_id", "class", "class_agreement",
        "plate_text", "is_valid_indian_format", "state_name", "violations",
        "violation_count", "total_fine", "currency", "speed_limit_kmh",
        "max_speed_kmh", "avg_speed_kmh", "speed_delta_kmh",
        "first_violation_frame", "first_violation_sec", "best_frame_id",
        "best_frame_area", "evidence_image_path"
    ]
    with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for r in unified_records:
            row = {
                "video_id": r["video_id"],
                "vehicle_id": r["vehicle_id"],
                "composite_id": r["composite_id"],
                "class": r["class"],
                "class_agreement": r["class_agreement"] if r["class_agreement"] is not None else "",
                "plate_text": r["plate"]["plate_text"] or "",
                "is_valid_indian_format": str(r["plate"]["is_valid_indian_format"]) if r["plate"]["is_valid_indian_format"] is not None else "",
                "state_name": r["plate"]["state_name"] or "",
                "violations": ";".join(r["violations"]),
                "violation_count": r["violation_count"],
                "total_fine": r["fine"]["total_fine"],
                "currency": r["fine"]["currency"],
                "speed_limit_kmh": r["speed"]["speed_limit_kmh"] if r["speed"]["speed_limit_kmh"] is not None else "",
                "max_speed_kmh": r["speed"]["max_speed_kmh"] if r["speed"]["max_speed_kmh"] is not None else "",
                "avg_speed_kmh": r["speed"]["avg_speed_kmh"] if r["speed"]["avg_speed_kmh"] is not None else "",
                "speed_delta_kmh": r["speed"]["speed_delta_kmh"] if r["speed"]["speed_delta_kmh"] is not None else "",
                "first_violation_frame": r["first_violation_frame"] if r["first_violation_frame"] is not None else "",
                "first_violation_sec": r["first_violation_sec"] if r["first_violation_sec"] is not None else "",
                "best_frame_id": r["evidence"]["best_frame_id"] if r["evidence"]["best_frame_id"] is not None else "",
                "best_frame_area": r["evidence"]["best_frame_area"] if r["evidence"]["best_frame_area"] is not None else "",
                "evidence_image_path": r["evidence"]["evidence_image_path"] or "",
            }
            writer.writerow(row)

    # Save Summary KPI JSON for Dashboard top scorecards
    kpi_summary = {
        "video_id": stem,
        "total_vehicles_tracked": len(unified_records),
        "total_violating_vehicles": sum(1 for r in unified_records if r["violation_count"] > 0),
        "total_violations_detected": sum(r["violation_count"] for r in unified_records),
        "violations_by_type": {
            "speeding": sum(1 for r in unified_records if "speeding" in r["violations"]),
            "tailgating": sum(1 for r in unified_records if "tailgating" in r["violations"]),
            "lane_change": sum(1 for r in unified_records if "lane_change" in r["violations"]),
            "wrong_side": sum(1 for r in unified_records if "wrong_side" in r["violations"]),
        },
        "vehicles_with_multiple_violations": sum(1 for r in unified_records if r["violation_count"] > 1),
        "total_prototype_fines": round(sum(r["fine"]["total_fine"] for r in unified_records), 2),
        "currency": fine_engine.currency,
        "plates_localized_count": sum(1 for r in unified_records if r["plate"]["plate_detected"]),
        "disclaimer": fine_engine.disclaimer,
    }
    kpi_json_path = out_dir / f"{stem}_summary_kpi.json"
    with open(kpi_json_path, "w", encoding="utf-8") as f:
        json.dump(kpi_summary, f, indent=2)

    log("=" * 70)
    log("PIPELINE COMPLETED SUCCESSFULLY!")
    log(f"  Total vehicles tracked       : {kpi_summary['total_vehicles_tracked']}")
    log(f"  Total violating vehicles     : {kpi_summary['total_violating_vehicles']}")
    log(f"  Total violations detected    : {kpi_summary['total_violations_detected']}")
    log(f"  Total prototype fines        : {kpi_summary['currency']} {kpi_summary['total_prototype_fines']:,.2f}")
    log(f"Generated Files:")
    log(f"  Dashboard JSON : {violators_json_path}")
    log(f"  Full JSON      : {unified_json_path}")
    log(f"  Full CSV       : {unified_csv_path}")
    log(f"  KPI Summary    : {kpi_json_path}")
    if (out_dir / "evidence_crops").exists():
        log(f"  Evidence Tagged Images : {out_dir / 'evidence_crops'}")
    if args.save_video and annotated_video.exists():
        log(f"  Video Render : {annotated_video}")
    log("=" * 70)


if __name__ == "__main__":
    main()
