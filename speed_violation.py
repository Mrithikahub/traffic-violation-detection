"""
Speed estimation + basic violation flagging, layered on the tracking output.

Reads the per-frame tracks CSV produced by detect_track.py and the source video,
estimates a speed per vehicle from frame-to-frame box movement, flags vehicles
over a configurable limit, and renders an annotated video.

Deliberately decoupled from detection: changing the speed limit or the scale
assumption re-runs in seconds instead of re-running YOLO over the whole video.

    python detect_track.py --video <in.mp4> --model <weights> --out-dir out
    python speed_violation.py --video <in.mp4> \
        --tracks out/<stem>_tracks.csv --out-dir out --speed-limit 60 --save-video

=============================== ACCURACY ===============================
Speeds are INDICATIVE ONLY. There is no camera calibration, so pixels are
converted to metres with an assumed scale. Treat the numbers as a relative
ranking ("this vehicle is moving much faster than that one"), not as
measurements. Do not use them for enforcement.

Three specific limitations:

1. SCALE IS ASSUMED, NOT MEASURED. Default derived from a standard ~3.5 m lane
   spanning ~310 px near the bottom of frame. Override with --mpp-near/--mpp-far.

2. PERSPECTIVE. A fixed metres-per-pixel is wrong for a camera looking down a
   road: the same real distance covers far fewer pixels further away. We
   interpolate the scale linearly by image row between --mpp-far (at the top of
   the measured band) and --mpp-near (at the bottom). Still an approximation --
   true perspective is non-linear -- but it stops distant vehicles reading ~0.

3. IMAGE-PLANE MOTION ONLY. Movement is measured in the image, so a vehicle
   heading straight at the camera is heavily foreshortened and reads slow.
   Cross-frame motion is measured most reliably.
========================================================================
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detect_track import CLASS_COLORS

# Output column order. Extends the tracks schema documented in OUTPUT_FORMAT.md.
CSV_COLUMNS = [
    "frame_id", "vehicle_id", "class", "x1", "y1", "x2", "y2",
    "confidence", "speed_kmh", "violation",
]

VIOLATION_COLOR = (0, 0, 255)   # BGR red


def parse_args():
    p = argparse.ArgumentParser(
        description="Speed estimation and violation flagging over tracked output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Source video (any file).")
    p.add_argument("--tracks", required=True,
                   help="Per-frame CSV from detect_track.py.")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--fps", type=float, default=None,
                   help="Override frame rate. Defaults to the video's own value; "
                        "required if the source reports none.")

    # --- scale assumption -------------------------------------------------
    p.add_argument("--mpp-near", type=float, default=0.0113,
                   help="Metres per pixel at the BOTTOM of the measured band "
                        "(near the camera). Default assumes a 3.5 m lane over "
                        "~310 px.")
    p.add_argument("--mpp-far", type=float, default=0.0450,
                   help="Metres per pixel at the TOP of the measured band (far "
                        "from the camera). Larger because distant objects are "
                        "smaller on screen.")
    p.add_argument("--constant-mpp", type=float, default=None,
                   help="Use one fixed metres-per-pixel everywhere instead of "
                        "interpolating by row. Simpler, less realistic.")
    p.add_argument("--band-top", type=float, default=0.15,
                   help="Top of the measured band as a fraction of frame height.")
    p.add_argument("--band-bottom", type=float, default=1.0,
                   help="Bottom of the measured band as a fraction of height.")

    # --- smoothing / flagging --------------------------------------------
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Frames over which displacement is measured. Larger is "
                        "smoother but lags real changes.")
    p.add_argument("--speed-limit", type=float, default=60.0,
                   help="km/h above which a vehicle is flagged.")
    p.add_argument("--min-violation-frames", type=int, default=3,
                   help="Frames a vehicle must stay over the limit before it "
                        "counts. Suppresses one-frame spikes from box jitter.")
    p.add_argument("--min-track-frames", type=int, default=4,
                   help="Tracks shorter than this get no speed (too little "
                        "evidence); they are written with an empty speed.")
    p.add_argument("--max-plausible-kmh", type=float, default=150.0,
                   help="Speeds above this are treated as tracking glitches and "
                        "discarded rather than reported.")

    p.add_argument("--homography", default=None,
                   help="Path to a 3x3 .npy mapping image pixels to ground-plane "
                        "metres (see calibrate_homography.py). When given, speeds "
                        "are computed from real ground coordinates using each "
                        "box's BOTTOM-CENTRE (where the vehicle meets the road) "
                        "and the --mpp-* row model is ignored.")
    p.add_argument("--min-ref-y", type=float, default=None,
                   help="Skip speed for samples whose reference point sits ABOVE "
                        "this image row (smaller y = further from camera). Use "
                        "the top of the region the homography was actually "
                        "fitted over: beyond it the mapping is extrapolating and "
                        "a one-pixel error projects to metres, producing "
                        "implausible speeds. Boxes are still detected and drawn, "
                        "they just carry no speed.")
    p.add_argument("--max-world-jump-m", type=float, default=40.0,
                   help="Discard a sample if the ground-plane displacement over "
                        "the window exceeds this. Guards against points near the "
                        "horizon, where small pixel errors project to huge "
                        "distances.")
    p.add_argument("--ignore-region", action="append", default=None,
                   metavar="x1,y1,x2,y2",
                   help="Rectangle to exclude, in source pixels. Repeatable. Use "
                        "for player overlays or picture-in-picture insets that "
                        "would otherwise be detected as real traffic.")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--annot-scale", type=float, default=1.0)
    return p.parse_args()


def parse_regions(specs):
    out = []
    for s in specs or []:
        parts = [float(v) for v in s.replace(" ", "").split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--ignore-region needs x1,y1,x2,y2 (got '{s}')")
        x1, y1, x2, y2 = parts
        out.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return out


def in_regions(box, regions):
    """True if the box centre falls inside any ignored rectangle."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in regions)


def make_scale_fn(args, height):
    """metres-per-pixel as a function of image row."""
    if args.constant_mpp is not None:
        k = args.constant_mpp
        return (lambda y: k), f"constant {k} m/px"
    y_top = args.band_top * height
    y_bot = args.band_bottom * height
    span = max(y_bot - y_top, 1e-6)

    def scale(y):
        t = (float(y) - y_top) / span          # 0 at band top, 1 at bottom
        t = min(max(t, 0.0), 1.0)              # clamp outside the band
        return args.mpp_far + t * (args.mpp_near - args.mpp_far)

    desc = (f"linear by row: {args.mpp_far} m/px at y={y_top:.0f} -> "
            f"{args.mpp_near} m/px at y={y_bot:.0f}")
    return scale, desc


def main():
    args = parse_args()
    tracks_path = Path(args.tracks)
    video_path = Path(args.video)
    for p in (tracks_path, video_path):
        if not p.exists():
            raise SystemExit(f"ERROR: not found: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    regions = parse_regions(args.ignore_region)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps or float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0:
        raise SystemExit("ERROR: video reports no frame rate; pass --fps")

    Hm = None
    if args.homography:
        Hm = np.load(args.homography)
        if Hm.shape != (3, 3):
            raise SystemExit(f"homography must be 3x3, got {Hm.shape}")
        scale_desc = (f"HOMOGRAPHY ground-plane calibration from "
                      f"{Path(args.homography).name} (bottom-centre reference)")
        scale_at = None
    else:
        scale_at, scale_desc = make_scale_fn(args, height)

    rows = []
    with open(tracks_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            if regions and in_regions(box, regions):
                continue
            rows.append({
                "frame_id": int(r["frame_id"]),
                "vehicle_id": int(r["vehicle_id"]),
                "class": r["class"],
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "confidence": float(r["confidence"]),
            })

    print(f"Video       : {video_path.name}  {width}x{height} @ {fps:.2f} fps")
    print(f"Tracks      : {tracks_path.name}  ({len(rows)} rows kept)")
    print(f"Scale       : {scale_desc}")
    print(f"Speed limit : {args.speed_limit:.0f} km/h "
          f"(sustained {args.min_violation_frames}+ frames)")
    if regions:
        print(f"Ignoring    : {len(regions)} region(s) {regions}")

    # ---- speed per track -------------------------------------------------
    by_track = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            by_track[r["vehicle_id"]].append(r)

    speeds = {}          # (frame_id, vehicle_id) -> km/h
    k = max(1, args.smooth_window)
    for tid, recs in by_track.items():
        recs.sort(key=lambda r: r["frame_id"])
        if len(recs) < args.min_track_frames:
            continue
        fid = np.array([r["frame_id"] for r in recs], dtype=float)
        # image-space reference row, used for the validity band regardless of
        # which scale model is active
        ref_y = np.array([r["y2"] for r in recs], dtype=float)
        if Hm is not None:
            # Bottom-centre is where the vehicle contacts the road, so it is the
            # only point on the box that actually lies on the calibrated ground
            # plane. The box centre floats above it and would project long.
            ref = np.float32([[(r["x1"] + r["x2"]) / 2, r["y2"]] for r in recs])
            wpts = cv2.perspectiveTransform(ref.reshape(-1, 1, 2),
                                            Hm).reshape(-1, 2)
            cx, cy = wpts[:, 0], wpts[:, 1]      # already metres
        else:
            cx = np.array([(r["x1"] + r["x2"]) / 2 for r in recs])
            cy = np.array([(r["y1"] + r["y2"]) / 2 for r in recs])

        for i in range(len(recs)):
            # Require a FULL look-back window. With a partial one (the first few
            # frames of a track) dt is a frame or two, so ordinary box jitter
            # divides into a huge apparent speed -- which showed up as spurious
            # "violations" at t=0.03s on every long track. Better to report no
            # speed for the first k frames than a fabricated one.
            if i < k:
                continue
            j = i - k
            # Both ends of the window must lie inside the calibrated band,
            # otherwise part of the displacement is measured where the
            # homography is extrapolating.
            if args.min_ref_y is not None and (ref_y[i] < args.min_ref_y
                                               or ref_y[j] < args.min_ref_y):
                continue
            dt = (fid[i] - fid[j]) / fps
            if dt <= 0:
                continue
            if Hm is not None:
                # cx/cy are already ground-plane metres
                dist_m = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
                if dist_m > args.max_world_jump_m:
                    continue          # near-horizon projection blow-up
            else:
                # Convert at the midpoint row, so the scale matches where the
                # vehicle actually was over this interval.
                mpp = scale_at((cy[i] + cy[j]) / 2)
                dist_m = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j])) * mpp
            kmh = dist_m / dt * 3.6
            if kmh <= args.max_plausible_kmh:
                speeds[(recs[i]["frame_id"], tid)] = kmh

    # ---- violations ------------------------------------------------------
    over = defaultdict(int)
    for (f_id, tid), v in speeds.items():
        if v > args.speed_limit:
            over[tid] += 1
    violators = {tid for tid, n in over.items() if n >= args.min_violation_frames}

    for r in rows:
        v = speeds.get((r["frame_id"], r["vehicle_id"]))
        r["speed_kmh"] = round(v, 1) if v is not None else ""
        r["violation"] = int(r["vehicle_id"] in violators
                             and v is not None and v > args.speed_limit)

    out_csv = out_dir / f"{stem}_tracks_speed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows([{c: r[c] for c in CSV_COLUMNS} for r in rows])
    print(f"\nWrote {out_csv}  ({len(rows)} rows)")

    # one row per flagged vehicle
    summary = []
    for tid in sorted(violators):
        vs = [v for (f_id, t), v in speeds.items() if t == tid]
        frames = sorted(f_id for (f_id, t) in speeds if t == tid)
        # The moment that matters is when it FIRST EXCEEDED the limit, not when
        # the track first had any speed at all.
        vframes = sorted(f_id for (f_id, t), v in speeds.items()
                         if t == tid and v > args.speed_limit)
        cls = max(set(r["class"] for r in by_track[tid]),
                  key=[r["class"] for r in by_track[tid]].count)
        summary.append({
            "vehicle_id": tid, "class": cls,
            "max_speed_kmh": round(max(vs), 1),
            "mean_speed_kmh": round(float(np.mean(vs)), 1),
            "frames_over_limit": over[tid],
            "first_violation_frame": vframes[0],
            "first_violation_sec": round(vframes[0] / fps, 2),
            "track_first_frame": frames[0], "track_last_frame": frames[-1],
        })
    viol_csv = out_dir / f"{stem}_violations.csv"
    with open(viol_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else
                           ["vehicle_id", "class", "max_speed_kmh",
                            "mean_speed_kmh", "frames_over_limit",
                            "first_violation_frame", "first_violation_sec",
                            "track_first_frame", "track_last_frame"])
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {viol_csv}  ({len(summary)} flagged)")

    meta = {
        "source_video": str(video_path), "tracks_csv": str(tracks_path),
        "width": width, "height": height, "fps": fps,
        "scale_model": scale_desc,
        "calibration_method": "homography" if Hm is not None else "row_interpolated_mpp",
        "homography_file": args.homography,
        "homography_matrix": Hm.tolist() if Hm is not None else None,
        "CALIBRATION_PROVENANCE": (
            "STANDARD IRC ROAD MEASUREMENTS - NOT SITE-VERIFIED. The exact "
            "location of this camera is unknown, so no distance in this scene "
            "was physically measured or checked against a map. The ground-plane "
            "scale is anchored on IRC lane-marking pitch (3.0 m dash + 4.5 m "
            "gap = 7.5 m) and IRC urban lane width 3.5 m, applied to lane "
            "markings detected in frame 154. Every speed inherits any error in "
            "those assumptions: if the real pitch or lane width differs, all "
            "speeds scale proportionally. Site verification (a measured "
            "distance, or the road identified on a map) is required before "
            "these numbers can be called measurements."
        ) if Hm is not None else None,
        "irc_assumptions": {
            "standard": "IRC 35 (Code of Practice for Road Markings) / IRC 86 "
                        "(urban lane width)",
            "lane_width_m": 3.5,
            "dash_pitch_m": 7.5,
            "dash_pitch_note": "IRC 3.0 m dash + 4.5 m gap. Pitch is the scale "
                               "anchor and validated to 7.52 m mean (sd 0.22) "
                               "across 5 measured intervals.",
            "solved_dash_length_m": 3.85,
            "solved_dash_note": "Dash:gap split was SOLVED from the image, not "
                                "assumed: forcing the IRC 3.0 m dash gave "
                                "systematically inconsistent geometry. Measured "
                                "dash ~3.85 m with ~3.65 m gap. Pitch (the "
                                "scale-setting quantity) still matches IRC.",
            "calibration_frame": 154,
            "reprojection_residual_m": {"mean": 0.163, "max": 0.490},
        } if Hm is not None else None,
        "mpp_near": args.mpp_near, "mpp_far": args.mpp_far,
        "constant_mpp": args.constant_mpp,
        "smooth_window": args.smooth_window,
        "min_ref_y": args.min_ref_y,
        "min_ref_y_note": (
            f"Speed reported only where the box bottom-centre is at row "
            f">= {args.min_ref_y}, i.e. inside the region the homography was "
            f"fitted over (lane dashes spanned y~268-503 in frame 154). "
            f"Vehicles further away are still detected and tracked, but carry "
            f"no speed rather than an extrapolated one."
            if args.min_ref_y is not None else
            "No validity band applied: speeds include samples where the scale "
            "model is extrapolating beyond its fitted region."),
        "speed_limit_kmh": args.speed_limit,
        "min_violation_frames": args.min_violation_frames,
        "max_plausible_kmh": args.max_plausible_kmh,
        "ignored_regions": regions,
        "vehicles_with_speed": len({t for (_, t) in speeds}),
        "violating_vehicles": len(violators),
        "WARNING": (
            "Speeds derived from IRC-standard road geometry, NOT from a "
            "site-verified measurement. Absolute values are only as correct as "
            "those standards are for this specific road. Not valid for "
            "enforcement."
            if Hm is not None else
            "Speeds are indicative only - assumed scale, no camera "
            "calibration. Not valid for enforcement."),
    }
    (out_dir / f"{stem}_speed_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / f'{stem}_speed_meta.json'}")

    report(speeds, by_track, violators, args, summary, Hm is not None)

    if args.save_video:
        render(cap, rows, out_dir / f"{stem}_speed.mp4", fps, args, violators,
               Hm is not None)
    cap.release()


def report(speeds, by_track, violators, args, summary, calibrated=False):
    print("\n" + "=" * 64)
    print("SPEED / VIOLATION SUMMARY")
    print("=" * 64)
    if not speeds:
        print("No speeds computed - no tracks long enough.")
        return
    vals = np.array(list(speeds.values()))
    print(f"Tracked vehicles        : {len(by_track)}")
    print(f"  with a speed estimate : {len({t for (_, t) in speeds})}")
    if args.min_ref_y is not None:
        print(f"  (restricted to the calibrated band: reference row "
              f">= {args.min_ref_y:.0f};")
        print(f"   more distant vehicles are tracked but deliberately "
              f"carry no speed)")
    print(f"Speed (km/h)            : mean {vals.mean():.1f}, "
          f"median {np.median(vals):.1f}, p90 {np.percentile(vals, 90):.1f}, "
          f"max {vals.max():.1f}")
    print(f"Over {args.speed_limit:.0f} km/h            : "
          f"{int((vals > args.speed_limit).sum())} of {len(vals)} "
          f"frame-samples ({100 * (vals > args.speed_limit).mean():.1f}%)")
    print(f"VEHICLES FLAGGED        : {len(violators)}")
    for s in summary[:10]:
        print(f"   #{s['vehicle_id']:<5} {s['class']:<9} "
              f"max {s['max_speed_kmh']:>5.1f} km/h  "
              f"mean {s['mean_speed_kmh']:>5.1f}  "
              f"{s['frames_over_limit']:>2} frames over  "
              f"first exceeded at t={s['first_violation_sec']}s "
              f"(frame {s['first_violation_frame']})")
    if calibrated:
        print("\nGround-plane calibrated (homography). Scale comes from STANDARD")
        print("IRC road geometry, NOT a site-verified measurement - the road was")
        print("never identified or measured. Speeds scale linearly with the")
        print("assumed 7.5 m lane-marking pitch and 3.5 m lane width, so a road")
        print("built to different dimensions shifts every number here.")
        print("Not valid for enforcement.")
    else:
        print("\nSpeeds are INDICATIVE ONLY: assumed metres-per-pixel, no camera")
        print("calibration. Use for relative comparison and demo, not enforcement.")
    print("=" * 64)


def render(cap, rows, out_path, fps, args, violators, calibrated=False):
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame_id"]].append(r)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = None
    s = args.annot_scale
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if s != 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s,
                               interpolation=cv2.INTER_CUBIC)

        for r in by_frame.get(idx, []):
            viol = bool(r["violation"])
            color = VIOLATION_COLOR if viol else CLASS_COLORS.get(
                r["class"], (255, 255, 255))
            x1, y1 = int(r["x1"] * s), int(r["y1"] * s)
            x2, y2 = int(r["x2"] * s), int(r["y2"] * s)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if viol else 2)

            vid = r["vehicle_id"]
            spd = r["speed_kmh"]
            label = f"{'?' if vid == -1 else f'#{vid}'} {r['class']}"
            if spd != "":
                label += f" {spd:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 5, ty + 2),
                          color, -1)
            cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if viol:
                vt = "VIOLATION"
                (vw, vh), _ = cv2.getTextSize(vt, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6, 2)
                by = y2 + vh + 6 if y2 + vh + 10 < frame.shape[0] else y1 - 6
                cv2.rectangle(frame, (x1, by - vh - 4), (x1 + vw + 6, by + 3),
                              VIOLATION_COLOR, -1)
                cv2.putText(frame, vt, (x1 + 3, by), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

        n_v = sum(1 for r in by_frame.get(idx, []) if r["violation"])
        note = ("[ground-plane calibrated - IRC standard geometry, "
                "NOT site-verified]" if calibrated else
                "[speeds indicative - assumed scale, uncalibrated]")
        banner = (f"limit {args.speed_limit:.0f} km/h   "
                  f"frame {idx}   flagged now: {n_v}   {note}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
