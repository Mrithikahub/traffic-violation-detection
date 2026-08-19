"""
Tailgating detection, layered on the tracking output.

Reads the per-frame tracks CSV from detect_track.py, pairs each vehicle with the
vehicle directly ahead of it in its own direction of travel, and flags pairs
that hold an unsafe following gap for a sustained period.

Standalone and decoupled from detection, like speed_violation.py, wrong_lane.py
and lane_change.py:

    python detect_track.py --video <in.mp4> --model <weights> --out-dir out
    python tailgating.py --video <in.mp4> --tracks out/<stem>_tracks.csv \
        --out-dir out --save-video

WHY THE GAP IS MEASURED IN FOLLOWER-LENGTHS
    A gap of "40 pixels" means nothing without knowing how far away the vehicle
    is. Dividing by the follower's own box length makes the measure scale-free:
    "half a car-length behind" is the same judgement near or far, and needs no
    camera calibration. That is what lets this layer run on any footage,
    including videos with no homography.

    If a homography and speed CSV are supplied, TIME HEADWAY (seconds) is also
    reported -- the measure traffic engineering actually uses -- but the flag
    itself stays on the scale-free metric so behaviour is identical everywhere.

=============================== ASSUMPTIONS ===============================
1. DIRECTION IS PER-VEHICLE, not assumed from the scene. Each vehicle's
   heading comes from its own recent motion, so "ahead" means ahead in ITS
   direction of travel. This is what lets the layer work on a motorway with
   two carriageways running opposite ways, without any zone definition.

2. BOX LENGTH PROXIES VEHICLE LENGTH. In this camera geometry vehicles are
   seen mostly end-on, so box height approximates length. For a vehicle
   crossing the view sideways this proxy is wrong, and its gaps should not be
   trusted.

3. QUEUES ARE NOT TAILGATING. Stationary or crawling vehicles sit close
   together legitimately. Both vehicles must exceed --min-speed-lengths before
   a pair can be flagged, otherwise every traffic jam reads as mass tailgating.
   That threshold is in vehicle-lengths per frame, not pixels: a pixel
   threshold is depth-dependent, so a distant vehicle could never meet it
   however fast it is actually travelling.

4. NO GROUND TRUTH. Nothing here is validated against observed driver
   behaviour; the threshold is an engineering choice, not a legal standard.
===========================================================================
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from detect_track import CLASS_COLORS

CSV_COLUMNS = [
    "frame_id", "vehicle_id", "class", "x1", "y1", "x2", "y2", "confidence",
    "leader_id", "gap_lengths", "gap_px", "headway_sec", "tailgating",
]

COLOR_TAILGATE = (255, 255, 0)      # cyan - distinct from speed/wrong-side/lane


def parse_args():
    p = argparse.ArgumentParser(
        description="Tailgating detection over tracked output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--out-dir", default="outputs")

    p.add_argument("--gap-lengths", type=float, default=1.5,
                   help="Flag when centre-to-centre separation falls below this "
                        "many follower-lengths. 1.0 is bumper contact, so 1.5 "
                        "means about half a vehicle-length of clear space. "
                        "Engineering choice, not a legal standard.")
    p.add_argument("--min-frames", type=int, default=15,
                   help="Frames the SAME pair must hold the gap. Rejects "
                        "momentary closes from lane changes and box jitter.")
    p.add_argument("--lateral-tolerance", type=float, default=0.6,
                   help="How far off-axis the leader may sit, as a fraction of "
                        "the follower's width, to count as the same lane.")
    p.add_argument("--min-speed-lengths", type=float, default=0.010,
                   help="Both vehicles must move at least this many of their "
                        "OWN lengths per frame. Scale-free, unlike a pixel "
                        "threshold, which a distant vehicle can never meet "
                        "however fast it is really going. Stops queues and "
                        "parked vehicles reading as tailgating. Default set "
                        "from the observed distribution on this footage: "
                        "stationary vehicles sit near 0.0004, the median "
                        "moving vehicle at 0.019, so 0.010 is ~25x the "
                        "stationary noise floor while retaining slow traffic. "
                        "At 30fps this is roughly 4 km/h for a 4 m car.")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Frames over which heading and speed are measured.")
    p.add_argument("--max-pair-iou", type=float, default=0.45,
                   help="Reject a pair whose boxes overlap more than this. Two "
                        "heavily-overlapping boxes of similar size are usually "
                        "ONE vehicle tracked twice, which otherwise reads as a "
                        "vehicle tailgating itself at a near-zero gap.")
    p.add_argument("--min-box-px", type=float, default=25.0,
                   help="Ignore vehicles smaller than this; far-field boxes are "
                        "too coarse for a reliable gap.")
    p.add_argument("--min-track-frames", type=int, default=10)

    p.add_argument("--homography", default=None,
                   help="Optional 3x3 .npy. Enables gap in metres.")
    p.add_argument("--speed-csv", default=None,
                   help="Optional _tracks_speed.csv. With --homography, enables "
                        "time headway in seconds.")
    p.add_argument("--ignore-region", action="append", default=None,
                   metavar="x1,y1,x2,y2")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--annot-scale", type=float, default=1.0)
    return p.parse_args()


def parse_regions(specs):
    out = []
    for s in specs or []:
        v = [float(x) for x in s.replace(" ", "").split(",")]
        if len(v) != 4:
            raise SystemExit(f"--ignore-region needs x1,y1,x2,y2 (got '{s}')")
        out.append((min(v[0], v[2]), min(v[1], v[3]),
                    max(v[0], v[2]), max(v[1], v[3])))
    return out


def main():
    args = parse_args()
    for p in (args.video, args.tracks):
        if not Path(p).exists():
            raise SystemExit(f"ERROR: not found: {p}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    regions = parse_regions(args.ignore_region)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    Hm = np.load(args.homography) if args.homography else None
    speeds = {}
    if args.speed_csv and Path(args.speed_csv).exists():
        for r in csv.DictReader(open(args.speed_csv, encoding="utf-8")):
            if r.get("speed_kmh"):
                speeds[(int(r["frame_id"]), int(r["vehicle_id"]))] = \
                    float(r["speed_kmh"])

    rows = []
    with open(args.tracks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(a <= cx <= c and d <= cy <= e for a, d, c, e in regions):
                continue
            rows.append({"frame_id": int(r["frame_id"]),
                         "vehicle_id": int(r["vehicle_id"]),
                         "class": r["class"], "x1": b[0], "y1": b[1],
                         "x2": b[2], "y2": b[3],
                         "confidence": float(r["confidence"])})

    print(f"Video   : {Path(args.video).name}  {width}x{height} @ {fps:.2f} fps")
    print(f"Tracks  : {Path(args.tracks).name}  ({len(rows)} rows kept)")
    print(f"Rule    : gap < {args.gap_lengths} follower-lengths (1.0 = contact), "
          f"same lane (±{args.lateral_tolerance}w),")
    print(f"          both moving >{args.min_speed_lengths} lengths/frame,")
    print(f"          sustained {args.min_frames}+ frames for the SAME pair")

    # ---- per-vehicle heading and speed ----------------------------------
    by_track = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            by_track[r["vehicle_id"]].append(r)
    for v in by_track.values():
        v.sort(key=lambda r: r["frame_id"])

    k = max(1, args.smooth_window)
    motion = {}          # (frame, vid) -> (ux, uy, lengths_per_frame)
    for tid, recs in by_track.items():
        if len(recs) < args.min_track_frames:
            continue
        # Ground-contact point throughout: the same reference used for the gap,
        # so heading and separation are measured on one consistent geometry.
        gx = np.array([(r["x1"] + r["x2"]) / 2 for r in recs])
        gy = np.array([r["y2"] for r in recs])
        hh = np.array([max(r["y2"] - r["y1"], 1.0) for r in recs])
        for i in range(len(recs)):
            j = max(0, i - k)
            if i == j:
                continue
            dx, dy = gx[i] - gx[j], gy[i] - gy[j]
            n = float(np.hypot(dx, dy))
            df = recs[i]["frame_id"] - recs[j]["frame_id"]
            if n < 1e-6 or df <= 0:
                continue
            # speed normalised by the vehicle's own size => scale-free
            motion[(recs[i]["frame_id"], tid)] = (
                dx / n, dy / n, (n / df) / hh[i])

    # ---- pair the follower with the vehicle directly ahead ---------------
    by_frame = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            by_frame[r["frame_id"]].append(r)

    pair_frames = Counter()
    pair_ious = defaultdict(list)
    pair_best = {}
    per_sample = {}
    stats = Counter()

    for f, items in by_frame.items():
        for a in items:
            m = motion.get((f, a["vehicle_id"]))
            if m is None:
                continue
            ux, uy, spd = m
            ah = a["y2"] - a["y1"]
            aw = a["x2"] - a["x1"]
            if ah < args.min_box_px:
                stats["skip_small_box"] += 1
                continue
            if spd < args.min_speed_lengths:
                stats["skip_follower_slow"] += 1
                continue
            acx, acy = (a["x1"] + a["x2"]) / 2, a["y2"]     # ground contact

            best = None
            for b in items:
                if b["vehicle_id"] == a["vehicle_id"]:
                    continue
                bh = b["y2"] - b["y1"]
                if bh < args.min_box_px:
                    continue
                bcx, bcy = (b["x1"] + b["x2"]) / 2, b["y2"]  # ground contact
                dx, dy = bcx - acx, bcy - acy
                along = dx * ux + dy * uy          # + => ahead of A
                lateral = abs(dx * (-uy) + dy * ux)
                if along <= 0:
                    continue
                if lateral > args.lateral_tolerance * aw:
                    continue
                if best is None or along < best[0]:
                    best = (along, b)
            if best is None:
                continue
            along, b = best
            mb = motion.get((f, b["vehicle_id"]))
            if mb is None or mb[2] < args.min_speed_lengths:
                stats["skip_leader_slow"] += 1
                continue
            # Two boxes that overlap heavily are almost always the same
            # physical vehicle picked up twice by the tracker.
            ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
            ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (aw * ah + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                     - inter)
            pair_iou = inter / union if union > 0 else 0.0
            if pair_iou > args.max_pair_iou:
                stats["skip_overlapping_pair"] += 1
                continue

            # Centre-to-centre separation in follower-lengths. Measured between
            # ground-contact points, so perspective cannot drive it negative the
            # way a bumper-to-bumper estimate in image space does. ~1.0 is
            # bumper contact; larger means more clear space.
            gap_px = along
            gap_lengths = along / ah
            rec = {"leader_id": b["vehicle_id"],
                   "gap_lengths": round(float(gap_lengths), 3),
                   "gap_px": round(float(gap_px), 1),
                   "headway_sec": ""}
            if Hm is not None:
                pa = cv2.perspectiveTransform(
                    np.float32([[[acx, a["y2"]]]]), Hm).reshape(2)
                pb = cv2.perspectiveTransform(
                    np.float32([[[bcx, b["y2"]]]]), Hm).reshape(2)
                gap_m = float(np.hypot(*(pb - pa)))
                rec["gap_m"] = round(gap_m, 2)
                kmh = speeds.get((f, a["vehicle_id"]))
                if kmh and kmh > 1:
                    rec["headway_sec"] = round(gap_m / (kmh / 3.6), 2)
            per_sample[(f, a["vehicle_id"])] = rec
            stats["judged"] += 1

            if gap_lengths <= args.gap_lengths:
                key = (a["vehicle_id"], b["vehicle_id"])
                pair_frames[key] += 1
                pair_ious[key].append(pair_iou)
                cur = pair_best.get(key)
                if cur is None or gap_lengths < cur["min_gap_lengths"]:
                    pair_best[key] = {
                        "min_gap_lengths": round(float(gap_lengths), 3),
                        "at_frame": int(f),
                        "follower_class": a["class"], "leader_class": b["class"],
                        "min_gap_px": round(float(gap_px), 1),
                    }

    # Pair-level duplicate rejection. The per-frame IoU guard above cannot
    # catch a pair that IS the same vehicle throughout: its overlap fluctuates,
    # so individual frames dip under the limit and it accumulates enough to be
    # flagged. Judge the pair on its median overlap instead of any one frame.
    dup_limit = args.max_pair_iou * 0.5
    flagged = {}
    for p, n in pair_frames.items():
        if n < args.min_frames:
            continue
        med = float(np.median(pair_ious[p])) if pair_ious[p] else 0.0
        if med > dup_limit:
            stats["skip_pair_persistent_overlap"] += 1
            continue
        flagged[p] = n

    for r in rows:
        s = per_sample.get((r["frame_id"], r["vehicle_id"]))
        r["leader_id"] = s["leader_id"] if s else ""
        r["gap_lengths"] = s["gap_lengths"] if s else ""
        r["gap_px"] = s["gap_px"] if s else ""
        r["headway_sec"] = s["headway_sec"] if s else ""
        r["tailgating"] = int(bool(
            s and (r["vehicle_id"], s["leader_id"]) in flagged
            and s["gap_lengths"] <= args.gap_lengths))

    out_csv = out_dir / f"{stem}_tailgating.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows([{c: r[c] for c in CSV_COLUMNS} for r in rows])
    print(f"\nWrote {out_csv}  ({len(rows)} rows)")

    summary = []
    for (fid, lid), n in sorted(flagged.items(), key=lambda kv: -kv[1]):
        d = pair_best[(fid, lid)]
        summary.append({"follower_id": fid, "leader_id": lid,
                        "follower_class": d["follower_class"],
                        "leader_class": d["leader_class"],
                        "frames_tailgating": n,
                        "duration_sec": round(n / fps, 2),
                        "min_gap_lengths": d["min_gap_lengths"],
                        "min_gap_px": d["min_gap_px"],
                        "at_frame": d["at_frame"],
                        "at_sec": round(d["at_frame"] / fps, 2)})
    ev_csv = out_dir / f"{stem}_tailgating_events.csv"
    with open(ev_csv, "w", newline="", encoding="utf-8") as f:
        cols = (list(summary[0]) if summary else
                ["follower_id", "leader_id", "follower_class", "leader_class",
                 "frames_tailgating", "duration_sec", "min_gap_lengths",
                 "min_gap_px", "at_frame", "at_sec"])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {ev_csv}  ({len(summary)} flagged pairs)")

    gaps = [s["gap_lengths"] for s in per_sample.values()]
    meta = {
        "source_video": str(args.video), "tracks_csv": str(args.tracks),
        "width": width, "height": height, "fps": fps,
        "metric": "centre-to-centre separation in follower-lengths (scale-free; ~1.0 = bumper contact)",
        "gap_definition": "separation between the two vehicles' ground-contact "
                          "points, projected on the follower's direction of "
                          "travel, divided by the follower's box height",
        "reference_point": "box bottom-centre for ground position; box height "
                           "as a proxy for vehicle length",
        "gap_lengths_threshold": args.gap_lengths,
        "min_frames_same_pair": args.min_frames,
        "lateral_tolerance_widths": args.lateral_tolerance,
        "min_speed_lengths_per_frame": args.min_speed_lengths,
        "max_pair_iou": args.max_pair_iou,
        "smooth_window": args.smooth_window,
        "min_box_px": args.min_box_px,
        "homography_used": bool(Hm is not None),
        "speed_csv_used": bool(speeds),
        "ignored_regions": regions,
        "samples_judged": stats["judged"],
        "skipped": {k: v for k, v in stats.items() if k != "judged"},
        "gap_distribution_lengths": {
            "p10": round(float(np.percentile(gaps, 10)), 2),
            "median": round(float(np.median(gaps)), 2),
            "p90": round(float(np.percentile(gaps, 90)), 2),
        } if gaps else {},
        "pairs_ever_close": len(pair_frames),
        "pairs_flagged": len(flagged),
        "ASSUMPTIONS": (
            "Gap is measured in the image using box height as a proxy for "
            "vehicle length, and each vehicle's own motion to decide which way "
            "is 'ahead'. Scale-free by construction, so no camera calibration "
            "is required and the same threshold applies at any distance. A "
            "vehicle crossing the view sideways breaks the length proxy and its "
            "gaps should not be trusted."
        ),
        "LIMITATIONS": (
            "The threshold is an ENGINEERING CHOICE, not a legal standard, and "
            "is not validated against observed driver behaviour. Queues are "
            "excluded by requiring both vehicles to be moving, so slow-moving "
            "congestion is deliberately not reported. Vehicles too small, too "
            "slow, or too briefly tracked receive no verdict and are NOT "
            "counted as safe."
        ),
    }
    (out_dir / f"{stem}_tailgating_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / f'{stem}_tailgating_meta.json'}")

    report(stats, gaps, pair_frames, flagged, summary, args, fps)

    if args.save_video:
        render(cap, rows, out_dir / f"{stem}_tailgating.mp4", fps, args)
    cap.release()


def report(stats, gaps, pair_frames, flagged, summary, args, fps):
    W = 70
    print("\n" + "=" * W)
    print("TAILGATING SUMMARY")
    print("=" * W)
    print(f"Follower-samples judged : {stats['judged']:,}")
    for k, v in stats.most_common():
        if k != "judged":
            print(f"   skipped: {k:<22}{v:>8,}")
    if gaps:
        g = np.array(gaps)
        print(f"\nGap (follower-lengths)  : p10 {np.percentile(g,10):.2f}, "
              f"median {np.median(g):.2f}, p90 {np.percentile(g,90):.2f}")
        print(f"Samples under {args.gap_lengths}       : "
              f"{int((g <= args.gap_lengths).sum()):,} "
              f"({100*(g <= args.gap_lengths).mean():.1f}%)")
    print(f"\nPairs ever below limit  : {len(pair_frames)}")
    print(f"PAIRS FLAGGED (sustained {args.min_frames}+ fr): {len(flagged)}")
    for s in summary[:12]:
        print(f"   #{s['follower_id']:<5} {s['follower_class']:<8} behind "
              f"#{s['leader_id']:<5} {s['leader_class']:<8} "
              f"{s['frames_tailgating']:>3}fr ({s['duration_sec']:>4.1f}s)  "
              f"min gap {s['min_gap_lengths']:.2f} lengths  t={s['at_sec']}s")
    if not summary:
        print("   (none sustained the gap long enough)")
    print("\nThreshold is an engineering choice, not a legal standard.")
    print("Queues excluded: both vehicles must be moving.")
    print("=" * W)


def render(cap, rows, out_path, fps, args):
    by_frame = defaultdict(list)
    pos = {}
    for r in rows:
        by_frame[r["frame_id"]].append(r)
        pos[(r["frame_id"], r["vehicle_id"])] = r

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
        n = 0
        for r in by_frame.get(idx, []):
            tg = bool(r["tailgating"])
            n += tg
            color = COLOR_TAILGATE if tg else CLASS_COLORS.get(
                r["class"], (255, 255, 255))
            x1, y1 = int(r["x1"] * s), int(r["y1"] * s)
            x2, y2 = int(r["x2"] * s), int(r["y2"] * s)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if tg else 2)

            lab = f"#{r['vehicle_id']} {r['class']}"
            if r["gap_lengths"] != "":
                lab += f" gap {float(r['gap_lengths']):.2f}L"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 5, ty + 2),
                          color, -1)
            cv2.putText(frame, lab, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if tg:
                lead = pos.get((idx, r["leader_id"]))
                if lead:
                    a = (int((r["x1"] + r["x2"]) / 2 * s), int(r["y1"] * s))
                    b = (int((lead["x1"] + lead["x2"]) / 2 * s),
                         int(lead["y2"] * s))
                    cv2.arrowedLine(frame, a, b, COLOR_TAILGATE, 2,
                                    tipLength=0.25)
                t = "TAILGATING"
                (bw, bh), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.55, 2)
                yy = (y2 + bh + 6 if y2 + bh + 10 < frame.shape[0] else y1 - 6)
                cv2.rectangle(frame, (x1, yy - bh - 4), (x1 + bw + 6, yy + 3),
                              COLOR_TAILGATE, -1)
                cv2.putText(frame, t, (x1 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 2, cv2.LINE_AA)

        banner = (f"frame {idx}   TAILGATING (cyan): {n}   "
                  f"gap < {args.gap_lengths} follower-lengths for "
                  f"{args.min_frames}+ frames   "
                  f"[scale-free metric; threshold is an engineering choice]")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (w, h))
        writer.write(frame)
    if writer is not None:
        writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
