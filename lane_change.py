"""
Lane-assignment and lane-change event detection, layered on the tracking output.

Reads the per-frame tracks CSV from detect_track.py, projects each vehicle's
ground-contact point onto the calibrated road plane, assigns it to a lane, and
logs transitions between lanes.

Lane changes are logged as EVENTS, not violations -- changing lane is normal
driving. Frequency is reported so erratic weaving can be spotted, but nothing
here asserts illegality.

Standalone and decoupled from detection, like speed_violation.py and
wrong_lane.py:

    python lane_change.py --video <in.mp4> --tracks out/<stem>_tracks.csv \
        --lanes lanes.json --homography out/homography.npy --out-dir out \
        --save-video --speed-csv ... --wrong-lane-csv ...

LANES FILE
    {"boundaries_world_x": [-7.91, -0.04, 3.24],
     "lane_names": ["lane_outer", "lane_inner"],
     "along_road_span_m": [-8.0, 34.0]}

    Boundaries are lateral offsets on the calibrated ground plane, ascending.
    N boundaries define N-1 lanes.

=============================== ASSUMPTIONS ===============================
Lane boundaries come from lane-marking dashes detected in one frame and
projected to the ground plane via the homography. They inherit every
assumption in that calibration (IRC standard geometry, not site-verified),
and they are valid only for this camera framing.

Boundaries are only as trustworthy as the homography is at their lateral
offset. The calibration was fitted over a narrow band, so boundaries far
outside it are extrapolated -- which is why only well-supported lines should
be used. A zone spanning more than ~4 m covers more than one physical lane,
and changes *within* such a zone are invisible to this layer.

Jitter rejection: a vehicle must remain in a lane for --min-frames-in-lane
consecutive frames for that lane to count as "settled". Only transitions
between settled runs are logged, so a box wobbling across a boundary produces
no event.
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
    "world_x", "world_y", "lane", "lane_settled", "lane_change_event",
]

COLOR_CHANGE = (0, 255, 255)    # yellow  - lane change event
COLOR_WRONG = (255, 0, 255)     # magenta - wrong side  (wrong_lane.py)
COLOR_SPEED = (0, 0, 255)       # red     - speeding    (speed_violation.py)
LANE_COLORS = [(90, 200, 255), (120, 255, 160), (255, 190, 90),
               (255, 130, 220), (170, 170, 255)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Lane assignment and lane-change events over tracked output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--lanes", required=True, help="Lane definition JSON.")
    p.add_argument("--homography", required=True,
                   help="3x3 .npy image->ground-plane, from calibrate_homography.py")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--speed-csv", default=None,
                   help="Optional _tracks_speed.csv to overlay speed violations.")
    p.add_argument("--wrong-lane-csv", default=None,
                   help="Optional _wrong_lane.csv to overlay wrong-side flags.")
    p.add_argument("--min-frames-in-lane", type=int, default=6,
                   help="Consecutive frames in a lane before it counts as "
                        "settled. Raise to reject more boundary jitter, at the "
                        "cost of missing quick changes.")
    p.add_argument("--min-track-frames", type=int, default=12,
                   help="Tracks shorter than this are not assessed.")
    p.add_argument("--erratic-changes", type=int, default=3,
                   help="Vehicles with at least this many changes are reported "
                        "as frequent changers. Reported only, NOT a violation.")
    p.add_argument("--event-display-frames", type=int, default=20,
                   help="How long the LANE CHANGE label stays on screen.")
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


def load_lanes(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    b = [float(x) for x in spec["boundaries_world_x"]]
    if len(b) < 2:
        raise SystemExit("need >=2 boundaries to define a lane")
    if b != sorted(b):
        raise SystemExit("boundaries_world_x must be ascending")
    names = spec.get("lane_names") or [f"lane_{i+1}" for i in range(len(b) - 1)]
    if len(names) != len(b) - 1:
        raise SystemExit(f"{len(b)} boundaries define {len(b)-1} lanes, "
                         f"but {len(names)} names given")
    span = spec.get("along_road_span_m", [-1e9, 1e9])
    return b, names, [float(span[0]), float(span[1])]


def settled_runs(seq, min_len):
    """Run-length encode, keeping only runs long enough to count as settled.

    Short runs are the signature of a box jittering across a boundary, so they
    are dropped entirely rather than treated as a lane occupancy.
    """
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j][1] == seq[i][1]:
            j += 1
        if seq[i][1] is not None and (j - i) >= min_len:
            runs.append({"lane": seq[i][1], "start": seq[i][0],
                         "end": seq[j - 1][0], "n": j - i})
        i = j
    return runs


def main():
    args = parse_args()
    for p in (args.video, args.tracks, args.lanes, args.homography):
        if not Path(p).exists():
            raise SystemExit(f"ERROR: not found: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    bounds, lane_names, span = load_lanes(args.lanes)
    Hm = np.load(args.homography)
    if Hm.shape != (3, 3):
        raise SystemExit(f"homography must be 3x3, got {Hm.shape}")
    regions = parse_regions(args.ignore_region)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    widths = [round(b - a, 2) for a, b in zip(bounds, bounds[1:])]
    print(f"Video   : {Path(args.video).name}  {width}x{height} @ {fps:.2f} fps")
    print(f"Lanes   : " + ", ".join(
        f"{n} [{a:.2f},{b:.2f}) w={w}m" + ("  <-- >1 physical lane" if w > 4.5 else "")
        for n, a, b, w in zip(lane_names, bounds, bounds[1:], widths)))
    print(f"Span    : along-road {span[0]} to {span[1]} m")
    print(f"Rules   : settled after {args.min_frames_in_lane} frames in a lane")

    rows = []
    with open(args.tracks, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if any(a <= cx <= c and b <= cy <= d for a, b, c, d in regions):
                continue
            rows.append({"frame_id": int(r["frame_id"]),
                         "vehicle_id": int(r["vehicle_id"]),
                         "class": r["class"],
                         "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                         "confidence": float(r["confidence"])})

    # ---- world position + lane per sample --------------------------------
    pts = np.float32([[(r["x1"] + r["x2"]) / 2, r["y2"]] for r in rows])
    world = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hm).reshape(-1, 2)

    skipped = Counter()
    lane_samples = Counter()
    for r, (wx, wy) in zip(rows, world):
        r["world_x"], r["world_y"] = round(float(wx), 2), round(float(wy), 2)
        r["lane"] = ""
        if not (span[0] <= wy <= span[1]):
            skipped["outside_along_road_span"] += 1
            continue
        for name, a, b in zip(lane_names, bounds, bounds[1:]):
            if a <= wx < b:
                r["lane"] = name
                lane_samples[name] += 1
                break
        else:
            skipped["outside_lane_boundaries"] += 1

    # ---- settled runs and transitions ------------------------------------
    by_track = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            by_track[r["vehicle_id"]].append(r)

    events = []
    settled_flag = {}
    changes_per_vehicle = Counter()
    for tid, recs in by_track.items():
        recs.sort(key=lambda r: r["frame_id"])
        if len(recs) < args.min_track_frames:
            skipped["track_too_short"] += len(recs)
            continue
        seq = [(r["frame_id"], r["lane"] or None) for r in recs]
        runs = settled_runs(seq, args.min_frames_in_lane)
        for run in runs:
            for f in range(run["start"], run["end"] + 1):
                settled_flag[(f, tid)] = run["lane"]
        for a, b in zip(runs, runs[1:]):
            if a["lane"] == b["lane"]:
                continue
            cls = Counter(r["class"] for r in recs).most_common(1)[0][0]
            events.append({
                "vehicle_id": tid, "class": cls,
                "from_lane": a["lane"], "to_lane": b["lane"],
                "change_frame": b["start"],
                "change_sec": round(b["start"] / fps, 2),
                "frames_in_from": a["n"], "frames_in_to": b["n"],
            })
            changes_per_vehicle[tid] += 1

    ev_by_key = defaultdict(list)
    for e in events:
        ev_by_key[(e["change_frame"], e["vehicle_id"])].append(e)

    for r in rows:
        r["lane_settled"] = settled_flag.get((r["frame_id"], r["vehicle_id"]), "")
        r["lane_change_event"] = 1 if ev_by_key.get(
            (r["frame_id"], r["vehicle_id"])) else 0

    out_csv = out_dir / f"{stem}_lane_change.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows([{c: r[c] for c in CSV_COLUMNS} for r in rows])
    print(f"\nWrote {out_csv}  ({len(rows)} rows)")

    ev_csv = out_dir / f"{stem}_lane_change_events.csv"
    cols = ["vehicle_id", "class", "from_lane", "to_lane", "change_frame",
            "change_sec", "frames_in_from", "frames_in_to"]
    with open(ev_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(events, key=lambda e: e["change_frame"]))
    print(f"Wrote {ev_csv}  ({len(events)} events)")

    erratic = {t: n for t, n in changes_per_vehicle.items()
               if n >= args.erratic_changes}
    meta = {
        "source_video": str(args.video), "tracks_csv": str(args.tracks),
        "homography_file": args.homography,
        "width": width, "height": height, "fps": fps,
        "reference_point": "bottom-centre of box (ground contact)",
        "boundaries_world_x": bounds,
        "lane_names": lane_names,
        "lane_widths_m": widths,
        "along_road_span_m": span,
        "min_frames_in_lane": args.min_frames_in_lane,
        "min_track_frames": args.min_track_frames,
        "erratic_threshold_changes": args.erratic_changes,
        "samples_per_lane": dict(lane_samples),
        "skipped_samples": dict(skipped),
        "vehicles_assessed": len([t for t, r in by_track.items()
                                  if len(r) >= args.min_track_frames]),
        "lane_change_events": len(events),
        "vehicles_changing_lane": len(changes_per_vehicle),
        "frequent_changers": {str(k): v for k, v in sorted(erratic.items())},
        "EVENTS_NOT_VIOLATIONS": (
            "Lane changes are logged as EVENTS. Changing lane is normal, legal "
            "driving; nothing in this layer asserts a violation. The "
            "'frequent_changers' list is descriptive only."
        ),
        "ASSUMPTIONS": (
            "Lane boundaries are lateral offsets on the ground plane derived "
            "from lane-marking dashes in one frame, projected via the "
            "homography. They inherit that calibration's assumptions (IRC "
            "standard geometry, NOT site-verified) and are valid only for this "
            "camera framing. Boundaries far from the laterally calibrated band "
            "are extrapolated and less reliable."
            + (f" NOTE: lane(s) {[n for n, w in zip(lane_names, widths) if w > 4.5]} "
               f"span more than ~4.5 m, i.e. more than one physical lane; "
               f"changes within such a zone are NOT detectable."
               if any(w > 4.5 for w in widths) else "")
        ),
        "LIMITATIONS": (
            "A vehicle must settle in a lane for several consecutive frames "
            "before it counts, so quick weaves may be missed and boundary "
            "jitter is deliberately discarded. Vehicles outside the lane "
            "boundaries or the along-road span get no lane assignment and are "
            "not assessed - they are not counted as staying in lane."
        ),
    }
    (out_dir / f"{stem}_lane_change_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / f'{stem}_lane_change_meta.json'}")

    report(lane_samples, skipped, events, changes_per_vehicle, erratic,
           lane_names, bounds, widths, args)

    if args.save_video:
        render(cap, rows, ev_by_key, bounds, lane_names, Hm, span,
               out_dir / f"{stem}_lane_change.mp4", fps, args)
    cap.release()


def report(lane_samples, skipped, events, changes, erratic, names, bounds,
           widths, args):
    W = 68
    print("\n" + "=" * W)
    print("LANE ASSIGNMENT / LANE-CHANGE EVENTS")
    print("=" * W)
    print("Samples per lane:")
    for n, a, b, w in zip(names, bounds, bounds[1:], widths):
        note = "   <-- spans >1 physical lane" if w > 4.5 else ""
        print(f"   {n:<16}[{a:>6.2f},{b:>6.2f}) {w:>5.2f}m "
              f"{lane_samples.get(n, 0):>6}{note}")
    if skipped:
        print("\nSamples not assessed (NOT counted as staying in lane):")
        for k, v in skipped.most_common():
            print(f"   {k:<28}{v:>7}")
    print(f"\nVehicles that changed lane : {len(changes)}")
    print(f"LANE CHANGE EVENTS         : {len(events)}")
    for e in sorted(events, key=lambda e: e["change_frame"])[:15]:
        print(f"   #{e['vehicle_id']:<5} {e['class']:<9} "
              f"{e['from_lane']:>12} -> {e['to_lane']:<12} "
              f"t={e['change_sec']:>5}s  "
              f"({e['frames_in_from']}f -> {e['frames_in_to']}f)")
    if not events:
        print("   (none - no vehicle settled in two different lanes)")
    if erratic:
        print(f"\nFrequent changers (>={args.erratic_changes}, descriptive only):")
        for t, n in sorted(erratic.items(), key=lambda kv: -kv[1]):
            print(f"   vehicle #{t}: {n} changes")
    print("\nLane changes are EVENTS, not violations.")
    print("=" * W)


def render(cap, rows, ev_by_key, bounds, names, Hm, span, out_path, fps, args):
    speed, wrong = {}, {}
    if args.speed_csv and Path(args.speed_csv).exists():
        for r in csv.DictReader(open(args.speed_csv, encoding="utf-8")):
            speed[(int(r["frame_id"]), int(r["vehicle_id"]))] = (
                r["speed_kmh"], r.get("violation", "0") == "1")
        print(f"Overlaying speed from {Path(args.speed_csv).name}")
    if args.wrong_lane_csv and Path(args.wrong_lane_csv).exists():
        for r in csv.DictReader(open(args.wrong_lane_csv, encoding="utf-8")):
            if r.get("wrong_side") == "1":
                wrong[(int(r["frame_id"]), int(r["vehicle_id"]))] = True
        print(f"Overlaying wrong-side from {Path(args.wrong_lane_csv).name}")

    Hinv = np.linalg.inv(Hm)
    ys = np.linspace(span[0], span[1], 60)

    def to_img(world_pts):
        return cv2.perspectiveTransform(
            np.float32(world_pts).reshape(-1, 1, 2), Hinv).reshape(-1, 2)

    boundary_px = [to_img([[x, y] for y in ys]) for x in bounds]

    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame_id"]].append(r)
    # keep the label up for a while after the event so it is visible
    active = defaultdict(list)
    for (f, tid), evs in ev_by_key.items():
        for d in range(args.event_display_frames):
            active[(f + d, tid)] = evs

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

        overlay = frame.copy()
        for i in range(len(bounds) - 1):
            poly = np.vstack([boundary_px[i],
                              boundary_px[i + 1][::-1]]) * s
            cv2.fillPoly(overlay, [poly.astype(np.int32)],
                         LANE_COLORS[i % len(LANE_COLORS)])
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
        for bp in boundary_px:
            p = (bp * s).astype(np.int32)
            p = p[(p[:, 0] > -5000) & (p[:, 0] < 5000)]
            if len(p) > 1:
                cv2.polylines(frame, [p], False, (255, 255, 255), 2)

        n_ev = 0
        for r in by_frame.get(idx, []):
            ev = active.get((idx, r["vehicle_id"]))
            kmh, speeding = speed.get((idx, r["vehicle_id"]), ("", False))
            ws = wrong.get((idx, r["vehicle_id"]), False)
            n_ev += bool(ev)

            if ev:
                color = COLOR_CHANGE
            elif ws:
                color = COLOR_WRONG
            elif speeding:
                color = COLOR_SPEED
            else:
                color = CLASS_COLORS.get(r["class"], (255, 255, 255))

            x1, y1 = int(r["x1"] * s), int(r["y1"] * s)
            x2, y2 = int(r["x2"] * s), int(r["y2"] * s)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          3 if (ev or ws or speeding) else 2)

            lab = f"#{r['vehicle_id']} {r['class']}"
            if r["lane_settled"]:
                lab += f" [{r['lane_settled']}]"
            if kmh not in ("", None):
                lab += f" {float(kmh):.0f}km/h"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 5, ty + 2),
                          color, -1)
            cv2.putText(frame, lab, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)

            tags = []
            if ev:
                e = ev[0]
                tags.append((f"LANE CHANGE {e['from_lane']}>{e['to_lane']}",
                             COLOR_CHANGE, (0, 0, 0)))
            if ws:
                tags.append(("WRONG SIDE", COLOR_WRONG, (255, 255, 255)))
            if speeding:
                tags.append(("SPEED", COLOR_SPEED, (255, 255, 255)))
            oy = y2 + 4
            for t, col, fg in tags:
                (bw, bh), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 2)
                yy = oy + bh + 2 if oy + bh + 6 < frame.shape[0] else y1 - 6
                cv2.rectangle(frame, (x1, yy - bh - 4), (x1 + bw + 6, yy + 3),
                              col, -1)
                cv2.putText(frame, t, (x1 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, fg, 2, cv2.LINE_AA)
                oy = yy + 6

        banner = (f"frame {idx}   LANE CHANGE (yellow): {n_ev}   "
                  f"wrong side = magenta   speeding = red   "
                  f"[lanes from calibrated markings; changes are events, "
                  f"not violations]")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
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
