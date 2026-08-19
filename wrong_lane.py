"""
Wrong-side / wrong-direction detection, layered on the tracking output.

Reads the per-frame tracks CSV from detect_track.py, assigns each vehicle to a
user-defined zone, compares its actual direction of travel against the expected
direction for that zone, and flags sustained opposing movement.

Standalone and decoupled from detection, like speed_violation.py: changing zones
or thresholds re-runs in seconds without re-running YOLO.

    python detect_track.py --video <in.mp4> --model <weights> --out-dir out
    python wrong_lane.py --video <in.mp4> --tracks out/<stem>_tracks.csv \
        --zones zones.json --out-dir out --save-video \
        --speed-csv out/<stem>_tracks_speed.csv     # optional, overlays both

ZONES FILE
    {"zones": [
       {"name": "main_carriageway",
        "polygon": [[200,180],[1280,180],[1280,624],[200,624]],
        "expected_direction": "toward_camera"},
       ...]}

    expected_direction: "toward_camera" (+y, down the image),
                        "away_from_camera" (-y), or an explicit [dx, dy]
                        vector in pixels for diagonal flows.

=============================== ASSUMPTIONS ===============================
Zone boundaries and expected directions are HAND-SPECIFIED for one camera
view. They are not derived from the road, so they are only valid for this
framing -- any pan, zoom or different camera invalidates them.

Direction is measured in the IMAGE plane, not on the ground plane. That is
sufficient here because the test is a sign comparison (moving with vs against
the flow), which perspective does not invert.

Two guards stop noise becoming violations:
  * a vehicle must move at least --min-displacement-px over the window, so a
    stationary vehicle's random box jitter is never given a direction;
  * the opposing direction must persist for --min-violation-frames, so a
    single bad frame cannot flag a vehicle.
A vehicle that fails either test is reported as "no verdict", not as compliant.
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
    "zone", "move_dx", "move_dy", "alignment", "wrong_side",
]

COLOR_WRONG = (255, 0, 255)     # magenta - wrong side
COLOR_SPEED = (0, 0, 255)       # red - speeding (matches speed_violation.py)
COLOR_BOTH = (0, 165, 255)      # orange - both at once

NAMED_DIRECTIONS = {
    # image coords: +y points DOWN the frame, i.e. toward the camera
    "toward_camera": (0.0, 1.0),
    "away_from_camera": (0.0, -1.0),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Wrong-side / wrong-direction detection over tracked output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True, help="CSV from detect_track.py.")
    p.add_argument("--zones", required=True, help="Zone definition JSON.")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--speed-csv", default=None,
                   help="Optional _tracks_speed.csv, so speed and wrong-side "
                        "violations can be drawn on one video.")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Frames over which travel direction is measured.")
    p.add_argument("--min-displacement-px", type=float, default=6.0,
                   help="Minimum movement over the window before a direction is "
                        "assigned. Below this a vehicle is effectively "
                        "stationary and its heading is noise.")
    p.add_argument("--wrong-cos-threshold", type=float, default=-0.3,
                   help="Flag a sample when cos(angle between movement and "
                        "expected direction) falls below this. -1 is exactly "
                        "reversed, 0 is perpendicular. The default (~107 deg) "
                        "leaves lane-changes and turns unflagged.")
    p.add_argument("--min-violation-frames", type=int, default=5,
                   help="Frames of opposing travel before a vehicle is flagged.")
    p.add_argument("--min-track-frames", type=int, default=8,
                   help="Tracks shorter than this get no verdict.")
    p.add_argument("--ignore-region", action="append", default=None,
                   metavar="x1,y1,x2,y2",
                   help="Rectangle to exclude (player overlays, insets).")
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


def load_zones(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    zones = []
    for z in spec["zones"]:
        d = z["expected_direction"]
        if isinstance(d, str):
            if d not in NAMED_DIRECTIONS:
                raise SystemExit(f"unknown expected_direction '{d}'; use "
                                 f"{sorted(NAMED_DIRECTIONS)} or [dx,dy]")
            vec = np.float32(NAMED_DIRECTIONS[d])
        else:
            vec = np.float32(d)
        n = float(np.linalg.norm(vec))
        if n == 0:
            raise SystemExit(f"zone '{z['name']}' has a zero direction vector")
        zones.append({"name": z["name"],
                      "polygon": np.array(z["polygon"], np.int32),
                      "direction_spec": d,
                      "unit": vec / n})
    return zones


def zone_of(pt, zones):
    for z in zones:
        if cv2.pointPolygonTest(z["polygon"], (float(pt[0]), float(pt[1])),
                                False) >= 0:
            return z
    return None


def main():
    args = parse_args()
    for p in (args.tracks, args.video, args.zones):
        if not Path(p).exists():
            raise SystemExit(f"ERROR: not found: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    zones = load_zones(args.zones)
    regions = parse_regions(args.ignore_region)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

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

    print(f"Video   : {Path(args.video).name}  {width}x{height} @ {fps:.2f} fps")
    print(f"Tracks  : {Path(args.tracks).name}  ({len(rows)} rows kept)")
    print(f"Zones   : " + ", ".join(
        f"{z['name']}->{z['direction_spec']}" for z in zones))
    print(f"Rules   : >={args.min_displacement_px}px movement, "
          f"cos<{args.wrong_cos_threshold}, "
          f"sustained {args.min_violation_frames}+ frames")

    # ---- per-sample direction vs expectation -----------------------------
    by_track = defaultdict(list)
    for r in rows:
        if r["vehicle_id"] != -1:
            by_track[r["vehicle_id"]].append(r)

    verdict = {}          # (frame_id, vid) -> dict
    zone_samples = Counter()
    no_verdict = Counter()
    k = max(1, args.smooth_window)

    for tid, recs in by_track.items():
        recs.sort(key=lambda r: r["frame_id"])
        if len(recs) < args.min_track_frames:
            no_verdict["track_too_short"] += len(recs)
            continue
        # bottom-centre: the vehicle's ground contact, so zone assignment
        # reflects the lane it is actually in rather than where its roof is
        px = np.array([(r["x1"] + r["x2"]) / 2 for r in recs])
        py = np.array([r["y2"] for r in recs])
        fid = np.array([r["frame_id"] for r in recs], dtype=float)

        for i in range(len(recs)):
            if i < k:
                no_verdict["window_warmup"] += 1
                continue
            j = i - k
            dx, dy = px[i] - px[j], py[i] - py[j]
            dist = float(np.hypot(dx, dy))
            z = zone_of((px[i], py[i]), zones)
            if z is None:
                no_verdict["outside_all_zones"] += 1
                continue
            zone_samples[z["name"]] += 1
            if dist < args.min_displacement_px:
                no_verdict["below_min_displacement"] += 1
                continue
            align = float(np.dot(np.float32([dx, dy]) / dist, z["unit"]))
            verdict[(recs[i]["frame_id"], tid)] = {
                "zone": z["name"], "dx": dx, "dy": dy, "alignment": align,
                "wrong": align < args.wrong_cos_threshold,
            }

    # ---- persistence -----------------------------------------------------
    wrong_count = Counter()
    for (f_id, tid), v in verdict.items():
        if v["wrong"]:
            wrong_count[tid] += 1
    violators = {t for t, n in wrong_count.items()
                 if n >= args.min_violation_frames}

    for r in rows:
        v = verdict.get((r["frame_id"], r["vehicle_id"]))
        r["zone"] = v["zone"] if v else ""
        r["move_dx"] = round(v["dx"], 2) if v else ""
        r["move_dy"] = round(v["dy"], 2) if v else ""
        r["alignment"] = round(v["alignment"], 3) if v else ""
        r["wrong_side"] = int(bool(v and v["wrong"]
                                   and r["vehicle_id"] in violators))

    out_csv = out_dir / f"{stem}_wrong_lane.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows([{c: r[c] for c in CSV_COLUMNS} for r in rows])
    print(f"\nWrote {out_csv}  ({len(rows)} rows)")

    summary = []
    for tid in sorted(violators):
        vs = [v for (f, t), v in verdict.items() if t == tid]
        wf = sorted(f for (f, t), v in verdict.items() if t == tid and v["wrong"])
        zc = Counter(v["zone"] for v in vs)
        cls = Counter(r["class"] for r in by_track[tid]).most_common(1)[0][0]
        summary.append({
            "vehicle_id": tid, "class": cls,
            "zone": zc.most_common(1)[0][0],
            "wrong_frames": wrong_count[tid],
            "total_judged_frames": len(vs),
            "mean_alignment": round(float(np.mean([v["alignment"] for v in vs])), 3),
            "first_wrong_frame": wf[0],
            "first_wrong_sec": round(wf[0] / fps, 2),
            "last_wrong_frame": wf[-1],
        })
    viol_csv = out_dir / f"{stem}_wrong_lane_violations.csv"
    with open(viol_csv, "w", newline="", encoding="utf-8") as f:
        cols = (list(summary[0]) if summary else
                ["vehicle_id", "class", "zone", "wrong_frames",
                 "total_judged_frames", "mean_alignment", "first_wrong_frame",
                 "first_wrong_sec", "last_wrong_frame"])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {viol_csv}  ({len(summary)} flagged)")

    meta = {
        "source_video": str(args.video), "tracks_csv": str(args.tracks),
        "width": width, "height": height, "fps": fps,
        "zones": [{"name": z["name"], "polygon": z["polygon"].tolist(),
                   "expected_direction": z["direction_spec"],
                   "unit_vector_image_coords": z["unit"].tolist()}
                  for z in zones],
        "reference_point": "bottom-centre of box (ground contact)",
        "smooth_window_frames": args.smooth_window,
        "min_displacement_px": args.min_displacement_px,
        "wrong_cos_threshold": args.wrong_cos_threshold,
        "min_violation_frames": args.min_violation_frames,
        "min_track_frames": args.min_track_frames,
        "ignored_regions": regions,
        "samples_per_zone": dict(zone_samples),
        "no_verdict_reasons": dict(no_verdict),
        "vehicles_judged": len({t for (_, t) in verdict}),
        "vehicles_flagged": len(violators),
        "ASSUMPTIONS": (
            "Zone polygons and expected directions are HAND-SPECIFIED for this "
            "camera view only; they encode a human judgement about which way "
            "traffic should flow in each region, not anything measured from the "
            "road. They are invalid for any other camera, framing, or if this "
            "camera moves. Direction is measured in the image plane (a sign "
            "test, which perspective cannot invert), using the box "
            "bottom-centre as the vehicle's ground contact point."
        ),
        "LIMITATIONS": (
            "A vehicle is flagged only when it moves far enough to have a "
            "meaningful heading AND opposes the expected direction for several "
            "consecutive frames. Stationary, slow, briefly-tracked, or "
            "out-of-zone vehicles receive NO VERDICT - they are not counted as "
            "compliant. Reversing, U-turns, and vehicles legitimately crossing "
            "between zones will register as wrong-side; this layer tests "
            "direction only, not intent or legality."
        ),
    }
    (out_dir / f"{stem}_wrong_lane_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / f'{stem}_wrong_lane_meta.json'}")

    report(zone_samples, no_verdict, verdict, violators, summary, args)

    if args.save_video:
        render(cap, rows, zones, out_dir / f"{stem}_wrong_lane.mp4", fps, args)
    cap.release()


def report(zone_samples, no_verdict, verdict, violators, summary, args):
    W = 66
    print("\n" + "=" * W)
    print("WRONG-SIDE / DIRECTION SUMMARY")
    print("=" * W)
    print("Judged samples per zone:")
    for name, n in zone_samples.most_common():
        wrong = sum(1 for v in verdict.values()
                    if v["zone"] == name and v["wrong"])
        print(f"   {name:<24}{n:>7}  opposing: {wrong:>5} "
              f"({100 * wrong / n if n else 0:.1f}%)")
    if no_verdict:
        print("\nSamples with NO verdict (not counted as compliant):")
        for reason, n in no_verdict.most_common():
            print(f"   {reason:<24}{n:>7}")
    print(f"\nVehicles judged        : {len({t for (_, t) in verdict})}")
    print(f"VEHICLES FLAGGED       : {len(violators)}")
    for s in summary[:15]:
        print(f"   #{s['vehicle_id']:<5} {s['class']:<9} {s['zone']:<22} "
              f"{s['wrong_frames']:>3}/{s['total_judged_frames']:<3} frames  "
              f"align {s['mean_alignment']:>6.2f}  "
              f"first t={s['first_wrong_sec']}s")
    if not summary:
        print("   (none - all judged vehicles travelled with the expected flow)")
    print("\nZones and directions are hand-specified for this camera view.")
    print("Direction only: U-turns and legitimate crossings also register.")
    print("=" * W)


def render(cap, rows, zones, out_path, fps, args):
    speed = {}
    if args.speed_csv and Path(args.speed_csv).exists():
        with open(args.speed_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                speed[(int(r["frame_id"]), int(r["vehicle_id"]))] = (
                    r["speed_kmh"], r.get("violation", "0") == "1")
        print(f"Overlaying speed violations from {Path(args.speed_csv).name}")

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

        # zone outlines, so the rule being applied is visible
        for z in zones:
            cv2.polylines(frame, [(z["polygon"] * s).astype(np.int32)], True,
                          (90, 90, 90), 1)

        n_wrong = n_speed = 0
        for r in by_frame.get(idx, []):
            wrong = bool(r["wrong_side"])
            kmh, speeding = speed.get((idx, r["vehicle_id"]), ("", False))
            n_wrong += wrong
            n_speed += speeding
            if wrong and speeding:
                color, tags = COLOR_BOTH, ["WRONG SIDE", "SPEED"]
            elif wrong:
                color, tags = COLOR_WRONG, ["WRONG SIDE"]
            elif speeding:
                color, tags = COLOR_SPEED, ["VIOLATION"]
            else:
                color, tags = CLASS_COLORS.get(r["class"], (255, 255, 255)), []

            x1, y1 = int(r["x1"] * s), int(r["y1"] * s)
            x2, y2 = int(r["x2"] * s), int(r["y2"] * s)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if tags else 2)

            label = f"#{r['vehicle_id']} {r['class']}"
            if kmh not in ("", None):
                label += f" {float(kmh):.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 5, ty + 2),
                          color, -1)
            cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

            oy = y2 + 4
            for t in tags:
                col = COLOR_WRONG if t == "WRONG SIDE" else COLOR_SPEED
                if len(tags) > 1:
                    col = COLOR_BOTH
                (bw, bh), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.55, 2)
                yy = oy + bh + 2 if oy + bh + 6 < frame.shape[0] else y1 - 6
                cv2.rectangle(frame, (x1, yy - bh - 4), (x1 + bw + 6, yy + 3),
                              col, -1)
                cv2.putText(frame, t, (x1 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)
                oy = yy + 6

        banner = (f"frame {idx}   WRONG SIDE (magenta): {n_wrong}   "
                  f"SPEED (red): {n_speed}   both = orange   "
                  f"[zones hand-specified for this view]")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
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
