"""
Feasibility scan: which violation types actually have candidate events?

Reads existing tracks CSVs and reports CANDIDATES for four violation types, so
detection logic is built for things the footage contains rather than guessed at.

This is a screening tool, not a detector. Thresholds are deliberately loose to
surface anything worth looking at; every hit needs human confirmation. A high
count means "worth building for", not "N violations occurred".

    python violation_scan.py --tracks outputs_scan/*_tracks.csv --video-dir ...

SCANS
  1. multi-rider   bikes carrying more than one person. Requires a PERSON pass
                   (--person-model), because neither vehicle model has a person
                   class -- counted by overlap of person boxes with bike boxes.
  2. tailgating    vehicle pairs holding a small longitudinal gap, measured in
                   multiples of the follower's own length so it is scale-free
                   and needs no calibration.
  3. u-turn        tracks whose heading reverses by more than --uturn-degrees
                   and stays reversed.
  4. stopped       tracks that barely move for a sustained period WHILE other
                   traffic is moving (so a jammed frame is not flagged).
"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Screen footage for candidate violation events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", nargs="+", required=True,
                   help="Tracks CSV paths or globs.")
    p.add_argument("--video-dir", default=None,
                   help="Where the source videos live (for the person pass).")
    p.add_argument("--person-model", default=None,
                   help="Model with a COCO 'person' class, e.g. yolov8s.pt. "
                        "Without it the multi-rider scan is skipped rather "
                        "than guessed.")
    p.add_argument("--person-stride", type=int, default=5,
                   help="Sample every Nth frame for the person pass.")
    p.add_argument("--out", default="violation_scan_report.md")
    p.add_argument("--json-out", default="violation_scan.json")

    # --- thresholds (loose by design) -------------------------------------
    p.add_argument("--min-track-frames", type=int, default=10)
    p.add_argument("--tailgate-gap-lengths", type=float, default=0.6,
                   help="Gap below this many follower-lengths counts as close.")
    p.add_argument("--tailgate-frames", type=int, default=15,
                   help="Frames the pair must hold that gap.")
    p.add_argument("--uturn-degrees", type=float, default=120.0)
    p.add_argument("--uturn-min-px", type=float, default=8.0,
                   help="Movement needed before a heading is trusted.")
    p.add_argument("--stopped-px", type=float, default=6.0,
                   help="Max movement over the window to count as stationary.")
    p.add_argument("--stopped-frames", type=int, default=60,
                   help="Frames stationary before it is a candidate.")
    p.add_argument("--scene-moving-frac", type=float, default=0.35,
                   help="Fraction of other vehicles that must be moving for a "
                        "stationary vehicle to be interesting (filters jams).")
    p.add_argument("--multi-rider-iou", type=float, default=0.25,
                   help="Person/bike overlap (as fraction of person box) to "
                        "count the person as on that bike.")
    return p.parse_args()


def load_tracks(path):
    by_track = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tid = int(r["vehicle_id"])
            if tid == -1:
                continue
            by_track[tid].append({
                "f": int(r["frame_id"]), "cls": r["class"],
                "x1": float(r["x1"]), "y1": float(r["y1"]),
                "x2": float(r["x2"]), "y2": float(r["y2"]),
                "conf": float(r["confidence"]),
            })
    for v in by_track.values():
        v.sort(key=lambda r: r["f"])
    return by_track


def centres(recs):
    cx = np.array([(r["x1"] + r["x2"]) / 2 for r in recs])
    cy = np.array([r["y2"] for r in recs])          # ground contact
    return cx, cy


def scan_uturn(by_track, args):
    hits = []
    for tid, recs in by_track.items():
        if len(recs) < args.min_track_frames:
            continue
        cx, cy = centres(recs)
        k = 5
        headings = []
        for i in range(k, len(recs)):
            dx, dy = cx[i] - cx[i - k], cy[i] - cy[i - k]
            if np.hypot(dx, dy) >= args.uturn_min_px:
                headings.append((recs[i]["f"], np.arctan2(dy, dx)))
        if len(headings) < 4:
            continue
        # largest angle between any early and any later heading
        best = 0.0
        best_f = None
        for i in range(len(headings) - 1):
            for j in range(i + 1, len(headings)):
                d = abs(np.degrees(np.angle(
                    np.exp(1j * (headings[j][1] - headings[i][1])))))
                if d > best:
                    best, best_f = d, headings[j][0]
        if best >= args.uturn_degrees:
            hits.append({"vehicle_id": tid, "class": recs[0]["cls"],
                         "max_heading_change_deg": round(float(best), 1),
                         "at_frame": int(best_f),
                         "track_frames": len(recs)})
    return sorted(hits, key=lambda h: -h["max_heading_change_deg"])


def scan_stopped(by_track, args):
    # per-frame: how much is the rest of the scene moving?
    move_by_frame = defaultdict(list)
    for recs in by_track.values():
        cx, cy = centres(recs)
        for i in range(1, len(recs)):
            move_by_frame[recs[i]["f"]].append(
                float(np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1])))

    hits = []
    for tid, recs in by_track.items():
        if len(recs) < args.stopped_frames:
            continue
        cx, cy = centres(recs)
        run = 0
        best_run, best_start = 0, None
        for i in range(1, len(recs)):
            j = max(0, i - 10)
            disp = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
            if disp <= args.stopped_px:
                run += 1
                if run > best_run:
                    best_run, best_start = run, recs[i - run + 1]["f"]
            else:
                run = 0
        if best_run < args.stopped_frames:
            continue
        # was the rest of the scene moving during that stretch?
        frames = range(best_start, best_start + best_run)
        fracs = []
        for f in frames:
            m = move_by_frame.get(f, [])
            if len(m) >= 3:
                fracs.append(float(np.mean([x > 1.5 for x in m])))
        scene = float(np.mean(fracs)) if fracs else 0.0
        if scene < args.scene_moving_frac:
            continue                       # whole scene stopped -> jam, not a stop
        hits.append({"vehicle_id": tid, "class": recs[0]["cls"],
                     "stationary_frames": int(best_run),
                     "from_frame": int(best_start),
                     "scene_moving_frac": round(scene, 2),
                     "track_frames": len(recs)})
    return sorted(hits, key=lambda h: -h["stationary_frames"])


def scan_tailgating(by_track, args):
    by_frame = defaultdict(list)
    for tid, recs in by_track.items():
        for r in recs:
            by_frame[r["f"]].append((tid, r))

    close = Counter()
    detail = {}
    for f, items in by_frame.items():
        for i in range(len(items)):
            ti, ri = items[i]
            hi = ri["y2"] - ri["y1"]
            if hi < 15:
                continue
            cxi = (ri["x1"] + ri["x2"]) / 2
            for j in range(len(items)):
                if i == j:
                    continue
                tj, rj = items[j]
                cxj = (rj["x1"] + rj["x2"]) / 2
                # same lane-ish: horizontal centres within half a box width
                if abs(cxi - cxj) > 0.5 * (ri["x2"] - ri["x1"]):
                    continue
                # j ahead of i (further up the image = smaller y)
                gap = ri["y1"] - rj["y2"]
                if gap < 0:
                    continue
                if gap <= args.tailgate_gap_lengths * hi:
                    key = (ti, tj)
                    close[key] += 1
                    d = detail.setdefault(key, {"min_gap_lengths": 9e9})
                    g = gap / hi
                    if g < d["min_gap_lengths"]:
                        d["min_gap_lengths"] = round(float(g), 2)
                        d["at_frame"] = int(f)
    hits = []
    for (ti, tj), n in close.items():
        if n >= args.tailgate_frames:
            hits.append({"follower_id": ti, "leader_id": tj,
                         "frames_close": int(n),
                         **detail[(ti, tj)]})
    return sorted(hits, key=lambda h: -h["frames_close"])


def scan_multi_rider(video_path, by_track, args):
    """Persons overlapping each bike box. Needs a model with a person class."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(args.person_model)
    bikes_by_frame = defaultdict(list)
    for tid, recs in by_track.items():
        for r in recs:
            if r["cls"] == "bike" and (r["y2"] - r["y1"]) >= 40:
                bikes_by_frame[r["f"]].append((tid, r))
    if not bikes_by_frame:
        return [], 0

    cap = cv2.VideoCapture(str(video_path))
    per_bike = defaultdict(list)
    checked = 0
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.person_stride or idx not in bikes_by_frame:
            continue
        res = model.predict(frame, classes=[0], conf=0.25, imgsz=640,
                            device="cpu", verbose=False)[0]
        checked += 1
        people = (res.boxes.xyxy.cpu().numpy()
                  if res.boxes is not None and len(res.boxes) else [])
        for tid, r in bikes_by_frame[idx]:
            n = 0
            for (px1, py1, px2, py2) in people:
                ix1, iy1 = max(px1, r["x1"]), max(py1, r["y1"])
                ix2, iy2 = min(px2, r["x2"]), min(py2, r["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                parea = max((px2 - px1) * (py2 - py1), 1e-6)
                if inter / parea >= args.multi_rider_iou:
                    n += 1
            per_bike[tid].append(n)
    cap.release()

    hits = []
    for tid, counts in per_bike.items():
        if not counts:
            continue
        mx = int(max(counts))
        if mx >= 2:
            hits.append({"vehicle_id": tid, "max_persons_on_bike": mx,
                         "frames_with_2plus": int(sum(1 for c in counts if c >= 2)),
                         "samples": len(counts)})
    return sorted(hits, key=lambda h: (-h["max_persons_on_bike"],
                                       -h["frames_with_2plus"])), checked


def main():
    args = parse_args()
    paths = []
    for t in args.tracks:
        paths.extend(sorted(glob.glob(t)))
    if not paths:
        raise SystemExit("no tracks CSVs matched")

    results = {}
    for p in paths:
        name = Path(p).name.replace("_tracks.csv", "")
        by_track = load_tracks(p)
        n_tracks = len(by_track)
        r = {"tracks": n_tracks,
             "uturn": scan_uturn(by_track, args),
             "stopped": scan_stopped(by_track, args),
             "tailgating": scan_tailgating(by_track, args),
             "multi_rider": [], "person_frames_checked": 0,
             "bike_tracks": sum(1 for v in by_track.values()
                                if v[0]["cls"] == "bike")}
        if args.person_model and args.video_dir:
            vids = list(Path(args.video_dir).glob(f"{name}.*"))
            if vids:
                hits, checked = scan_multi_rider(vids[0], by_track, args)
                r["multi_rider"], r["person_frames_checked"] = hits, checked
        results[name] = r
        print(f"{name[-24:]:<26} tracks {n_tracks:>4}  "
              f"u-turn {len(r['uturn']):>3}  stopped {len(r['stopped']):>3}  "
              f"tailgate {len(r['tailgating']):>4}  "
              f"multi-rider {len(r['multi_rider']):>3}", flush=True)

    Path(args.json_out).write_text(json.dumps(results, indent=2),
                                   encoding="utf-8")
    write_report(results, args)


def write_report(results, args):
    L, w = [], None
    out = []
    def w(s=""):
        out.append(s)

    w("# Violation Feasibility Scan")
    w("")
    w("Which violation types actually have candidate events in the available "
      "footage. **This is a screening tool, not a detector** — thresholds are "
      "loose to surface anything worth a look, and every hit needs human "
      "confirmation. Counts mean \"worth building for\", not \"N violations "
      "occurred\".")
    w("")
    w("| Scan | Rule used |")
    w("|---|---|")
    w(f"| Multi-rider | ≥2 person boxes overlapping one bike box "
      f"(≥{args.multi_rider_iou} of the person box inside) |")
    w(f"| Tailgating | gap ≤ {args.tailgate_gap_lengths}× follower length, "
      f"same lane, ≥{args.tailgate_frames} frames |")
    w(f"| U-turn | heading reverses ≥{args.uturn_degrees}° within one track |")
    w(f"| Stopped | ≤{args.stopped_px}px movement for ≥{args.stopped_frames} "
      f"frames while ≥{args.scene_moving_frac:.0%} of other traffic moves |")
    w("")
    w("## Summary")
    w("")
    w("| Video | tracks | bikes | multi-rider | tailgating | u-turn | stopped |")
    w("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        mr = (str(len(r["multi_rider"])) if r["person_frames_checked"]
              else "n/a")
        w(f"| `{name[-24:]}` | {r['tracks']} | {r['bike_tracks']} | {mr} | "
          f"{len(r['tailgating'])} | {len(r['uturn'])} | {len(r['stopped'])} |")
    w("")
    tot = {k: sum(len(r[k]) for r in results.values())
           for k in ("multi_rider", "tailgating", "uturn", "stopped")}
    w("## Totals across scanned footage")
    w("")
    for k, v in tot.items():
        vids = sum(1 for r in results.values() if r[k])
        w(f"- **{k.replace('_', '-')}**: {v} candidates across "
          f"{vids}/{len(results)} videos")
    w("")
    for name, r in results.items():
        if not any(r[k] for k in ("multi_rider", "tailgating", "uturn",
                                  "stopped")):
            continue
        w(f"## {name}")
        w("")
        for k, label in (("multi_rider", "Multi-rider"),
                         ("tailgating", "Tailgating"),
                         ("uturn", "U-turn"), ("stopped", "Stopped")):
            if not r[k]:
                continue
            w(f"**{label}** ({len(r[k])})")
            w("")
            for h in r[k][:6]:
                w(f"- `{json.dumps(h)}`")
            if len(r[k]) > 6:
                w(f"- …and {len(r[k]) - 6} more")
            w("")
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
