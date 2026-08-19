"""
Combined violation renderer — all layers on one output video.

Merges the per-frame outputs of the independent violation layers by
(frame_id, vehicle_id) and draws them together, so a vehicle triggering more
than one violation shows every flag it earned rather than only the first.

    python combine_violations.py --video <in.mp4> \
        --tracks out/<stem>_tracks.csv \
        --speed-csv out/<stem>_tracks_speed.csv \
        --wrong-lane-csv out/<stem>_wrong_lane.csv \
        --lane-change-csv out/<stem>_lane_change.csv \
        --tailgating-csv out/<stem>_tailgating.csv \
        --out-dir out

Every layer is optional: pass only the CSVs that exist and the rest are simply
absent from the render, so this works part-way through a pipeline.

Multiple simultaneous flags are drawn as NESTED rectangles, one per violation,
each in its own colour, plus a stacked tag under the box. A single blended
colour would be ambiguous; nesting keeps each violation individually readable.

Nothing here re-derives any violation. It is a presentation layer over
decisions the individual scripts already made and documented, so the video
cannot disagree with the CSVs.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# BGR, matching each layer's own renderer so colours stay consistent
LAYERS = [
    # key,        tag,      colour,            text colour
    ("speed",     "SPEED",  (0, 0, 255),       (255, 255, 255)),
    ("wrong",     "WRONG",  (255, 0, 255),     (255, 255, 255)),
    ("lane",      "LANE",   (0, 255, 255),     (0, 0, 0)),
    ("tail",      "TAIL",   (255, 255, 0),     (0, 0, 0)),
]
LAYER_LABEL = {"speed": "speeding", "wrong": "wrong side",
               "lane": "lane change", "tail": "tailgating"}
CONTEXT_COLOR = (130, 130, 130)   # tracked but unflagged


def parse_args():
    p = argparse.ArgumentParser(
        description="Render all violation layers onto one video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True, help="Base per-frame boxes.")
    p.add_argument("--speed-csv", default=None)
    p.add_argument("--wrong-lane-csv", default=None)
    p.add_argument("--lane-change-csv", default=None)
    p.add_argument("--tailgating-csv", default=None)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--name", default=None,
                   help="Output basename. Defaults to <video stem>_all_violations.")
    p.add_argument("--event-hold-frames", type=int, default=20,
                   help="How long a momentary event (lane change) stays "
                        "on screen after it fires.")
    p.add_argument("--ignore-region", action="append", default=None,
                   metavar="x1,y1,x2,y2")
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


def read_csv(path):
    if not path or not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    for p in (args.video, args.tracks):
        if not Path(p).exists():
            raise SystemExit(f"ERROR: not found: {p}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    name = args.name or f"{stem}_all_violations"
    regions = parse_regions(args.ignore_region)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    base = read_csv(args.tracks)
    rows = []
    for r in base:
        b = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if any(a <= cx <= c and d <= cy <= e for a, d, c, e in regions):
            continue
        rows.append({"frame_id": int(r["frame_id"]),
                     "vehicle_id": int(r["vehicle_id"]),
                     "class": r["class"], "x1": b[0], "y1": b[1],
                     "x2": b[2], "y2": b[3]})

    # ---- merge each layer's verdict by (frame, vehicle) -------------------
    flags = defaultdict(dict)      # (frame, vid) -> {layer: True}
    extra = defaultdict(dict)      # (frame, vid) -> display fields
    present = {}

    sp = read_csv(args.speed_csv)
    present["speed"] = sp is not None
    for r in sp or []:
        k = (int(r["frame_id"]), int(r["vehicle_id"]))
        if r.get("speed_kmh"):
            extra[k]["kmh"] = float(r["speed_kmh"])
        if r.get("violation") == "1":
            flags[k]["speed"] = True

    wl = read_csv(args.wrong_lane_csv)
    present["wrong"] = wl is not None
    for r in wl or []:
        if r.get("wrong_side") == "1":
            flags[(int(r["frame_id"]), int(r["vehicle_id"]))]["wrong"] = True

    lc = read_csv(args.lane_change_csv)
    present["lane"] = lc is not None
    for r in lc or []:
        k = (int(r["frame_id"]), int(r["vehicle_id"]))
        if r.get("lane_settled"):
            extra[k]["lane"] = r["lane_settled"]
        if r.get("lane_change_event") == "1":
            # a change is instantaneous; hold it briefly so it is visible
            for d in range(args.event_hold_frames):
                flags[(k[0] + d, k[1])]["lane"] = True

    tg = read_csv(args.tailgating_csv)
    present["tail"] = tg is not None
    for r in tg or []:
        k = (int(r["frame_id"]), int(r["vehicle_id"]))
        if r.get("gap_lengths"):
            extra[k]["gap"] = float(r["gap_lengths"])
        if r.get("headway_sec"):
            extra[k]["hw"] = float(r["headway_sec"])
        if r.get("tailgating") == "1":
            flags[k]["tail"] = True
            if r.get("leader_id") not in ("", None):
                extra[k]["leader"] = int(r["leader_id"])

    missing = [LAYER_LABEL[k] for k, v in present.items() if not v]
    print(f"Video  : {Path(args.video).name} @ {fps:.2f} fps")
    print(f"Layers : " + ", ".join(LAYER_LABEL[k] for k, v in present.items() if v))
    if missing:
        print(f"MISSING: {', '.join(missing)} (not drawn)")

    by_frame = defaultdict(list)
    pos = {}
    for r in rows:
        by_frame[r["frame_id"]].append(r)
        pos[(r["frame_id"], r["vehicle_id"])] = r

    # ---- stats -----------------------------------------------------------
    veh_layers = defaultdict(set)
    for (f, vid), d in flags.items():
        for k in d:
            veh_layers[vid].add(k)
    counts = Counter()
    for vid, ks in veh_layers.items():
        for k in ks:
            counts[k] += 1
    multi = {vid: sorted(ks) for vid, ks in veh_layers.items() if len(ks) > 1}

    # ---- render ----------------------------------------------------------
    out_path = out_dir / f"{name}.mp4"
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = None
    s = args.annot_scale
    idx = -1
    peak = Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if s != 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s,
                               interpolation=cv2.INTER_CUBIC)
        live = Counter()

        for r in by_frame.get(idx, []):
            key = (idx, r["vehicle_id"])
            act = [(k, tag, col, fg) for k, tag, col, fg in LAYERS
                   if flags.get(key, {}).get(k)]
            for k, _, _, _ in act:
                live[k] += 1

            x1, y1 = int(r["x1"] * s), int(r["y1"] * s)
            x2, y2 = int(r["x2"] * s), int(r["y2"] * s)

            if not act:
                # Unflagged traffic is drawn dim and thin. The per-class
                # palette is deliberately NOT used here: car-orange and
                # rickshaw-yellow are close enough to the lane-change and
                # tailgating colours that a viewer reads them as flags.
                cv2.rectangle(frame, (x1, y1), (x2, y2), CONTEXT_COLOR, 1)
                cv2.putText(frame, f"#{r['vehicle_id']}", (x1 + 1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, CONTEXT_COLOR, 1,
                            cv2.LINE_AA)
                continue

            # nested rectangles, one per violation, so several stay legible
            for n, (_, _, col, _) in enumerate(act):
                o = n * 3
                cv2.rectangle(frame, (x1 - o, y1 - o), (x2 + o, y2 + o), col, 2)

            e = extra.get(key, {})
            lab = f"#{r['vehicle_id']} {r['class']}"
            if "kmh" in e:
                lab += f" {e['kmh']:.0f}km/h"
            if "gap" in e and any(k == "tail" for k, _, _, _ in act):
                lab += f" gap{e['gap']:.1f}L"
            head_col = act[0][2]
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = y1 - 6 if y1 - th - 8 >= 0 else y2 + th + 8
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 5, ty + 3),
                          head_col, -1)
            cv2.putText(frame, lab, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if any(k == "tail" for k, _, _, _ in act) and "leader" in e:
                lead = pos.get((idx, e["leader"]))
                if lead:
                    a = (int((r["x1"] + r["x2"]) / 2 * s), int(r["y1"] * s))
                    b = (int((lead["x1"] + lead["x2"]) / 2 * s),
                         int(lead["y2"] * s))
                    cv2.arrowedLine(frame, a, b, (255, 255, 0), 2,
                                    tipLength=0.25)

            oy = y2 + 5
            for _, tag, col, fg in act:
                (bw, bh), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 2)
                yy = oy + bh + 2 if oy + bh + 6 < frame.shape[0] else y1 - 8
                cv2.rectangle(frame, (x1, yy - bh - 4), (x1 + bw + 6, yy + 3),
                              col, -1)
                cv2.putText(frame, tag, (x1 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, fg, 2, cv2.LINE_AA)
                oy = yy + 5

        for k, v in live.items():
            peak[k] = max(peak[k], v)

        # ---- banner: live counts + colour legend -------------------------
        H = frame.shape[0]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (18, 18, 18), -1)
        cv2.putText(frame, f"frame {idx}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (230, 230, 230), 1, cv2.LINE_AA)
        x = 110
        for k, tag, col, _ in LAYERS:
            if not present.get(k):
                continue
            cv2.rectangle(frame, (x, 8), (x + 16, 22), col, -1)
            txt = f"{LAYER_LABEL[k]}: {live.get(k, 0)}"
            cv2.putText(frame, txt, (x + 22, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x += 22 + tw + 26
        cv2.rectangle(frame, (x, 8), (x + 16, 22), CONTEXT_COLOR, 1)
        cv2.putText(frame, "tracked, no violation", (x + 22, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                    cv2.LINE_AA)
        cv2.putText(frame,
                    "multiple flags = nested boxes   |   speeds & lane rules "
                    "rest on documented assumptions - see KNOWN_LIMITATIONS.md",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1,
                    cv2.LINE_AA)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (w, h))
        writer.write(frame)
    if writer is not None:
        writer.release()
    cap.release()

    summary = {
        "video": str(args.video), "output": str(out_path), "fps": fps,
        "layers_present": {LAYER_LABEL[k]: v for k, v in present.items()},
        "vehicles_flagged_per_layer": {LAYER_LABEL[k]: counts.get(k, 0)
                                       for k, _, _, _ in LAYERS},
        "peak_simultaneous_per_layer": {LAYER_LABEL[k]: peak.get(k, 0)
                                        for k, _, _, _ in LAYERS},
        "vehicles_with_multiple_violation_types": len(multi),
        "multi_violation_detail": {str(v): [LAYER_LABEL[x] for x in ks]
                                   for v, ks in sorted(multi.items())},
        "NOTE": "Presentation layer only. Every flag is read from the "
                "individual layers' CSVs; nothing is re-derived here, so this "
                "video cannot disagree with them. Thresholds and assumptions "
                "are documented in each layer's _meta.json and in "
                "KNOWN_LIMITATIONS.md.",
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")

    print(f"\nVehicles flagged per layer:")
    for k, _, _, _ in LAYERS:
        if present.get(k):
            print(f"   {LAYER_LABEL[k]:<14}{counts.get(k, 0):>4}   "
                  f"peak simultaneous on screen: {peak.get(k, 0)}")
    print(f"\nVehicles with MORE THAN ONE violation type: {len(multi)}")
    for vid, ks in list(sorted(multi.items()))[:12]:
        print(f"   #{vid:<6}{', '.join(LAYER_LABEL[k] for k in ks)}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {out_dir / f'{name}_summary.json'}")


if __name__ == "__main__":
    main()
