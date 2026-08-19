"""
Cross-domain generalisation test: stock vs fine-tuned detector.

Runs two models over a folder of videos and reports per-video detection rates,
class distributions and confidence, so domain specialisation can be quantified
rather than asserted.

Detection only (no tracking): the question is what the detector emits on
unfamiliar footage, and tracking would add time without informing that.

Resumable -- one JSON per (video, model) written on completion, so an
interrupted run continues where it stopped.

    python generalisation_test.py --videos data/teammate_dataset/ML_videos \
        --out-dir results_generalisation \
        --model-a yolov8s.pt \
        --model-b runs/detect/runs/bmd45_ft/weights/best.pt \
        --control "Screen Recording 2026-08-17 215744.mp4"

The control video is one KNOWN to be in the fine-tuned model's own domain. It
is measured identically and reported separately, so "model B behaves oddly
here" can be separated from "model B is simply broken".
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detect_track import build_class_map

# Classes that cannot legitimately occur in non-Indian traffic. Any of these on
# Western motorway footage is a hallucination by construction.
DOMAIN_SPECIFIC_CLASSES = {"rickshaw"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two detectors across a folder of videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--videos", required=True)
    p.add_argument("--pattern", default="*.mp4")
    p.add_argument("--out-dir", default="results_generalisation")
    p.add_argument("--model-a", default="yolov8s.pt", help="Baseline (stock).")
    p.add_argument("--model-b",
                   default="runs/detect/runs/bmd45_ft/weights/best.pt",
                   help="Fine-tuned.")
    p.add_argument("--control", default=None,
                   help="Filename that IS in model B's domain; reported apart "
                        "from the cross-domain set.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Higher than the pipeline default: this measures what "
                        "the model asserts confidently, not its weak tail.")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", default="GENERALISATION_REPORT.md")
    return p.parse_args()


def run_one(model, class_map, path, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    keep = sorted(class_map)
    cls_count = Counter()
    confs = defaultdict(list)
    frames = seen = empty = 0
    t0 = time.time()
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.stride:
            continue
        seen += 1
        res = model.predict(frame, classes=keep, conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, device=args.device,
                            verbose=False)[0]
        n = 0
        if res.boxes is not None and len(res.boxes):
            for cl, cf in zip(res.boxes.cls.cpu().numpy().astype(int),
                              res.boxes.conf.cpu().numpy()):
                name = class_map.get(int(cl))
                if name:
                    cls_count[name] += 1
                    confs[name].append(float(cf))
                    n += 1
        if n == 0:
            empty += 1
        frames += n
    cap.release()
    tot = sum(cls_count.values())
    return {
        "video": path.name,
        "frames_processed": seen,
        "detections": tot,
        "dets_per_frame": round(tot / seen, 3) if seen else 0,
        "frames_empty": empty,
        "frames_empty_pct": round(100 * empty / seen, 2) if seen else 0,
        "class_counts": dict(cls_count),
        "class_conf_mean": {k: round(float(np.mean(v)), 4)
                            for k, v in confs.items()},
        "mean_conf": round(float(np.mean(
            [c for v in confs.values() for c in v])), 4) if tot else 0,
        "domain_specific_detections": sum(
            cls_count.get(c, 0) for c in DOMAIN_SPECIFIC_CLASSES),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    args = parse_args()
    vids = sorted(Path(args.videos).glob(args.pattern))
    if not vids:
        raise SystemExit(f"no videos matching {args.pattern} in {args.videos}")
    if args.limit:
        vids = vids[:args.limit]
    out_dir = Path(args.out_dir)
    (out_dir / "per_video").mkdir(parents=True, exist_ok=True)

    models = {"A_stock": args.model_a, "B_finetuned": args.model_b}
    loaded, maps = {}, {}
    for tag, path in models.items():
        m = YOLO(path)
        loaded[tag] = m
        maps[tag] = build_class_map(m)
        print(f"{tag}: {path}")
        print(f"   classes: {[maps[tag][k] for k in sorted(maps[tag])]}")

    todo = [(v, t) for v in vids for t in models]
    done = 0
    print(f"\n{len(vids)} videos x {len(models)} models = {len(todo)} runs "
          f"(conf={args.conf}, stride={args.stride})\n", flush=True)

    t_start = time.time()
    for i, (v, tag) in enumerate(todo, 1):
        jf = out_dir / "per_video" / f"{v.stem}__{tag}.json"
        if jf.exists() and not args.force:
            done += 1
            continue
        r = run_one(loaded[tag], maps[tag], v, args)
        if r is None:
            print(f"[{i}/{len(todo)}] FAILED to open {v.name}", flush=True)
            continue
        r["model"] = tag
        r["model_path"] = models[tag]
        r["is_control"] = (args.control is not None and v.name == args.control)
        jf.write_text(json.dumps(r, indent=2), encoding="utf-8")
        done += 1
        el = time.time() - t_start
        eta = el / max(done, 1) * (len(todo) - done)
        print(f"[{i}/{len(todo)}] {tag:<12}{v.name[-22:]:<24}"
              f"{r['dets_per_frame']:>6.2f}/fr  conf {r['mean_conf']:.2f}  "
              f"rickshaw {r['domain_specific_detections']:>4}  "
              f"{r['seconds']:>5.0f}s  ETA {eta/60:.0f}m", flush=True)

    aggregate(out_dir, args)


def aggregate(out_dir, args):
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out_dir / "per_video").glob("*.json"))]
    if not recs:
        raise SystemExit("no per-video results found")

    by = defaultdict(dict)
    for r in recs:
        by[r["video"]][r["model"]] = r

    rows = []
    for vid, m in sorted(by.items()):
        a, b = m.get("A_stock"), m.get("B_finetuned")
        if not (a and b):
            continue
        rows.append({
            "video": vid,
            "is_control": a.get("is_control", False),
            "frames": a["frames_processed"],
            "A_dets_per_frame": a["dets_per_frame"],
            "B_dets_per_frame": b["dets_per_frame"],
            "ratio_B_over_A": round(b["dets_per_frame"] / a["dets_per_frame"], 2)
                              if a["dets_per_frame"] else None,
            "A_mean_conf": a["mean_conf"],
            "B_mean_conf": b["mean_conf"],
            "A_empty_pct": a["frames_empty_pct"],
            "B_empty_pct": b["frames_empty_pct"],
            "B_rickshaw": b["domain_specific_detections"],
            "B_rickshaw_pct": round(100 * b["domain_specific_detections"]
                                    / b["detections"], 2) if b["detections"] else 0,
        })

    csv_path = out_dir / "per_video_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    cross = [r for r in rows if not r["is_control"]]
    ctrl = [r for r in rows if r["is_control"]]

    def pool(tag, subset):
        cc, tot, fr = Counter(), 0, 0
        conf = []
        for r in recs:
            if r["model"] != tag:
                continue
            if (r.get("is_control", False)) != (subset == "control"):
                continue
            cc.update(r["class_counts"])
            tot += r["detections"]
            fr += r["frames_processed"]
            if r["detections"]:
                conf.append(r["mean_conf"] * r["detections"])
        return cc, tot, fr, (sum(conf) / tot if tot else 0)

    summary = {}
    for subset in ("cross_domain", "control"):
        summary[subset] = {}
        for tag in ("A_stock", "B_finetuned"):
            cc, tot, fr, mc = pool(tag, subset)
            summary[subset][tag] = {
                "videos": len(cross) if subset == "cross_domain" else len(ctrl),
                "frames": fr, "detections": tot,
                "dets_per_frame": round(tot / fr, 3) if fr else 0,
                "mean_conf": round(mc, 4),
                "class_counts": dict(cc),
                "class_pct": {k: round(100 * v / tot, 2)
                              for k, v in cc.items()} if tot else {},
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'}")
    write_report(summary, rows, args)


def write_report(summary, rows, args):
    L = []
    w = L.append
    cd = summary["cross_domain"]
    ct = summary.get("control", {})
    a, b = cd["A_stock"], cd["B_finetuned"]

    w("# Cross-Domain Generalisation Test")
    w("")
    w("Stock vs fine-tuned detector on footage from outside the fine-tuning "
      "domain. Detection only, no tracking.")
    w("")
    w(f"- **Cross-domain set:** {a['videos']} videos, {a['frames']:,} frames "
      f"(Western motorway)")
    if ct and ct.get("A_stock", {}).get("videos"):
        w(f"- **Control:** {ct['A_stock']['videos']} video "
          f"({ct['A_stock']['frames']:,} frames) from the fine-tuned model's "
          f"own domain")
    w(f"- **Settings:** conf={args.conf}, imgsz={args.imgsz}, "
      f"stride={args.stride}, device={args.device}")
    w("")
    w("## Headline")
    w("")
    w("| Metric | A: stock yolov8s | B: fine-tuned | |")
    w("|---|---|---|---|")
    w(f"| Detections/frame | {a['dets_per_frame']} | {b['dets_per_frame']} | "
      f"{'B higher' if b['dets_per_frame'] > a['dets_per_frame'] else 'A higher'} |")
    w(f"| Total detections | {a['detections']:,} | {b['detections']:,} | |")
    w(f"| Mean confidence | {a['mean_conf']:.3f} | {b['mean_conf']:.3f} | |")
    w("")
    rk = b["class_counts"].get("rickshaw", 0)
    w("## Domain-specific hallucination")
    w("")
    w(f"`rickshaw` cannot occur on Western motorways, so every such detection "
      f"is wrong by construction.")
    w("")
    w(f"- **Fine-tuned model emitted `rickshaw` {rk:,} times** "
      f"({b['class_pct'].get('rickshaw', 0)}% of its detections) across the "
      f"cross-domain set.")
    w(f"- Stock model cannot emit it (no such class).")
    w("")
    w("## Class distribution (cross-domain)")
    w("")
    w("| Class | A stock | A % | B fine-tuned | B % |")
    w("|---|---|---|---|---|")
    for c in sorted(set(a["class_counts"]) | set(b["class_counts"])):
        w(f"| `{c}` | {a['class_counts'].get(c, 0):,} | "
          f"{a['class_pct'].get(c, 0)}% | {b['class_counts'].get(c, 0):,} | "
          f"{b['class_pct'].get(c, 0)}% |")
    w("")
    if ct and ct.get("B_finetuned", {}).get("detections"):
        ca, cb = ct["A_stock"], ct["B_finetuned"]
        w("## Control (in-domain Indian footage)")
        w("")
        w("| Metric | A stock | B fine-tuned |")
        w("|---|---|---|")
        w(f"| Detections/frame | {ca['dets_per_frame']} | {cb['dets_per_frame']} |")
        w(f"| Mean confidence | {ca['mean_conf']:.3f} | {cb['mean_conf']:.3f} |")
        w(f"| `rickshaw` | n/a | {cb['class_counts'].get('rickshaw', 0):,} "
          f"({cb['class_pct'].get('rickshaw', 0)}%) |")
        w("")
        w("On its own domain the fine-tuned model's `rickshaw` detections are "
          "legitimate. The same class on motorway footage is not. Comparing "
          "the two rates separates 'specialised' from 'broken'.")
        w("")
    w("## Per-video")
    w("")
    w("| Video | frames | A /fr | B /fr | B/A | A conf | B conf | B rickshaw |")
    w("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["B_rickshaw"])):
        tag = " *(control)*" if r["is_control"] else ""
        w(f"| `{r['video'][-22:]}`{tag} | {r['frames']} | "
          f"{r['A_dets_per_frame']} | {r['B_dets_per_frame']} | "
          f"{r['ratio_B_over_A']} | {r['A_mean_conf']:.2f} | "
          f"{r['B_mean_conf']:.2f} | **{r['B_rickshaw']}** |")
    w("")
    w("## Caveat")
    w("")
    w("There are no ground-truth boxes for this footage, so neither recall nor "
      "precision is measured here. What is measured is what each model "
      "*asserts*. The `rickshaw` count is the exception: it is wrong by "
      "construction regardless of ground truth, which is why it is the "
      "cleanest available evidence of domain specialisation.")
    w("")
    w("These are screen recordings of video players, so some detections may "
      "fall on player UI or picture-in-picture insets. That affects both "
      "models equally and does not bias the comparison.")

    Path(args.report).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
