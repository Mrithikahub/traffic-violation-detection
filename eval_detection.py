"""
Evaluate the detector against a COCO ground-truth dataset.

Our own footage has no ground truth, so every earlier number described what the
model FOUND, never what it MISSED. This script measures real recall/precision
against annotated data, and separates two very different failure modes:

  * localisation failure - the model never fired on the vehicle at all
  * classification failure - it found the box but gave it the wrong label

That split matters for deciding what fine-tuning would fix.

Classes in a dataset like BMD-45 are finer-grained than COCO's, so they are
folded into this project's canonical car/bike/bus/truck scheme. Classes with no
COCO equivalent at all (auto-rickshaws) are EXCLUDED rather than force-mapped,
and reported separately as a coverage gap - a pretrained COCO model cannot
detect them under any label, so scoring them as misses would conflate
"model performed badly" with "class does not exist in the label space".

Usage:
    python eval_detection.py --images data/bmd45/images \
        --annotations data/bmd45/annotations.coco.json --model yolov8s.pt
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from detect_track import CANONICAL_CLASSES, build_class_map

# --------------------------------------------------------------------------
# Dataset class -> canonical class
# --------------------------------------------------------------------------
# Mapping requested explicitly for this project.
SPECIFIED_MAP = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "LCV": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Two-wheeler": "bike",
    "Bicycle": "bike",
    # Auto-rickshaw. Stock COCO weights cannot produce this, so it is scored
    # only when the evaluated model actually has the class (see `producible`
    # below) -- otherwise it is reported as a coverage gap, exactly as before.
    "Three-wheeler": "rickshaw",
}

# Not covered by the requested mapping. COCO has no van/minibus/tempo class, so
# these are folded into the nearest canonical class by judgement. Flagged in the
# report, and the headline metric is also printed with them excluded so their
# influence is visible rather than buried.
ASSUMED_MAP = {
    "Van": "car",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
}

DATASET_MAP = {**SPECIFIED_MAP, **ASSUMED_MAP}

# Which canonical classes a given model can emit is a property of the MODEL, not
# of the dataset: stock COCO weights have no rickshaw class, our fine-tuned
# weights do. Scoring a class the model structurally cannot produce would
# conflate "performed badly" with "class absent from the label space", so those
# boxes are withheld and reported as a coverage gap instead. Deriving this per
# model keeps before/after comparisons honest in both directions.
def unmatchable_classes(producible):
    """Dataset class names whose canonical target this model cannot emit."""
    return {name for name, canon in DATASET_MAP.items() if canon not in producible}


# Kept for prepare_dataset.py, which needs the training-time view: with a
# taxonomy that includes rickshaw, nothing is inherently unmatchable.
UNMATCHABLE = set()

IOU_MATCH = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure recall/precision against COCO ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", required=True, help="Directory of images.")
    p.add_argument("--annotations", required=True, help="COCO annotations JSON.")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only N images.")
    p.add_argument("--json-out", default=None, help="Write metrics to this JSON file.")
    return p.parse_args()


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ba = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ba[None, :] - inter, 1e-9)


def greedy_match(dets, gts, class_aware):
    """Highest-IoU-first one-to-one matching. Returns (matched_det, matched_gt)."""
    if not dets or not gts:
        return set(), set()
    M = iou_matrix([d[0] for d in dets], [g[0] for g in gts])
    if class_aware:
        for i, d in enumerate(dets):
            for j, g in enumerate(gts):
                if d[1] != g[1]:
                    M[i, j] = 0.0
    md, mg = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
    for i, j in order:
        if M[i, j] < IOU_MATCH:
            break
        if i in md or j in mg:
            continue
        md.add(int(i))
        mg.add(int(j))
    return md, mg


def main():
    args = parse_args()
    coco = json.load(open(args.annotations, encoding="utf-8"))
    cid2name = {c["id"]: c["name"] for c in coco["categories"]}

    unknown = {cid2name[c["id"]] for c in coco["categories"]
               if cid2name[c["id"]] not in DATASET_MAP}
    if unknown:
        print(f"WARNING: dataset classes with no mapping rule, excluded: "
              f"{sorted(unknown)}\n")

    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    images = coco["images"]
    if args.limit:
        images = images[:args.limit]

    model = YOLO(args.model)
    # Works for both pretrained COCO weights and our fine-tuned model.
    class_map = build_class_map(model)
    keep = sorted(class_map)
    img_dir = Path(args.images)
    producible = set(class_map.values())
    unmatchable = unmatchable_classes(producible)
    print(f"class map ({len(class_map)} ids): "
          f"{ {k: class_map[k] for k in sorted(class_map)} }")
    print(f"model can emit: {sorted(producible)}")
    print(f"excluded as unmatchable by this model: {sorted(unmatchable) or 'none'}")

    # counters
    stats = {
        "agn": Counter(), "aware": Counter(),
        "gt_per_canon": Counter(), "hit_agn_canon": Counter(),
        "hit_aware_canon": Counter(),
        "gt_per_dataset": Counter(), "hit_agn_dataset": Counter(),
        "gt_assumed": 0, "hit_agn_assumed": 0, "hit_aware_assumed": 0,
        "unmatchable_boxes": 0, "total_gt_boxes": 0,
        "det_on_unmatchable": 0, "det_total_raw": 0,
        "pred_canon": Counter(),
        "confusion": Counter(),
        "unmatchable_by_class": Counter(),
    }

    print(f"Evaluating {len(images)} images with {args.model} "
          f"(conf={args.conf}, imgsz={args.imgsz})\n")

    for n, im in enumerate(images, 1):
        path = img_dir / im["file_name"]
        if not path.exists():
            continue

        gts_map, gts_unmatch, assumed_idx = [], [], set()
        for a in ann_by_img[im["id"]]:
            name = cid2name[a["category_id"]]
            x, y, w, h = a["bbox"]
            box = [x, y, x + w, y + h]
            stats["total_gt_boxes"] += 1
            if name in unmatchable:
                gts_unmatch.append(box)
                stats["unmatchable_boxes"] += 1
                stats["unmatchable_by_class"][name] += 1
                continue
            canon = DATASET_MAP.get(name)
            if canon is None:
                continue
            if name in ASSUMED_MAP:
                assumed_idx.add(len(gts_map))
                stats["gt_assumed"] += 1
            gts_map.append((box, canon, name))
            stats["gt_per_canon"][canon] += 1
            stats["gt_per_dataset"][name] += 1

        res = model.predict(str(path), classes=keep, conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, device=args.device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            for (x1, y1, x2, y2), cl in zip(res.boxes.xyxy.cpu().numpy(),
                                            res.boxes.cls.cpu().numpy().astype(int)):
                canon = class_map.get(int(cl))
                if canon:
                    dets.append(([float(x1), float(y1), float(x2), float(y2)], canon))
        stats["det_total_raw"] += len(dets)

        # A detection that landed on an auto-rickshaw is neither a true positive
        # (it is not in the scored GT) nor a false positive (the model was not
        # wrong to see a vehicle there). Withhold it from scoring entirely,
        # mirroring how UA-DETRAC treats its ignored regions.
        if gts_unmatch and dets:
            Mu = iou_matrix([d[0] for d in dets], gts_unmatch)
            keep_det = []
            for i, d in enumerate(dets):
                if Mu[i].max() >= IOU_MATCH:
                    stats["det_on_unmatchable"] += 1
                else:
                    keep_det.append(d)
            dets = keep_det

        for _, c in dets:
            stats["pred_canon"][c] += 1

        gts_pairs = [(g[0], g[1]) for g in gts_map]
        for mode, aware in (("agn", False), ("aware", True)):
            md, mg = greedy_match(dets, gts_pairs, class_aware=aware)
            stats[mode]["TP"] += len(mg)
            stats[mode]["FP"] += len(dets) - len(md)
            stats[mode]["FN"] += len(gts_pairs) - len(mg)
            for j in mg:
                canon, dsname = gts_map[j][1], gts_map[j][2]
                if mode == "agn":
                    stats["hit_agn_canon"][canon] += 1
                    stats["hit_agn_dataset"][dsname] += 1
                    if j in assumed_idx:
                        stats["hit_agn_assumed"] += 1
                else:
                    stats["hit_aware_canon"][canon] += 1
                    if j in assumed_idx:
                        stats["hit_aware_assumed"] += 1

        # what the model called the GT boxes it localised but mislabelled
        md, mg = greedy_match(dets, gts_pairs, class_aware=False)
        if dets and gts_pairs:
            M = iou_matrix([d[0] for d in dets], [g[0] for g in gts_pairs])
            for i in md:
                j = int(np.argmax(M[i]))
                if M[i, j] >= IOU_MATCH and dets[i][1] != gts_pairs[j][1]:
                    stats["confusion"][f"{gts_pairs[j][1]} -> {dets[i][1]}"] += 1

        if n % 50 == 0 or n == len(images):
            print(f"  [{n}/{len(images)}]", flush=True)

    report(stats, args)


def rate(num, den):
    return num / den if den else 0.0


def report(s, args):
    W = 74
    print("\n" + "=" * W)
    print("DETECTION EVALUATION vs GROUND TRUTH")
    print("=" * W)
    print(f"model={args.model}  conf={args.conf}  imgsz={args.imgsz}")

    scored_gt = s["agn"]["TP"] + s["agn"]["FN"]
    print(f"\nGround-truth boxes total      : {s['total_gt_boxes']:,}")
    print(f"  scored (mappable to COCO)   : {scored_gt:,}")
    print(f"  EXCLUDED, no COCO equivalent: {s['unmatchable_boxes']:,}")

    print("\n" + "-" * W)
    print("1. RECALL / PRECISION  (mappable classes only)")
    print("-" * W)
    for mode, title in (("agn", "localisation only (any class)"),
                        ("aware", "localisation + correct class")):
        TP, FP, FN = s[mode]["TP"], s[mode]["FP"], s[mode]["FN"]
        r, p = rate(TP, TP + FN), rate(TP, TP + FP)
        f1 = rate(2 * r * p, r + p) if (r + p) else 0
        print(f"  {title:<32} recall {r:6.1%}   precision {p:6.1%}   F1 {f1:6.1%}")
    print(f"\n  detections withheld (landed on an excluded class): "
          f"{s['det_on_unmatchable']:,}")

    # headline with the assumed mappings removed, to show they do not carry it
    ta = s["gt_assumed"]
    if ta:
        TP, FN = s["agn"]["TP"], s["agn"]["FN"]
        r_excl = rate(TP - s["hit_agn_assumed"], (TP + FN) - ta)
        print(f"  recall excluding the 3 assumed-mapping classes "
              f"({ta} boxes): {r_excl:.1%}")

    print("\n" + "-" * W)
    print("2. COVERAGE GAP  (classes THIS model cannot emit under any label)")
    print("-" * W)
    gap = rate(s["unmatchable_boxes"], s["total_gt_boxes"])
    if s["unmatchable_boxes"]:
        for name, n in s["unmatchable_by_class"].most_common():
            print(f"  {name:<32}: {n:,} boxes")
        print(f"  share of ALL ground-truth boxes  : {gap:.1%}")
        print(f"\n  Not counted as misses above: the model has no such class, so no")
        print(f"  threshold or tuning can recover them - only fine-tuning on a")
        print(f"  taxonomy that includes them.")
    else:
        print(f"  NONE - this model can emit every mapped class, including")
        print(f"  rickshaw. The earlier coverage gap is closed, and those boxes")
        print(f"  are now SCORED in section 1 rather than excluded.")

    print("\n" + "-" * W)
    print("3. PER-CLASS BREAKDOWN")
    print("-" * W)
    print(f"  {'canonical':<10}{'GT':>8}{'found':>8}{'recall':>9}"
          f"{'correct':>9}{'cls-acc':>9}")
    print("  " + "-" * (W - 4))
    for c in CANONICAL_CLASSES:
        g = s["gt_per_canon"][c]
        if not g:
            continue
        ha, hw = s["hit_agn_canon"][c], s["hit_aware_canon"][c]
        print(f"  {c:<10}{g:>8,}{ha:>8,}{rate(ha,g):>9.1%}"
              f"{hw:>9,}{rate(hw,ha):>9.1%}")
    print("\n  cls-acc = of the boxes it localised, how many it also labelled right")

    print(f"\n  {'BMD-45 class':<18}{'GT':>8}{'found':>8}{'recall':>9}   -> canonical")
    print("  " + "-" * (W - 4))
    for name, g in s["gt_per_dataset"].most_common():
        h = s["hit_agn_dataset"][name]
        tag = DATASET_MAP[name] + (" (assumed)" if name in ASSUMED_MAP else "")
        print(f"  {name:<18}{g:>8,}{h:>8,}{rate(h,g):>9.1%}   -> {tag}")

    if s["confusion"]:
        print(f"\n  most common label errors on correctly-located boxes:")
        for k, v in s["confusion"].most_common(6):
            print(f"    {k:<22}{v:>6,}")

    print("=" * W)

    if args.json_out:
        out = {
            "model": args.model, "conf": args.conf, "imgsz": args.imgsz,
            "total_gt_boxes": s["total_gt_boxes"],
            "scored_gt_boxes": scored_gt,
            "unmatchable_boxes": s["unmatchable_boxes"],
            "coverage_gap_pct": round(100 * gap, 2),
            "localisation": {k: s["agn"][k] for k in ("TP", "FP", "FN")},
            "with_class": {k: s["aware"][k] for k in ("TP", "FP", "FN")},
            "gt_per_canonical": dict(s["gt_per_canon"]),
            "hit_per_canonical": dict(s["hit_agn_canon"]),
            "hit_with_class_per_canonical": dict(s["hit_aware_canon"]),
            "gt_per_dataset_class": dict(s["gt_per_dataset"]),
            "hit_per_dataset_class": dict(s["hit_agn_dataset"]),
            "label_errors": dict(s["confusion"]),
            "dataset_map": DATASET_MAP,
            "assumed_map": ASSUMED_MAP,
            "unmatchable_classes": sorted(UNMATCHABLE),
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
