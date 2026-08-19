"""
Run the detection + tracking pipeline over every clip in the dataset.

Reuses process_video() from detect_track.py, so the per-clip outputs are byte
for byte what a single-video run produces. Loads the model once and resets the
tracker between clips rather than reloading weights 254 times.

Resumable: a clip is considered done when its <clip>_stats.json exists. That
file is written last, so a clip interrupted midway is redone on the next run.

Usage:
    python batch_run.py                       # run everything, resume as needed
    python batch_run.py --limit 5             # smoke test on the first 5 clips
    python batch_run.py --aggregate-only      # rebuild the report from existing results
    python batch_run.py --force               # ignore existing results, redo all
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from detect_track import (
    AGREEMENT_FLOOR,
    CANONICAL_CLASSES,
    add_pipeline_args,
    build_tracker_cfg,
    process_video,
    reset_tracker,
)

STATS_NAME = "{stem}_stats.json"

# The clip that was checked by eye (boxes cropped per track, ~40 vehicles visible
# vs 17 detected). The report locates it within the dataset-wide distribution so
# that observation can be judged against the whole set rather than assumed typical.
REFERENCE_CLIP = "cctv052x2004080516x01638"


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch detection + tracking over a folder of clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-dir", default="data/video",
                   help="Folder of input clips.")
    p.add_argument("--pattern", default="*.avi", help="Glob for clip files.")
    p.add_argument("--out-dir", default="results",
                   help="Root output folder. One subfolder per clip.")
    p.add_argument("--info", default="data/info.txt",
                   help="Dataset metadata (traffic density / weather) used to break the "
                        "aggregates down by condition. Skipped if missing.")
    p.add_argument("--report", default="BASELINE_REPORT.md",
                   help="Markdown summary to write at the end.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many clips (after skipping finished ones).")
    p.add_argument("--force", action="store_true",
                   help="Reprocess clips that already have results.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Do not run the model; just rebuild the aggregate from the "
                        "per-clip stats already on disk.")
    p.add_argument("--save-video", action="store_true",
                   help="Also write an annotated mp4 per clip. This is ~2 MB per clip, "
                        "so roughly 500 MB across the dataset.")
    add_pipeline_args(p)
    # The whole dataset has a corrupted first frame, so default to skipping it
    # here even though detect_track.py defaults to 0 for arbitrary videos.
    p.set_defaults(start_frame=1)
    return p.parse_args()


def log(msg, fh=None):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def load_conditions(info_path):
    """clip stem -> {traffic density, weather} from the dataset's label file.

    Handles both the day set's info.txt and the night set's listing.txt. They
    share a column layout (name, date, time, view, day/night, weather, start
    frame, frames, traffic class, comments) but NOT a delimiter: listing.txt
    mixes tabs and runs of spaces, so splitting on "\\t" silently drops 5 of its
    21 rows. Splitting on any whitespace is safe because only the trailing
    comments column can contain spaces, and we never read past column 8.
    """
    path = Path(info_path)
    if not path.exists():
        return {}
    out = {}
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) >= 9:
            out[f[0]] = {"traffic": f[8].strip(), "weather": f[5].strip()}
        else:
            skipped += 1
    if skipped:
        print(f"WARNING: {skipped} unparseable row(s) in {path}; those clips will "
              f"have no condition labels in the aggregate.", flush=True)
    return out


def run_batch(args):
    video_dir = Path(args.video_dir)
    clips = sorted(video_dir.glob(args.pattern))
    if not clips:
        sys.exit(f"ERROR: no files matching {args.pattern} in {video_dir}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pending = []
    done_already = 0
    for c in clips:
        stats_file = out_root / c.stem / STATS_NAME.format(stem=c.stem)
        if stats_file.exists() and not args.force:
            done_already += 1
        else:
            pending.append(c)
    if args.limit is not None:
        pending = pending[:args.limit]

    with open(out_root / "batch_log.txt", "a", encoding="utf-8") as fh:
        log(f"found {len(clips)} clips in {video_dir}", fh)
        log(f"already complete: {done_already}   to process now: {len(pending)}", fh)
        if not pending:
            log("nothing to do; going straight to aggregation", fh)
            return

        log(f"model={args.model} imgsz={args.imgsz} conf={args.conf} "
            f"device={args.device} start_frame={args.start_frame}", fh)
        model = YOLO(args.model)
        tracker_cfg = build_tracker_cfg(args, out_root)
        log(f"tracker cfg: {Path(tracker_cfg).name}", fh)

        conditions = load_conditions(args.info) if args.info else {}
        t_start = time.time()
        durations = []
        failures = []

        for i, clip in enumerate(pending, 1):
            clip_dir = out_root / clip.stem
            t0 = time.time()
            try:
                # Without this each clip would inherit the previous clip's track
                # IDs and Kalman state -- see reset_tracker's docstring.
                reset_tracker(model)
                res = process_video(clip, args, clip_dir, model, tracker_cfg,
                                    save_video=args.save_video)
            except Exception as exc:                      # noqa: BLE001
                failures.append((clip.stem, repr(exc)))
                log(f"[{i}/{len(pending)}] {clip.stem}  FAILED: {exc!r}", fh)
                continue

            stats = res["stats"]
            stats["conditions"] = conditions.get(clip.stem, {})
            # Written last: its presence is what marks the clip complete.
            with open(clip_dir / STATS_NAME.format(stem=clip.stem), "w",
                      encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

            dt = time.time() - t0
            durations.append(dt)
            eta = timedelta(seconds=int(np.mean(durations) * (len(pending) - i)))
            log(f"[{i}/{len(pending)}] {clip.stem}  "
                f"dets={stats['total_detections']:>4}  "
                f"{stats.get('dets_per_frame_mean', 0):>5.1f}/fr  "
                f"tracks={stats.get('n_tracks', 0):>3}  "
                f"{dt:>5.1f}s  ETA {eta}", fh)

        log(f"processed {len(durations)} clips in "
            f"{timedelta(seconds=int(time.time() - t_start))}", fh)
        if failures:
            log(f"{len(failures)} FAILURES:", fh)
            for stem, err in failures:
                log(f"    {stem}: {err}", fh)


def collect_stats(out_root):
    stats = []
    for f in sorted(Path(out_root).glob(f"*/*_stats.json")):
        with open(f, encoding="utf-8") as fh:
            stats.append(json.load(fh))
    return stats


def aggregate(stats):
    """Pool the per-clip stats into dataset-wide numbers."""
    a = {
        "clips": len(stats),
        "frames": sum(s["frames_processed"] for s in stats),
        "frames_empty": sum(s["frames_empty"] for s in stats),
        "detections": sum(s["total_detections"] for s in stats),
        "processing_seconds": round(sum(s["processing_seconds"] for s in stats), 1),
    }

    # Detections per frame: per-clip means, so each clip counts once.
    dpf = np.array([s.get("dets_per_frame_mean", 0.0) for s in stats])
    a["dets_per_frame"] = {
        "mean_of_clips": round(float(dpf.mean()), 2),
        "median_of_clips": round(float(np.median(dpf)), 2),
        "min": round(float(dpf.min()), 2),
        "max": round(float(dpf.max()), 2),
        "p10": round(float(np.percentile(dpf, 10)), 2),
        "p90": round(float(np.percentile(dpf, 90)), 2),
        "pooled": round(a["detections"] / a["frames"], 2) if a["frames"] else 0,
    }

    # Confidence, weighted by detections so big clips are not under-counted.
    tot = a["detections"]
    if tot:
        a["conf_mean_weighted"] = round(
            sum(s.get("conf_mean", 0) * s["total_detections"] for s in stats) / tot, 4)
        a["conf_pct_below_040"] = round(
            sum(s.get("conf_pct_below_040", 0) * s["total_detections"]
                for s in stats) / tot, 2)

    det_cls = Counter()
    veh_cls = Counter()
    for s in stats:
        det_cls.update(s.get("class_counts", {}))
        veh_cls.update(s.get("vehicle_class_counts", {}))
    a["class_counts_detections"] = {c: det_cls.get(c, 0) for c in CANONICAL_CLASSES}
    a["class_counts_vehicles"] = {c: veh_cls.get(c, 0) for c in CANONICAL_CLASSES}

    # Tracking / fragmentation, pooled exactly from the retained per-clip lists.
    lengths = np.array([x for s in stats for x in s.get("track_lengths", [])])
    a["tracks"] = int(sum(s.get("n_tracks", 0) for s in stats))
    a["tracked_detections"] = int(lengths.sum()) if len(lengths) else 0
    a["untracked_detections"] = sum(s.get("untracked_detections", 0) for s in stats)
    if len(lengths):
        a["track_len"] = {
            "mean": round(float(lengths.mean()), 2),
            "median": int(np.median(lengths)),
            "p90": int(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        }
        a["tracks_le_2_frames"] = int((lengths <= 2).sum())
        a["tracks_le_2_frames_pct"] = round(100 * float((lengths <= 2).mean()), 2)
        a["tracks_single_frame"] = int((lengths == 1).sum())

    a["tracks_changing_class"] = sum(s.get("tracks_changing_class", 0) for s in stats)
    a["tracks_changing_class_pct"] = (
        round(100 * a["tracks_changing_class"] / a["tracks"], 2) if a["tracks"] else 0)
    pairs = Counter()
    for s in stats:
        pairs.update(s.get("class_flip_pairs", {}))
    a["class_flip_pairs"] = dict(pairs.most_common())

    # class_agreement across every tracked vehicle in the dataset.
    agree = np.array([x for s in stats for x in s.get("agreements", [])])
    if len(agree):
        a["agreement"] = {
            "vehicles": int(len(agree)),
            "mean": round(float(agree.mean()), 4),
            "median": round(float(np.median(agree)), 4),
            "unanimous": int((agree >= 0.999).sum()),
            "unanimous_pct": round(100 * float((agree >= 0.999).mean()), 2),
            "below_floor": int((agree < AGREEMENT_FLOOR).sum()),
            "below_floor_pct": round(100 * float((agree < AGREEMENT_FLOOR).mean()), 2),
        }

    # Breakdown by the dataset's own condition labels, so "17 detections/frame"
    # can be read against how much traffic was actually present.
    by = defaultdict(list)
    for s in stats:
        cond = s.get("conditions") or {}
        if cond.get("traffic"):
            by[("traffic", cond["traffic"])].append(s)
        if cond.get("weather"):
            by[("weather", cond["weather"])].append(s)
    a["by_condition"] = {}
    for (kind, value), group in sorted(by.items()):
        g_dpf = np.array([s.get("dets_per_frame_mean", 0.0) for s in group])
        g_tracks = sum(s.get("n_tracks", 0) for s in group)
        g_flip = sum(s.get("tracks_changing_class", 0) for s in group)
        g_len = np.array([x for s in group for x in s.get("track_lengths", [])])
        a["by_condition"][f"{kind}:{value}"] = {
            "clips": len(group),
            "dets_per_frame_mean": round(float(g_dpf.mean()), 2),
            "tracks": g_tracks,
            "tracks_changing_class_pct": round(100 * g_flip / g_tracks, 2) if g_tracks else 0,
            "tracks_le_2_frames_pct": (
                round(100 * float((g_len <= 2).mean()), 2) if len(g_len) else 0),
            "conf_mean": round(float(np.mean([s.get("conf_mean", 0) for s in group])), 4),
        }
    return a


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def write_report(a, stats, args, path):
    cfg = f"`{args.model}`, `imgsz={args.imgsz}`, `conf={args.conf}`, `{args.tracker}`"
    worst = sorted(stats, key=lambda s: s.get("dets_per_frame_mean", 0))[:5]
    best = sorted(stats, key=lambda s: -s.get("dets_per_frame_mean", 0))[:5]
    dcls = a["class_counts_detections"]
    vcls = a["class_counts_vehicles"]
    dtot = max(sum(dcls.values()), 1)
    vtot = max(sum(vcls.values()), 1)

    L = []
    w = L.append
    w("# Baseline Report — Vehicle Detection + Tracking")
    w("")
    w("Pre-fine-tuning baseline for the detection/tracking module, measured over the "
      "**whole** UCSD TrafficDB set. Re-run `batch_run.py` after any change and compare "
      "against these numbers.")
    w("")
    w(f"- **Generated:** {datetime.now():%Y-%m-%d %H:%M}")
    w(f"- **Config:** {cfg}, `start_frame={args.start_frame}`, `device={args.device}`")
    w(f"- **Clips:** {a['clips']} · **Frames:** {a['frames']:,} · "
      f"**Runtime:** {timedelta(seconds=int(a['processing_seconds']))}")
    w("")
    w("Every number here comes from the model's own output. The dataset has **no "
      "ground-truth boxes**, so nothing below measures recall — a vehicle that is never "
      "detected is invisible to all of it. Treat these as a reproducible baseline to "
      "compare against, not as accuracy.")
    w("")
    w("## Headline")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Total detections | {a['detections']:,} |")
    w(f"| Total tracked vehicles | {a['tracks']:,} |")
    w(f"| Detections per frame (pooled) | {a['dets_per_frame']['pooled']} |")
    w(f"| Mean confidence | {a.get('conf_mean_weighted', 0):.3f} |")
    w(f"| Detections below 0.40 confidence | {a.get('conf_pct_below_040', 0):.1f}% |")
    w(f"| Frames with zero detections | {a['frames_empty']:,} "
      f"({pct(a['frames_empty'], a['frames'])}) |")
    w(f"| Tracks lasting ≤2 frames | {a.get('tracks_le_2_frames', 0):,} "
      f"({a.get('tracks_le_2_frames_pct', 0):.1f}%) |")
    w(f"| Tracks changing class mid-track | {a['tracks_changing_class']:,} "
      f"({a['tracks_changing_class_pct']:.1f}%) |")
    w("")
    w("## Class distribution")
    w("")
    w("Per detection, and per tracked vehicle after the majority vote. The vehicle "
      "column is the one to quote — it counts each physical vehicle once.")
    w("")
    w("| Class | Detections | % | Vehicles | % |")
    w("|---|---|---|---|---|")
    for c in CANONICAL_CLASSES:
        w(f"| `{c}` | {dcls[c]:,} | {100 * dcls[c] / dtot:.1f}% | "
          f"{vcls[c]:,} | {100 * vcls[c] / vtot:.1f}% |")
    w("")
    # State exactly what the counts show: detection-level and vehicle-level
    # totals can disagree, since a lone stray detection is voted away.
    if not vcls.get("bike"):
        if not dcls.get("bike"):
            w("**`bike` is never emitted.** Zero motorcycle or bicycle detections across "
              "the entire dataset.")
        else:
            w(f"**`bike` survives in no vehicle.** Just {dcls['bike']} bike "
              f"detection(s) dataset-wide, none of which won its track's majority vote, "
              f"so the per-vehicle count is 0. Treat any `bike` in the per-frame CSV as "
              f"noise.")
        w("")
        w("This is highway footage, so the class is effectively unused. Downstream "
          "modules should still handle it, but must not depend on it.")
        w("")
    w("## Class agreement")
    w("")
    ag = a.get("agreement", {})
    w(f"`class_agreement` is the fraction of a track's frames that voted for its final "
      f"class, over all {ag.get('vehicles', 0):,} tracked vehicles.")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Mean agreement | {ag.get('mean', 0):.3f} |")
    w(f"| Median agreement | {ag.get('median', 0):.3f} |")
    w(f"| Unanimous (=1.0) | {ag.get('unanimous', 0):,} "
      f"({ag.get('unanimous_pct', 0):.1f}%) |")
    w(f"| Below {AGREEMENT_FLOOR} (unreliable) | {ag.get('below_floor', 0):,} "
      f"({ag.get('below_floor_pct', 0):.1f}%) |")
    w("")
    if a["class_flip_pairs"]:
        w("Most common confusions, counted per track:")
        w("")
        w("| Confusion | Tracks |")
        w("|---|---|")
        for pair, n in list(a["class_flip_pairs"].items())[:6]:
            w(f"| `{pair}` | {n:,} |")
        w("")
    w("## Detections per frame, across clips")
    w("")
    d = a["dets_per_frame"]
    # State where the manually-inspected clip actually falls rather than asserting
    # a conclusion the numbers might not support.
    ref = next((s for s in stats if s["clip"] == REFERENCE_CLIP), None)
    if ref:
        rv = ref.get("dets_per_frame_mean", 0)
        rank = sum(1 for s in stats if s.get("dets_per_frame_mean", 0) <= rv)
        w(f"`{REFERENCE_CLIP}` — the clip inspected frame by frame, where roughly 40 "
          f"vehicles were visible against {rv:.1f} detected — sits at "
          f"**{rv:.1f} detections/frame**, higher than {rank - 1} of the other "
          f"{len(stats) - 1} clips, putting it in the top "
          f"{100 - 100 * (rank - 1) / max(len(stats) - 1, 1):.0f}% by detection "
          f"density. Clips range from {d['min']} to {d['max']}.")
        w("")
        w("So the shortfall seen there is not a low outlier — it is at or above typical "
          "for this dataset. But note this metric counts what the model *found*, not "
          "what was present: with no ground truth, the size of the gap can only be "
          "established by eye, clip by clip.")
    else:
        w(f"Clips range from {d['min']} to {d['max']} detections/frame. This counts what "
          f"the model found, not what was present — with no ground truth the true miss "
          f"rate cannot be derived from these numbers.")
    w("")
    w("| Metric | Detections/frame |")
    w("|---|---|")
    w(f"| Pooled (all detections / all frames) | {d['pooled']} |")
    w(f"| Mean across clips | {d['mean_of_clips']} |")
    w(f"| Median across clips | {d['median_of_clips']} |")
    w(f"| 10th percentile | {d['p10']} |")
    w(f"| 90th percentile | {d['p90']} |")
    w(f"| Min / Max clip | {d['min']} / {d['max']} |")
    w("")
    w("Highest-density clips:")
    w("")
    w("| Clip | Dets/frame | Tracks |")
    w("|---|---|---|")
    for s in best:
        w(f"| `{s['clip']}` | {s.get('dets_per_frame_mean', 0)} | {s.get('n_tracks', 0)} |")
    w("")
    w("Lowest-density clips:")
    w("")
    w("| Clip | Dets/frame | Tracks |")
    w("|---|---|---|")
    for s in worst:
        w(f"| `{s['clip']}` | {s.get('dets_per_frame_mean', 0)} | {s.get('n_tracks', 0)} |")
    w("")
    w("## Track fragmentation")
    w("")
    tl = a.get("track_len", {})
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Tracked vehicles | {a['tracks']:,} |")
    w(f"| Mean track length (frames) | {tl.get('mean', 0)} |")
    w(f"| Median track length | {tl.get('median', 0)} |")
    w(f"| 90th percentile | {tl.get('p90', 0)} |")
    w(f"| Longest track | {tl.get('max', 0)} |")
    w(f"| Single-frame tracks | {a.get('tracks_single_frame', 0):,} "
      f"({pct(a.get('tracks_single_frame', 0), a['tracks'])}) |")
    w(f"| Tracks ≤2 frames | {a.get('tracks_le_2_frames', 0):,} "
      f"({a.get('tracks_le_2_frames_pct', 0):.1f}%) |")
    w(f"| Detections with no track ID | {a['untracked_detections']:,} "
      f"({pct(a['untracked_detections'], a['detections'])}) |")
    w("")
    w("Clips are ~52 frames (5.2 s at 10 fps), so a track can never exceed ~52. A short "
      "track is either a vehicle that genuinely entered or left the frame, or an ID that "
      "was dropped and re-issued. These two cannot be separated without ground truth.")
    w("")
    if a["by_condition"]:
        w("## By condition")
        w("")
        w("From the dataset's own labels in `info.txt`. Density matters when reading "
          "detections/frame — a light-traffic clip *should* have fewer.")
        w("")
        w("| Condition | Clips | Dets/frame | Tracks | Class-flip % | ≤2-frame tracks % | Mean conf |")
        w("|---|---|---|---|---|---|---|")
        for k, v in a["by_condition"].items():
            w(f"| `{k}` | {v['clips']} | {v['dets_per_frame_mean']} | {v['tracks']:,} | "
              f"{v['tracks_changing_class_pct']:.1f}% | "
              f"{v['tracks_le_2_frames_pct']:.1f}% | {v['conf_mean']:.3f} |")
        w("")
    w("## Reproducing")
    w("")
    w("```bash")
    w("python batch_run.py")
    w("```")
    w("")
    w("Resumable — finished clips are skipped. `--aggregate-only` rebuilds this report "
      "from existing per-clip results without re-running the model. Per-clip outputs are "
      f"in `{args.out_dir}/<clip>/`; the schema is documented in `OUTPUT_FORMAT.md`.")
    w("")

    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    args = parse_args()
    if not args.aggregate_only:
        run_batch(args)

    stats = collect_stats(args.out_dir)
    if not stats:
        sys.exit(f"ERROR: no per-clip stats found under {args.out_dir}")

    a = aggregate(stats)
    agg_path = Path(args.out_dir) / "aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2)
    write_report(a, stats, args, args.report)

    print(f"\nAggregated {a['clips']} clips  |  {a['detections']:,} detections  |  "
          f"{a['tracks']:,} tracked vehicles")
    print(f"Wrote {agg_path}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
