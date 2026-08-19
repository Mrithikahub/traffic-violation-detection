# Baseline Report — Vehicle Detection + Tracking

Pre-fine-tuning baseline for the detection/tracking module, measured over the **whole** UCSD TrafficDB set. Re-run `batch_run.py` after any change and compare against these numbers.

- **Generated:** 2026-08-12 22:40
- **Config:** `yolo11s.pt`, `imgsz=640`, `conf=0.1`, `bytetrack.yaml`, `start_frame=1`, `device=cpu`
- **Clips:** 254 · **Frames:** 13,063 · **Runtime:** 0:52:50

Every number here comes from the model's own output. The dataset has **no ground-truth boxes**, so nothing below measures recall — a vehicle that is never detected is invisible to all of it. Treat these as a reproducible baseline to compare against, not as accuracy.

## Headline

| Metric | Value |
|---|---|
| Total detections | 78,976 |
| Total tracked vehicles | 6,858 |
| Detections per frame (pooled) | 6.05 |
| Mean confidence | 0.486 |
| Detections below 0.40 confidence | 34.6% |
| Frames with zero detections | 170 (1.3%) |
| Tracks lasting ≤2 frames | 3,004 (43.8%) |
| Tracks changing class mid-track | 1,108 (16.2%) |

## Class distribution

Per detection, and per tracked vehicle after the majority vote. The vehicle column is the one to quote — it counts each physical vehicle once.

| Class | Detections | % | Vehicles | % |
|---|---|---|---|---|
| `car` | 64,953 | 82.2% | 5,421 | 79.0% |
| `bike` | 28 | 0.0% | 14 | 0.2% |
| `bus` | 2,810 | 3.6% | 266 | 3.9% |
| `truck` | 11,185 | 14.2% | 1,157 | 16.9% |

## Class agreement

`class_agreement` is the fraction of a track's frames that voted for its final class, over all 6,858 tracked vehicles.

| Metric | Value |
|---|---|
| Mean agreement | 0.962 |
| Median agreement | 1.000 |
| Unanimous (=1.0) | 5,750 (83.8%) |
| Below 0.6 (unreliable) | 208 (3.0%) |

Most common confusions, counted per track:

| Confusion | Tracks |
|---|---|
| `car <-> truck` | 827 |
| `bus <-> truck` | 244 |
| `bus <-> car` | 110 |
| `bike <-> car` | 8 |

## Detections per frame, across clips

`cctv052x2004080516x01638` — the clip inspected frame by frame, where roughly 40 vehicles were visible against 14.5 detected — sits at **14.5 detections/frame**, higher than 250 of the other 253 clips, putting it in the top 1% by detection density. Clips range from 1.06 to 16.85.

So the shortfall seen there is not a low outlier — it is at or above typical for this dataset. But note this metric counts what the model *found*, not what was present: with no ground truth, the size of the gap can only be established by eye, clip by clip.

| Metric | Detections/frame |
|---|---|
| Pooled (all detections / all frames) | 6.05 |
| Mean across clips | 6.06 |
| Median across clips | 4.62 |
| 10th percentile | 2.36 |
| 90th percentile | 11.71 |
| Min / Max clip | 1.06 / 16.85 |

Highest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080616x00047` | 16.85 | 41 |
| `cctv052x2004080616x00049` | 14.84 | 39 |
| `cctv052x2004080516x01641` | 14.76 | 35 |
| `cctv052x2004080516x01638` | 14.5 | 44 |
| `cctv052x2004080616x00046` | 13.73 | 38 |

Lowest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080610x01878` | 1.06 | 2 |
| `cctv052x2004080609x01876` | 1.21 | 5 |
| `cctv052x2004080607x01842` | 1.37 | 0 |
| `cctv052x2004080607x01844` | 1.46 | 4 |
| `cctv052x2004080612x01910` | 1.6 | 1 |

## Track fragmentation

| Metric | Value |
|---|---|
| Tracked vehicles | 6,858 |
| Mean track length (frames) | 10.5 |
| Median track length | 4 |
| 90th percentile | 31 |
| Longest track | 52 |
| Single-frame tracks | 2,229 (32.5%) |
| Tracks ≤2 frames | 3,004 (43.8%) |
| Detections with no track ID | 6,992 (8.9%) |

Clips are ~52 frames (5.2 s at 10 fps), so a track can never exceed ~52. A short track is either a vehicle that genuinely entered or left the frame, or an ID that was dropped and re-issued. These two cannot be separated without ground truth.

## By condition

From the dataset's own labels in `info.txt`. Density matters when reading detections/frame — a light-traffic clip *should* have fewer.

| Condition | Clips | Dets/frame | Tracks | Class-flip % | ≤2-frame tracks % | Mean conf |
|---|---|---|---|---|---|---|
| `traffic:heavy` | 44 | 10.43 | 1,381 | 14.7% | 29.8% | 0.481 |
| `traffic:light` | 165 | 3.68 | 3,778 | 16.6% | 53.8% | 0.441 |
| `traffic:medium` | 45 | 10.5 | 1,699 | 16.2% | 32.9% | 0.516 |
| `weather:clear` | 51 | 7.49 | 1,628 | 14.9% | 41.2% | 0.491 |
| `weather:overcast` | 125 | 7.16 | 3,815 | 17.4% | 40.6% | 0.494 |
| `weather:rain` | 78 | 3.35 | 1,415 | 14.3% | 55.5% | 0.390 |

## Reproducing

```bash
python batch_run.py
```

Resumable — finished clips are skipped. `--aggregate-only` rebuilds this report from existing per-clip results without re-running the model. Per-clip outputs are in `results_v11/<clip>/`; the schema is documented in `OUTPUT_FORMAT.md`.
