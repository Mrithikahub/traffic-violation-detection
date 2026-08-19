# Baseline Report — Vehicle Detection + Tracking

Pre-fine-tuning baseline for the detection/tracking module, measured over the **whole** UCSD TrafficDB set. Re-run `batch_run.py` after any change and compare against these numbers.

- **Generated:** 2026-08-06 22:34
- **Config:** `yolov8s.pt`, `imgsz=640`, `conf=0.1`, `bytetrack.yaml`, `start_frame=1`, `device=cpu`
- **Clips:** 254 · **Frames:** 13,063 · **Runtime:** 0:53:54

Every number here comes from the model's own output. The dataset has **no ground-truth boxes**, so nothing below measures recall — a vehicle that is never detected is invisible to all of it. Treat these as a reproducible baseline to compare against, not as accuracy.

## Headline

| Metric | Value |
|---|---|
| Total detections | 92,922 |
| Total tracked vehicles | 7,916 |
| Detections per frame (pooled) | 7.11 |
| Mean confidence | 0.498 |
| Detections below 0.40 confidence | 31.6% |
| Frames with zero detections | 253 (1.9%) |
| Tracks lasting ≤2 frames | 3,344 (42.2%) |
| Tracks changing class mid-track | 1,432 (18.1%) |

## Class distribution

Per detection, and per tracked vehicle after the majority vote. The vehicle column is the one to quote — it counts each physical vehicle once.

| Class | Detections | % | Vehicles | % |
|---|---|---|---|---|
| `car` | 76,429 | 82.3% | 6,360 | 80.3% |
| `bike` | 1 | 0.0% | 0 | 0.0% |
| `bus` | 3,737 | 4.0% | 375 | 4.7% |
| `truck` | 12,755 | 13.7% | 1,181 | 14.9% |

**`bike` survives in no vehicle.** Just 1 bike detection(s) dataset-wide, none of which won its track's majority vote, so the per-vehicle count is 0. Treat any `bike` in the per-frame CSV as noise.

This is highway footage, so the class is effectively unused. Downstream modules should still handle it, but must not depend on it.

## Class agreement

`class_agreement` is the fraction of a track's frames that voted for its final class, over all 7,916 tracked vehicles.

| Metric | Value |
|---|---|
| Mean agreement | 0.955 |
| Median agreement | 1.000 |
| Unanimous (=1.0) | 6,484 (81.9%) |
| Below 0.6 (unreliable) | 292 (3.7%) |

Most common confusions, counted per track:

| Confusion | Tracks |
|---|---|
| `car <-> truck` | 1,116 |
| `bus <-> truck` | 271 |
| `bus <-> car` | 129 |
| `bike <-> car` | 1 |

## Detections per frame, across clips

`cctv052x2004080516x01638` — the clip inspected frame by frame, where roughly 40 vehicles were visible against 17.4 detected — sits at **17.4 detections/frame**, higher than 247 of the other 253 clips, putting it in the top 2% by detection density. Clips range from 0.6 to 19.42.

So the shortfall seen there is not a low outlier — it is at or above typical for this dataset. But note this metric counts what the model *found*, not what was present: with no ground truth, the size of the gap can only be established by eye, clip by clip.

| Metric | Detections/frame |
|---|---|
| Pooled (all detections / all frames) | 7.11 |
| Mean across clips | 7.13 |
| Median across clips | 4.82 |
| 10th percentile | 2.22 |
| 90th percentile | 15.11 |
| Min / Max clip | 0.6 / 19.42 |

Highest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080517x01659` | 19.42 | 33 |
| `cctv052x2004080616x00055` | 18.4 | 45 |
| `cctv052x2004080516x01641` | 18.02 | 45 |
| `cctv052x2004080516x01646` | 17.83 | 28 |
| `cctv052x2004080616x00047` | 17.63 | 45 |

Lowest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080611x01906` | 0.6 | 4 |
| `cctv052x2004080607x01842` | 0.85 | 0 |
| `cctv052x2004080611x01904` | 1.19 | 7 |
| `cctv052x2004080610x01878` | 1.23 | 3 |
| `cctv052x2004080609x01876` | 1.37 | 5 |

## Track fragmentation

| Metric | Value |
|---|---|
| Tracked vehicles | 7,916 |
| Mean track length (frames) | 10.9 |
| Median track length | 4 |
| 90th percentile | 34 |
| Longest track | 52 |
| Single-frame tracks | 2,389 (30.2%) |
| Tracks ≤2 frames | 3,344 (42.2%) |
| Detections with no track ID | 6,602 (7.1%) |

Clips are ~52 frames (5.2 s at 10 fps), so a track can never exceed ~52. A short track is either a vehicle that genuinely entered or left the frame, or an ID that was dropped and re-issued. These two cannot be separated without ground truth.

## By condition

From the dataset's own labels in `info.txt`. Density matters when reading detections/frame — a light-traffic clip *should* have fewer.

| Condition | Clips | Dets/frame | Tracks | Class-flip % | ≤2-frame tracks % | Mean conf |
|---|---|---|---|---|---|---|
| `traffic:heavy` | 44 | 13.9 | 1,710 | 18.6% | 27.2% | 0.513 |
| `traffic:light` | 165 | 3.84 | 4,246 | 16.0% | 53.7% | 0.425 |
| `traffic:medium` | 45 | 12.57 | 1,960 | 22.1% | 30.5% | 0.526 |
| `weather:clear` | 51 | 9.48 | 2,004 | 20.3% | 40.2% | 0.505 |
| `weather:overcast` | 125 | 8.49 | 4,492 | 17.6% | 38.6% | 0.498 |
| `weather:rain` | 78 | 3.4 | 1,420 | 16.6% | 56.8% | 0.363 |

## Reproducing

```bash
python batch_run.py
```

Resumable — finished clips are skipped. `--aggregate-only` rebuilds this report from existing per-clip results without re-running the model. Per-clip outputs are in `results/<clip>/`; the schema is documented in `OUTPUT_FORMAT.md`.
