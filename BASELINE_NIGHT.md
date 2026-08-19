# Baseline Report — Vehicle Detection + Tracking

Pre-fine-tuning baseline for the detection/tracking module, measured over the **whole** UCSD TrafficDB set. Re-run `batch_run.py` after any change and compare against these numbers.

- **Generated:** 2026-08-06 23:17
- **Config:** `yolov8s.pt`, `imgsz=640`, `conf=0.1`, `bytetrack.yaml`, `start_frame=1`, `device=cpu`
- **Clips:** 21 · **Frames:** 1,085 · **Runtime:** 0:03:56

Every number here comes from the model's own output. The dataset has **no ground-truth boxes**, so nothing below measures recall — a vehicle that is never detected is invisible to all of it. Treat these as a reproducible baseline to compare against, not as accuracy.

## Headline

| Metric | Value |
|---|---|
| Total detections | 385 |
| Total tracked vehicles | 20 |
| Detections per frame (pooled) | 0.35 |
| Mean confidence | 0.237 |
| Detections below 0.40 confidence | 86.2% |
| Frames with zero detections | 798 (73.5%) |
| Tracks lasting ≤2 frames | 17 (85.0%) |
| Tracks changing class mid-track | 0 (0.0%) |

## Class distribution

Per detection, and per tracked vehicle after the majority vote. The vehicle column is the one to quote — it counts each physical vehicle once.

| Class | Detections | % | Vehicles | % |
|---|---|---|---|---|
| `car` | 384 | 99.7% | 20 | 100.0% |
| `bike` | 0 | 0.0% | 0 | 0.0% |
| `bus` | 0 | 0.0% | 0 | 0.0% |
| `truck` | 1 | 0.3% | 0 | 0.0% |

**`bike` is never emitted.** Zero motorcycle or bicycle detections across the entire dataset.

This is highway footage, so the class is effectively unused. Downstream modules should still handle it, but must not depend on it.

## Class agreement

`class_agreement` is the fraction of a track's frames that voted for its final class, over all 20 tracked vehicles.

| Metric | Value |
|---|---|
| Mean agreement | 1.000 |
| Median agreement | 1.000 |
| Unanimous (=1.0) | 20 (100.0%) |
| Below 0.6 (unreliable) | 0 (0.0%) |

## Detections per frame, across clips

Clips range from 0.02 to 1.38 detections/frame. This counts what the model found, not what was present — with no ground truth the true miss rate cannot be derived from these numbers.

| Metric | Detections/frame |
|---|---|
| Pooled (all detections / all frames) | 0.35 |
| Mean across clips | 0.36 |
| Median across clips | 0.24 |
| 10th percentile | 0.08 |
| 90th percentile | 0.69 |
| Min / Max clip | 0.02 / 1.38 |

Highest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080521x01718` | 1.38 | 8 |
| `cctv052x2004080521x01715` | 1.0 | 3 |
| `cctv052x2004080521x01717` | 0.69 | 3 |
| `cctv052x2004080621x00128` | 0.62 | 1 |
| `cctv052x2004080521x01716` | 0.56 | 3 |

Lowest-density clips:

| Clip | Dets/frame | Tracks |
|---|---|---|
| `cctv052x2004080622x00143` | 0.02 | 0 |
| `cctv052x2004080521x01719` | 0.06 | 0 |
| `cctv052x2004080622x00138` | 0.08 | 0 |
| `cctv052x2004080521x01713` | 0.12 | 0 |
| `cctv052x2004080621x00127` | 0.12 | 0 |

## Track fragmentation

| Metric | Value |
|---|---|
| Tracked vehicles | 20 |
| Mean track length (frames) | 2.05 |
| Median track length | 1 |
| 90th percentile | 4 |
| Longest track | 11 |
| Single-frame tracks | 13 (65.0%) |
| Tracks ≤2 frames | 17 (85.0%) |
| Detections with no track ID | 344 (89.4%) |

Clips are ~52 frames (5.2 s at 10 fps), so a track can never exceed ~52. A short track is either a vehicle that genuinely entered or left the frame, or an ID that was dropped and re-issued. These two cannot be separated without ground truth.

## By condition

From the dataset's own labels in `info.txt`. Density matters when reading detections/frame — a light-traffic clip *should* have fewer.

| Condition | Clips | Dets/frame | Tracks | Class-flip % | ≤2-frame tracks % | Mean conf |
|---|---|---|---|---|---|---|
| `traffic:heavy` | 3 | 0.88 | 14 | 0.0% | 78.6% | 0.293 |
| `traffic:light` | 12 | 0.24 | 3 | 0.0% | 100.0% | 0.178 |
| `traffic:medium` | 6 | 0.33 | 3 | 0.0% | 100.0% | 0.204 |
| `weather:overcast` | 17 | 0.36 | 18 | 0.0% | 83.3% | 0.205 |
| `weather:rain` | 4 | 0.35 | 2 | 0.0% | 100.0% | 0.187 |

## Reproducing

```bash
python batch_run.py
```

Resumable — finished clips are skipped. `--aggregate-only` rebuilds this report from existing per-clip results without re-running the model. Per-clip outputs are in `results_night/<clip>/`; the schema is documented in `OUTPUT_FORMAT.md`.
