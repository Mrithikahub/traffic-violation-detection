# Traffic Violation Detection — Vehicle Detection & Tracking Module

Vehicle detection, multi-object tracking, and four violation-detection layers for
traffic CCTV footage.

This repository is the **detection and tracking half** of a larger traffic
violation system. It turns video into a stable, documented per-frame table of
tracked vehicles, then layers violation logic on top of that table. Downstream
modules (number-plate recognition, violation rule engines) consume the CSV
contract in **[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md)** — that document is the
interface, and it is the one to read first if you are building on this.

---

## What's built

**Detection + tracking** — `detect_track.py`

- YOLOv8s (Ultralytics) per-frame detection, CPU-capable.
- ByteTrack for persistent `vehicle_id`s across frames.
- Five canonical classes: `car`, `bike`, `bus`, `truck`, `rickshaw`.
- Accepts video files or numbered image sequences.
- Outputs per-frame CSV/JSON, a per-vehicle majority-voted class table, and an
  optional annotated video.

**Four violation layers**, each a standalone script that reads the tracks CSV
and writes its own CSV + metadata. They do not import each other and can be run
in any order, or not at all:

| Layer | Script | Flags |
|---|---|---|
| Speed estimation + speeding | `speed_violation.py` | Homography-calibrated km/h against a configurable limit |
| Wrong-side driving | `wrong_lane.py` | Motion opposing the expected direction of a carriageway zone |
| Lane changes | `lane_change.py` | Settled lane-to-lane transitions, jitter-rejected |
| Tailgating | `tailgating.py` | Following gap below 1.5 follower-lengths |

**Combined renderer** — `combine_violations.py` draws all four layers onto one
video, with multiple simultaneous flags per vehicle shown as nested boxes.

**Supporting tools** — batch processing over hundreds of clips (`batch_run.py`),
ground-truth evaluation (`eval_detection.py`), homography calibration
(`calibrate_homography.py`), dataset conversion and fine-tuning
(`prepare_dataset.py`, `train_finetune.py`), and cross-domain A/B testing
(`generalisation_test.py`).

---

## Install

CPU-only machine:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On a CUDA machine, drop the `--index-url` line and install torch normally.
Stock YOLO weights download automatically on first run.

---

## How to run

### 1. Detection + tracking

```bash
python detect_track.py --video data/video/teammate_video.mp4 --model yolov8s.pt --out-dir outputs_demo --save-video
```

This produces `outputs_demo/teammate_video_tracks.csv` — the input to
everything below, and the file described in
[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

### 2. Violation layers

Each layer takes the tracks CSV and is independent of the others:

```bash
python speed_violation.py --video data/video/teammate_video.mp4 --tracks outputs_demo/teammate_video_tracks.csv --homography outputs_demo/homography.npy --speed-limit 60 --min-ref-y 268 --max-plausible-kmh 100 --out-dir outputs_calib
```

```bash
python wrong_lane.py --video data/video/teammate_video.mp4 --tracks outputs_demo/teammate_video_tracks.csv --zones outputs_demo/zones.json --out-dir outputs_calib
```

```bash
python lane_change.py --video data/video/teammate_video.mp4 --tracks outputs_demo/teammate_video_tracks.csv --homography outputs_demo/homography.npy --lanes outputs_demo/lanes.json --out-dir outputs_calib
```

```bash
python tailgating.py --video data/video/teammate_video.mp4 --tracks outputs_demo/teammate_video_tracks.csv --out-dir outputs_calib
```

### 3. Combined demo video

```bash
python combine_violations.py --video data/video/teammate_video.mp4 --tracks outputs_demo/teammate_video_tracks.csv --speed-csv outputs_calib/teammate_video_tracks_speed.csv --wrong-lane-csv outputs_calib/teammate_video_wrong_lane.csv --lane-change-csv outputs_calib/teammate_video_lane_change.csv --tailgating-csv outputs_calib/teammate_video_tailgating.csv --out-dir outputs_calib --ignore-region "0,355,325,575"
```

### 4. Batch over a whole dataset

```bash
python batch_run.py --video-dir data/video --pattern "*.avi" --out-dir results --report BASELINE_REPORT.md
```

Resumable — completed clips are skipped on restart, and `--aggregate-only`
rebuilds the summary from per-clip stats already on disk without re-running the
model.

---

## For teammates building on this

Read **[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md)**. It is the contract, and if the
schema ever changes it changes there first. The short version:

`<name>_tracks.csv` has exactly eight columns, in this order:

```csv
frame_id,vehicle_id,class,x1,y1,x2,y2,confidence
1,1,car,142.31,88.04,171.55,110.92,0.8421
1,2,truck,201.77,55.19,238.40,92.63,0.6013
```

Four things that will bite you if you assume otherwise:

- **`frame_id` has gaps.** Frames with no detections produce no rows.
- **`vehicle_id` is `-1`** for detections the tracker did not associate, and
  IDs are unique *within one video only*.
- **Per-row `class` flips.** For "what kind of vehicle is this", read
  `<name>_vehicles.csv`, which carries a majority-voted class per track.
- **`rickshaw` only appears from the fine-tuned weights.** Handle all five
  classes; do not assume it is present.

Boxes are absolute pixels in the source resolution, so you can crop directly
with `frame[int(y1):int(y2), int(x1):int(x2)]`.

Each violation layer appends its own columns to a copy of this table rather
than modifying it, so the base schema stays stable.

---

## Read this before trusting any number

**[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)** documents every assumption in
the pipeline, graded by severity. The load-bearing ones:

- **Speed rests on assumed scale.** The homography is built from IRC standard
  road dimensions (3.5 m lane, 3 m + 4.5 m dash pitch), not site survey. Speeds
  are internally consistent and plausible, not evidential.
- **Speed is only measured inside the calibrated band** (`y < 268`). Vehicles
  outside it are tracked but not speed-judged.
- **The fine-tuned model is specialised, not simply better.** On Western
  motorway footage its truck detection collapses (11.9% → 1.1% of detections)
  and it hallucinates `rickshaw` at ~1.1%. Use stock `yolov8s.pt` outside
  Indian urban footage.
- **Thresholds are engineering choices, not legal standards.** The 1.5-length
  tailgating gap and the 60 km/h limit are demo defaults.
- **No ground truth on the demo video.** Violation counts are unvalidated.

---

## Reports

| Document | Contents |
|---|---|
| [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) | **The schema contract.** Start here. |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Every assumption, graded by severity |
| [BASELINE_REPORT.md](BASELINE_REPORT.md) | YOLOv8s over 254 UCSD daytime clips |
| [BASELINE_NIGHT.md](BASELINE_NIGHT.md) | Night / low-light behaviour |
| [BASELINE_V11.md](BASELINE_V11.md) | YOLOv11s over the same 254 clips, for comparison against the v8s baseline |
| [FINETUNE_RESULTS.md](FINETUNE_RESULTS.md) | BMD-45 fine-tuning, before/after |
| [GENERALISATION_REPORT.md](GENERALISATION_REPORT.md) | 39-video, 78-run cross-domain test |
| [HELMET_DETECTION_EVALUATION.md](HELMET_DETECTION_EVALUATION.md) | Why helmet detection was not shipped |
| [VIOLATION_SCAN.md](VIOLATION_SCAN.md) | Feasibility screening for further layers |

Headline baseline (stock YOLOv8s, 254 clips, 13,063 frames): 92,922 detections,
7,916 tracks, mean confidence 0.498. Fine-tuning on 800 Bengaluru CCTV images
lifted recall-with-correct-class from 51.2% to 87.7% on held-out Indian
footage — with the generalisation cost noted above.

---

## What is not in this repo

Datasets (~4.7 GB), model weights (~395 MB), rendered videos, and the raw
per-clip output of the 254-clip batch runs are excluded by `.gitignore` —
all are regenerable, and the distilled results are kept in the reports and in
`aggregate.json`. What *is* kept: all code, all reports, the calibration files
(`homography.npy`, `lanes.json`, `zones.json`), every layer's `_meta.json` of
thresholds and assumptions, and the reference layer outputs in `outputs_calib/`
so the schema can be checked against real data.

---

## Status

Detection, tracking, and all four violation layers are working end to end on
Indian urban CCTV footage. Helmet detection was evaluated and deliberately not
shipped — pretrained models failed at the head sizes available in this footage
(see [HELMET_DETECTION_EVALUATION.md](HELMET_DETECTION_EVALUATION.md)). The
U-turn scan in `violation_scan.py` is known broken and needs a straightness
metric; it is not used by any shipped layer.
