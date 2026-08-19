# Vehicle Detection + Tracking — Output Format

Contract between the detection/tracking module and downstream modules
(violation detection, plate recognition). **If this schema changes, it changes
here first.**

Produced by `detect_track.py`. For an input video `<name>.avi` the module writes
into `--out-dir` (default `outputs/`):

| File | Contents |
|---|---|
| `<name>_tracks.csv` | One row per detection per frame. **The main output.** |
| `<name>_tracks.json` | Same rows, plus a `meta` block. |
| `<name>_vehicles.csv` | One row per tracked vehicle, with a majority-voted class. **Read this for "what kind of vehicle is this".** |
| `<name>_annotated.mp4` | Optional visual check (`--save-video`). Not an input to any module. |
| `_tracker_*.yaml` | Generated tracker config, written for reproducibility. Ignore it. |

---

## CSV schema

Exactly these eight columns, in this order, with a header row.

| Column | Type | Description |
|---|---|---|
| `frame_id` | int | 0-based index of the frame **in the source video**. Not a running counter — if `--start-frame 1` was used, the first row is `frame_id=1`. Frames with no detections produce no rows, so this column has gaps. |
| `vehicle_id` | int | Persistent ByteTrack ID. Stable across frames for the same vehicle. `-1` means the detection was not assigned to a track (see below). IDs are unique *within one video only*. |
| `class` | string | One of `car`, `bike`, `bus`, `truck`, `rickshaw`. Nothing else is ever emitted. **`rickshaw` only appears from our fine-tuned weights** — stock COCO weights have no such class. Handle all five; do not assume `rickshaw` is present. |
| `x1` | float | Left edge of the box, pixels, 2dp. |
| `y1` | float | Top edge, pixels. |
| `x2` | float | Right edge, pixels. |
| `y2` | float | Bottom edge, pixels. |
| `confidence` | float | Detector confidence, 0.0–1.0, 4dp. |

Box coordinates are **absolute pixels in the source video's resolution**, origin
top-left, `x1 < x2` and `y1 < y2`. They are not normalised and not scaled by
`--imgsz`. Crop directly with `frame[int(y1):int(y2), int(x1):int(x2)]`.

### Example

```csv
frame_id,vehicle_id,class,x1,y1,x2,y2,confidence
1,1,car,142.31,88.04,171.55,110.92,0.8421
1,2,truck,201.77,55.19,238.40,92.63,0.6013
2,1,car,143.02,91.77,172.88,114.60,0.8390
```

---

## Per-vehicle schema (`<name>_vehicles.csv`)

One row per `vehicle_id`. This exists because the per-frame `class` is noisy —
measured on this footage, 13–28% of tracks change class at least once while the
box stays correctly on the vehicle. **Do not classify a vehicle from one frame.**

| Column | Type | Description |
|---|---|---|
| `vehicle_id` | int | Matches `vehicle_id` in the per-frame CSV. |
| `class` | string | Majority vote across the track. Ties broken by summed confidence. |
| `class_agreement` | float | Fraction of the track's frames that voted for `class`. 1.0 = unanimous. **Treat anything below ~0.6 as an unreliable class.** |
| `n_frames` | int | How many frames the vehicle was tracked for. |
| `first_frame` / `last_frame` | int | Frame range of the track. |
| `duration_sec` | float | `n_frames / fps`. |
| `mean_confidence` / `max_confidence` | float | Detector confidence over the track. |

```csv
vehicle_id,class,class_agreement,n_frames,first_frame,last_frame,duration_sec,mean_confidence,max_confidence
1,truck,0.843,51,1,51,5.1,0.7712,0.8530
2,car,1.0,44,1,44,4.4,0.6903,0.7781
```

---

## JSON schema

```json
{
  "meta": {
    "source_video": "data/video/cctv052x2004080516x01638.avi",
    "width": 320, "height": 240, "fps": 10.0,
    "frames_processed": 52, "start_frame": 1,
    "model": "yolo11s.pt", "tracker": "bytetrack.yaml",
    "conf_threshold": 0.25, "iou_threshold": 0.5, "imgsz": 640,
    "classes": ["car", "bike", "bus", "truck"],
    "total_detections": 411, "unique_vehicle_ids": 27,
    "processing_seconds": 12.4
  },
  "detections": [
    { "frame_id": 1, "vehicle_id": 1, "class": "car",
      "x1": 142.31, "y1": 88.04, "x2": 171.55, "y2": 110.92,
      "confidence": 0.8421 }
  ]
}
```

`detections` holds the same records as the CSV rows, same field names, in frame
order. `meta` exists so you don't have to reopen the video to get `fps` or
resolution — everything needed to interpret the rows is in there.

---

## Things you need to know before building on this

**`vehicle_id = -1` is not an error.** ByteTrack withholds an ID until a track
is confirmed over a few frames, so the first appearance of a vehicle — and any
low-confidence detection — can land in the CSV without an ID. Decide explicitly
what your module does with these rows; the usual choice is to skip them.

**IDs are per-video.** ID 7 in two different videos is two different vehicles.
Key on `(video_name, vehicle_id)` if you aggregate across clips.

**IDs are not contiguous.** They start at 1 and increment, but lost tracks burn
IDs, so `unique_vehicle_ids` is not `max(vehicle_id)`.

**A track's class can change between frames.** The class is a per-frame
prediction, not a track-level property. Use `<name>_vehicles.csv` rather than
re-deriving this yourself, and check `class_agreement` before relying on it.
`car`↔`truck` is by far the most common confusion, then `bus`↔`truck`.

**`bike` is currently never emitted on the UCSD footage.** That is highway
footage; across the clips tested, essentially zero motorcycles or bicycles were
detected (1 detection in 92,922). Do not assume the class exists in that data —
but do handle it, since the schema allows it, and it is common in Indian footage.

**`rickshaw` depends on which weights produced the file.** Auto-rickshaws have no
COCO equivalent, so stock `yolov8s.pt` can never emit the class; only our
fine-tuned weights can. Check `meta.model` in the JSON if you need to know which
produced a given run. A file with no `rickshaw` rows does not mean there were no
rickshaws in the footage.

**Timestamps are derived, not stored.** `timestamp_sec = frame_id / meta.fps`.
Deliberately not a column so there is one source of truth for fps.

**No rows for empty frames.** Absence of a `frame_id` means nothing was
detected, which is not the same as nothing being present. On low-resolution
footage misses are common — do not read a gap as "the road was empty".

---

## Loading it

```python
import pandas as pd
df = pd.read_csv("outputs/cctv052x2004080516x01638_tracks.csv")
veh = pd.read_csv("outputs/cctv052x2004080516x01638_vehicles.csv")

tracked = df[df.vehicle_id != -1]              # drop unassigned detections
one = tracked[tracked.vehicle_id == 3]         # one vehicle's trajectory
one = one.sort_values("frame_id")
cx = (one.x1 + one.x2) / 2                     # box centre, for direction/speed
cy = (one.y1 + one.y2) / 2

# per-frame boxes with a trustworthy track-level class attached
reliable = veh[veh.class_agreement >= 0.6]
joined = tracked.merge(reliable[["vehicle_id", "class"]],
                       on="vehicle_id", suffixes=("_frame", "_vehicle"))
```

---

## Regenerating

```bash
python detect_track.py --video data/video/<name>.avi --start-frame 1 --save-video
```

`--start-frame 1` skips the corrupted first frame of the UCSD TrafficDB clips
(the README notes frame 1 carries a second, overlaid video signal).
Run `python detect_track.py --help` for the full set of flags.

Defaults are `yolo11s.pt`, `--imgsz 640`, `--conf 0.10`, tuned on this footage:

- The `s` size finds ~50% more vehicles than `n` for ~1.5x the runtime (measured
  on yolov8s vs yolov8n). Pass `--model yolov8s.pt` to reproduce the original
  v8 baseline in `BASELINE_REPORT.md`.
- `--imgsz 640` beats 960 and 1280. Upscaling 320x240 further *loses*
  detections — there is no extra detail to recover, and the enlarged blurry
  vehicles fall outside the size distribution the model was trained on.
- `--conf 0.10` also rewrites ByteTrack's `track_high_thresh` /
  `new_track_thresh`. Those default to 0.25 and gate the tracker independently
  of the detector, so lowering `--conf` alone would change nothing.

### If you modify the frame loop

`model.track(..., persist=True)` must be passed `persist=True` on **every**
call including the first. Ultralytics binds the value from the first call and
ignores it thereafter; a single `persist=False` makes the tracker reset every
frame, and `vehicle_id` silently degrades into "rank of this detection by
confidence" while still looking like valid output.
