# Integration Guide — Violation Output for Fine Estimation & Plate OCR

Companion to [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md). That document is the contract
for the **detection/tracking** output. This one covers the **violation layers**
and answers the questions the Fine Estimation and OCR modules need settled
before they can build against us.

Everything here was verified against the code and against real output files in
`outputs_calib/` and `outputs_demo/` — every sample row below is copied
verbatim, not invented.

- **Reference clip:** `data/video/teammate_video.mp4`, 1280×624, 30 fps, 413 frames
- **Layer outputs:** `outputs_calib/`
- **Detection output:** `outputs_demo/`

---

## 1. Read the summary files, not the per-frame files

Each layer writes **two** kinds of file:

- a **per-frame CSV** — one row per detection per frame, carrying that frame's
  raw measurements. Large, mostly unflagged rows.
- a **summary/event CSV** — **one row per violation**. This is what Fine
  Estimation should consume.

| Layer | Per-frame CSV | Summary CSV (one row per violation) |
|---|---|---|
| Speed | `<stem>_tracks_speed.csv` | `<stem>_violations.csv` |
| Wrong-side | `<stem>_wrong_lane.csv` | `<stem>_wrong_lane_violations.csv` |
| Lane-change | `<stem>_lane_change.csv` | `<stem>_lane_change_events.csv` |
| Tailgating | `<stem>_tailgating.csv` | `<stem>_tailgating_events.csv` |

> **Note the two naming inconsistencies.** The speed layer's per-frame file is
> `_tracks_speed.csv` (not `_speed.csv`), and its summary is `_violations.csv`
> (no layer prefix). The other three follow `<stem>_<layer>.csv`. Hardcode the
> exact names above rather than deriving them from a pattern.

---

## 2. Exact schemas

### 2.1 Per-frame CSVs

All four begin with the **identical 8-column base schema** from
`detect_track.py` (`CSV_COLUMNS`, line 79), in this order:

```
frame_id, vehicle_id, class, x1, y1, x2, y2, confidence
```

Then each appends its own columns:

**Speed** — `<stem>_tracks_speed.csv`

| Column | Type | Notes |
|---|---|---|
| `speed_kmh` | float, 1dp | **Empty for many rows.** Only computed where the box bottom-centre is at `y >= 268` (the calibrated band). |
| `violation` | 0 / 1 | 1 = this vehicle exceeded the limit for at least `min_violation_frames` (3). |

**Wrong-side** — `<stem>_wrong_lane.csv`

| Column | Type | Notes |
|---|---|---|
| `zone` | string | `main_carriageway`, `opposite_carriageway`, or empty if outside all zones. |
| `move_dx`, `move_dy` | float | Smoothed motion vector, pixels. |
| `alignment` | float | Cosine of motion against the zone's expected direction. −1 = fully opposed. |
| `wrong_side` | 0 / 1 | |

**Lane-change** — `<stem>_lane_change.csv`

| Column | Type | Notes |
|---|---|---|
| `world_x`, `world_y` | float | Ground-plane position in metres, via homography. |
| `lane` | string | Instantaneous lane: `outer_zone` or `inner_lane`. |
| `lane_settled` | string | Lane after jitter rejection. Empty while unsettled. |
| `lane_change_event` | 0 / 1 | 1 **only on the single frame** the change is registered. |

**Tailgating** — `<stem>_tailgating.csv`

| Column | Type | Notes |
|---|---|---|
| `leader_id` | int | `vehicle_id` of the vehicle in front. Empty if none. |
| `gap_lengths` | float | Gap in follower-lengths. **The primary severity metric.** ~1.0 ≈ bumper contact. |
| `gap_px` | float | Same gap in pixels. Depth-dependent — do not compare across the frame. |
| `headway_sec` | float | Time headway. **Populated on only ~11% of flagged rows** (needs a speed, which needs the calibrated band). |
| `tailgating` | 0 / 1 | |

### 2.2 Summary CSVs — one row per violation

```
<stem>_violations.csv          (speed)
vehicle_id, class, max_speed_kmh, mean_speed_kmh, frames_over_limit,
first_violation_frame, first_violation_sec, track_first_frame, track_last_frame

<stem>_wrong_lane_violations.csv
vehicle_id, class, zone, wrong_frames, total_judged_frames, mean_alignment,
first_wrong_frame, first_wrong_sec, last_wrong_frame

<stem>_lane_change_events.csv
vehicle_id, class, from_lane, to_lane, change_frame, change_sec,
frames_in_from, frames_in_to

<stem>_tailgating_events.csv
follower_id, leader_id, follower_class, leader_class, frames_tailgating,
duration_sec, min_gap_lengths, min_gap_px, at_frame, at_sec
```

### 2.3 Three fields that do not exist

| You may expect | Reality |
|---|---|
| `violation_type` | **No such field anywhere.** The type is implicit in which file and column you read. If you need a type string, you assign it. |
| `speed_limit` | **Not in any CSV.** Only in `<stem>_speed_meta.json` → `"speed_limit_kmh": 60.0`. Read it from there. |
| `timestamp` | **No per-frame CSV has a time column** — only `frame_id`. Seconds appear only in summary files (`first_violation_sec`, `change_sec`, `at_sec`). Compute `frame_id / fps`; `fps` is in every `_meta.json` (30.0 for this clip). |

---

## 3. Real sample output

Vehicle **20** at **frame 26**, genuinely speeding *and* tailgating at once.
Copied verbatim.

`outputs_calib/teammate_video_tracks_speed.csv`
```csv
frame_id,vehicle_id,class,x1,y1,x2,y2,confidence,speed_kmh,violation
26,20,bike,799.85,199.77,881.45,348.14,0.9157,94.5,1
```

`outputs_calib/teammate_video_tailgating.csv`
```csv
frame_id,vehicle_id,class,x1,y1,x2,y2,confidence,leader_id,gap_lengths,gap_px,headway_sec,tailgating
26,20,bike,799.85,199.77,881.45,348.14,0.9157,8,0.71,105.4,1.05,1
```

Summary rows — `teammate_video_violations.csv` and
`teammate_video_lane_change_events.csv`:
```csv
vehicle_id,class,max_speed_kmh,mean_speed_kmh,frames_over_limit,first_violation_frame,first_violation_sec,track_first_frame,track_last_frame
2072,car,92.4,62.9,21,379,12.63,376,412

vehicle_id,class,from_lane,to_lane,change_frame,change_sec,frames_in_from,frames_in_to
11,bike,outer_zone,inner_lane,59,1.97,59,22
```

Per-vehicle row — `outputs_demo/teammate_video_vehicles.csv`:
```csv
vehicle_id,class,class_agreement,n_frames,first_frame,last_frame,duration_sec,mean_confidence,max_confidence,best_frame_id,best_frame_area,best_frame_confidence
20,bike,1.0,55,0,54,1.83,0.8402,0.934,48,42022.9,0.9046
```

---

## 4. Multiple simultaneous violations

**One row per violation type, in separate files. Nothing is combined.** There is
no joined per-vehicle violation record produced anywhere in this codebase.

Vehicle 20 appears in the speed CSV *and* in the tailgating CSV at the same
`frame_id`, with no cross-reference between them.

On the reference clip:
- **24** frame-vehicle pairs are simultaneously speeding and tailgating
- **11** vehicles carry more than one violation type across the clip

### How to join

Per-frame, to find co-occurring violations:

```python
key = (int(row["frame_id"]), int(row["vehicle_id"]))
```

Per-vehicle, for fine assessment:

```python
key = int(row["vehicle_id"])          # but see the tailgating exception in §5
```

`combine_violations.py` already implements this join — its `main()` builds a
`flags` dict keyed by `(frame_id, vehicle_id)` mapping to the set of layers that
fired. Reuse that pattern rather than rewriting it.

`teammate_video_all_violations_summary.json` records the result, e.g.
`"20": ["speeding", "tailgating"]` — but it holds **counts only, not
per-violation records**. Use it as a cross-check, not as an integration source.

---

## 5. `vehicle_id` — consistent, with three caveats

Plain **base-10 integer**, no prefix, no zero-padding, in a column named
`vehicle_id` at **position 2** of every per-frame file. It is the ByteTrack ID
assigned by `detect_track.py`; every layer parses it with
`int(r["vehicle_id"])` and **never rewrites it**. It will link cleanly with
plate recognition.

Three things that will bite:

1. **`vehicle_id = -1`** means the detection was never associated to a track.
   All four layers exclude these from analysis (`if r["vehicle_id"] != -1`) but
   **still write the rows**, with empty violation fields. The reference clip
   happens to contain zero such rows — do not let that tempt you into skipping
   the guard, other footage has them.
2. **IDs are unique within one video only.** Not global. Two different videos
   both contain a vehicle 20. Any persisted fine record needs
   `(video_name, vehicle_id)` as its key.
3. **`<stem>_tailgating_events.csv` uses `follower_id`, not `vehicle_id`.** It
   is the only summary file that renames the column, because a row describes a
   *pair*. `follower_id` is the offending vehicle; `leader_id` is the vehicle in
   front and is **not** itself accused of anything.

---

## 6. Where to hook in

**There is no single function that produces a final violation record.** Each
layer writes independently from inside its own `main()`:

| Layer | File | Where the violation record is built |
|---|---|---|
| Speed | `speed_violation.py` | `main()`, `summary = []` loop → `_violations.csv` |
| Wrong-side | `wrong_lane.py` | `main()`, `for tid in sorted(violators)` loop |
| Lane-change | `lane_change.py` | `main()`, `events.append(...)` |
| Tailgating | `tailgating.py` | `main()`, `summary` loop over `flagged` pairs |

**The best hook point is `combine_violations.py`, in `main()`** — the block that
builds the `flags` and `extra` dicts keyed by `(frame_id, vehicle_id)`. That is
the only place in the codebase where all four layers are already joined into one
structure. It currently reduces that join to counts; Fine Estimation should
iterate it instead and emit one record per (vehicle, violation type).

Recommended shape for a new `fine_estimation.py`, following the same decoupled
pattern as the four layers — read CSVs, write CSVs, import nothing from them:

```
fine_estimation.py
  --tracks/--speed-csv/--wrong-lane-csv/--lane-change-csv/--tailgating-csv
  --plates-csv          (from the OCR module: vehicle_id -> plate string)
  --rules rules.json    (violation type -> fine amount; does not exist yet)
  -> <stem>_fines.csv   (one row per vehicle per violation type)
```

---

## 7. Fine values do not exist — build from scratch

Searched the whole project for `fine`, `penalty`, `rupee`, `INR`, `₹`,
`challan`, `MV Act`: **zero hits in project code.** There is no rule table, no
amount mapping, no severity tiering, no statute reference.

This is entirely unbuilt. What is needed:

- A violation-type vocabulary (the four layers do not emit type strings — §2.3)
- An amount per type, ideally externalised to a `rules.json` rather than hardcoded
- A policy for **compounding**: 11 vehicles on the reference clip trigger more
  than one type. Does a vehicle pay per violation, or the maximum, or a
  capped sum? This is a rules decision, not a technical one.
- A policy for **repeat frames**: a vehicle speeding for 47 consecutive frames is
  one violation, not 47. The summary CSVs already collapse this correctly —
  consume them, not the per-frame files (§1).

---

## 8. Overspeeding — actual speeds, not just a flag

Both the measured speed and the pass/fail flag are available:

| What | Where |
|---|---|
| Per-frame speed | `_tracks_speed.csv` → `speed_kmh` (float, 1dp) |
| Per-frame flag | `_tracks_speed.csv` → `violation` (0/1) |
| Per-vehicle peak/mean | `_violations.csv` → `max_speed_kmh`, `mean_speed_kmh` |
| Frames over limit | `_violations.csv` → `frames_over_limit` |
| **The applicable limit** | `_speed_meta.json` → `speed_limit_kmh` (60.0) — **not on any row** |

So an amount can be scaled by how far over the limit a vehicle was. Two
constraints:

- **`speed_kmh` is empty for many rows.** Speed is only computed inside the
  calibrated band (`y >= 268`). On the reference clip 48 vehicles received a
  speed and 22 violated; every other tracked vehicle carries no speed at all. A
  vehicle with no speed is **not** a vehicle that was not speeding.
- **These speeds are not enforcement-grade.** The homography is anchored on IRC
  standard road dimensions, not a site survey. `_speed_meta.json` carries an
  explicit `"WARNING": "... Not valid for enforcement."` Fines computed from
  them are demonstration figures. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
  §1 and §2.

---

## 9. Best frame for plate OCR

`<stem>_vehicles.csv` now names, for every tracked vehicle, the single frame
most likely to yield a readable plate:

| Column | Meaning |
|---|---|
| `best_frame_id` | Frame to crop for plate recognition |
| `best_frame_area` | Box area in px² at that frame — a legibility budget |
| `best_frame_confidence` | Detector confidence at that frame (**not** `max_confidence`) |

Typical OCR loop:

```python
import csv, cv2
veh = {int(r["vehicle_id"]): r
       for r in csv.DictReader(open("outputs_demo/teammate_video_vehicles.csv",
                                    encoding="utf-8"))}
boxes = {}
for r in csv.DictReader(open("outputs_demo/teammate_video_tracks.csv",
                             encoding="utf-8")):
    boxes[(int(r["frame_id"]), int(r["vehicle_id"]))] = r

cap = cv2.VideoCapture("data/video/teammate_video.mp4")
for vid, v in veh.items():
    if float(v["best_frame_area"]) < 5000:      # too small to resolve a plate
        continue
    f = int(v["best_frame_id"])
    b = boxes[(f, vid)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    crop = frame[int(float(b["y1"])):int(float(b["y2"])),
                 int(float(b["x1"])):int(float(b["x2"]))]
    # -> plate recognition on `crop`
```

### How the frame is chosen

`pick_best_frame()` in `detect_track.py`. Boxes touching the frame border are
discarded first — a vehicle halfway out of shot is often the *largest* box in
its track while showing no plate at all. The rest are scored
`sqrt(area) × confidence`: `sqrt` keeps the term proportional to plate *height*
rather than its square, and confidence proxies for a clean, unoccluded view. If
every box in a track is truncated, the border filter is dropped rather than
returning nothing.

### What it is not

This is a heuristic over detection geometry. **It never inspects pixels** — it
cannot see motion blur, glare, rain, a dirty plate, or a plate turned away from
the camera. It is a better starting point than "highest confidence" or "biggest
box", not a guarantee of legibility.

Measured over the 179 vehicles of the reference clip:

| | |
|---|---|
| Differs from a naive highest-confidence pick | 98 vehicles (55%) |
| Differs from a naive largest-box pick | 118 vehicles (66%) |
| …of those, specifically rejecting a frame-truncated box | 27 |
| Best-frame area: min / median / max | 462 / 2,789 / 290,754 px² |
| Vehicles whose best frame is ≥5000 px² | 68 of 179 (38%) |

**That last row is the important one.** On this footage only about **38% of
tracked vehicles are ever large enough for a plausible plate read**, even at
their best frame. Expect OCR coverage well below 100%, and treat a missing plate
as normal rather than as a failure. The same resolution ceiling is what caused
helmet detection to be dropped — see
[HELMET_DETECTION_EVALUATION.md](HELMET_DETECTION_EVALUATION.md).

---

## 10. Regenerating

`best_frame_*` was added after the reference outputs were first produced.
`outputs_demo/teammate_video_vehicles.csv` has been regenerated and contains the
new columns. **Any `_vehicles.csv` produced before this change lacks them** —
re-run `detect_track.py` on that footage to add them.

Nothing else changed: the per-frame schema, `vehicle_id` semantics, and all four
layer outputs are untouched, so no existing consumer breaks.
