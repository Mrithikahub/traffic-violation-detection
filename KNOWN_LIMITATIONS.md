# Known Limitations

Consolidated caveats for the vehicle detection, tracking and violation pipeline.
Written so that every number the system produces can be traced to the assumption
it rests on.

**One-line summary:** the pipeline detects, tracks and classifies reliably on
Indian CCTV footage; the *violation* layers are demonstrations built on
documented assumptions, and their absolute values are not enforcement-grade.

- **Date:** 2026-08-19
- **Components:** `detect_track.py`, `speed_violation.py`, `wrong_lane.py`,
  `lane_change.py`, `tailgating.py`, `calibrate_homography.py`,
  `generalisation_test.py`
- **Demo footage:** `teammate_video.mp4` — 1280×624, 30 fps, 413 frames (13.8 s)

---

## Severity at a glance

| # | Limitation | Affects | Severity |
|---|---|---|---|
| 1 | Speed scale assumed from IRC standards, never site-verified | all speeds | **High** |
| 2 | Speed only measured in the calibrated band (72% of samples dropped) | coverage | **High** |
| 3 | `outer_zone` spans ~2 physical lanes | lane changes | Medium |
| 4 | Helmet detection not implemented | feature absent | Medium |
| 5 | Detector specialised to Indian footage — truck detection collapses off-domain, and the failure is invisible in headline metrics | other datasets | **High** |
| 6 | Zones/lanes hand-specified for one camera view | portability | Medium |
| 7 | No ground truth on the demo video | all metrics | Medium |
| 8 | Demo thresholds are not legal limits | interpretation | Low |
| 9 | Source video artifacts (frozen intro, player inset) | demo only | Low |
| 10 | Tailgating gap uses box height as a vehicle-length proxy; threshold is an engineering choice | tailgating | Medium |

---

## 1. Speed scale is assumed, not measured — **High**

The homography that converts pixels to metres is anchored on **standard IRC road
geometry, not on any distance measured at this site**. The road was never
identified on a map, and nothing in the scene was physically measured.

| Quantity | Source |
|---|---|
| Lane width 3.5 m | **Assumed** — IRC 86, urban |
| Dash pitch 7.5 m | **Assumed** — IRC 35 (3.0 m dash + 4.5 m gap) |
| Dash length 3.85 m | **Solved from the image** — the IRC 3.0 m split gave inconsistent geometry |

**Consequence: every speed scales linearly with those assumptions.** If the real
lane-marking pitch is 6 m rather than 7.5 m, all reported speeds are ~20% too
high. A speed of "77 km/h" means "77 km/h *if* this road is built to IRC
standard dimensions".

**What was verified:** the fit is *internally* consistent — recovered pitch
**7.52 m mean (sd 0.22)** against the assumed 7.5 m, reprojection residual
0.163 m mean / 0.490 m max, fitted over 14 dash endpoints by least squares.
`calibrate_homography.py` refuses to emit a homography if recovered pitch drifts
more than 0.6 m.

That validates *the fit*, not *the assumption*. Internal consistency cannot
detect a systematically wrong standard.

> An early 4-point calibration passed casual inspection while being badly wrong
> (dash lengths recovered as 3.3–5.5 m instead of a constant 3 m). It was caught
> only because the fit was checked against geometry it had not been fitted to.
> A 4-point homography has zero redundancy and cannot reveal its own errors.

**Fix:** one measured distance on site, or identifying the road on a map.

---

## 2. Speed is measured only inside the calibrated band — **High**

Speeds are reported only where the vehicle's ground-contact point is at image
row **y ≥ 268**, the region the homography was actually fitted over (lane dashes
spanned y ≈ 268–503). Beyond it the mapping extrapolates, and near the horizon a
one-pixel tracking error projects to several metres.

| | Value |
|---|---|
| Detections in demo video | 10,590 |
| Samples with a speed | **1,492 (14%)** |
| Vehicles tracked | 169 |
| Vehicles with any speed | **48 (28%)** |

**Distant vehicles are still detected, tracked and drawn — they simply carry no
speed.** They are *not* counted as compliant.

Evidence this cutoff is necessary: without it, **5.67% of samples exceeded
100 km/h** (max 149.7), concentrated near the horizon. With the band plus
`--max-plausible-kmh 100`, the range is 0–95.7 km/h and all 22 flagged vehicles
sit inside the validated region.

**Fix:** calibrate over a larger span, using lane markings further up the road.

---

## 3. `outer_zone` spans about two physical lanes — Medium

Lane boundaries use only the **3 well-supported lane lines** (≥5 supporting
dashes) at ground-plane x = −7.91, −0.04, +3.24 m. A fourth line at x = −3.65 m
had only 3 supporting dashes and was excluded as unreliable.

| Lane | Width | Note |
|---|---|---|
| `outer_zone` | **7.87 m** | ≈ 2 physical lanes — the excluded line runs through its middle |
| `inner_lane` | 3.28 m | one lane |

**Consequence: a vehicle changing between the two physical lanes inside
`outer_zone` produces no event.** Reported lane changes are therefore a lower
bound. `lane_change.py` prints a warning for any zone wider than 4.5 m and
records it in metadata.

Coverage is also thin: of 10,590 samples only **850** received a lane assignment
(730 `outer_zone`, 120 `inner_lane`). The 6 logged events are a demonstration of
the mechanism, not a lane-usage study.

**Fix:** better lane-line extraction (multiple frames, or manual boundary
marking) to support all 4+ lines.

---

## 4. Helmet detection not implemented — Medium

Evaluated and **deliberately dropped**. Two pretrained models were tested on
crops around each tracked bike; both failed. Rider heads are ~15–30 px at this
camera distance, and the only model with a genuine no-helmet class produced
**one** such detection across 50 riders, at 0.17 confidence.

Shipping it would have flagged near-zero riders while looking like a working
detector reporting compliance.

Full evaluation, including the upscaling test and per-model numbers:
**`HELMET_DETECTION_EVALUATION.md`**.

**Fix:** footage where the rider's head spans ≳60 px, or a model trained on
overhead CCTV of two-wheelers.

---

## 5. The detector is specialised to Indian footage — **High**

*Severity raised from Medium after systematic testing: the failure is larger and
less detectable than a single clip suggested.*

`yolov8s` fine-tuned on BMD-45 (800 train / 200 val Bengaluru CCTV images,
5 classes incl. `rickshaw`). On its target domain it is far better than stock:

| Metric | Pretrained | Fine-tuned |
|---|---|---|
| Localisation recall | 61.2% | **90.6%** |
| Two-wheeler recall | 41.7% | **88.2%** |
| Rickshaw recall | not detectable | **91.3%** |
| `car → truck` errors | 101 | **22** |

Off its domain it degrades badly. Quantified by a controlled test — **39 videos,
78 runs, 18,105 frames per model**: 38 Western motorway clips plus the one
Indian clip retained as an **in-domain control**, both models at identical
settings (`conf=0.25`, `imgsz=640`, every frame). Full results in
**`GENERALISATION_REPORT.md`**; per-video figures in
`results_generalisation/per_video_comparison.csv`.

### 5a. The damage is invisible in headline metrics

| Metric (38 cross-domain videos) | Stock | Fine-tuned |
|---|---|---|
| Detections/frame | 8.17 | 8.80 |
| Total detections | 144,484 | 155,688 |
| Mean confidence | 0.616 | 0.689 |

Detection volume barely moves (per-video B/A ratio: median **1.07**, range
0.73–1.41). **Nothing in the aggregate counts signals a problem.** The damage is
entirely in *what the class labels mean*, so any check based on detection rate
alone would pass this model.

### 5b. Truck detection collapses — the most consequential finding

| Class | Stock | Fine-tuned | Change |
|---|---|---|---|
| `truck` | 17,242 (11.93%) | **1,674 (1.08%)** | **−90%** |
| `bus` | 3,987 (2.76%) | 14,094 (9.05%) | +3.3× |
| `bike` | 424 (0.29%) | 10,015 (6.43%) | +24× |
| `car` | 122,831 (85.01%) | 128,213 (82.35%) | ~flat |

On motorway footage dominated by articulated lorries, the fine-tuned model has
largely **stopped calling them trucks and calls them buses instead**.

Root cause is the training class balance: BMD-45 supplied only **220 truck
training boxes** against 3,973 for bike, and its "trucks" are Indian light
commercial vehicles, not European articulated lorries. Fine-tuning did not
merely *add* Indian classes — it **redefined `truck`** around a small, visually
different sample.

Implication beyond this project: a scarce class in fine-tuning data can be
degraded far outside the target domain, not just under-learned within it.

### 5c. Rickshaw hallucination is systematic and unfilterable

An auto-rickshaw cannot occur on a Western motorway, so every such detection is
wrong *by construction* — the one error measurable here without ground truth.

| | Cross-domain (38 videos) |
|---|---|
| Total `rickshaw` detections | **1,692** (1.09% of output) |
| Videos affected | **35 of 38 (92%)** |
| Videos clean | 3 |
| Median per video | 32 |
| Worst video | 168 detections; 9.98% of that clip's output |

**Raising the confidence threshold does not remove them.** These are emitted at
mean confidence **0.689** (per-video means up to 0.75) — the same band as
correct detections. There is no threshold that keeps the good and drops the bad.

The control separates *specialised* from *broken*: on its own Indian footage the
same model emits `rickshaw` at **4.64%**, which is correct behaviour. The class
works; it simply fires where it should not.

### 5d. What to do

**Use the fine-tuned weights only for Indian traffic; use stock `yolov8s`
everywhere else.** Both work via `--model`, and `build_class_map()` handles the
differing class index spaces automatically.

Do **not** rely on confidence filtering or detection-count sanity checks to
catch cross-domain misuse — 5a and 5c show both would pass it.

For one general model: mix non-Indian footage into training, and materially
increase the truck sample. The 220-box truck class is the clearest single lever.

---

## 6. Zones and lanes are hand-specified for one camera view — Medium

Wrong-side zones and lane boundaries encode a human judgement about this
specific framing. **They are invalid for any other camera, and for this camera
if it pans, zooms or is remounted.** Nothing detects that they have gone stale —
a moved camera would silently produce wrong verdicts.

Coverage on the demo video: 6,358 of 10,590 samples (60%) fell outside all
wrong-side zones and received **no verdict** — again, not counted as compliant.

The wrong-side layer reported **0 violations**. That was verified as a real
finding rather than a broken detector by re-running with the expected directions
inverted, which flagged **48 of 57 vehicles** — so the logic fires when there is
something to fire on.

It also tests **direction only**: a legitimate U-turn or a vehicle crossing
between zones would register as wrong-side. It does not model intent or legality.

---

## 7. No ground truth on the demo video — Medium

`teammate_video.mp4` has no annotations, so **detection recall on it cannot be
measured**. Reported detection counts describe what the model *found*, never
what it *missed*. Every quality report prints this caveat.

Where ground truth did exist (UA-DETRAC, BMD-45) it was used, and those numbers
appear in `FINETUNE_RESULTS.md`.

---

## 8. Demo thresholds are not legal limits — Low

The **60 km/h** speed threshold is a demonstration value chosen against the
observed speed distribution (median ≈50 km/h), **not** the posted limit for this
road, which is unknown. Likewise `--min-violation-frames 3` and the plausibility
cap at 100 km/h are engineering choices to suppress noise, not legal criteria.

Every threshold is recorded in each layer's `_meta.json`.

---

## 9. Source video artifacts — Low

`teammate_video.mp4` is a screen recording of a video player, which introduces
two artifacts handled explicitly:

- **Frozen opening ~10 frames.** Vehicle 8 sits at exactly cx=724.9 for frames
  0–9 before moving; playback starts around frame 10. Speeds in that window are
  correctly ~0.
- **Picture-in-picture inset** (bottom-left, the player's seek preview) mirrors
  the same road and generated phantom vehicles. Masked via
  `--ignore-region "0,355,325,575"` in every layer.

Also of note across datasets: the UCSD TrafficDB clips have a **corrupted first
frame** (hence `--start-frame 1`), and BMD-45 / UVH-26 are **independent stills,
not video** — usable for training, not for tracking.

---

## 10. Tailgating: scale-free metric, but assumption-laden — Medium

`tailgating.py` flags a vehicle following another too closely. It is the only
violation layer that needs **no camera calibration**, so it runs unchanged on
any footage — but its assumptions are worth stating exactly.

### 10a. What the gap actually measures

Separation between the two vehicles' **ground-contact points** (box
bottom-centre), projected onto the **follower's own direction of travel**, then
divided by the **follower's box height**:

```
gap_lengths = along_travel_separation / follower_box_height
```

So the unit is *follower vehicle-lengths*, and **≈1.0 means bumper contact**.
The default threshold of **1.5** therefore means roughly half a vehicle-length
of clear space.

Two consequences:

- **Scale-free.** "Half a car-length behind" is the same judgement near or far,
  so one threshold works at any distance and on any camera. This is why the
  layer transfers across domains where the speed layer cannot.
- **Box height proxies vehicle length.** Valid in this geometry, where vehicles
  are seen mostly end-on. **A vehicle crossing the view sideways breaks the
  proxy** and its gaps should not be trusted.

Direction is taken per-vehicle from its own recent motion, not assumed from the
scene — which is what lets it work on a motorway with two carriageways running
opposite ways, with no zone definition at all.

### 10b. The 1.5-length threshold is an engineering choice

**It is not a legal standard, and is not validated against observed driver
behaviour or any accident data.** It was chosen to sit meaningfully below the
observed median gap (1.71 lengths) while staying above bumper contact. A
different road, speed limit or vehicle mix would justify a different value.

Where a homography and speed CSV are supplied the layer *also* reports **time
headway in seconds** — the measure traffic engineering actually uses, against
which the "2-second rule" is the familiar benchmark. On the demo footage the
median headway is **1.33 s, with 70% of judged samples under 2 s**. The flag
itself deliberately stays on the scale-free metric so behaviour is identical on
uncalibrated footage.

### 10c. The speed gate was tuned from data, not guessed

Queued traffic sits close together legitimately, so both vehicles must be
moving before a pair can be flagged. The first implementation gated on **raw
pixels per frame, which is depth-dependent** — a distant vehicle can never meet
it however fast it is really travelling — and that discarded **81% of all
traffic**.

The gate is now in **vehicle-lengths per frame**, set from the measured
distribution on this footage:

| | lengths/frame |
|---|---|
| Stationary vehicles (p5) | 0.0004 |
| Median moving vehicle | 0.019 |
| **Gate (default)** | **0.010** |

That is ~25× the stationary noise floor while retaining slow traffic — roughly
4 km/h for a 4 m vehicle at 30 fps. Judged samples rose from 968 to **2,912**.

**Slow-moving congestion is therefore deliberately not reported.** That is a
design decision, not an oversight: a jam is not tailgating.

### 10d. Duplicate detections are rejected in two stages

A vehicle occasionally picked up under two track IDs presents as a vehicle
**tailgating itself** at a near-zero gap — one such pair appeared in testing
(IoU 0.69, near-identical box sizes, apparent gap 0.09 lengths).

- **Per-frame:** a pair whose boxes overlap above `--max-pair-iou` (0.45) is
  skipped for that frame.
- **Pair-level:** a pair whose *median* overlap across its close frames exceeds
  half that limit is rejected entirely.

The second stage is necessary because a pair that genuinely is one vehicle has
*fluctuating* overlap — individual frames dip under the per-frame limit and it
accumulates enough to be flagged anyway. Verified: **0 duplicate pairs remain**
in the demo output.

### 10e. Coverage and what is not judged

On the demo footage, of 8,871 follower-samples considered:

| Outcome | Samples |
|---|---|
| Judged | **2,912** |
| Skipped — follower too slow | 3,199 |
| Skipped — leader too slow | 585 |
| Skipped — box under 25 px | 175 |
| Skipped — overlapping pair | 18 |

Result: **105 pairs came below the limit at some point; 21 held it for the
required 15+ frames** and were flagged.

Vehicles that are too slow, too small, or too briefly tracked receive **no
verdict — they are not counted as driving safely**.

### 10f. Cross-domain behaviour

Run unchanged, with no calibration and no parameter changes, on a Western
motorway video: **10 pairs flagged**, gap distribution p10 1.12 / median 2.10
(versus 0.61 / 1.71 on the Indian footage — looser following, as expected on a
motorway). This is the only violation layer verified to transfer between
domains.

**No ground truth exists for tailgating in any of this footage**, so precision
and recall are unmeasured. What is measured is which pairs satisfy the stated
geometric rule.

---

## What this pipeline is, and is not

**Is:** a working detection + tracking system for Indian urban CCTV, with four
violation layers demonstrating speed estimation, wrong-side detection,
lane-change logging and tailgating on real footage, each with its assumptions
recorded machine-readably. Tailgating is the only one that needs no calibration
and has been verified to transfer to a different country's footage unchanged.

**Is not:** an enforcement system. Speeds rest on unverified standard
dimensions; zone rules are hand-drawn for one camera; tailgating thresholds are
engineering choices with no ground-truth validation; helmet detection is
absent. Every layer prints its caveats on the output video and writes them to
`_meta.json` — deliberately, so no number travels without its context.
