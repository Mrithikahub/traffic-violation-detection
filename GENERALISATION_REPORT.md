# Cross-Domain Generalisation Test

Stock vs fine-tuned detector on footage from outside the fine-tuning domain. Detection only, no tracking.

- **Cross-domain set:** 38 videos, 17,692 frames (Western motorway)
- **Control:** 1 video (413 frames) from the fine-tuned model's own domain
- **Settings:** conf=0.25, imgsz=640, stride=1, device=cpu

## Headline

| Metric | A: stock yolov8s | B: fine-tuned | |
|---|---|---|---|
| Detections/frame | 8.167 | 8.8 | B higher |
| Total detections | 144,484 | 155,688 | |
| Mean confidence | 0.616 | 0.689 | |

## Domain-specific hallucination

`rickshaw` cannot occur on Western motorways, so every such detection is wrong by construction.

- **Fine-tuned model emitted `rickshaw` 1,692 times** (1.09% of its detections) across the cross-domain set.
- Stock model cannot emit it (no such class).

## Class distribution (cross-domain)

| Class | A stock | A % | B fine-tuned | B % |
|---|---|---|---|---|
| `bike` | 424 | 0.29% | 10,015 | 6.43% |
| `bus` | 3,987 | 2.76% | 14,094 | 9.05% |
| `car` | 122,831 | 85.01% | 128,213 | 82.35% |
| `rickshaw` | 0 | 0% | 1,692 | 1.09% |
| `truck` | 17,242 | 11.93% | 1,674 | 1.08% |

## Control (in-domain Indian footage)

| Metric | A stock | B fine-tuned |
|---|---|---|
| Detections/frame | 19.567 | 27.305 |
| Mean confidence | 0.561 | 0.680 |
| `rickshaw` | n/a | 523 (4.64%) |

On its own domain the fine-tuned model's `rickshaw` detections are legitimate. The same class on motorway footage is not. Comparing the two rates separates 'specialised' from 'broken'.

## Per-video

| Video | frames | A /fr | B /fr | B/A | A conf | B conf | B rickshaw |
|---|---|---|---|---|---|---|---|
| ` 2026-08-17 215744.mp4` *(control)* | 413 | 19.567 | 27.305 | 1.4 | 0.56 | 0.68 | **523** |
| ` 2026-08-18 224410.mp4` | 522 | 17.544 | 21.946 | 1.25 | 0.61 | 0.71 | **168** |
| ` 2026-08-18 154942.mp4` | 417 | 10.029 | 10.405 | 1.04 | 0.70 | 0.75 | **114** |
| ` 2026-08-18 222200.mp4` | 434 | 2.18 | 2.447 | 1.12 | 0.64 | 0.70 | **106** |
| ` 2026-08-18 225058.mp4` | 559 | 11.292 | 12.086 | 1.07 | 0.68 | 0.71 | **103** |
| ` 2026-08-18 220610.mp4` | 456 | 7.138 | 6.452 | 0.9 | 0.58 | 0.64 | **98** |
| ` 2026-08-18 221051.mp4` | 454 | 9.291 | 11.236 | 1.21 | 0.65 | 0.74 | **94** |
| ` 2026-08-18 224217.mp4` | 498 | 5.882 | 6.47 | 1.1 | 0.59 | 0.60 | **89** |
| ` 2026-08-18 155112.mp4` | 540 | 12.73 | 13.235 | 1.04 | 0.59 | 0.67 | **79** |
| ` 2026-08-18 223641.mp4` | 433 | 12.139 | 12.956 | 1.07 | 0.60 | 0.68 | **78** |
| ` 2026-08-18 223802.mp4` | 423 | 12.314 | 12.574 | 1.02 | 0.58 | 0.69 | **78** |
| ` 2026-08-18 224316.mp4` | 487 | 10.376 | 11.864 | 1.14 | 0.66 | 0.74 | **78** |
| ` 2026-08-18 220034.mp4` | 471 | 2.244 | 2.276 | 1.01 | 0.71 | 0.75 | **72** |
| ` 2026-08-18 221149.mp4` | 395 | 9.706 | 11.354 | 1.17 | 0.68 | 0.75 | **62** |
| ` 2026-08-18 155346.mp4` | 427 | 8.82 | 8.78 | 1.0 | 0.59 | 0.66 | **56** |
| ` 2026-08-18 220450.mp4` | 416 | 7.954 | 9.248 | 1.16 | 0.59 | 0.63 | **56** |
| ` 2026-08-18 220913.mp4` | 397 | 8.856 | 6.431 | 0.73 | 0.57 | 0.58 | **47** |
| ` 2026-08-18 155302.mp4` | 429 | 14.142 | 15.485 | 1.09 | 0.59 | 0.70 | **42** |
| ` 2026-08-18 215700.mp4` | 546 | 13.383 | 13.725 | 1.03 | 0.59 | 0.68 | **39** |
| ` 2026-08-18 221830.mp4` | 396 | 6.495 | 5.879 | 0.91 | 0.58 | 0.66 | **32** |
| ` 2026-08-18 215512.mp4` | 405 | 14.635 | 15.694 | 1.07 | 0.60 | 0.71 | **31** |
| ` 2026-08-18 155146.mp4` | 417 | 15.508 | 17.326 | 1.12 | 0.60 | 0.69 | **29** |
| ` 2026-08-18 155441.mp4` | 425 | 14.704 | 15.252 | 1.04 | 0.60 | 0.70 | **24** |
| ` 2026-08-18 222121.mp4` | 845 | 2.715 | 3.29 | 1.21 | 0.65 | 0.69 | **22** |
| ` 2026-08-18 220759.mp4` | 615 | 9.4 | 9.525 | 1.01 | 0.63 | 0.62 | **20** |
| ` 2026-08-18 220359.mp4` | 591 | 3.173 | 3.188 | 1.0 | 0.70 | 0.76 | **19** |
| ` 2026-08-18 223454.mp4` | 431 | 3.039 | 3.566 | 1.17 | 0.64 | 0.71 | **12** |
| ` 2026-08-18 155221.mp4` | 410 | 13.315 | 14.263 | 1.07 | 0.58 | 0.71 | **7** |
| ` 2026-08-18 222441.mp4` | 519 | 1.474 | 2.075 | 1.41 | 0.59 | 0.73 | **7** |
| ` 2026-08-18 222522.mp4` | 437 | 2.256 | 2.476 | 1.1 | 0.64 | 0.73 | **7** |
| ` 2026-08-18 224131.mp4` | 423 | 8.076 | 8.832 | 1.09 | 0.61 | 0.61 | **7** |
| ` 2026-08-18 222618.mp4` | 349 | 2.602 | 2.926 | 1.12 | 0.65 | 0.72 | **5** |
| ` 2026-08-18 221929.mp4` | 484 | 8.122 | 8.324 | 1.02 | 0.62 | 0.63 | **4** |
| ` 2026-08-18 223723.mp4` | 445 | 12.816 | 13.254 | 1.03 | 0.58 | 0.71 | **4** |
| ` 2026-08-18 224607.mp4` | 466 | 6.957 | 8.384 | 1.21 | 0.65 | 0.65 | **2** |
| ` 2026-08-18 220146.mp4` | 507 | 2.15 | 2.039 | 0.95 | 0.60 | 0.69 | **1** |
| ` 2026-08-18 154903.mp4` | 332 | 6.37 | 7.72 | 1.21 | 0.64 | 0.66 | **0** |
| ` 2026-08-18 222018.mp4` | 457 | 2.775 | 3.468 | 1.25 | 0.70 | 0.72 | **0** |
| ` 2026-08-18 223410.mp4` | 434 | 2.023 | 2.035 | 1.01 | 0.63 | 0.73 | **0** |

## Caveat

There are no ground-truth boxes for this footage, so neither recall nor precision is measured here. What is measured is what each model *asserts*. The `rickshaw` count is the exception: it is wrong by construction regardless of ground truth, which is why it is the cleanest available evidence of domain specialisation.

These are screen recordings of video players, so some detections may fall on player UI or picture-in-picture insets. That affects both models equally and does not bias the comparison.