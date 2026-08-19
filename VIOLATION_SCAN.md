# Violation Feasibility Scan

Which violation types actually have candidate events in the available footage. **This is a screening tool, not a detector** — thresholds are loose to surface anything worth a look, and every hit needs human confirmation. Counts mean "worth building for", not "N violations occurred".

| Scan | Rule used |
|---|---|
| Multi-rider | ≥2 person boxes overlapping one bike box (≥0.25 of the person box inside) |
| Tailgating | gap ≤ 0.6× follower length, same lane, ≥15 frames |
| U-turn | heading reverses ≥120.0° within one track |
| Stopped | ≤6.0px movement for ≥60 frames while ≥35% of other traffic moves |

## Summary

| Video | tracks | bikes | multi-rider | tailgating | u-turn | stopped |
|---|---|---|---|---|---|---|
| `INDIAN_teammate_video` | 179 | 82 | 11 | 96 | 10 | 12 |
| `ording 2026-08-18 155146` | 126 | 2 | n/a | 17 | 1 | 0 |
| `ording 2026-08-18 155346` | 76 | 0 | n/a | 3 | 0 | 0 |
| `ording 2026-08-18 220610` | 111 | 0 | n/a | 2 | 0 | 1 |
| `ording 2026-08-18 220759` | 220 | 0 | n/a | 5 | 25 | 0 |
| `ording 2026-08-18 222121` | 53 | 0 | n/a | 2 | 0 | 0 |
| `ording 2026-08-18 222441` | 29 | 10 | 1 | 1 | 0 | 0 |
| `ording 2026-08-18 223410` | 13 | 0 | n/a | 0 | 0 | 0 |
| `ording 2026-08-18 223723` | 116 | 0 | n/a | 18 | 0 | 0 |
| `ording 2026-08-18 224410` | 94 | 1 | 0 | 23 | 0 | 0 |

## Totals across scanned footage

- **multi-rider**: 12 candidates across 2/10 videos
- **tailgating**: 167 candidates across 9/10 videos
- **uturn**: 36 candidates across 3/10 videos
- **stopped**: 13 candidates across 2/10 videos

## INDIAN_teammate_video

**Multi-rider** (11)

- `{"vehicle_id": 1597, "max_persons_on_bike": 3, "frames_with_2plus": 6, "samples": 20}`
- `{"vehicle_id": 1372, "max_persons_on_bike": 3, "frames_with_2plus": 4, "samples": 35}`
- `{"vehicle_id": 861, "max_persons_on_bike": 3, "frames_with_2plus": 2, "samples": 32}`
- `{"vehicle_id": 8, "max_persons_on_bike": 2, "frames_with_2plus": 4, "samples": 9}`
- `{"vehicle_id": 11, "max_persons_on_bike": 2, "frames_with_2plus": 4, "samples": 18}`
- `{"vehicle_id": 15, "max_persons_on_bike": 2, "frames_with_2plus": 3, "samples": 26}`
- …and 5 more

**Tailgating** (96)

- `{"follower_id": 1620, "leader_id": 2392, "frames_close": 73, "min_gap_lengths": 0.0, "at_frame": 335}`
- `{"follower_id": 3, "leader_id": 24, "frames_close": 69, "min_gap_lengths": 0.01, "at_frame": 0}`
- `{"follower_id": 1154, "leader_id": 2159, "frames_close": 69, "min_gap_lengths": 0.14, "at_frame": 282}`
- `{"follower_id": 22, "leader_id": 886, "frames_close": 63, "min_gap_lengths": 0.26, "at_frame": 132}`
- `{"follower_id": 6, "leader_id": 13, "frames_close": 60, "min_gap_lengths": 0.24, "at_frame": 20}`
- `{"follower_id": 2072, "leader_id": 886, "frames_close": 57, "min_gap_lengths": 0.0, "at_frame": 325}`
- …and 90 more

**U-turn** (10)

- `{"vehicle_id": 14, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 158, "track_frames": 413}`
- `{"vehicle_id": 25, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 206, "track_frames": 405}`
- `{"vehicle_id": 1790, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 408, "track_frames": 187}`
- `{"vehicle_id": 18, "class": "car", "max_heading_change_deg": 179.4, "at_frame": 131, "track_frames": 383}`
- `{"vehicle_id": 16, "class": "car", "max_heading_change_deg": 179.1, "at_frame": 291, "track_frames": 413}`
- `{"vehicle_id": 27, "class": "bus", "max_heading_change_deg": 166.1, "at_frame": 281, "track_frames": 400}`
- …and 4 more

**Stopped** (12)

- `{"vehicle_id": 16, "class": "car", "stationary_frames": 282, "from_frame": 1, "scene_moving_frac": 0.36, "track_frames": 413}`
- `{"vehicle_id": 886, "class": "bike", "stationary_frames": 276, "from_frame": 118, "scene_moving_frac": 0.36, "track_frames": 296}`
- `{"vehicle_id": 27, "class": "bus", "stationary_frames": 149, "from_frame": 1, "scene_moving_frac": 0.39, "track_frames": 400}`
- `{"vehicle_id": 25, "class": "car", "stationary_frames": 141, "from_frame": 28, "scene_moving_frac": 0.41, "track_frames": 405}`
- `{"vehicle_id": 14, "class": "car", "stationary_frames": 122, "from_frame": 291, "scene_moving_frac": 0.39, "track_frames": 413}`
- `{"vehicle_id": 515, "class": "bike", "stationary_frames": 110, "from_frame": 69, "scene_moving_frac": 0.4, "track_frames": 111}`
- …and 6 more

## Screen Recording 2026-08-18 155146

**Tailgating** (17)

- `{"follower_id": 1047, "leader_id": 863, "frames_close": 95, "min_gap_lengths": 0.03, "at_frame": 358}`
- `{"follower_id": 912, "leader_id": 352, "frames_close": 73, "min_gap_lengths": 0.24, "at_frame": 381}`
- `{"follower_id": 652, "leader_id": 352, "frames_close": 62, "min_gap_lengths": 0.02, "at_frame": 209}`
- `{"follower_id": 1414, "leader_id": 1198, "frames_close": 54, "min_gap_lengths": 0.02, "at_frame": 402}`
- `{"follower_id": 313, "leader_id": 1, "frames_close": 47, "min_gap_lengths": 0.19, "at_frame": 192}`
- `{"follower_id": 313, "leader_id": 146, "frames_close": 39, "min_gap_lengths": 0.02, "at_frame": 108}`
- …and 11 more

**U-turn** (1)

- `{"vehicle_id": 1, "class": "truck", "max_heading_change_deg": 171.4, "at_frame": 249, "track_frames": 331}`

## Screen Recording 2026-08-18 155346

**Tailgating** (3)

- `{"follower_id": 1, "leader_id": 2, "frames_close": 70, "min_gap_lengths": 0.14, "at_frame": 104}`
- `{"follower_id": 1245, "leader_id": 1071, "frames_close": 39, "min_gap_lengths": 0.07, "at_frame": 386}`
- `{"follower_id": 104, "leader_id": 3, "frames_close": 34, "min_gap_lengths": 0.29, "at_frame": 66}`

## Screen Recording 2026-08-18 220610

**Tailgating** (2)

- `{"follower_id": 112, "leader_id": 181, "frames_close": 37, "min_gap_lengths": 0.0, "at_frame": 101}`
- `{"follower_id": 935, "leader_id": 956, "frames_close": 21, "min_gap_lengths": 0.09, "at_frame": 378}`

**Stopped** (1)

- `{"vehicle_id": 717, "class": "truck", "stationary_frames": 96, "from_frame": 280, "scene_moving_frac": 0.69, "track_frames": 97}`

## Screen Recording 2026-08-18 220759

**Tailgating** (5)

- `{"follower_id": 1, "leader_id": 8, "frames_close": 48, "min_gap_lengths": 0.0, "at_frame": 47}`
- `{"follower_id": 597, "leader_id": 518, "frames_close": 19, "min_gap_lengths": 0.3, "at_frame": 499}`
- `{"follower_id": 7, "leader_id": 6, "frames_close": 18, "min_gap_lengths": 0.09, "at_frame": 25}`
- `{"follower_id": 1, "leader_id": 3, "frames_close": 17, "min_gap_lengths": 0.13, "at_frame": 16}`
- `{"follower_id": 388, "leader_id": 323, "frames_close": 16, "min_gap_lengths": 0.01, "at_frame": 255}`

**U-turn** (25)

- `{"vehicle_id": 518, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 495, "track_frames": 209}`
- `{"vehicle_id": 598, "class": "truck", "max_heading_change_deg": 180.0, "at_frame": 521, "track_frames": 179}`
- `{"vehicle_id": 640, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 519, "track_frames": 168}`
- `{"vehicle_id": 699, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 511, "track_frames": 96}`
- `{"vehicle_id": 765, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 509, "track_frames": 104}`
- `{"vehicle_id": 856, "class": "car", "max_heading_change_deg": 180.0, "at_frame": 572, "track_frames": 107}`
- …and 19 more

## Screen Recording 2026-08-18 222121

**Tailgating** (2)

- `{"follower_id": 161, "leader_id": 115, "frames_close": 27, "min_gap_lengths": 0.35, "at_frame": 493}`
- `{"follower_id": 115, "leader_id": 90, "frames_close": 21, "min_gap_lengths": 0.16, "at_frame": 425}`

## Screen Recording 2026-08-18 222441

**Multi-rider** (1)

- `{"vehicle_id": 69, "max_persons_on_bike": 2, "frames_with_2plus": 1, "samples": 2}`

**Tailgating** (1)

- `{"follower_id": 13, "leader_id": 5, "frames_close": 38, "min_gap_lengths": 0.33, "at_frame": 168}`

## Screen Recording 2026-08-18 223723

**Tailgating** (18)

- `{"follower_id": 846, "leader_id": 638, "frames_close": 100, "min_gap_lengths": 0.1, "at_frame": 307}`
- `{"follower_id": 94, "leader_id": 3, "frames_close": 79, "min_gap_lengths": 0.01, "at_frame": 102}`
- `{"follower_id": 13, "leader_id": 12, "frames_close": 64, "min_gap_lengths": 0.3, "at_frame": 77}`
- `{"follower_id": 846, "leader_id": 370, "frames_close": 62, "min_gap_lengths": 0.18, "at_frame": 314}`
- `{"follower_id": 795, "leader_id": 688, "frames_close": 42, "min_gap_lengths": 0.0, "at_frame": 235}`
- `{"follower_id": 243, "leader_id": 10, "frames_close": 37, "min_gap_lengths": 0.24, "at_frame": 169}`
- …and 12 more

## Screen Recording 2026-08-18 224410

**Tailgating** (23)

- `{"follower_id": 5, "leader_id": 6, "frames_close": 305, "min_gap_lengths": 0.0, "at_frame": 22}`
- `{"follower_id": 15, "leader_id": 315, "frames_close": 189, "min_gap_lengths": 0.0, "at_frame": 143}`
- `{"follower_id": 13, "leader_id": 15, "frames_close": 165, "min_gap_lengths": 0.0, "at_frame": 34}`
- `{"follower_id": 6, "leader_id": 14, "frames_close": 165, "min_gap_lengths": 0.0, "at_frame": 244}`
- `{"follower_id": 315, "leader_id": 1267, "frames_close": 138, "min_gap_lengths": 0.0, "at_frame": 254}`
- `{"follower_id": 3, "leader_id": 7, "frames_close": 120, "min_gap_lengths": 0.08, "at_frame": 22}`
- …and 17 more
