# Provenance of the committed outputs

The files in `outputs_calib/`, and their input
`outputs_demo/teammate_video_tracks.csv`, were generated with the **fine-tuned
YOLOv8s model** described in the paper's Methodology section. They were **not**
produced by the stock COCO-pretrained `yolov8s.pt`. These are the outputs behind
the paper's reported results: 179 vehicles tracked, 34 violating vehicles,
45 violations, and Rs. 44,187.00 in prototype fines.

## Detection and tracking run

Recorded in the run's own metadata (`outputs_demo/teammate_video_tracks.json`,
which is not committed because `*_tracks.json` is gitignored):

| Setting | Value |
|---|---|
| Weights | `runs/detect/runs/bmd45_ft/weights/best.pt` (YOLOv8s fine-tuned on BMD-45, epoch 18 of 20) |
| Weights file size | 22,494,122 bytes |
| Weights SHA-256 | `bf57e2fe7e36daa4b21a419c65a17c517f22cb50801caacd166a9e08ba88fffb` |
| Tracker | ByteTrack (`bytetrack.yaml`) |
| Confidence / NMS IoU / image size | 0.10 / 0.5 / 640 |
| Source | `data/video/teammate_video.mp4`, 1280×624, 30 fps, 413 frames from frame 0 |
| Result | 11,002 detections, 179 vehicle IDs |

The weights are not included in this repository. They are available from the
authors on request. Use the SHA-256 above to confirm that a copy you receive is
the exact file used.

## How the committed outputs chain together

All four violation layers' metadata files in this folder
(`*_speed_meta.json`, `*_wrong_lane_meta.json`, `*_lane_change_meta.json`,
`*_tailgating_meta.json`) record `outputs_demo/teammate_video_tracks.csv` as
their input. The fines in `teammate_video_fines.csv` and
`teammate_video_fines_summary.json` were computed from those layer outputs using
`fine_estimation/config/fine_rules.json`.

## Reproducing

Follow the `detect_track.py` command in the README with the fine-tuned weights,
then the four layer commands. Two ways to get different numbers:

- Using stock `yolov8s.pt` instead of the fine-tuned weights.
- Using `run_pipeline.py` with its default `--model yolov8n.pt`. This produces a
  separate, smaller run with different vehicle IDs and totals.
