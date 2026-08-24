# Fine Estimation Module

A modular, extensible rule evaluation and fine calculation engine for the **AI-Based Intelligent Vehicle Monitoring and Traffic Violation Detection System**.

> [!NOTE]
> **DISCLAIMER**: The fine amounts, multipliers, and compounding rules implemented in this module are **project-defined prototype rules for system demonstration and evaluation**. They do NOT constitute legally enforceable fines under the Motor Vehicles Act or any jurisdictional statute.

---

## Architecture Overview

```
fine_estimation/
├── README.md                  # Module documentation & usage guide
├── config/
│   └── fine_rules.json        # Configurable fine amounts, multipliers & policies
├── src/
│   ├── __init__.py            # Package exports (RuleEngine, FineCalculator)
│   ├── rule_engine.py         # Config parser, rule evaluation & compounding
│   └── fine_calculator.py     # Multi-source CSV ingestion, timing & output serialization
├── tests/
│   ├── __init__.py
│   └── test_fine_calculator.py # 11-scenario comprehensive test suite
└── examples/
    └── run_fine_estimation.py # Standalone runner & verification script
```

---

## Key Features

1. **Multi-Violation Compounding**:
   - Accurately joins independent violation summary CSVs (`_violations.csv`, `_wrong_lane_violations.csv`, `_lane_change_events.csv`, `_tailgating_events.csv`) on `vehicle_id`.
   - Supports configurable compounding policies: `"sum"`, `"max"`, or `"sum_with_cap"`.

2. **Overspeed Scaling via `max_speed_kmh`**:
   - Primary metric for speed penalty calculation is `max_speed_kmh`.
   - Dynamically fetches `speed_limit_kmh` from `<stem>_speed_meta.json`.
   - Safely handles vehicles with uncalculated speeds (outside calibrated homography zone).

3. **Multi-Video Identity Isolation**:
   - Vehicles are tracked and keyed using a composite identifier `(video_id, vehicle_id)` (e.g. `teammate_video_20`), preventing ID collisions across video clips.

4. **Preserves Violation Timing & Best Frame for OCR**:
   - Retains earliest violation frame (`first_violation_frame`) and timestamp (`first_violation_sec`).
   - Retains `best_frame_id`, `best_frame_area`, and `best_frame_confidence` from `<stem>_vehicles.csv` for downstream Plate Recognition (NPR) linking.

5. **Excludes Invalid Tracks**:
   - Automatically filters out untracked detections (`vehicle_id = -1` or `vehicle_id <= 0`).

---

## Configuration (`config/fine_rules.json`)

```json
{
  "compounding_policy": "sum",
  "max_total_fine_cap": 25000.0,
  "currency": "INR",
  "base_fines": {
    "speeding": { "car": 1000.0, "bike": 500.0, "bus": 2000.0, "truck": 2000.0, "rickshaw": 750.0, "default": 1000.0 },
    "wrong_side": { "car": 1500.0, "bike": 750.0, "bus": 2500.0, "truck": 2500.0, "rickshaw": 1000.0, "default": 1500.0 },
    "lane_change": { "car": 500.0, "bike": 250.0, "bus": 1000.0, "truck": 1000.0, "rickshaw": 500.0, "default": 500.0 },
    "tailgating": { "car": 750.0, "bike": 350.0, "bus": 1500.0, "truck": 1500.0, "rickshaw": 500.0, "default": 750.0 }
  },
  "overspeed_penalty": {
    "enabled": true,
    "metric": "max_speed_kmh",
    "excess_speed_multiplier_per_kmh": { "car": 25.0, "bike": 15.0, "bus": 40.0, "truck": 40.0, "rickshaw": 20.0, "default": 25.0 }
  }
}
```

---

## Usage

### 1. Programmatic API
```python
from fine_estimation.src import FineCalculator, RuleEngine

calculator = FineCalculator()

records = calculator.process_records(
    video_id="teammate_video",
    vehicles_csv="outputs_demo/teammate_video_vehicles.csv",
    speed_violations_csv="outputs_calib/teammate_video_violations.csv",
    wrong_lane_csv="outputs_calib/teammate_video_wrong_lane_violations.csv",
    lane_change_csv="outputs_calib/teammate_video_lane_change_events.csv",
    tailgating_csv="outputs_calib/teammate_video_tailgating_events.csv",
    speed_meta_json="outputs_calib/teammate_video_speed_meta.json",
)

csv_path, json_path = calculator.save_outputs(
    records=records,
    out_dir="outputs_calib",
    basename="teammate_video",
    only_violators=False,
)
```

### 2. Standalone CLI Runner
```bash
python -m fine_estimation.examples.run_fine_estimation \
    --video-id teammate_video \
    --calib-dir outputs_calib \
    --demo-dir outputs_demo \
    --out-dir outputs_calib
```

### 3. Running Unit Tests
```bash
python -m unittest discover -s fine_estimation/tests -p "test_*.py" -v
```
