"""
Fine Estimation Runner Example
==============================
Processes output files for a video and generates fine-enriched violation records.

Usage:
    python -m fine_estimation.examples.run_fine_estimation \\
        --video-id teammate_video \\
        --calib-dir outputs_calib \\
        --demo-dir outputs_demo \\
        --out-dir outputs_calib
"""

import argparse
import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fine_estimation.src.fine_calculator import FineCalculator
from fine_estimation.src.rule_engine import RuleEngine


def parse_args():
    p = argparse.ArgumentParser(
        description="Run fine estimation on detected traffic violations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-id", required=True, help="Video identifier (e.g. teammate_video)")
    p.add_argument("--calib-dir", default="outputs_calib", help="Directory containing violation summary CSVs and speed metadata")
    p.add_argument("--demo-dir", default="outputs_demo", help="Directory containing base vehicles.csv")
    p.add_argument("--out-dir", default="outputs_calib", help="Directory to save fine output files")
    p.add_argument("--rules-config", default=None, help="Path to custom fine_rules.json config")
    p.add_argument("--only-violators", action="store_true", default=False, help="Include only violating vehicles in CSV output")
    return p.parse_args()


def main():
    args = parse_args()
    calib_dir = Path(args.calib_dir)
    demo_dir = Path(args.demo_dir)
    out_dir = Path(args.out_dir)

    stem = args.video_id

    # Auto-resolve standard paths based on INTEGRATION.md
    vehicles_csv = demo_dir / f"{stem}_vehicles.csv"
    speed_csv = calib_dir / f"{stem}_violations.csv"
    wrong_lane_csv = calib_dir / f"{stem}_wrong_lane_violations.csv"
    lane_change_csv = calib_dir / f"{stem}_lane_change_events.csv"
    tailgating_csv = calib_dir / f"{stem}_tailgating_events.csv"
    speed_meta_json = calib_dir / f"{stem}_speed_meta.json"

    print(f"=== Fine Estimation Runner ===")
    print(f"Video ID          : {stem}")
    print(f"Vehicles CSV      : {vehicles_csv if vehicles_csv.exists() else 'None (auto-discover from violations)'}")
    print(f"Speed CSV         : {speed_csv if speed_csv.exists() else 'None'}")
    print(f"Wrong-lane CSV    : {wrong_lane_csv if wrong_lane_csv.exists() else 'None'}")
    print(f"Lane-change CSV   : {lane_change_csv if lane_change_csv.exists() else 'None'}")
    print(f"Tailgating CSV    : {tailgating_csv if tailgating_csv.exists() else 'None'}")
    print(f"Speed Meta JSON   : {speed_meta_json if speed_meta_json.exists() else 'None'}")
    print(f"Output Directory  : {out_dir}")

    engine = RuleEngine(config_path=args.rules_config)
    calculator = FineCalculator(rule_engine=engine)

    records = calculator.process_records(
        video_id=stem,
        vehicles_csv=vehicles_csv if vehicles_csv.exists() else None,
        speed_violations_csv=speed_csv if speed_csv.exists() else None,
        wrong_lane_csv=wrong_lane_csv if wrong_lane_csv.exists() else None,
        lane_change_csv=lane_change_csv if lane_change_csv.exists() else None,
        tailgating_csv=tailgating_csv if tailgating_csv.exists() else None,
        speed_meta_json=speed_meta_json if speed_meta_json.exists() else None,
    )

    csv_path, json_path = calculator.save_outputs(
        records=records,
        out_dir=out_dir,
        basename=stem,
        only_violators=args.only_violators,
    )

    violator_count = sum(1 for r in records if r["violation_count"] > 0)
    total_fines = sum(r["total_fine"] for r in records)

    print(f"\nEvaluation Results:")
    print(f"  Total vehicles tracked       : {len(records)}")
    print(f"  Violating vehicles           : {violator_count}")
    print(f"  Total prototype fine amount  : {engine.currency} {total_fines:,.2f}")
    print(f"\nGenerated files:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
