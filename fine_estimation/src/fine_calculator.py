"""
Fine Calculator
===============
Processes grouped traffic violation summary records, links speed metadata,
evaluates fines using the RuleEngine, and generates unified fine records.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .rule_engine import RuleEngine


class FineCalculator:
    """
    Computes prototype fine estimates for detected traffic violations.
    """

    def __init__(self, rule_engine: Optional[RuleEngine] = None, config_path: Optional[str or Path] = None):
        self.rule_engine = rule_engine or RuleEngine(config_path)

    @staticmethod
    def _read_csv(path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if not path:
            return []
        p = Path(path)
        if not p.exists():
            return []
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _read_json(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        if val is None or val == "":
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def process_records(
        self,
        video_id: str,
        vehicles_csv: Optional[Union[str, Path]] = None,
        speed_violations_csv: Optional[Union[str, Path]] = None,
        wrong_lane_csv: Optional[Union[str, Path]] = None,
        lane_change_csv: Optional[Union[str, Path]] = None,
        tailgating_csv: Optional[Union[str, Path]] = None,
        speed_meta_json: Optional[Union[str, Path]] = None,
        speed_limit_kmh: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes summary CSVs and metadata for a specific video and generates fine records.
        """
        # Determine speed limit from meta JSON or parameter
        resolved_speed_limit = speed_limit_kmh
        if resolved_speed_limit is None and speed_meta_json:
            meta = self._read_json(speed_meta_json)
            resolved_speed_limit = self._safe_float(meta.get("speed_limit_kmh"))

        # Vehicle registry: vehicle_id -> vehicle_data
        # Keyed by integer vehicle_id (must be > 0)
        vehicle_registry: Dict[int, Dict[str, Any]] = {}

        # 1. Ingest base vehicles CSV if provided
        veh_rows = self._read_csv(vehicles_csv)
        for r in veh_rows:
            vid = self._safe_int(r.get("vehicle_id"))
            if vid is None or vid <= 0:  # Skip invalid tracks (vehicle_id = -1)
                continue
            vehicle_registry[vid] = {
                "vehicle_id": vid,
                "class": r.get("class", "unknown"),
                "class_agreement": self._safe_float(r.get("class_agreement")),
                "best_frame_id": self._safe_int(r.get("best_frame_id")),
                "best_frame_area": self._safe_float(r.get("best_frame_area")),
                "best_frame_confidence": self._safe_float(r.get("best_frame_confidence")),
                "violations": set(),
                "violation_details": {},
                "max_speed_kmh": None,
                "mean_speed_kmh": None,
                "timing": {},
            }

        # Helper to ensure vehicle exists in registry
        def ensure_vehicle(vid: int, default_class: str = "unknown") -> Dict[str, Any]:
            if vid not in vehicle_registry:
                vehicle_registry[vid] = {
                    "vehicle_id": vid,
                    "class": default_class,
                    "class_agreement": None,
                    "best_frame_id": None,
                    "best_frame_area": None,
                    "best_frame_confidence": None,
                    "violations": set(),
                    "violation_details": {},
                    "max_speed_kmh": None,
                    "mean_speed_kmh": None,
                    "timing": {},
                }
            return vehicle_registry[vid]

        # 2. Ingest Speeding Violations
        speed_rows = self._read_csv(speed_violations_csv)
        for r in speed_rows:
            vid = self._safe_int(r.get("vehicle_id"))
            if vid is None or vid <= 0:
                continue
            v = ensure_vehicle(vid, r.get("class", "unknown"))
            if v["class"] == "unknown" and r.get("class"):
                v["class"] = r["class"]
            
            v["violations"].add("speeding")
            max_spd = self._safe_float(r.get("max_speed_kmh"))
            mean_spd = self._safe_float(r.get("mean_speed_kmh"))
            v["max_speed_kmh"] = max_spd
            v["mean_speed_kmh"] = mean_spd

            v["timing"]["speeding"] = {
                "first_violation_frame": self._safe_int(r.get("first_violation_frame")),
                "first_violation_sec": self._safe_float(r.get("first_violation_sec")),
                "frames_over_limit": self._safe_int(r.get("frames_over_limit")),
            }
            v["violation_details"]["speeding"] = {
                "max_speed_kmh": max_spd,
                "mean_speed_kmh": mean_spd,
                "frames_over_limit": self._safe_int(r.get("frames_over_limit")),
            }

        # 3. Ingest Wrong Side Violations
        wrong_rows = self._read_csv(wrong_lane_csv)
        for r in wrong_rows:
            vid = self._safe_int(r.get("vehicle_id"))
            if vid is None or vid <= 0:
                continue
            v = ensure_vehicle(vid, r.get("class", "unknown"))
            if v["class"] == "unknown" and r.get("class"):
                v["class"] = r["class"]
            
            v["violations"].add("wrong_side")
            v["timing"]["wrong_side"] = {
                "first_violation_frame": self._safe_int(r.get("first_wrong_frame")),
                "first_violation_sec": self._safe_float(r.get("first_wrong_sec")),
                "wrong_frames": self._safe_int(r.get("wrong_frames")),
            }
            v["violation_details"]["wrong_side"] = {
                "zone": r.get("zone", "opposite_carriageway"),
                "wrong_frames": self._safe_int(r.get("wrong_frames")),
                "mean_alignment": self._safe_float(r.get("mean_alignment")),
            }

        # 4. Ingest Lane Change Events
        lane_rows = self._read_csv(lane_change_csv)
        for r in lane_rows:
            vid = self._safe_int(r.get("vehicle_id"))
            if vid is None or vid <= 0:
                continue
            v = ensure_vehicle(vid, r.get("class", "unknown"))
            if v["class"] == "unknown" and r.get("class"):
                v["class"] = r["class"]

            v["violations"].add("lane_change")
            # If multiple lane changes occurred for same vehicle, preserve earliest timing
            if "lane_change" not in v["timing"]:
                v["timing"]["lane_change"] = {
                    "first_violation_frame": self._safe_int(r.get("change_frame")),
                    "first_violation_sec": self._safe_float(r.get("change_sec")),
                }
            v["violation_details"]["lane_change"] = {
                "from_lane": r.get("from_lane"),
                "to_lane": r.get("to_lane"),
                "change_frame": self._safe_int(r.get("change_frame")),
                "change_sec": self._safe_float(r.get("change_sec")),
            }

        # 5. Ingest Tailgating Events (follower_id is the offending vehicle)
        tail_rows = self._read_csv(tailgating_csv)
        for r in tail_rows:
            vid = self._safe_int(r.get("follower_id"))
            if vid is None or vid <= 0:
                continue
            v = ensure_vehicle(vid, r.get("follower_class", "unknown"))
            if v["class"] == "unknown" and r.get("follower_class"):
                v["class"] = r["follower_class"]

            v["violations"].add("tailgating")
            if "tailgating" not in v["timing"]:
                v["timing"]["tailgating"] = {
                    "first_violation_frame": self._safe_int(r.get("at_frame")),
                    "first_violation_sec": self._safe_float(r.get("at_sec")),
                }
            v["violation_details"]["tailgating"] = {
                "leader_id": self._safe_int(r.get("leader_id")),
                "leader_class": r.get("leader_class"),
                "min_gap_lengths": self._safe_float(r.get("min_gap_lengths")),
                "frames_tailgating": self._safe_int(r.get("frames_tailgating")),
                "duration_sec": self._safe_float(r.get("duration_sec")),
            }

        # 6. Calculate Fines for all registered vehicles
        results: List[Dict[str, Any]] = []
        for vid in sorted(vehicle_registry.keys()):
            v = vehicle_registry[vid]
            v_class = v["class"]
            violations_list = sorted(list(v["violations"]))

            # Evaluate itemized breakdown for each violation
            fine_breakdown: Dict[str, Dict[str, Any]] = {}
            for v_type in violations_list:
                breakdown = self.rule_engine.evaluate_violation(
                    violation_type=v_type,
                    vehicle_class=v_class,
                    max_speed_kmh=v["max_speed_kmh"],
                    speed_limit_kmh=resolved_speed_limit,
                    event_details=v["violation_details"].get(v_type, {}),
                )
                fine_breakdown[v_type] = breakdown

            total_fine = self.rule_engine.compound_fines(fine_breakdown)

            # Determine primary timing (earliest violation frame/sec across all violations)
            first_frame: Optional[int] = None
            first_sec: Optional[float] = None
            for t_info in v["timing"].values():
                ff = t_info.get("first_violation_frame")
                fs = t_info.get("first_violation_sec")
                if ff is not None:
                    first_frame = ff if first_frame is None else min(first_frame, ff)
                if fs is not None:
                    first_sec = fs if first_sec is None else min(first_sec, fs)

            composite_id = f"{video_id}_{vid}"
            record: Dict[str, Any] = {
                "video_id": video_id,
                "vehicle_id": vid,
                "composite_id": composite_id,
                "class": v_class,
                "class_agreement": v["class_agreement"],
                "best_frame_id": v["best_frame_id"],
                "best_frame_area": v["best_frame_area"],
                "best_frame_confidence": v["best_frame_confidence"],
                "speed_limit_kmh": resolved_speed_limit,
                "max_speed_kmh": v["max_speed_kmh"],
                "avg_speed_kmh": v["mean_speed_kmh"],
                "violations": violations_list,
                "violation_count": len(violations_list),
                "fine_breakdown": fine_breakdown,
                "total_fine": total_fine,
                "currency": self.rule_engine.currency,
                "compounding_policy": self.rule_engine.compounding_policy,
                "first_violation_frame": first_frame,
                "first_violation_sec": first_sec,
                "timing_by_violation": v["timing"],
                "disclaimer": self.rule_engine.disclaimer,
            }
            results.append(record)

        return results

    def save_outputs(
        self,
        records: List[Dict[str, Any]],
        out_dir: Union[str, Path],
        basename: str,
        only_violators: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Saves calculated fine records to CSV and JSON files.
        """
        out_path_dir = Path(out_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)

        target_records = [r for r in records if r["violation_count"] > 0] if only_violators else records

        csv_file = out_path_dir / f"{basename}_fines.csv"
        json_file = out_path_dir / f"{basename}_fines_summary.json"

        # Write CSV
        csv_headers = [
            "video_id",
            "vehicle_id",
            "composite_id",
            "class",
            "violations",
            "violation_count",
            "total_fine",
            "currency",
            "speed_limit_kmh",
            "max_speed_kmh",
            "avg_speed_kmh",
            "first_violation_frame",
            "first_violation_sec",
            "best_frame_id",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for r in target_records:
                row = {
                    "video_id": r["video_id"],
                    "vehicle_id": r["vehicle_id"],
                    "composite_id": r["composite_id"],
                    "class": r["class"],
                    "violations": ";".join(r["violations"]),
                    "violation_count": r["violation_count"],
                    "total_fine": r["total_fine"],
                    "currency": r["currency"],
                    "speed_limit_kmh": r["speed_limit_kmh"] if r["speed_limit_kmh"] is not None else "",
                    "max_speed_kmh": r["max_speed_kmh"] if r["max_speed_kmh"] is not None else "",
                    "avg_speed_kmh": r["avg_speed_kmh"] if r["avg_speed_kmh"] is not None else "",
                    "first_violation_frame": r["first_violation_frame"] if r["first_violation_frame"] is not None else "",
                    "first_violation_sec": r["first_violation_sec"] if r["first_violation_sec"] is not None else "",
                    "best_frame_id": r["best_frame_id"] if r["best_frame_id"] is not None else "",
                }
                writer.writerow(row)

        # Write JSON Summary
        summary_payload = {
            "video_id": basename,
            "total_vehicles_evaluated": len(records),
            "total_violating_vehicles": sum(1 for r in records if r["violation_count"] > 0),
            "total_fines_assessed": round(sum(r["total_fine"] for r in records), 2),
            "currency": self.rule_engine.currency,
            "compounding_policy": self.rule_engine.compounding_policy,
            "disclaimer": self.rule_engine.disclaimer,
            "records": target_records,
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return csv_file, json_file
