"""
Rule Engine for Fine Estimation
================================
Loads and applies project-defined prototype fine calculation rules based on
vehicle class, violation type, excess speed (via max_speed_kmh), and compounding policy.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RuleEngine:
    """Evaluates fine amounts according to configured prototype rules."""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fine_rules.json"

    def __init__(self, config_path: Optional[str or Path] = None):
        cfg_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Fine rules config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.rules_meta = self.config.get("rules_meta", {})
        self.disclaimer = self.rules_meta.get(
            "disclaimer",
            "Project-defined prototype fine calculation. Not valid for legal enforcement."
        )
        self.compounding_policy = self.config.get("compounding_policy", "sum")
        self.max_total_fine_cap = float(self.config.get("max_total_fine_cap", 25000.0))
        self.currency = self.config.get("currency", "INR")
        self.base_fines = self.config.get("base_fines", {})
        self.overspeed_cfg = self.config.get("overspeed_penalty", {})

    def get_base_fine(self, violation_type: str, vehicle_class: str) -> float:
        """Returns base fine for a given violation type and vehicle class."""
        v_rules = self.base_fines.get(violation_type, {})
        if not v_rules:
            return 0.0
        v_class = (vehicle_class or "default").lower()
        return float(v_rules.get(v_class, v_rules.get("default", 0.0)))

    def calculate_overspeed_penalty(
        self,
        vehicle_class: str,
        max_speed_kmh: Optional[float],
        speed_limit_kmh: Optional[float],
    ) -> Tuple[float, float, str]:
        """
        Calculates excess speed penalty based on max_speed_kmh and speed_limit_kmh.

        Returns:
            (excess_speed_penalty, speed_delta_kmh, details_string)
        """
        if not self.overspeed_cfg.get("enabled", True):
            return 0.0, 0.0, "Overspeed scaling disabled"

        if max_speed_kmh is None or speed_limit_kmh is None:
            return 0.0, 0.0, "Speed measurement unavailable"

        try:
            max_spd = float(max_speed_kmh)
            spd_lim = float(speed_limit_kmh)
        except (ValueError, TypeError):
            return 0.0, 0.0, "Invalid speed numeric values"

        delta = round(max_spd - spd_lim, 2)
        if delta <= 0.0:
            return 0.0, 0.0, f"Speed {max_spd:.1f} km/h within limit {spd_lim:.1f} km/h"

        multipliers = self.overspeed_cfg.get("excess_speed_multiplier_per_kmh", {})
        v_class = (vehicle_class or "default").lower()
        mult = float(multipliers.get(v_class, multipliers.get("default", 25.0)))

        penalty = round(delta * mult, 2)
        details = f"{delta:.1f} km/h over {spd_lim:.1f} km/h limit (max: {max_spd:.1f} km/h, rate: {mult:.1f}/km/h)"
        return penalty, delta, details

    def evaluate_violation(
        self,
        violation_type: str,
        vehicle_class: str,
        max_speed_kmh: Optional[float] = None,
        speed_limit_kmh: Optional[float] = None,
        event_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates fine for a single violation instance.
        """
        base = self.get_base_fine(violation_type, vehicle_class)
        breakdown: Dict[str, Any] = {
            "base_fine": base,
            "subtotal": base,
            "details": "",
        }

        if violation_type == "speeding":
            penalty, delta, det_str = self.calculate_overspeed_penalty(
                vehicle_class, max_speed_kmh, speed_limit_kmh
            )
            breakdown["excess_speed_penalty"] = penalty
            breakdown["speed_delta_kmh"] = delta
            breakdown["subtotal"] = round(base + penalty, 2)
            breakdown["details"] = det_str
        elif violation_type == "wrong_side":
            zone = (event_details or {}).get("zone", "opposite_carriageway")
            breakdown["details"] = f"Wrong lane driving in {zone}"
        elif violation_type == "lane_change":
            from_l = (event_details or {}).get("from_lane", "")
            to_l = (event_details or {}).get("to_lane", "")
            if from_l and to_l:
                breakdown["details"] = f"Unsafe lane change from {from_l} to {to_l}"
            else:
                breakdown["details"] = "Unsafe lane change event"
        elif violation_type == "tailgating":
            gap = (event_details or {}).get("min_gap_lengths")
            if gap is not None:
                breakdown["details"] = f"Tailgating min gap {float(gap):.2f} vehicle lengths"
            else:
                breakdown["details"] = "Tailgating violation"
        else:
            breakdown["details"] = f"Violation: {violation_type}"

        return breakdown

    def compound_fines(self, fine_breakdown: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculates total fine across all violations according to compounding policy.
        """
        if not fine_breakdown:
            return 0.0

        subtotals = [float(v.get("subtotal", 0.0)) for v in fine_breakdown.values()]

        if self.compounding_policy == "max":
            total = max(subtotals) if subtotals else 0.0
        elif self.compounding_policy == "sum_with_cap":
            total = min(sum(subtotals), self.max_total_fine_cap)
        else:  # default 'sum'
            total = sum(subtotals)

        return round(total, 2)
