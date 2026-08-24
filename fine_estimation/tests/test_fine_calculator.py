"""
Unit and Integration Tests for Fine Estimation Module
=====================================================
Covers all 11 required test scenarios:
- TEST 1: Valid vehicle with overspeeding.
- TEST 2: Valid vehicle with no violation.
- TEST 3: Valid vehicle with wrong-side violation.
- TEST 4: Valid vehicle with lane-change violation.
- TEST 5: Valid vehicle with tailgating violation.
- TEST 6: One vehicle with multiple violations.
- TEST 7: vehicle_id = -1 -> must be skipped.
- TEST 8: Vehicle with missing speed -> handled safely.
- TEST 9: Two different videos containing same vehicle_id -> must not be merged incorrectly.
- TEST 10: Speed exactly equal to speed limit -> verify expected rule behaviour.
- TEST 11: Speed just above speed limit -> verify overspeed rule.
"""

import json
import tempfile
import unittest
from pathlib import Path

from fine_estimation.src.rule_engine import RuleEngine
from fine_estimation.src.fine_calculator import FineCalculator


class TestFineEstimation(unittest.TestCase):

    def setUp(self):
        self.engine = RuleEngine()
        self.calc = FineCalculator(rule_engine=self.engine)

    def test_01_valid_vehicle_overspeeding(self):
        """TEST 1: Valid vehicle with overspeeding (max_speed_kmh)."""
        # Car with max_speed = 80.0 km/h against speed_limit = 60.0 km/h
        # Base car speeding fine = 1000.0, excess = 20 km/h * 25.0/kmh = 500.0 -> Total = 1500.0
        breakdown = self.engine.evaluate_violation(
            violation_type="speeding",
            vehicle_class="car",
            max_speed_kmh=80.0,
            speed_limit_kmh=60.0,
        )
        self.assertEqual(breakdown["base_fine"], 1000.0)
        self.assertEqual(breakdown["speed_delta_kmh"], 20.0)
        self.assertEqual(breakdown["excess_speed_penalty"], 500.0)
        self.assertEqual(breakdown["subtotal"], 1500.0)

    def test_02_valid_vehicle_no_violation(self):
        """TEST 2: Valid vehicle with no violation (0 fine)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            veh_csv = Path(tmpdir) / "test_vehicles.csv"
            veh_csv.write_text(
                "vehicle_id,class,class_agreement,n_frames,first_frame,last_frame,duration_sec,mean_confidence,max_confidence,best_frame_id,best_frame_area,best_frame_confidence\n"
                "10,car,1.0,20,0,19,0.67,0.9,0.95,5,12000.0,0.95\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                vehicles_csv=veh_csv,
                speed_limit_kmh=60.0,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 10)
            self.assertEqual(rec["violation_count"], 0)
            self.assertEqual(rec["total_fine"], 0.0)
            self.assertEqual(len(rec["violations"]), 0)

    def test_03_valid_vehicle_wrong_side(self):
        """TEST 3: Valid vehicle with wrong-side violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_csv = Path(tmpdir) / "test_wrong_lane_violations.csv"
            wrong_csv.write_text(
                "vehicle_id,class,zone,wrong_frames,total_judged_frames,mean_alignment,first_wrong_frame,first_wrong_sec,last_wrong_frame\n"
                "5,car,opposite_carriageway,30,30,-0.95,45,1.5,75\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                wrong_lane_csv=wrong_csv,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 5)
            self.assertIn("wrong_side", rec["violations"])
            self.assertEqual(rec["total_fine"], 1500.0)  # base car wrong-side fine
            self.assertEqual(rec["first_violation_frame"], 45)
            self.assertEqual(rec["first_violation_sec"], 1.5)

    def test_04_valid_vehicle_lane_change(self):
        """TEST 4: Valid vehicle with lane-change violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lane_csv = Path(tmpdir) / "test_lane_change_events.csv"
            lane_csv.write_text(
                "vehicle_id,class,from_lane,to_lane,change_frame,change_sec,frames_in_from,frames_in_to\n"
                "11,bike,outer_zone,inner_lane,59,1.97,59,22\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                lane_change_csv=lane_csv,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 11)
            self.assertIn("lane_change", rec["violations"])
            self.assertEqual(rec["total_fine"], 250.0)  # base bike lane-change fine
            self.assertEqual(rec["first_violation_frame"], 59)
            self.assertEqual(rec["first_violation_sec"], 1.97)

    def test_05_valid_vehicle_tailgating(self):
        """TEST 5: Valid vehicle with tailgating violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tail_csv = Path(tmpdir) / "test_tailgating_events.csv"
            tail_csv.write_text(
                "follower_id,leader_id,follower_class,leader_class,frames_tailgating,duration_sec,min_gap_lengths,min_gap_px,at_frame,at_sec\n"
                "20,8,bike,bike,27,0.9,0.679,98.3,24,0.8\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                tailgating_csv=tail_csv,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 20)
            self.assertIn("tailgating", rec["violations"])
            self.assertEqual(rec["total_fine"], 350.0)  # base bike tailgating fine
            self.assertEqual(rec["first_violation_frame"], 24)
            self.assertEqual(rec["first_violation_sec"], 0.8)

    def test_06_vehicle_with_multiple_violations(self):
        """TEST 6: One vehicle with multiple violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            speed_csv = Path(tmpdir) / "test_violations.csv"
            speed_csv.write_text(
                "vehicle_id,class,max_speed_kmh,mean_speed_kmh,frames_over_limit,first_violation_frame,first_violation_sec,track_first_frame,track_last_frame\n"
                "20,bike,90.0,55.0,15,26,0.87,26,54\n",
                encoding="utf-8"
            )
            tail_csv = Path(tmpdir) / "test_tailgating_events.csv"
            tail_csv.write_text(
                "follower_id,leader_id,follower_class,leader_class,frames_tailgating,duration_sec,min_gap_lengths,min_gap_px,at_frame,at_sec\n"
                "20,8,bike,bike,27,0.9,0.679,98.3,24,0.8\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                speed_violations_csv=speed_csv,
                tailgating_csv=tail_csv,
                speed_limit_kmh=60.0,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 20)
            self.assertEqual(set(rec["violations"]), {"speeding", "tailgating"})
            # Speeding: 500 (base bike) + 30 km/h * 15/kmh = 450 -> 950
            # Tailgating: 350 (base bike)
            # Total = 950 + 350 = 1300.0
            self.assertEqual(rec["fine_breakdown"]["speeding"]["subtotal"], 950.0)
            self.assertEqual(rec["fine_breakdown"]["tailgating"]["subtotal"], 350.0)
            self.assertEqual(rec["total_fine"], 1300.0)
            # Earliest frame is 24 (tailgating at frame 24, speed at frame 26)
            self.assertEqual(rec["first_violation_frame"], 24)

    def test_07_invalid_vehicle_id_skipped(self):
        """TEST 7: vehicle_id = -1 -> must be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            speed_csv = Path(tmpdir) / "test_violations.csv"
            speed_csv.write_text(
                "vehicle_id,class,max_speed_kmh,mean_speed_kmh,frames_over_limit,first_violation_frame,first_violation_sec,track_first_frame,track_last_frame\n"
                "-1,car,90.0,55.0,15,26,0.87,26,54\n"
                "15,car,80.0,50.0,10,12,0.4,12,30\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                speed_violations_csv=speed_csv,
                speed_limit_kmh=60.0,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["vehicle_id"], 15)

    def test_08_vehicle_with_missing_speed_handled_safely(self):
        """TEST 8: Vehicle with missing speed -> handled safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_csv = Path(tmpdir) / "test_wrong_lane_violations.csv"
            wrong_csv.write_text(
                "vehicle_id,class,zone,wrong_frames,total_judged_frames,mean_alignment,first_wrong_frame,first_wrong_sec,last_wrong_frame\n"
                "99,truck,opposite_carriageway,10,10,-0.99,100,3.33,110\n",
                encoding="utf-8"
            )
            records = self.calc.process_records(
                video_id="test_vid",
                wrong_lane_csv=wrong_csv,
                speed_limit_kmh=60.0,
            )
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["vehicle_id"], 99)
            self.assertIsNone(rec["max_speed_kmh"])
            self.assertEqual(rec["total_fine"], 2500.0)  # base truck wrong-side fine

    def test_09_two_different_videos_same_vehicle_id(self):
        """TEST 9: Two different videos containing same vehicle_id -> must not be merged incorrectly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            speed_csv1 = Path(tmpdir) / "vid1_violations.csv"
            speed_csv1.write_text(
                "vehicle_id,class,max_speed_kmh,mean_speed_kmh,frames_over_limit,first_violation_frame,first_violation_sec,track_first_frame,track_last_frame\n"
                "25,car,80.0,70.0,10,10,0.33,1,50\n",
                encoding="utf-8"
            )
            speed_csv2 = Path(tmpdir) / "vid2_violations.csv"
            speed_csv2.write_text(
                "vehicle_id,class,max_speed_kmh,mean_speed_kmh,frames_over_limit,first_violation_frame,first_violation_sec,track_first_frame,track_last_frame\n"
                "25,bike,90.0,75.0,15,20,0.66,1,50\n",
                encoding="utf-8"
            )
            records_vid1 = self.calc.process_records(video_id="clip_A", speed_violations_csv=speed_csv1, speed_limit_kmh=60.0)
            records_vid2 = self.calc.process_records(video_id="clip_B", speed_violations_csv=speed_csv2, speed_limit_kmh=60.0)

            self.assertEqual(records_vid1[0]["composite_id"], "clip_A_25")
            self.assertEqual(records_vid1[0]["class"], "car")

            self.assertEqual(records_vid2[0]["composite_id"], "clip_B_25")
            self.assertEqual(records_vid2[0]["class"], "bike")

            self.assertNotEqual(records_vid1[0]["composite_id"], records_vid2[0]["composite_id"])

    def test_10_speed_exactly_equal_to_limit(self):
        """TEST 10: Speed exactly equal to speed limit -> verify expected rule behaviour (0 overspeed penalty)."""
        breakdown = self.engine.evaluate_violation(
            violation_type="speeding",
            vehicle_class="car",
            max_speed_kmh=60.0,
            speed_limit_kmh=60.0,
        )
        self.assertEqual(breakdown["speed_delta_kmh"], 0.0)
        self.assertEqual(breakdown["excess_speed_penalty"], 0.0)
        self.assertEqual(breakdown["subtotal"], 1000.0)  # only base fine

    def test_11_speed_just_above_speed_limit(self):
        """TEST 11: Speed just above speed limit -> verify overspeed rule."""
        breakdown = self.engine.evaluate_violation(
            violation_type="speeding",
            vehicle_class="car",
            max_speed_kmh=60.5,
            speed_limit_kmh=60.0,
        )
        # delta = 0.5 km/h, car rate = 25.0/kmh -> 12.5 penalty
        self.assertEqual(breakdown["speed_delta_kmh"], 0.5)
        self.assertEqual(breakdown["excess_speed_penalty"], 12.5)
        self.assertEqual(breakdown["subtotal"], 1012.5)


if __name__ == "__main__":
    unittest.main()
