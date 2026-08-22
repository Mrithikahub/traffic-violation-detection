"""
Unit Tests for Dual-Mode (Indian + International) License Plate Post-Processor
"""

import os
import sys

# Add npr_module root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postprocessor import IndianPlatePostProcessor


def test_standard_indian_plates():
    processor = IndianPlatePostProcessor()
    
    test_cases = [
        ("TN38AB1234", "TN38AB1234", True, "STANDARD", "Tamil Nadu"),
        ("MH12DE1433", "MH12DE1433", True, "STANDARD", "Maharashtra"),
        ("DL1CAB1234", "DL1CAB1234", True, "STANDARD", "Delhi"),
        ("KA01MJ9999", "KA01MJ9999", True, "STANDARD", "Karnataka"),
        ("UP32BN5678", "UP32BN5678", True, "STANDARD", "Uttar Pradesh"),
        ("GJ06AQ0001", "GJ06AQ0001", True, "STANDARD", "Gujarat"),
    ]
    
    print("Testing Standard Indian Plates...")
    for raw, expected_std, expected_valid, expected_fmt, expected_state in test_cases:
        res = processor.validate_and_parse(raw)
        assert res["plate_type"] == "INDIAN", f"Expected INDIAN plate_type for {raw}, got {res['plate_type']}"
        assert res["standardized_text"] == expected_std, f"Failed for {raw}: got {res['standardized_text']}"
        assert res["is_valid"] == expected_valid, f"Validity failed for {raw}"
        assert res["format_type"] == expected_fmt, f"Format failed for {raw}"
        assert res["state_name"] == expected_state, f"State failed for {raw}"
        print(f"  [PASS] {raw} -> {res['standardized_text']} ({res['state_name']}, type={res['plate_type']})")


def test_bharat_series():
    processor = IndianPlatePostProcessor()
    
    bh_cases = [
        ("22BH1234AA", "22BH1234AA", True, "BH_SERIES"),
        ("21BH9999Z", "21BH9999Z", True, "BH_SERIES"),
        ("23BH0001AB", "23BH0001AB", True, "BH_SERIES")
    ]
    
    print("\nTesting Bharat (BH) Series...")
    for raw, expected_std, expected_valid, expected_fmt in bh_cases:
        res = processor.validate_and_parse(raw)
        assert res["plate_type"] == "INDIAN", f"Expected INDIAN for {raw}, got {res['plate_type']}"
        assert res["standardized_text"] == expected_std, f"Failed for {raw}: got {res['standardized_text']}"
        assert res["is_valid"] == expected_valid, f"Validity failed for {raw}"
        print(f"  [PASS] {raw} -> {res['standardized_text']} (BH Series)")


def test_character_disambiguation_and_noise():
    processor = IndianPlatePostProcessor()
    
    noisy_cases = [
        # Digit '0' instead of 'O' or vice-versa
        ("TN38ABIZ34", "TN38AB1234"),      # 'I'->'1', 'Z'->'2' in last 4
        ("IND TN 38 AB 1234", "TN38AB1234"),# Spaces and IND watermark
        ("MH-12-DE-1433", "MH12DE1433"),    # Hyphens
        ("DL1CAB123A", "DL1CAB1234"),      # 'A'->'4' in last 4 digits
        ("KA01MJOOO1", "KA01MJ0001"),      # 'O'->'0' in last 4 digits
        ("TN3BAB1234", "TN38AB1234"),      # 'B'->'8' in district code
    ]
    
    print("\nTesting Character Disambiguation & Noise Cleaning on Indian Plates...")
    for raw, expected_std in noisy_cases:
        res = processor.validate_and_parse(raw)
        assert res["plate_type"] == "INDIAN"
        assert res["standardized_text"] == expected_std, f"Failed for {raw}: got {res['standardized_text']}, expected {expected_std}"
        assert res["is_valid"] is True, f"Expected valid for {raw}"
        print(f"  [PASS] {raw:22s} -> {res['standardized_text']}")


def test_foreign_plates_preservation():
    processor = IndianPlatePostProcessor()

    foreign_cases = [
        ("BD21SMX", "BD21SMX", "FOREIGN"),  # UK format
        ("BT63UYN", "BT63UYN", "FOREIGN"),  # UK format
        ("7ABC123", "7ABC123", "FOREIGN"),  # US California format
        ("ABC1234", "ABC1234", "FOREIGN"),  # US standard
        ("1234ABC", "1234ABC", "FOREIGN"),  # European format
    ]

    print("\nTesting Foreign Plates (Non-Destructive OCR Preservation)...")
    for raw, expected_text, expected_type in foreign_cases:
        res = processor.validate_and_parse(raw)
        assert res["plate_type"] == expected_type, f"Expected {expected_type} for {raw}, got {res['plate_type']}"
        assert res["standardized_text"] == expected_text, f"Text corrupted! Expected {expected_text}, got {res['standardized_text']}"
        assert res["is_valid"] is False, "Foreign plate should not pass Indian regex"
        assert res["raw_text"] == raw, "Raw text must be preserved"
        print(f"  [PASS] {raw:14s} -> text='{res['standardized_text']}' (type={res['plate_type']}, is_valid_indian={res['is_valid']})")


def test_unknown_and_noisy_plates():
    processor = IndianPlatePostProcessor()

    unknown_cases = [
        ("XYZ", "XYZ"),               # Short fragment
        ("123", "123"),               # Digits only
        ("SAMPLE99", "SAMPLE99"),     # Arbitrary text
        ("", ""),                     # Empty
    ]

    print("\nTesting Unknown / Unclassified Plates...")
    for raw, expected_text in unknown_cases:
        res = processor.validate_and_parse(raw)
        assert res["plate_type"] == "UNKNOWN", f"Expected UNKNOWN for {raw}, got {res['plate_type']}"
        assert res["is_valid"] is False, f"Expected is_valid=False for {raw}"
        assert res["raw_text"] == raw
        print(f"  [PASS] '{raw:10s}' -> text='{res['standardized_text']}' (type={res['plate_type']})")


if __name__ == "__main__":
    test_standard_indian_plates()
    test_bharat_series()
    test_character_disambiguation_and_noise()
    test_foreign_plates_preservation()
    test_unknown_and_noisy_plates()
    print("\nAll Post-Processor tests passed successfully!")

