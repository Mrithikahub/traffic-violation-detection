"""
Indian & International License Plate Post-Processing & Validation Module
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import re
from typing import Dict, Any, Optional, Tuple, List


class IndianPlatePostProcessor:
    """
    Validates, corrects, and standardizes OCR output strings against 
    Indian Motor Vehicle Act registration patterns and Bharat Series (BH) formats,
    while non-destructively preserving and categorizing Foreign and Unknown plate formats.
    """

    # Official 28 Indian States + 8 Union Territories (+ legacy codes)
    VALID_STATE_CODES = {
        # States
        "AP": "Andhra Pradesh",
        "AR": "Arunachal Pradesh",
        "AS": "Assam",
        "BR": "Bihar",
        "CG": "Chhattisgarh",
        "GA": "Goa",
        "GJ": "Gujarat",
        "HR": "Haryana",
        "HP": "Himachal Pradesh",
        "JH": "Jharkhand",
        "KA": "Karnataka",
        "KL": "Kerala",
        "MP": "Madhya Pradesh",
        "MH": "Maharashtra",
        "MN": "Manipur",
        "ML": "Meghalaya",
        "MZ": "Mizoram",
        "NL": "Nagaland",
        "OD": "Odisha",
        "OR": "Odisha (Legacy)",
        "PB": "Punjab",
        "RJ": "Rajasthan",
        "SK": "Sikkim",
        "TN": "Tamil Nadu",
        "TS": "Telangana",
        "TR": "Tripura",
        "UP": "Uttar Pradesh",
        "UK": "Uttarakhand",
        "UA": "Uttarakhand (Legacy)",
        "WB": "West Bengal",
        # Union Territories
        "AN": "Andaman & Nicobar",
        "CH": "Chandigarh",
        "DN": "Dadra & Nagar Haveli",
        "DD": "Daman & Diu",
        "DH": "Dadra, Nagar Haveli, Daman & Diu",
        "DL": "Delhi",
        "JK": "Jammu & Kashmir",
        "LA": "Ladakh",
        "LD": "Lakshadweep",
        "PY": "Puducherry"
    }

    # Common OCR letter-to-digit and digit-to-letter confusion matrices
    DIGIT_TO_LETTER_MAP = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'J',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
        '9': 'P'
    }

    LETTER_TO_DIGIT_MAP = {
        'O': '0',
        'D': '0',
        'Q': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'E': '3',
        'A': '4',
        'S': '5',
        'G': '6',
        'b': '6',
        'T': '7',
        'B': '8',
        'g': '9',
        'q': '9',
        'P': '9'
    }

    # Indian Regex Patterns
    # 1. Standard Indian RTO: [State: 2 letters] [District: 1-2 digits] [Series: 0-3 letters] [Number: 4 digits]
    STANDARD_REGEX = re.compile(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$')
    
    # 2. Bharat (BH) Series: [Year: 2 digits] BH [Number: 4 digits] [Series: 1-2 letters]
    BH_SERIES_REGEX = re.compile(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$')

    # Recognized Positive Foreign Plate Regex Patterns
    FOREIGN_PATTERNS = [
        # UK Standard (e.g., BD21SMX, BT63UYN)
        re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$'),
        # US Standard 7-char formats (e.g., 7ABC123, ABC1234, 123ABC4)
        re.compile(r'^[0-9][A-Z]{3}[0-9]{3}$'),
        re.compile(r'^[A-Z]{3}[0-9]{4}$'),
        re.compile(r'^[0-9]{3}[A-Z]{3}$'),
        re.compile(r'^[0-9]{3}[A-Z]{4}$'),
        # European standard hyphenated/spaced formats (e.g., B-1234-AB, 1234-ABC)
        re.compile(r'^[0-9]{4}[A-Z]{3}$'),
        re.compile(r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,2}$')
    ]

    def __init__(self):
        pass

    def clean_raw_text(self, text: str) -> str:
        """
        Removes special characters, spaces, punctuation, IND watermark artifacts,
        and converts to uppercase.
        """
        if not text:
            return ""
        cleaned = text.upper().strip()
        # Remove common country/watermark prefixes like 'IND' if clearly a prefix
        if cleaned.startswith("IND") and len(cleaned) > 6:
            cleaned = cleaned[3:]
        # Remove any non-alphanumeric character
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        return cleaned

    def _looks_like_indian(self, cleaned_text: str) -> bool:
        """
        Conservative heuristic: Checks if text strongly resembles an Indian plate.
        Returns True if:
        1. Starts with a valid 2-letter Indian state code and has length 7-11
        2. Matches BH-series structure (e.g. 2 digits + BH + 4 digits)
        3. Starts with an OCR state code typo (e.g. 0L for DL, 1N for TN) followed by digits
        """
        n = len(cleaned_text)
        if n < 7 or n > 12:
            return False

        prefix2 = cleaned_text[:2]
        if prefix2 in self.VALID_STATE_CODES:
            # Check if 3rd char is a digit or common OCR digit typo (O, I, Z, A, S, B)
            if cleaned_text[2].isdigit() or cleaned_text[2] in "OIZASB":
                return True

        # Check BH series pattern
        if n >= 8 and (cleaned_text[0].isdigit() or cleaned_text[1].isdigit()) and ('BH' in cleaned_text[2:4] or '8H' in cleaned_text[2:4]):
            return True

        # Known OCR typos for state codes
        typo_state_prefixes = {"0L", "0D", "1N", "1S", "K4", "M8", "U9"}
        if prefix2 in typo_state_prefixes and (cleaned_text[2].isdigit() or cleaned_text[2] in "OIZASB"):
            return True

        return False

    def _matches_positive_foreign(self, cleaned_text: str) -> bool:
        """
        Checks if cleaned text positively matches a known international plate structure.
        """
        if not cleaned_text or len(cleaned_text) < 4:
            return False
        for pattern in self.FOREIGN_PATTERNS:
            if pattern.match(cleaned_text):
                return True
        return False

    def correct_positional_characters(self, text: str) -> str:
        """
        Applies domain-specific positional disambiguation based on Indian plate syntax rules:
        - Characters 0-1 (State code) must be letters.
        - Characters 2-3 (District RTO) must be digits.
        - Last 4 characters must be digits (for standard plates).
        """
        cleaned = self.clean_raw_text(text)
        if len(cleaned) < 6:
            return cleaned

        chars = list(cleaned)
        n = len(chars)

        # Check for BH series pattern: begins with 2 digits followed by 'BH'
        if n >= 8 and (chars[0].isdigit() or chars[1].isdigit()) and (''.join(chars[2:4]) in ['BH', '8H', 'B#', 'RH']):
            chars[0] = self.LETTER_TO_DIGIT_MAP.get(chars[0], chars[0])
            chars[1] = self.LETTER_TO_DIGIT_MAP.get(chars[1], chars[1])
            chars[2] = 'B'
            chars[3] = 'H'
            # 4 digits
            for i in range(4, min(8, n)):
                chars[i] = self.LETTER_TO_DIGIT_MAP.get(chars[i], chars[i])
            # Last letters
            for i in range(8, n):
                chars[i] = self.DIGIT_TO_LETTER_MAP.get(chars[i], chars[i])
            return ''.join(chars)

        # Standard Indian Format Rule:
        # 1. State Code (index 0, 1) -> Must be Alphabetic
        for i in range(min(2, n)):
            if chars[i].isdigit():
                chars[i] = self.DIGIT_TO_LETTER_MAP.get(chars[i], chars[i])

        # If State code is close to a valid state, rectify it
        state_candidate = ''.join(chars[:2])
        if state_candidate not in self.VALID_STATE_CODES:
            rect_map = {"0L": "DL", "0D": "OD", "1N": "TN", "1S": "TS", "K4": "KA", "M8": "MH", "U9": "UP"}
            if state_candidate in rect_map:
                chars[0] = rect_map[state_candidate][0]
                chars[1] = rect_map[state_candidate][1]

        # 2. District code & series disambiguation
        # Case A: Length 10 (Standard 2-digit RTO + 2-letter series + 4-digit number: AA 00 AA 0000)
        if n == 10:
            chars[2] = self.LETTER_TO_DIGIT_MAP.get(chars[2], chars[2])
            chars[3] = self.LETTER_TO_DIGIT_MAP.get(chars[3], chars[3])
            chars[4] = self.DIGIT_TO_LETTER_MAP.get(chars[4], chars[4])
            chars[5] = self.DIGIT_TO_LETTER_MAP.get(chars[5], chars[5])
        # Case B: Length 9 (e.g. DL 1 CA 1234 or TN 38 A 1234)
        elif n == 9:
            chars[2] = self.LETTER_TO_DIGIT_MAP.get(chars[2], chars[2])
            if chars[3].isalpha() and chars[4].isalpha():
                chars[3] = self.DIGIT_TO_LETTER_MAP.get(chars[3], chars[3])
                chars[4] = self.DIGIT_TO_LETTER_MAP.get(chars[4], chars[4])
            else:
                chars[3] = self.LETTER_TO_DIGIT_MAP.get(chars[3], chars[3])
                chars[4] = self.DIGIT_TO_LETTER_MAP.get(chars[4], chars[4])
        # Case C: Length >= 11 (e.g. 3-letter series like DL 01 CAB 1234)
        elif n >= 11:
            chars[2] = self.LETTER_TO_DIGIT_MAP.get(chars[2], chars[2])
            chars[3] = self.LETTER_TO_DIGIT_MAP.get(chars[3], chars[3])
            for i in range(4, n - 4):
                chars[i] = self.DIGIT_TO_LETTER_MAP.get(chars[i], chars[i])
        elif n >= 4:
            chars[2] = self.LETTER_TO_DIGIT_MAP.get(chars[2], chars[2])

        # 3. Last 4 characters -> Must be Digits
        for i in range(max(0, n - 4), n):
            chars[i] = self.LETTER_TO_DIGIT_MAP.get(chars[i], chars[i])

        return ''.join(chars)

    def validate_and_parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Runs conservative multi-format validation and parsing.
        
        Plate Type Decision:
        - INDIAN: Strong match/resemblance to Indian RTO or BH series.
        - FOREIGN: Positive evidence of known foreign plate syntax.
        - UNKNOWN: OCR text exists but cannot be confidently categorized.
        
        Raw OCR text is always preserved in `raw_text` and `cleaned_text`.
        """
        cleaned = self.clean_raw_text(raw_text)

        result: Dict[str, Any] = {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "standardized_text": cleaned,
            "plate_type": "UNKNOWN",
            "is_valid": False,
            "format_type": "UNKNOWN",
            "state_code": None,
            "state_name": None,
            "district_code": None,
            "series_code": None,
            "unique_number": None,
            "validation_score": 0.0
        }

        if not cleaned:
            return result

        # 1. Check if it looks like an Indian plate
        if self._looks_like_indian(cleaned):
            # Apply Indian-specific positional disambiguation
            corrected = self.correct_positional_characters(cleaned)
            result["standardized_text"] = corrected
            result["plate_type"] = "INDIAN"

            # Check Standard Indian Format
            std_match = self.STANDARD_REGEX.match(corrected)
            if std_match:
                state, district, series, number = std_match.groups()
                is_valid_state = state in self.VALID_STATE_CODES
                result["format_type"] = "STANDARD"
                result["state_code"] = state
                result["state_name"] = self.VALID_STATE_CODES.get(state, "Unknown State")
                result["district_code"] = district
                result["series_code"] = series
                result["unique_number"] = number
                result["is_valid"] = is_valid_state
                result["validation_score"] = 1.0 if is_valid_state else 0.85
                return result

            # Check Bharat (BH) Series Format
            bh_match = self.BH_SERIES_REGEX.match(corrected)
            if bh_match:
                year, number, series = bh_match.groups()
                result["format_type"] = "BH_SERIES"
                result["state_code"] = "BH"
                result["state_name"] = "Bharat Series (All India)"
                result["district_code"] = year
                result["series_code"] = series
                result["unique_number"] = number
                result["is_valid"] = True
                result["validation_score"] = 1.0
                return result

            # Partial Indian match (e.g. valid state code prefix but OCR noise in number)
            state_sub = corrected[:2]
            if state_sub in self.VALID_STATE_CODES:
                result["format_type"] = "NON_STANDARD_INDIAN"
                result["state_code"] = state_sub
                result["state_name"] = self.VALID_STATE_CODES.get(state_sub)
                result["is_valid"] = False
                result["validation_score"] = 0.50
                return result

        # 2. Check for Positive Foreign Plate Format
        if self._matches_positive_foreign(cleaned):
            result["plate_type"] = "FOREIGN"
            result["format_type"] = "FOREIGN_STANDARD"
            result["standardized_text"] = cleaned  # Unmodified raw cleaned text
            result["is_valid"] = False            # Not an Indian format
            result["validation_score"] = 0.80
            return result

        # 3. Fallback: UNKNOWN (do not assume foreign or Indian)
        result["plate_type"] = "UNKNOWN"
        result["format_type"] = "UNKNOWN"
        result["standardized_text"] = cleaned
        result["is_valid"] = False
        result["validation_score"] = 0.20 if len(cleaned) >= 4 else 0.0

        return result

