"""
Tests for Camelot Wheel logic.
"""

import unittest
from src.audio.camelot import CamelotWheel


class TestCamelotWheel(unittest.TestCase):
    """Test cases for CamelotWheel class."""

    def test_key_to_camelot_major(self):
        """Test major key conversion."""
        self.assertEqual(CamelotWheel.key_to_camelot("C major"), "8B")
        self.assertEqual(CamelotWheel.key_to_camelot("G major"), "9B")
        self.assertEqual(CamelotWheel.key_to_camelot("A major"), "11B")

    def test_key_to_camelot_minor(self):
        """Test minor key conversion."""
        self.assertEqual(CamelotWheel.key_to_camelot("A minor"), "8A")
        self.assertEqual(CamelotWheel.key_to_camelot("E minor"), "9A")
        self.assertEqual(CamelotWheel.key_to_camelot("F# minor"), "11A")

    def test_parse_camelot(self):
        """Test Camelot code parsing."""
        self.assertEqual(CamelotWheel.parse_camelot("8A"), (8, "A"))
        self.assertEqual(CamelotWheel.parse_camelot("12B"), (12, "B"))
        self.assertIsNone(CamelotWheel.parse_camelot("13A"))
        self.assertIsNone(CamelotWheel.parse_camelot("invalid"))

    def test_distance_same_key(self):
        """Test distance for same key."""
        self.assertEqual(CamelotWheel.calculate_distance("8A", "8A"), 0)
        self.assertEqual(CamelotWheel.calculate_distance("12B", "12B"), 0)

    def test_distance_relative_major_minor(self):
        """Test distance for relative major/minor (same number)."""
        self.assertEqual(CamelotWheel.calculate_distance("8A", "8B"), 1)
        self.assertEqual(CamelotWheel.calculate_distance("5B", "5A"), 1)

    def test_distance_adjacent_numbers(self):
        """Test distance for adjacent numbers same letter."""
        self.assertEqual(CamelotWheel.calculate_distance("8A", "9A"), 1)
        self.assertEqual(CamelotWheel.calculate_distance("8A", "7A"), 1)

    def test_distance_wrap_around(self):
        """Test distance wrapping (1 and 12 are adjacent)."""
        self.assertEqual(CamelotWheel.calculate_distance("1A", "12A"), 1)
        self.assertEqual(CamelotWheel.calculate_distance("12B", "1B"), 1)

    def test_distance_opposite(self):
        """Test distance for opposite keys."""
        self.assertEqual(CamelotWheel.calculate_distance("1A", "7A"), 6)
        self.assertEqual(CamelotWheel.calculate_distance("6B", "12B"), 6)

    def test_key_score_perfect(self):
        """Test perfect match score."""
        self.assertEqual(CamelotWheel.get_key_score("8A", "8A"), 100.0)

    def test_key_score_adjacent(self):
        """Test adjacent key score."""
        self.assertEqual(CamelotWheel.get_key_score("8A", "8B"), 85.0)
        self.assertEqual(CamelotWheel.get_key_score("8A", "9A"), 85.0)

    def test_compatible_keys(self):
        """Test compatible keys list."""
        compatible = CamelotWheel.get_compatible_keys("8A")

        # Should include same key, relative major, and adjacent numbers
        codes = [c[0] for c in compatible]
        self.assertIn("8A", codes)  # Same
        self.assertIn("8B", codes)  # Relative major
        self.assertIn("7A", codes)  # Previous
        self.assertIn("9A", codes)  # Next


if __name__ == "__main__":
    unittest.main()
