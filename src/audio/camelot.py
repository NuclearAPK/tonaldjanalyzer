"""
Camelot Wheel implementation for musical key compatibility.

The Camelot Wheel is a visual representation of musical keys arranged in a circle,
making it easy to identify compatible keys for harmonic mixing.

Structure:
- Numbers 1-12 arranged in a circle
- Each number has two modes: A (minor) and B (major)
- Adjacent keys (same number different letter, or +/-1 same letter) are compatible
"""

from typing import Optional, Tuple


class CamelotWheel:
    """Handles Camelot notation conversion and compatibility calculations."""

    # Mapping from musical key to Camelot notation
    # Format: (key_name, mode) -> camelot_code
    KEY_TO_CAMELOT = {
        # Minor keys (A)
        ('A', 'minor'): '8A',
        ('E', 'minor'): '9A',
        ('B', 'minor'): '10A',
        ('F#', 'minor'): '11A',
        ('Gb', 'minor'): '11A',
        ('C#', 'minor'): '12A',
        ('Db', 'minor'): '12A',
        ('G#', 'minor'): '1A',
        ('Ab', 'minor'): '1A',
        ('D#', 'minor'): '2A',
        ('Eb', 'minor'): '2A',
        ('A#', 'minor'): '3A',
        ('Bb', 'minor'): '3A',
        ('F', 'minor'): '4A',
        ('C', 'minor'): '5A',
        ('G', 'minor'): '6A',
        ('D', 'minor'): '7A',

        # Major keys (B)
        ('C', 'major'): '8B',
        ('G', 'major'): '9B',
        ('D', 'major'): '10B',
        ('A', 'major'): '11B',
        ('E', 'major'): '12B',
        ('B', 'major'): '1B',
        ('F#', 'major'): '2B',
        ('Gb', 'major'): '2B',
        ('C#', 'major'): '3B',
        ('Db', 'major'): '3B',
        ('G#', 'major'): '4B',
        ('Ab', 'major'): '4B',
        ('D#', 'major'): '5B',
        ('Eb', 'major'): '5B',
        ('A#', 'major'): '6B',
        ('Bb', 'major'): '6B',
        ('F', 'major'): '7B',
    }

    # Reverse mapping for display purposes
    CAMELOT_TO_KEY = {
        '1A': 'Ab minor', '1B': 'B major',
        '2A': 'Eb minor', '2B': 'Gb major',
        '3A': 'Bb minor', '3B': 'Db major',
        '4A': 'F minor', '4B': 'Ab major',
        '5A': 'C minor', '5B': 'Eb major',
        '6A': 'G minor', '6B': 'Bb major',
        '7A': 'D minor', '7B': 'F major',
        '8A': 'A minor', '8B': 'C major',
        '9A': 'E minor', '9B': 'G major',
        '10A': 'B minor', '10B': 'D major',
        '11A': 'F# minor', '11B': 'A major',
        '12A': 'C# minor', '12B': 'E major',
    }

    @classmethod
    def key_to_camelot(cls, key: str) -> Optional[str]:
        """
        Convert a musical key string to Camelot notation.

        Args:
            key: Musical key string (e.g., "C major", "A minor", "F# minor")

        Returns:
            Camelot code (e.g., "8B", "8A", "11A") or None if not recognized
        """
        if not key:
            return None

        # Parse key string
        parts = key.strip().split()
        if len(parts) < 2:
            return None

        root = parts[0]
        mode = parts[-1].lower()

        # Normalize mode
        if mode in ('min', 'm'):
            mode = 'minor'
        elif mode in ('maj', 'M'):
            mode = 'major'

        return cls.KEY_TO_CAMELOT.get((root, mode))

    @classmethod
    def parse_camelot(cls, camelot: str) -> Optional[Tuple[int, str]]:
        """
        Parse Camelot code into number and letter.

        Args:
            camelot: Camelot code (e.g., "8A", "12B")

        Returns:
            Tuple of (number, letter) or None if invalid
        """
        if not camelot or len(camelot) < 2:
            return None

        try:
            letter = camelot[-1].upper()
            number = int(camelot[:-1])

            if letter not in ('A', 'B') or not 1 <= number <= 12:
                return None

            return (number, letter)
        except ValueError:
            return None

    @classmethod
    def calculate_distance(cls, camelot1: str, camelot2: str) -> Optional[int]:
        """
        Calculate the distance between two Camelot codes on the wheel.

        Distance rules:
        - Same code: 0
        - Same number, different letter (A<->B): 1
        - Adjacent numbers, same letter: 1
        - Two steps on wheel: 2
        - etc.

        Args:
            camelot1: First Camelot code
            camelot2: Second Camelot code

        Returns:
            Distance (0-6) or None if codes are invalid
        """
        parsed1 = cls.parse_camelot(camelot1)
        parsed2 = cls.parse_camelot(camelot2)

        if not parsed1 or not parsed2:
            return None

        num1, letter1 = parsed1
        num2, letter2 = parsed2

        # Calculate circular distance for numbers (1-12 wheel)
        num_distance = min(
            abs(num1 - num2),
            12 - abs(num1 - num2)
        )

        # Letter distance (A<->B = 1 step)
        letter_distance = 0 if letter1 == letter2 else 1

        # Combined distance
        # Same number, different letter is considered 1 step (relative major/minor)
        if num_distance == 0:
            return letter_distance

        # Different numbers with same letter - just number distance
        if letter_distance == 0:
            return num_distance

        # Different numbers AND different letters - combined distance
        # This is less compatible, so we add them
        return num_distance + letter_distance

    @classmethod
    def get_compatible_keys(cls, camelot: str) -> list:
        """
        Get list of compatible Camelot codes for a given code.

        Returns codes with distance 0-1 (most compatible).

        Args:
            camelot: Camelot code

        Returns:
            List of compatible Camelot codes sorted by compatibility
        """
        parsed = cls.parse_camelot(camelot)
        if not parsed:
            return []

        num, letter = parsed
        compatible = []

        # Same key (distance 0)
        compatible.append((camelot, 0))

        # Relative major/minor (same number, different letter)
        other_letter = 'B' if letter == 'A' else 'A'
        compatible.append((f"{num}{other_letter}", 1))

        # Adjacent numbers (same letter)
        prev_num = 12 if num == 1 else num - 1
        next_num = 1 if num == 12 else num + 1
        compatible.append((f"{prev_num}{letter}", 1))
        compatible.append((f"{next_num}{letter}", 1))

        return compatible

    @classmethod
    def get_key_score(cls, camelot1: str, camelot2: str) -> float:
        """
        Calculate compatibility score (0-100) between two Camelot codes.

        Args:
            camelot1: First Camelot code (usually master track)
            camelot2: Second Camelot code

        Returns:
            Score from 0 (incompatible) to 100 (perfect match)
        """
        distance = cls.calculate_distance(camelot1, camelot2)

        if distance is None:
            return 0.0

        # Score mapping based on distance
        score_map = {
            0: 100.0,  # Same key - perfect
            1: 85.0,   # Adjacent - excellent
            2: 60.0,   # Two steps - good
            3: 35.0,   # Three steps - fair
            4: 15.0,   # Four steps - poor
            5: 5.0,    # Five steps - very poor
            6: 0.0,    # Opposite - incompatible
        }

        return score_map.get(distance, 0.0)
