"""
Compatibility calculator for track matching based on BPM and key.
"""

from typing import Optional, Tuple
from .track import Track
from ..audio.camelot import CamelotWheel


class CompatibilityCalculator:
    """
    Calculates compatibility scores between tracks based on BPM and key.

    Formula: score = key_weight * key_score + bpm_weight * bpm_score
    """

    # Default weights
    KEY_WEIGHT = 0.6
    BPM_WEIGHT = 0.4

    # BPM tolerance thresholds
    BPM_PERFECT_THRESHOLD = 0.02  # 2% - perfect match
    BPM_GOOD_THRESHOLD = 0.06    # 6% - still mixable
    BPM_MAX_THRESHOLD = 0.10    # 10% - maximum useful range

    def __init__(self, key_weight: float = 0.6, bpm_weight: float = 0.4):
        """
        Initialize calculator with custom weights.

        Args:
            key_weight: Weight for key compatibility (0-1)
            bpm_weight: Weight for BPM compatibility (0-1)
        """
        self.key_weight = key_weight
        self.bpm_weight = bpm_weight

    def calculate_bpm_score(self, bpm1: float, bpm2: float) -> float:
        """
        Calculate BPM compatibility score.

        Takes into account that some tracks might be at double/half tempo
        (e.g., 140 BPM dubstep vs 70 BPM half-time).

        Args:
            bpm1: First track BPM
            bpm2: Second track BPM

        Returns:
            Score from 0 to 100
        """
        if bpm1 <= 0 or bpm2 <= 0:
            return 0.0

        # Check normal difference
        diff_normal = abs(bpm1 - bpm2) / bpm1

        # Check double/half tempo relationships
        diff_double = abs(bpm1 - bpm2 * 2) / bpm1
        diff_half = abs(bpm1 - bpm2 / 2) / bpm1

        # Use the smallest difference
        diff = min(diff_normal, diff_double, diff_half)

        # Calculate score
        if diff <= self.BPM_PERFECT_THRESHOLD:
            return 100.0
        elif diff <= self.BPM_GOOD_THRESHOLD:
            # Linear interpolation from 100 to 70
            t = (diff - self.BPM_PERFECT_THRESHOLD) / (self.BPM_GOOD_THRESHOLD - self.BPM_PERFECT_THRESHOLD)
            return 100.0 - (t * 30.0)
        elif diff <= self.BPM_MAX_THRESHOLD:
            # Linear interpolation from 70 to 30
            t = (diff - self.BPM_GOOD_THRESHOLD) / (self.BPM_MAX_THRESHOLD - self.BPM_GOOD_THRESHOLD)
            return 70.0 - (t * 40.0)
        else:
            # Beyond max threshold - rapid falloff
            over = diff - self.BPM_MAX_THRESHOLD
            return max(0.0, 30.0 - (over * 200))

    def calculate_key_score(self, camelot1: str, camelot2: str) -> float:
        """
        Calculate key compatibility score using Camelot wheel.

        Args:
            camelot1: First track's Camelot notation
            camelot2: Second track's Camelot notation

        Returns:
            Score from 0 to 100
        """
        if not camelot1 or not camelot2:
            return 50.0  # Neutral score if key unknown

        return CamelotWheel.get_key_score(camelot1, camelot2)

    def calculate_compatibility(self, master: Track, other: Track) -> float:
        """
        Calculate overall compatibility score between two tracks.

        Uses effective_bpm (with multiplier applied) for BPM comparison.

        Args:
            master: The reference track
            other: The track to compare

        Returns:
            Score from 0 to 100
        """
        # Calculate individual scores
        key_score = self.calculate_key_score(master.camelot, other.camelot)

        # BPM score using effective BPM (with multiplier)
        master_bpm = master.effective_bpm
        other_bpm = other.effective_bpm

        if master_bpm and other_bpm:
            bpm_score = self.calculate_bpm_score(master_bpm, other_bpm)
        else:
            bpm_score = 50.0  # Neutral if BPM unknown

        # Weighted combination
        total = (self.key_weight * key_score) + (self.bpm_weight * bpm_score)

        return round(total, 1)

    def update_track_compatibility(self, master: Track, tracks: list):
        """
        Update compatibility scores for all tracks relative to master.

        Args:
            master: The master track
            tracks: List of all tracks to update
        """
        for track in tracks:
            if track == master or track.file_path == master.file_path:
                track.compatibility_score = None
                track.is_master = True
            else:
                track.is_master = False
                track.compatibility_score = self.calculate_compatibility(master, track)

    @staticmethod
    def get_compatibility_color(score: Optional[float]) -> Tuple[int, int, int]:
        """
        Get RGB color for a compatibility score.

        Color scale:
        - 80-100%: Green (#4CAF50)
        - 60-80%: Yellow-Green (#8BC34A)
        - 40-60%: Yellow (#FFEB3B)
        - 20-40%: Orange (#FF9800)
        - 0-20%: Red (#F44336)

        Args:
            score: Compatibility score (0-100) or None

        Returns:
            RGB tuple (r, g, b)
        """
        if score is None:
            return (128, 128, 128)  # Gray for no score

        if score >= 80:
            return (76, 175, 80)    # Green
        elif score >= 60:
            return (139, 195, 74)   # Yellow-Green
        elif score >= 40:
            return (255, 235, 59)   # Yellow
        elif score >= 20:
            return (255, 152, 0)    # Orange
        else:
            return (244, 67, 54)    # Red

    @staticmethod
    def get_compatibility_label(score: Optional[float]) -> str:
        """
        Get human-readable label for compatibility score.

        Args:
            score: Compatibility score (0-100) or None

        Returns:
            Label string
        """
        if score is None:
            return "N/A"

        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Bad"
