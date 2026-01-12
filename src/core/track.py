from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import numpy as np


# Available BPM multipliers
BPM_MULTIPLIERS = [0.5, 1.0, 2.0]


@dataclass
class Track:
    """Represents an audio track with its metadata and analysis results."""

    file_path: Path
    filename: str = field(init=False)
    duration: float = 0.0  # seconds
    bpm: Optional[float] = None
    bpm_multiplier: float = 1.0  # BPM multiplier (0.5, 1.0, or 2.0)
    key: Optional[str] = None  # Musical key (e.g., "C major", "A minor")
    camelot: Optional[str] = None  # Camelot notation (e.g., "8A", "9B")
    compatibility_score: Optional[float] = None  # 0-100 (harmonic + BPM)
    content_score: Optional[float] = None  # 0-100 (content similarity)
    combined_score: Optional[float] = None  # 0-100 (overall match)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)  # Audio embedding
    is_master: bool = False
    is_analyzed: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        self.filename = self.file_path.name

    @property
    def duration_str(self) -> str:
        """Returns duration as MM:SS string."""
        if self.duration <= 0:
            return "--:--"
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes}:{seconds:02d}"

    @property
    def effective_bpm(self) -> Optional[float]:
        """Returns BPM adjusted by multiplier."""
        if self.bpm is None:
            return None
        return self.bpm * self.bpm_multiplier

    @property
    def bpm_str(self) -> str:
        """Returns effective BPM as string with multiplier indicator."""
        if self.bpm is None:
            return "--"
        eff_bpm = self.effective_bpm
        if self.bpm_multiplier == 1.0:
            return f"{eff_bpm:.0f}"
        elif self.bpm_multiplier == 0.5:
            return f"{eff_bpm:.0f} (x0.5)"
        else:
            return f"{eff_bpm:.0f} (x2)"

    @property
    def original_bpm_str(self) -> str:
        """Returns original BPM as string."""
        if self.bpm is None:
            return "--"
        return f"{self.bpm:.0f}"

    def cycle_bpm_multiplier(self):
        """Cycle through BPM multipliers: 1.0 -> 2.0 -> 0.5 -> 1.0"""
        if self.bpm_multiplier == 1.0:
            self.bpm_multiplier = 2.0
        elif self.bpm_multiplier == 2.0:
            self.bpm_multiplier = 0.5
        else:
            self.bpm_multiplier = 1.0

    @property
    def camelot_str(self) -> str:
        """Returns Camelot notation or placeholder."""
        return self.camelot if self.camelot else "--"

    @property
    def compatibility_str(self) -> str:
        """Returns compatibility as percentage string."""
        if self.compatibility_score is None:
            return "--"
        return f"{self.compatibility_score:.0f}%"

    @property
    def content_str(self) -> str:
        """Returns content similarity as percentage string."""
        if self.content_score is None:
            return "--"
        return f"{self.content_score:.0f}%"

    @property
    def combined_str(self) -> str:
        """Returns combined score as percentage string."""
        if self.combined_score is None:
            return "--"
        return f"{self.combined_score:.0f}%"

    @property
    def has_embedding(self) -> bool:
        """Check if embedding is available."""
        return self.embedding is not None

    def reset_compatibility(self):
        """Reset compatibility scores (used when master track changes)."""
        self.compatibility_score = None
        self.content_score = None
        self.combined_score = None
        self.is_master = False
