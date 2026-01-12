"""
Audio analysis module for BPM and key detection using librosa.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa

from .camelot import CamelotWheel
from .metadata import get_metadata_handler
from .embeddings import get_audio_embeddings
from ..core.cache import get_cache


class AudioAnalyzer:
    """Analyzes audio files to extract BPM, key, and duration."""

    # Krumhansl-Schmuckler key profiles for key detection
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Note names for key detection
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, sr: int = 22050, use_cache: bool = True, use_metadata: bool = True,
                 extract_embeddings: bool = True):
        """
        Initialize the analyzer.

        Args:
            sr: Sample rate to use for analysis (default 22050 for speed)
            use_cache: Whether to use cached results (default True)
            use_metadata: Whether to read/write MP3 metadata (default True)
            extract_embeddings: Whether to extract audio embeddings for content matching (default True)
        """
        self.sr = sr
        self.use_cache = use_cache
        self.use_metadata = use_metadata
        self.extract_embeddings = extract_embeddings
        self._cache = get_cache() if use_cache else None
        self._metadata = get_metadata_handler() if use_metadata else None
        self._embeddings = get_audio_embeddings() if extract_embeddings else None

    def analyze_file(self, file_path: Path, force_reanalyze: bool = False) -> dict:
        """
        Analyze a single audio file.

        Args:
            file_path: Path to the audio file
            force_reanalyze: If True, ignore cache/metadata and re-analyze

        Returns:
            Dictionary with keys: duration, bpm, key, camelot, embedding, error, from_cache, from_metadata
        """
        file_path = Path(file_path)

        if not force_reanalyze:
            # Check MP3 metadata first (highest priority)
            if self._metadata:
                metadata = self._metadata.read_metadata(file_path)
                if metadata:
                    # Try to get cached embedding
                    if self._cache:
                        metadata['embedding'] = self._cache.get_embedding(file_path)
                        self._cache.save_result(file_path, metadata)
                    return metadata

            # Check cache second
            if self._cache:
                cached = self._cache.get_cached_result(file_path)
                if cached:
                    # Also try to get cached embedding
                    cached['embedding'] = self._cache.get_embedding(file_path)
                    return cached

        result = {
            'duration': 0.0,
            'bpm': None,
            'key': None,
            'camelot': None,
            'embedding': None,
            'error': None,
            'from_cache': False,
            'from_metadata': False
        }

        try:
            # Load audio file (mono, resampled)
            y, sr = librosa.load(str(file_path), sr=self.sr, mono=True)

            # Calculate duration
            result['duration'] = librosa.get_duration(y=y, sr=sr)

            # Detect BPM
            result['bpm'] = self._detect_bpm(y, sr)

            # Detect key
            key = self._detect_key(y, sr)
            result['key'] = key

            # Convert to Camelot notation
            if key:
                result['camelot'] = CamelotWheel.key_to_camelot(key)

            # Save to cache
            if self._cache and not result['error']:
                self._cache.save_result(file_path, result)

            # Write to MP3 metadata (for MP3 files)
            if self._metadata and not result['error']:
                self._metadata.write_metadata(file_path, result)

        except Exception as e:
            result['error'] = str(e)

        return result

    def extract_embedding(self, file_path: Path) -> dict:
        """
        Extract audio embedding for a file (separate from main analysis for performance).

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with embedding or None
        """
        file_path = Path(file_path)

        # Check cache first
        if self._cache:
            cached_embedding = self._cache.get_embedding(file_path)
            if cached_embedding is not None:
                return {'embedding': cached_embedding, 'from_cache': True}

        # Extract new embedding
        if self._embeddings and self._embeddings.is_available():
            embedding = self._embeddings.extract_embedding(file_path)
            if embedding is not None:
                # Save to cache
                if self._cache:
                    self._cache.save_embedding(file_path, embedding)
                return {'embedding': embedding, 'from_cache': False}

        return {'embedding': None, 'from_cache': False}

    def _detect_bpm(self, y: np.ndarray, sr: int) -> Optional[float]:
        """
        Detect BPM using librosa's beat tracking.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Estimated BPM or None
        """
        try:
            # Use librosa's beat tracker
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Handle numpy array return type in newer versions
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else float(tempo)

            # Round to reasonable precision
            return round(float(tempo), 1)
        except Exception:
            return None

    def _detect_key(self, y: np.ndarray, sr: int) -> Optional[str]:
        """
        Detect musical key using Krumhansl-Schmuckler algorithm.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Key string (e.g., "C major", "A minor") or None
        """
        try:
            # Compute chromagram
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

            # Average chroma across time
            chroma_avg = np.mean(chromagram, axis=1)

            # Normalize
            chroma_avg = chroma_avg / np.max(chroma_avg)

            # Find best matching key using correlation
            best_corr = -1
            best_key = None
            best_mode = None

            for i in range(12):
                # Rotate profiles to test each root note
                major_rotated = np.roll(self.MAJOR_PROFILE, i)
                minor_rotated = np.roll(self.MINOR_PROFILE, i)

                # Calculate correlation
                major_corr = np.corrcoef(chroma_avg, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_avg, minor_rotated)[0, 1]

                if major_corr > best_corr:
                    best_corr = major_corr
                    best_key = self.NOTE_NAMES[i]
                    best_mode = 'major'

                if minor_corr > best_corr:
                    best_corr = minor_corr
                    best_key = self.NOTE_NAMES[i]
                    best_mode = 'minor'

            if best_key and best_mode:
                return f"{best_key} {best_mode}"

            return None

        except Exception:
            return None

    def analyze_files_parallel(self, file_paths: list,
                                max_workers: int = 4,
                                progress_callback=None,
                                force_reanalyze: bool = False) -> dict:
        """
        Analyze multiple files in parallel.

        Args:
            file_paths: List of file paths to analyze
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback(completed, total) for progress updates
            force_reanalyze: If True, ignore cache and re-analyze all files

        Returns:
            Dictionary mapping file_path -> analysis result
        """
        results = {}
        total = len(file_paths)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.analyze_file, path, force_reanalyze): path
                for path in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    results[path] = {
                        'duration': 0.0,
                        'bpm': None,
                        'key': None,
                        'camelot': None,
                        'error': str(e),
                        'from_cache': False
                    }

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def invalidate_cache(self, file_path: Path = None):
        """
        Invalidate cache entries.

        Args:
            file_path: Specific file to invalidate, or None to invalidate all
        """
        if self._cache:
            if file_path:
                self._cache.invalidate(file_path)
            else:
                self._cache.invalidate_all()


def is_supported_format(file_path: Path) -> bool:
    """Check if the file format is supported."""
    supported = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    return file_path.suffix.lower() in supported
