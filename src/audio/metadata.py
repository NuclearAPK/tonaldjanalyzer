"""
MP3 metadata handling for storing and retrieving analysis results.
Uses ID3 tags (TXXX frames) to store BPM, key, and Camelot notation.
"""

from pathlib import Path
from typing import Optional, Dict, Any

try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TXXX, TIT2, TBPM, TKEY, TCON
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


# Custom TXXX frame descriptions for our metadata
TXXX_BPM = "TONAL_DJ_BPM"
TXXX_KEY = "TONAL_DJ_KEY"
TXXX_CAMELOT = "TONAL_DJ_CAMELOT"
TXXX_DURATION = "TONAL_DJ_DURATION"


class MetadataHandler:
    """Handles reading and writing analysis metadata to MP3 files."""

    @staticmethod
    def is_mp3(file_path: Path) -> bool:
        """Check if file is an MP3."""
        return file_path.suffix.lower() == '.mp3'

    @staticmethod
    def can_handle_metadata() -> bool:
        """Check if metadata handling is available."""
        return MUTAGEN_AVAILABLE

    def read_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read analysis metadata from MP3 file.

        Args:
            file_path: Path to the MP3 file

        Returns:
            Dictionary with keys: duration, bpm, key, camelot, from_metadata
            or None if metadata not found or file is not MP3
        """
        if not MUTAGEN_AVAILABLE:
            return None

        if not self.is_mp3(file_path):
            return None

        try:
            audio = MP3(str(file_path))

            # Check if we have ID3 tags
            if audio.tags is None:
                return None

            # Read our custom TXXX frames
            bpm = self._get_txxx_value(audio.tags, TXXX_BPM)
            key = self._get_txxx_value(audio.tags, TXXX_KEY)
            camelot = self._get_txxx_value(audio.tags, TXXX_CAMELOT)
            duration = self._get_txxx_value(audio.tags, TXXX_DURATION)

            # Only return if we have at least BPM and key
            if bpm is not None and key is not None:
                return {
                    'duration': float(duration) if duration else audio.info.length,
                    'bpm': float(bpm),
                    'key': key,
                    'camelot': camelot,
                    'error': None,
                    'from_metadata': True
                }

            return None

        except Exception:
            return None

    def write_metadata(self, file_path: Path, result: Dict[str, Any]) -> bool:
        """
        Write analysis metadata to MP3 file.

        Args:
            file_path: Path to the MP3 file
            result: Analysis result dictionary

        Returns:
            True if successful, False otherwise
        """
        if not MUTAGEN_AVAILABLE:
            return False

        if not self.is_mp3(file_path):
            return False

        try:
            audio = MP3(str(file_path))

            # Create ID3 tags if they don't exist
            if audio.tags is None:
                audio.add_tags()

            # Write our custom TXXX frames
            if result.get('bpm') is not None:
                self._set_txxx_value(audio.tags, TXXX_BPM, str(result['bpm']))

            if result.get('key') is not None:
                self._set_txxx_value(audio.tags, TXXX_KEY, result['key'])

            if result.get('camelot') is not None:
                self._set_txxx_value(audio.tags, TXXX_CAMELOT, result['camelot'])

            if result.get('duration') is not None:
                self._set_txxx_value(audio.tags, TXXX_DURATION, str(result['duration']))

            # Also write standard BPM tag for compatibility with other software
            if result.get('bpm') is not None:
                audio.tags.delall('TBPM')
                audio.tags.add(TBPM(encoding=3, text=[str(int(round(result['bpm'])))]))

            # Write standard key tag
            if result.get('key') is not None:
                audio.tags.delall('TKEY')
                audio.tags.add(TKEY(encoding=3, text=[result['key']]))

            audio.save()
            return True

        except Exception:
            return False

    def _get_txxx_value(self, tags, description: str) -> Optional[str]:
        """Get value from a TXXX frame by description."""
        frame_id = f"TXXX:{description}"
        if frame_id in tags:
            frame = tags[frame_id]
            if frame.text:
                return str(frame.text[0])
        return None

    def _set_txxx_value(self, tags, description: str, value: str):
        """Set value in a TXXX frame by description."""
        frame_id = f"TXXX:{description}"
        # Remove existing frame if present
        if frame_id in tags:
            del tags[frame_id]
        # Add new frame
        tags.add(TXXX(encoding=3, desc=description, text=[value]))

    def read_genre(self, file_path: Path) -> Optional[str]:
        """
        Read genre from MP3 metadata (TCON tag).

        Args:
            file_path: Path to the MP3 file

        Returns:
            Genre string or None if not found
        """
        if not MUTAGEN_AVAILABLE:
            return None

        if not self.is_mp3(file_path):
            return None

        try:
            audio = MP3(str(file_path))

            if audio.tags is None:
                return None

            # Read TCON (Content Type / Genre) tag
            if 'TCON' in audio.tags:
                genre = audio.tags['TCON']
                if genre.text:
                    return str(genre.text[0])

            return None

        except Exception:
            return None


# Global instance
_metadata_handler: Optional[MetadataHandler] = None


def get_metadata_handler() -> MetadataHandler:
    """Get the global metadata handler instance."""
    global _metadata_handler
    if _metadata_handler is None:
        _metadata_handler = MetadataHandler()
    return _metadata_handler
