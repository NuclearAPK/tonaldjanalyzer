"""
Audio player module using pygame mixer.
"""

from pathlib import Path
from typing import Optional, Callable
import threading
import time

import pygame


class AudioPlayer:
    """Simple audio player with basic controls."""

    def __init__(self):
        """Initialize the audio player."""
        self._initialized = False
        self._current_file: Optional[Path] = None
        self._is_playing = False
        self._is_paused = False
        self._duration = 0.0
        self._position = 0.0
        self._volume = 0.7
        self._position_callback: Optional[Callable[[float], None]] = None
        self._end_callback: Optional[Callable[[], None]] = None
        self._position_thread: Optional[threading.Thread] = None
        self._stop_thread = False

        self._init_pygame()

    def _init_pygame(self):
        """Initialize pygame mixer."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self._initialized = True
        except Exception as e:
            print(f"Failed to initialize audio: {e}")
            self._initialized = False

    def load(self, file_path: Path) -> bool:
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            self.stop()
            pygame.mixer.music.load(str(file_path))
            self._current_file = file_path
            self._position = 0.0

            # Get duration (pygame doesn't provide this directly)
            # We'll rely on the Track's duration from analysis
            return True

        except Exception as e:
            print(f"Failed to load audio: {e}")
            return False

    def play(self):
        """Start or resume playback."""
        if not self._initialized or not self._current_file:
            return

        try:
            if self._is_paused:
                pygame.mixer.music.unpause()
                self._is_paused = False
            else:
                pygame.mixer.music.play()
                self._start_position_tracking()

            self._is_playing = True

        except Exception as e:
            print(f"Failed to play: {e}")

    def pause(self):
        """Pause playback."""
        if not self._initialized or not self._is_playing:
            return

        try:
            pygame.mixer.music.pause()
            self._is_paused = True
            self._is_playing = False
        except Exception as e:
            print(f"Failed to pause: {e}")

    def stop(self):
        """Stop playback."""
        if not self._initialized:
            return

        try:
            self._stop_position_tracking()
            pygame.mixer.music.stop()
            self._is_playing = False
            self._is_paused = False
            self._position = 0.0
        except Exception as e:
            print(f"Failed to stop: {e}")

    def seek(self, position: float):
        """
        Seek to a position in the track.

        Args:
            position: Position in seconds
        """
        if not self._initialized or not self._current_file:
            return

        try:
            was_playing = self._is_playing

            # pygame.mixer.music doesn't support seeking directly for all formats
            # For MP3, we need to reload and start from position
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=position)

            if not was_playing:
                pygame.mixer.music.pause()
                self._is_paused = True
            else:
                self._is_playing = True
                self._start_position_tracking()

            self._position = position

        except Exception as e:
            print(f"Failed to seek: {e}")

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self._initialized:
            return

        self._volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self._volume)

    def get_volume(self) -> float:
        """Get current volume level."""
        return self._volume

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if not self._initialized or not self._is_playing:
            return self._position

        # pygame returns position in milliseconds
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms > 0:
            return self._position + (pos_ms / 1000.0)
        return self._position

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing and not self._is_paused

    def is_paused(self) -> bool:
        """Check if audio is paused."""
        return self._is_paused

    def set_position_callback(self, callback: Optional[Callable[[float], None]]):
        """Set callback for position updates during playback."""
        self._position_callback = callback

    def set_end_callback(self, callback: Optional[Callable[[], None]]):
        """Set callback for when playback ends."""
        self._end_callback = callback

    def _start_position_tracking(self):
        """Start background thread for position tracking."""
        self._stop_thread = False
        self._position_thread = threading.Thread(target=self._track_position, daemon=True)
        self._position_thread.start()

    def _stop_position_tracking(self):
        """Stop position tracking thread."""
        self._stop_thread = True
        if self._position_thread and self._position_thread.is_alive():
            self._position_thread.join(timeout=0.5)

    def _track_position(self):
        """Background thread for tracking playback position."""
        while not self._stop_thread:
            if self._is_playing and not self._is_paused:
                # Check if playback ended
                if not pygame.mixer.music.get_busy():
                    self._is_playing = False
                    if self._end_callback:
                        self._end_callback()
                    break

                # Update position
                if self._position_callback:
                    self._position_callback(self.get_position())

            time.sleep(0.1)

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self._initialized:
            pygame.mixer.quit()
            self._initialized = False
