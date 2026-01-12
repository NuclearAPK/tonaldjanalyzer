"""
Audio player widget with playback controls.
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSlider, QLabel, QStyle
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from ..audio.player import AudioPlayer
from ..core.track import Track


class PlayerWidget(QWidget):
    """Widget with audio playback controls."""

    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = AudioPlayer()
        self._current_track: Optional[Track] = None
        self._duration = 0.0

        self._setup_ui()
        self._setup_timer()
        self._connect_signals()

    def _setup_ui(self):
        """Create UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Track info
        info_layout = QHBoxLayout()

        self._track_label = QLabel("No track loaded")
        self._track_label.setObjectName("titleLabel")
        info_layout.addWidget(self._track_label)

        info_layout.addStretch()

        self._time_label = QLabel("--:-- / --:--")
        info_layout.addWidget(self._time_label)

        layout.addLayout(info_layout)

        # Progress slider
        self._progress_slider = QSlider(Qt.Horizontal)
        self._progress_slider.setMinimum(0)
        self._progress_slider.setMaximum(1000)
        self._progress_slider.setValue(0)
        self._progress_slider.setEnabled(False)
        layout.addWidget(self._progress_slider)

        # Controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self._play_button = QPushButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._play_button.setFixedSize(40, 40)
        self._play_button.setEnabled(False)
        controls_layout.addWidget(self._play_button)

        # Stop button
        self._stop_button = QPushButton()
        self._stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self._stop_button.setFixedSize(40, 40)
        self._stop_button.setEnabled(False)
        controls_layout.addWidget(self._stop_button)

        controls_layout.addSpacing(20)

        # Volume
        volume_label = QLabel("Volume:")
        controls_layout.addWidget(volume_label)

        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setMinimum(0)
        self._volume_slider.setMaximum(100)
        self._volume_slider.setValue(70)
        self._volume_slider.setFixedWidth(100)
        controls_layout.addWidget(self._volume_slider)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

    def _setup_timer(self):
        """Setup timer for position updates."""
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_position)

    def _connect_signals(self):
        """Connect UI signals."""
        self._play_button.clicked.connect(self._toggle_playback)
        self._stop_button.clicked.connect(self._stop)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        self._progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self._progress_slider.sliderReleased.connect(self._on_slider_released)

        self._player.set_end_callback(self._on_playback_ended)

    def load_track(self, track: Track, auto_play: bool = False):
        """Load a track for playback."""
        # Stop current playback first
        if self._player.is_playing():
            self._player.stop()
            self._timer.stop()

        self._current_track = track
        self._duration = track.duration

        # Reset position
        self._progress_slider.setValue(0)
        self._update_time_display(0)

        if self._player.load(track.file_path):
            self._track_label.setText(track.filename)
            self._progress_slider.setEnabled(True)
            self._play_button.setEnabled(True)
            self._stop_button.setEnabled(True)

            # Auto-play if requested
            if auto_play:
                self._player.play()
                self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self._timer.start()
                self.playback_started.emit()
            else:
                self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self._track_label.setText("Failed to load track")
            self._progress_slider.setEnabled(False)
            self._play_button.setEnabled(False)
            self._stop_button.setEnabled(False)

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self._player.is_playing():
            self._player.pause()
            self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self._timer.stop()
        else:
            self._player.play()
            self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self._timer.start()
            self.playback_started.emit()

    def _stop(self):
        """Stop playback."""
        self._player.stop()
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._timer.stop()
        self._progress_slider.setValue(0)
        self._update_time_display(0)
        self.playback_stopped.emit()

    def _on_volume_changed(self, value: int):
        """Handle volume slider change."""
        self._player.set_volume(value / 100.0)

    def _on_slider_pressed(self):
        """Handle slider press - pause position updates."""
        self._timer.stop()

    def _on_slider_released(self):
        """Handle slider release - seek to position."""
        if self._duration > 0:
            position = (self._progress_slider.value() / 1000.0) * self._duration
            self._player.seek(position)

            if self._player.is_playing():
                self._timer.start()

    def _update_position(self):
        """Update position display from player."""
        if not self._player.is_playing():
            return

        position = self._player.get_position()
        self._update_time_display(position)

        if self._duration > 0:
            slider_value = int((position / self._duration) * 1000)
            self._progress_slider.blockSignals(True)
            self._progress_slider.setValue(slider_value)
            self._progress_slider.blockSignals(False)

    def _update_time_display(self, position: float):
        """Update time label."""
        pos_str = self._format_time(position)
        dur_str = self._format_time(self._duration) if self._duration > 0 else "--:--"
        self._time_label.setText(f"{pos_str} / {dur_str}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        if seconds < 0:
            return "--:--"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def _on_playback_ended(self):
        """Handle playback end."""
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._timer.stop()
        self._progress_slider.setValue(0)
        self._update_time_display(0)
        self.playback_stopped.emit()

    def cleanup(self):
        """Clean up resources."""
        self._timer.stop()
        self._player.cleanup()
