"""
Main application window.
"""

import csv
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from .track_table import TrackTable
from .player_widget import PlayerWidget
from .styles import MAIN_STYLESHEET
from ..core.track import Track
from ..core.compatibility import CompatibilityCalculator
from ..core.cache import get_cache
from ..audio.analyzer import AudioAnalyzer, is_supported_format


class AnalysisWorker(QThread):
    """Background worker for audio analysis."""

    progress = pyqtSignal(int, int)  # completed, total
    track_analyzed = pyqtSignal(object, dict)  # track, results
    finished = pyqtSignal()

    def __init__(self, tracks: List[Track], force_reanalyze: bool = False, parent=None):
        super().__init__(parent)
        self._tracks = tracks
        self._force_reanalyze = force_reanalyze
        self._analyzer = AudioAnalyzer()

    def run(self):
        """Analyze tracks in background."""
        total = len(self._tracks)

        for i, track in enumerate(self._tracks):
            if self.isInterruptionRequested():
                break

            result = self._analyzer.analyze_file(track.file_path, self._force_reanalyze)
            self.track_analyzed.emit(track, result)
            self.progress.emit(i + 1, total)

        self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._tracks: List[Track] = []
        self._master_track: Optional[Track] = None
        self._compatibility_calc = CompatibilityCalculator()
        self._analysis_worker: Optional[AnalysisWorker] = None

        self._setup_ui()
        self._connect_signals()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        """Create UI layout."""
        self.setWindowTitle("Tonal DJ - Track Compatibility Analyzer")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header with buttons
        header_layout = QHBoxLayout()

        self._load_button = QPushButton("Load Tracks")
        self._load_button.clicked.connect(self._on_load_tracks)
        header_layout.addWidget(self._load_button)

        self._clear_button = QPushButton("Clear All")
        self._clear_button.clicked.connect(self._on_clear_tracks)
        header_layout.addWidget(self._clear_button)

        self._reanalyze_button = QPushButton("Reanalyze All")
        self._reanalyze_button.setToolTip("Force re-analyze all tracks (ignore cache)")
        self._reanalyze_button.clicked.connect(self._on_reanalyze_all)
        self._reanalyze_button.setEnabled(False)
        header_layout.addWidget(self._reanalyze_button)

        header_layout.addSpacing(20)

        self._master_button = QPushButton("Set as Master")
        self._master_button.setObjectName("masterButton")
        self._master_button.clicked.connect(self._on_set_master)
        self._master_button.setEnabled(False)
        header_layout.addWidget(self._master_button)

        self._sort_button = QPushButton("Sort by Match")
        self._sort_button.clicked.connect(self._on_sort_by_compatibility)
        self._sort_button.setEnabled(False)
        header_layout.addWidget(self._sort_button)

        header_layout.addStretch()

        self._export_button = QPushButton("Export CSV")
        self._export_button.clicked.connect(self._on_export_csv)
        self._export_button.setEnabled(False)
        header_layout.addWidget(self._export_button)

        layout.addLayout(header_layout)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # Track table
        self._track_table = TrackTable()
        layout.addWidget(self._track_table)

        # Status bar
        status_layout = QHBoxLayout()

        self._status_label = QLabel("Drag & drop audio files or click 'Load Tracks'")
        self._status_label.setObjectName("statusLabel")
        status_layout.addWidget(self._status_label)

        status_layout.addStretch()

        self._track_count_label = QLabel("0 tracks")
        status_layout.addWidget(self._track_count_label)

        layout.addLayout(status_layout)

        # Player widget
        self._player_widget = PlayerWidget()
        layout.addWidget(self._player_widget)

    def _connect_signals(self):
        """Connect signals and slots."""
        self._track_table.track_selected.connect(self._on_track_selected)
        self._track_table.master_track_changed.connect(self._set_master_track)
        self._track_table.play_requested.connect(self._on_play_track)
        self._track_table.bpm_multiplier_changed.connect(self._on_bpm_multiplier_changed)
        self._track_table.reanalyze_requested.connect(self._on_reanalyze_track)

    def _on_bpm_multiplier_changed(self, track):
        """Handle BPM multiplier change - save to cache and recalculate compatibility."""
        # Save BPM multiplier to cache
        cache = get_cache()
        cache.save_bpm_multiplier(track.file_path, track.bpm_multiplier)

        # Recalculate compatibility
        if self._master_track:
            self._update_compatibility()

    def _on_load_tracks(self):
        """Open file dialog to load tracks."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*.*)"
        )

        if files:
            self._add_files([Path(f) for f in files])

    def _add_files(self, file_paths: List[Path]):
        """Add files and start analysis."""
        # Filter supported formats
        valid_files = [f for f in file_paths if is_supported_format(f)]

        if not valid_files:
            QMessageBox.warning(
                self,
                "No Valid Files",
                "No supported audio files found. Supported formats: MP3, WAV, FLAC, OGG, M4A"
            )
            return

        # Create track objects
        new_tracks = [Track(file_path=f) for f in valid_files]
        self._tracks.extend(new_tracks)

        # Add to table
        self._track_table.add_tracks(new_tracks)

        # Update UI
        self._update_track_count()
        self._export_button.setEnabled(True)

        # Start analysis
        self._start_analysis(new_tracks)

    def _start_analysis(self, tracks: List[Track], force_reanalyze: bool = False):
        """Start background analysis of tracks."""
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.requestInterruption()
            self._analysis_worker.wait()

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._load_button.setEnabled(False)
        self._reanalyze_button.setEnabled(False)
        status_text = "Re-analyzing tracks..." if force_reanalyze else "Analyzing tracks..."
        self._status_label.setText(status_text)

        self._analysis_worker = AnalysisWorker(tracks, force_reanalyze)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.track_analyzed.connect(self._on_track_analyzed)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.start()

    def _on_analysis_progress(self, completed: int, total: int):
        """Update progress bar."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(completed)
        self._status_label.setText(f"Analyzing: {completed}/{total}")

    def _on_track_analyzed(self, track: Track, result: dict):
        """Handle single track analysis complete."""
        track.duration = result['duration']
        track.bpm = result['bpm']
        track.key = result['key']
        track.camelot = result['camelot']
        track.error = result['error']
        track.is_analyzed = True

        # Apply saved BPM multiplier from cache if available
        if 'bpm_multiplier' in result and result['bpm_multiplier'] is not None:
            track.bpm_multiplier = result['bpm_multiplier']

        # Update table
        self._track_table.update_track(track)

        # Update compatibility if master is set
        if self._master_track:
            self._update_compatibility()

    def _on_analysis_finished(self):
        """Handle all analysis complete."""
        self._progress_bar.setVisible(False)
        self._load_button.setEnabled(True)
        self._reanalyze_button.setEnabled(len(self._tracks) > 0)
        self._master_button.setEnabled(len(self._tracks) > 0)
        self._sort_button.setEnabled(self._master_track is not None)
        self._status_label.setText("Analysis complete")

    def _on_reanalyze_all(self):
        """Force re-analyze all tracks (ignore cache)."""
        if not self._tracks:
            return

        reply = QMessageBox.question(
            self,
            "Reanalyze All Tracks",
            f"Re-analyze all {len(self._tracks)} tracks?\nThis will ignore cached data and may take some time.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._start_analysis(self._tracks, force_reanalyze=True)

    def _on_reanalyze_track(self, track: Track):
        """Force re-analyze a single track (ignore cache)."""
        self._start_analysis([track], force_reanalyze=True)

    def _on_track_selected(self, track: Track):
        """Handle track selection."""
        self._master_button.setEnabled(True)

    def _on_set_master(self):
        """Set selected track as master."""
        track = self._track_table.get_selected_track()
        if track:
            self._set_master_track(track)

    def _set_master_track(self, track: Track):
        """Set a track as the master track."""
        # Clear previous master
        if self._master_track:
            self._master_track.is_master = False

        self._master_track = track
        track.is_master = True
        track.compatibility_score = None

        self._update_compatibility()
        self._sort_button.setEnabled(True)
        self._status_label.setText(f"Master: {track.filename}")

    def _update_compatibility(self):
        """Recalculate compatibility for all tracks."""
        if not self._master_track:
            return

        self._compatibility_calc.update_track_compatibility(
            self._master_track,
            self._tracks
        )
        self._track_table.update_all_compatibility()

    def _on_sort_by_compatibility(self):
        """Sort tracks by compatibility score."""
        self._track_table.sort_by_compatibility()

    def _on_play_track(self, track: Track):
        """Play a track."""
        self._player_widget.load_track(track)

    def _on_clear_tracks(self):
        """Clear all tracks."""
        if self._tracks:
            reply = QMessageBox.question(
                self,
                "Clear All",
                "Remove all tracks?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._tracks.clear()
                self._master_track = None
                self._track_table.clear_tracks()
                self._update_track_count()
                self._master_button.setEnabled(False)
                self._sort_button.setEnabled(False)
                self._export_button.setEnabled(False)
                self._reanalyze_button.setEnabled(False)
                self._status_label.setText("All tracks cleared")

    def _on_export_csv(self):
        """Export track list to CSV."""
        if not self._tracks:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            "tracks.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Filename', 'Duration', 'Original BPM', 'BPM Multiplier', 'Effective BPM', 'Key', 'Camelot', 'Match %', 'Is Master'])

                    for track in self._tracks:
                        writer.writerow([
                            track.filename,
                            track.duration_str,
                            track.original_bpm_str,
                            f"x{track.bpm_multiplier}",
                            track.bpm_str.split()[0] if track.bpm else '--',  # Just the number
                            track.key or '',
                            track.camelot_str,
                            track.compatibility_str,
                            'Yes' if track.is_master else 'No'
                        ])

                self._status_label.setText(f"Exported to {file_path}")

            except Exception as e:
                QMessageBox.warning(self, "Export Failed", str(e))

    def _update_track_count(self):
        """Update track count label."""
        count = len(self._tracks)
        self._track_count_label.setText(f"{count} track{'s' if count != 1 else ''}")

    # Drag and drop support
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop."""
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                files.append(Path(url.toLocalFile()))

        if files:
            self._add_files(files)

    def closeEvent(self, event):
        """Handle window close."""
        # Stop analysis if running
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.requestInterruption()
            self._analysis_worker.wait()

        # Clean up player
        self._player_widget.cleanup()

        event.accept()
