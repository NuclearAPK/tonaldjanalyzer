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
from .settings_dialog import SettingsDialog
from .styles import MAIN_STYLESHEET
from ..core.track import Track
from ..core.compatibility import CompatibilityCalculator
from ..core.cache import get_cache
from ..core.settings import get_settings
from ..core.localization import tr
from ..core.logger import log_error, log_info, log_warning
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
        self._analyzer = AudioAnalyzer(extract_embeddings=False)  # Don't extract embeddings during basic analysis

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


class EmbeddingWorker(QThread):
    """Background worker for extracting audio embeddings and classifying style."""

    progress = pyqtSignal(int, int)  # completed, total
    embedding_extracted = pyqtSignal(object, dict)  # track, result dict
    finished = pyqtSignal()
    status = pyqtSignal(str)  # status message
    error = pyqtSignal(str)  # error message

    def __init__(self, tracks: List[Track], parent=None):
        super().__init__(parent)
        self._tracks = tracks
        self._analyzer = None  # Lazy init in thread

    def run(self):
        """Extract embeddings and classify style for tracks in background."""
        try:
            # Initialize analyzer in worker thread
            self._analyzer = AudioAnalyzer()
            total = len(self._tracks)

            self.status.emit(tr('loading_ai_model'))

            for i, track in enumerate(self._tracks):
                if self.isInterruptionRequested():
                    break

                self.status.emit(tr('analyzing_style', current=i + 1, total=total))
                try:
                    # Pass original BPM for genre correction (e.g., Breakbeat vs DnB)
                    # Use original BPM, not effective_bpm (which has multiplier applied)
                    result = self._analyzer.extract_embedding(track.file_path, bpm=track.bpm)
                    self.embedding_extracted.emit(track, result)
                except Exception as e:
                    self.status.emit(f"Error on {track.filename}: {e}")
                    self.embedding_extracted.emit(track, {'embedding': None, 'style': None})
                    log_error(f"Error analyzing {track.filename}: {e}")
                self.progress.emit(i + 1, total)

        except Exception as e:
            log_error(f"Embedding worker error: {e}")
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._tracks: List[Track] = []
        self._master_track: Optional[Track] = None
        self._compatibility_calc = CompatibilityCalculator()
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._embedding_worker: Optional[EmbeddingWorker] = None

        self._setup_ui()
        self._setup_icon()
        self._connect_signals()
        self.setAcceptDrops(True)

    def _setup_icon(self):
        """Set application icon."""
        from pathlib import Path
        from PyQt5.QtGui import QIcon

        # Try to load icon from assets folder
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            # Try PNG as fallback
            png_path = Path(__file__).parent.parent.parent / "assets" / "icon.png"
            if png_path.exists():
                self.setWindowIcon(QIcon(str(png_path)))

    def _setup_ui(self):
        """Create UI layout."""
        self.setWindowTitle(tr('app_title'))
        self.setMinimumSize(900, 600)
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header with buttons
        header_layout = QHBoxLayout()

        self._load_button = QPushButton(tr('load_tracks'))
        self._load_button.clicked.connect(self._on_load_tracks)
        header_layout.addWidget(self._load_button)

        self._clear_button = QPushButton(tr('clear_all'))
        self._clear_button.clicked.connect(self._on_clear_tracks)
        header_layout.addWidget(self._clear_button)

        self._reanalyze_button = QPushButton(tr('reanalyze_all'))
        self._reanalyze_button.setToolTip("Force re-analyze all tracks (ignore cache)")
        self._reanalyze_button.clicked.connect(self._on_reanalyze_all)
        self._reanalyze_button.setEnabled(False)
        header_layout.addWidget(self._reanalyze_button)

        header_layout.addSpacing(20)

        self._master_button = QPushButton(tr('set_master'))
        self._master_button.setObjectName("masterButton")
        self._master_button.clicked.connect(self._on_set_master)
        self._master_button.setEnabled(False)
        header_layout.addWidget(self._master_button)

        self._sort_button = QPushButton(tr('sort_match'))
        self._sort_button.clicked.connect(self._on_sort_by_compatibility)
        self._sort_button.setEnabled(False)
        header_layout.addWidget(self._sort_button)

        header_layout.addSpacing(20)

        self._content_button = QPushButton(tr('analyze_ai'))
        self._content_button.setToolTip("Extract audio features using AI for content-based matching")
        self._content_button.clicked.connect(self._on_analyze_content)
        self._content_button.setEnabled(False)
        header_layout.addWidget(self._content_button)

        header_layout.addStretch()

        self._export_button = QPushButton(tr('export_csv'))
        self._export_button.clicked.connect(self._on_export_csv)
        self._export_button.setEnabled(False)
        header_layout.addWidget(self._export_button)

        self._settings_button = QPushButton(tr('settings'))
        self._settings_button.clicked.connect(self._on_open_settings)
        header_layout.addWidget(self._settings_button)

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

        self._status_label = QLabel(tr('drop_files_here'))
        self._status_label.setObjectName("statusLabel")
        status_layout.addWidget(self._status_label)

        status_layout.addStretch()

        self._track_count_label = QLabel(tr('tracks_loaded', count=0))
        status_layout.addWidget(self._track_count_label)

        layout.addLayout(status_layout)

        # Player widget
        self._player_widget = PlayerWidget()
        layout.addWidget(self._player_widget)

        # Connect player signals
        self._player_widget.track_finished.connect(self._on_track_finished)

    def _connect_signals(self):
        """Connect signals and slots."""
        self._track_table.track_selected.connect(self._on_track_selected)
        self._track_table.master_track_changed.connect(self._set_master_track)
        self._track_table.play_requested.connect(self._on_play_track)
        self._track_table.bpm_multiplier_changed.connect(self._on_bpm_multiplier_changed)
        self._track_table.reanalyze_requested.connect(self._on_reanalyze_track)
        self._track_table.content_analyze_requested.connect(self._on_content_analyze_track)
        self._track_table.track_removed.connect(self._on_track_removed)
        self._track_table.edit_metadata_requested.connect(self._on_edit_metadata)

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
            tr('select_audio_files'),
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
        self._status_label.setText(tr('analyzing', current=completed, total=total))

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

        # Read genre from metadata
        from ..audio.metadata import get_metadata_handler
        metadata = get_metadata_handler()
        metadata_genre = metadata.read_genre(track.file_path)
        if metadata_genre:
            track.genre = metadata_genre
            track.genre_from_metadata = True

        # Apply cached embedding and style if available
        if result.get('embedding') is not None:
            track.embedding = result['embedding']
        if result.get('style') is not None:
            track.style = result['style']  # This will set mood (and genre if not from metadata)

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
        self._content_button.setEnabled(len(self._tracks) > 0)
        self._status_label.setText(tr('analysis_complete'))

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

    def _on_analyze_content(self):
        """Extract content features using AI for all tracks."""
        if not self._tracks:
            return

        # Check which tracks need embedding extraction
        tracks_to_process = [t for t in self._tracks if t.embedding is None]

        if not tracks_to_process:
            QMessageBox.information(
                self,
                "Content Analysis",
                "All tracks already have content features extracted."
            )
            return

        # Preload AI model in main thread to avoid Qt/torch thread conflicts
        self._status_label.setText(tr('loading_ai_model'))
        QApplication.processEvents()  # Update UI

        from ..audio.embeddings import get_audio_embeddings
        embeddings = get_audio_embeddings()
        if not embeddings.is_available():
            from ..audio.embeddings import get_load_error
            error = get_load_error() or "Unknown error"
            QMessageBox.warning(self, tr('ai_analysis_error'), f"{tr('failed_to_analyze')}\n{error}")
            self._status_label.setText(tr('ai_analysis_error'))
            return

        self._start_embedding_extraction(tracks_to_process)

    def _start_embedding_extraction(self, tracks: List[Track]):
        """Start background embedding extraction."""
        if self._embedding_worker and self._embedding_worker.isRunning():
            self._embedding_worker.requestInterruption()
            self._embedding_worker.wait()

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._load_button.setEnabled(False)
        self._content_button.setEnabled(False)
        self._status_label.setText(tr('loading_ai_model'))

        self._embedding_worker = EmbeddingWorker(tracks)
        self._embedding_worker.progress.connect(self._on_embedding_progress)
        self._embedding_worker.embedding_extracted.connect(self._on_embedding_extracted)
        self._embedding_worker.finished.connect(self._on_embedding_finished)
        self._embedding_worker.status.connect(self._on_embedding_status)
        self._embedding_worker.error.connect(self._on_embedding_error)
        self._embedding_worker.start()

    def _on_embedding_progress(self, completed: int, total: int):
        """Update progress bar for embedding extraction."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(completed)

    def _on_embedding_status(self, status: str):
        """Update status label during embedding extraction."""
        self._status_label.setText(status)

    def _on_embedding_extracted(self, track: Track, result: dict):
        """Handle single track embedding extraction complete."""
        # Make a copy to avoid potential torch/numpy memory issues
        import numpy as np
        embedding = result.get('embedding')
        track.embedding = np.array(embedding, copy=True) if embedding is not None else None

        # Set genre/mood from result
        if result.get('genre_from_metadata'):
            track.genre = result.get('genre')
            track.genre_from_metadata = True
        elif result.get('genre'):
            track.genre = result.get('genre')
        track.mood = result.get('mood')

        # Update compatibility if master is set
        if self._master_track:
            self._update_compatibility()

        # Always update track display (for AI indicator and style)
        self._track_table.update_track(track)

    def _on_embedding_finished(self):
        """Handle all embedding extraction complete."""
        # Wait for worker thread to fully terminate
        if self._embedding_worker:
            self._embedding_worker.wait()

        # Force garbage collection to clean up torch resources
        import gc
        gc.collect()

        self._progress_bar.setVisible(False)
        self._load_button.setEnabled(True)
        self._content_button.setEnabled(True)

        # Count how many tracks have embeddings
        with_embeddings = sum(1 for t in self._tracks if t.embedding is not None)
        self._status_label.setText(tr('ai_analysis_complete'))

    def _on_embedding_error(self, error_msg: str):
        """Handle embedding extraction error."""
        self._progress_bar.setVisible(False)
        self._load_button.setEnabled(True)
        self._content_button.setEnabled(True)
        self._status_label.setText(tr('ai_analysis_error'))
        QMessageBox.warning(self, tr('ai_analysis_error'), f"{tr('failed_to_analyze')}\n{error_msg}")

    def _on_reanalyze_track(self, track: Track):
        """Force re-analyze a single track (ignore cache)."""
        self._start_analysis([track], force_reanalyze=True)

    def _on_content_analyze_track(self, track: Track, force: bool = True):
        """Analyze AI content for a single track."""
        if self._embedding_worker and self._embedding_worker.isRunning():
            self._status_label.setText(tr('content_analysis_in_progress'))
            return

        # Clear cached AI data if forcing re-analysis
        if force:
            cache = get_cache()
            cache.clear_ai_data(track.file_path)
            track.embedding = None
            track.style = None

        self._status_label.setText(tr('analyzing_content', filename=track.filename))
        self._start_embedding_extraction([track])

    def _on_track_removed(self, track: Track):
        """Handle track removal."""
        # Remove from internal list
        if track in self._tracks:
            self._tracks.remove(track)

        # If removed track was master, clear master
        if self._master_track == track:
            self._master_track = None
            self._sort_button.setEnabled(False)
            self._status_label.setText(tr('master_removed'))

        # Update UI
        self._update_track_count()

        # Update button states
        if not self._tracks:
            self._master_button.setEnabled(False)
            self._export_button.setEnabled(False)
            self._reanalyze_button.setEnabled(False)

    def _on_edit_metadata(self, track: Track):
        """Open metadata editor for track."""
        from .metadata_dialog import MetadataDialog

        dialog = MetadataDialog(track.file_path, self)
        if dialog.exec_() == MetadataDialog.Accepted:
            # Reload track metadata after editing
            self._status_label.setText(tr('metadata_saved', filename=track.filename))
            # Re-read genre from metadata and update track
            from ..audio.metadata import get_metadata_handler
            metadata = get_metadata_handler()
            new_genre = metadata.read_genre(track.file_path)
            if new_genre:
                track.genre = new_genre
                track.genre_from_metadata = True
            self._track_table.update_track(track)

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

        # Auto-sort by compatibility descending (master will be first)
        self._track_table.sort_by_compatibility(ascending=False)

        self._status_label.setText(tr('master_set', name=track.filename))

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
        """Play a track (double-click) - reset position and auto-play."""
        self._player_widget.load_track(track, auto_play=True)

    def _on_track_finished(self, finished_track: Track):
        """Handle track finished - auto-play next if enabled."""
        settings = get_settings()
        if not settings.get_auto_play_next():
            return

        # Find next track in table
        all_tracks = self._track_table.get_all_tracks()
        if not all_tracks:
            return

        # Find index of finished track
        try:
            # Get tracks in current table order (respects sorting)
            current_idx = -1
            for row in range(self._track_table.rowCount()):
                track_at_row = self._track_table._get_track_at_row(row)
                if track_at_row == finished_track:
                    current_idx = row
                    break

            if current_idx >= 0 and current_idx < self._track_table.rowCount() - 1:
                # Get next track in table order
                next_track = self._track_table._get_track_at_row(current_idx + 1)
                if next_track:
                    # Select and play next track
                    self._track_table.selectRow(current_idx + 1)
                    self._player_widget.load_track(next_track, auto_play=True)
                    self._status_label.setText(tr('auto_playing', filename=next_track.filename))
        except Exception as e:
            log_error(f"Auto-play error: {e}")

    def _on_clear_tracks(self):
        """Clear all tracks."""
        if self._tracks:
            reply = QMessageBox.question(
                self,
                tr('clear_all'),
                tr('clear_all_confirm'),
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
                self._content_button.setEnabled(False)
                self._status_label.setText(tr('ready'))

    def _on_export_csv(self):
        """Export track list to CSV."""
        if not self._tracks:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr('export_csv_title'),
            "tracks.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Filename', 'Duration', 'Original BPM', 'BPM Multiplier', 'Effective BPM', 'Key', 'Camelot', 'Harmonic %', 'Content %', 'Match %', 'Is Master'])

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
                            track.content_str,
                            track.combined_str,
                            'Yes' if track.is_master else 'No'
                        ])

                self._status_label.setText(tr('exported_to', filename=file_path))

            except Exception as e:
                log_error(f"Export error: {e}")
                QMessageBox.warning(self, tr('export_error'), str(e))

    def _on_open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec_()

    def _update_track_count(self):
        """Update track count label."""
        count = len(self._tracks)
        self._track_count_label.setText(tr('tracks_loaded', count=count))

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

        # Stop embedding extraction if running
        if self._embedding_worker and self._embedding_worker.isRunning():
            self._embedding_worker.requestInterruption()
            self._embedding_worker.wait()

        # Clean up player
        self._player_widget.cleanup()

        event.accept()
