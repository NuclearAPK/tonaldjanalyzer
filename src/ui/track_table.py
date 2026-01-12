"""
Track table widget for displaying and managing tracks.
"""

from typing import Optional, List
from pathlib import Path

from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush


class CompatibilityTableWidgetItem(QTableWidgetItem):
    """Custom table item for compatibility column that supports proper sorting."""

    def __init__(self, text: str, score: float = None, is_master: bool = False):
        super().__init__(text)
        self._score = score
        self._is_master = is_master

    def __lt__(self, other):
        """Custom comparison for sorting - master track always first."""
        if not isinstance(other, CompatibilityTableWidgetItem):
            return super().__lt__(other)

        # Master track is always "greatest" so it appears first in descending sort
        if self._is_master:
            return False
        if other._is_master:
            return True

        # None scores go to the end
        if self._score is None and other._score is None:
            return False
        if self._score is None:
            return True
        if other._score is None:
            return False

        return self._score < other._score

    def update_data(self, text: str, score: float = None, is_master: bool = False):
        """Update item data."""
        self.setText(text)
        self._score = score
        self._is_master = is_master

from ..core.track import Track
from .styles import get_color_for_score, COLORS


class TrackTable(QTableWidget):
    """Table widget for displaying track list with compatibility info."""

    # Signals
    track_selected = pyqtSignal(Track)
    master_track_changed = pyqtSignal(Track)
    play_requested = pyqtSignal(Track)
    bpm_multiplier_changed = pyqtSignal(Track)  # Emitted when BPM multiplier changes
    reanalyze_requested = pyqtSignal(Track)  # Emitted when reanalyze is requested

    # Column indices
    COL_NAME = 0
    COL_DURATION = 1
    COL_BPM = 2
    COL_KEY = 3
    COL_CAMELOT = 4
    COL_COMPAT = 5

    COLUMNS = ['Name', 'Duration', 'BPM', 'Key', 'Camelot', 'Match']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: List[Track] = []
        self._sort_ascending: bool = False  # Track current sort order
        self._setup_table()

    def _setup_table(self):
        """Configure table appearance and behavior."""
        # Set columns
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)

        # Appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)

        # Column sizing
        header = self.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_DURATION, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_BPM, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_KEY, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_CAMELOT, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_COMPAT, QHeaderView.ResizeToContents)

        # Sorting
        self.setSortingEnabled(True)
        self.sortByColumn(self.COL_NAME, Qt.AscendingOrder)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Selection changed
        self.itemSelectionChanged.connect(self._on_selection_changed)

        # Double-click to play
        self.doubleClicked.connect(self._on_double_click)

    def add_track(self, track: Track):
        """Add a track to the table."""
        self._tracks.append(track)
        self._add_track_row(track)

    def add_tracks(self, tracks: List[Track]):
        """Add multiple tracks to the table."""
        self.setSortingEnabled(False)
        for track in tracks:
            self._tracks.append(track)
            self._add_track_row(track)
        self.setSortingEnabled(True)

    def _add_track_row(self, track: Track):
        """Add a row for a track."""
        row = self.rowCount()
        self.insertRow(row)

        # Store track reference in first cell
        name_item = QTableWidgetItem(track.filename)
        name_item.setData(Qt.UserRole, track)
        self.setItem(row, self.COL_NAME, name_item)

        # Other columns
        self.setItem(row, self.COL_DURATION, QTableWidgetItem(track.duration_str))
        self.setItem(row, self.COL_BPM, QTableWidgetItem(track.bpm_str))
        self.setItem(row, self.COL_KEY, QTableWidgetItem(track.key or '--'))
        self.setItem(row, self.COL_CAMELOT, QTableWidgetItem(track.camelot_str))

        # Compatibility column (with custom sorting support)
        compat_item = CompatibilityTableWidgetItem(
            track.compatibility_str,
            track.compatibility_score,
            track.is_master
        )
        self.setItem(row, self.COL_COMPAT, compat_item)

        # Set row color based on compatibility
        self._update_row_color(row, track)

    def update_track(self, track: Track):
        """Update display for a specific track."""
        row = self._find_track_row(track)
        if row < 0:
            return

        # Update cells
        self.item(row, self.COL_DURATION).setText(track.duration_str)
        self.item(row, self.COL_BPM).setText(track.bpm_str)
        self.item(row, self.COL_KEY).setText(track.key or '--')
        self.item(row, self.COL_CAMELOT).setText(track.camelot_str)

        # Update compatibility item with proper sorting data
        compat_item = self.item(row, self.COL_COMPAT)
        if isinstance(compat_item, CompatibilityTableWidgetItem):
            compat_item.update_data(track.compatibility_str, track.compatibility_score, track.is_master)
        else:
            compat_item.setText(track.compatibility_str)

        self._update_row_color(row, track)

    def update_all_compatibility(self):
        """Update compatibility display for all tracks."""
        for row in range(self.rowCount()):
            track = self._get_track_at_row(row)
            if track:
                compat_item = self.item(row, self.COL_COMPAT)
                if isinstance(compat_item, CompatibilityTableWidgetItem):
                    compat_item.update_data(track.compatibility_str, track.compatibility_score, track.is_master)
                else:
                    compat_item.setText(track.compatibility_str)
                self._update_row_color(row, track)

    def _update_row_color(self, row: int, track: Track):
        """Update row background color based on compatibility score."""
        if track.is_master:
            color = QColor(*COLORS['master'])
        elif track.compatibility_score is not None:
            color = QColor(*get_color_for_score(track.compatibility_score))
        else:
            color = QColor(*COLORS['neutral'])

        # Apply color with transparency
        color.setAlpha(80)
        brush = QBrush(color)

        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(brush)

    def _find_track_row(self, track: Track) -> int:
        """Find row index for a track."""
        for row in range(self.rowCount()):
            item = self.item(row, self.COL_NAME)
            if item and item.data(Qt.UserRole) == track:
                return row
        return -1

    def _get_track_at_row(self, row: int) -> Optional[Track]:
        """Get track object from row index."""
        item = self.item(row, self.COL_NAME)
        if item:
            return item.data(Qt.UserRole)
        return None

    def get_selected_track(self) -> Optional[Track]:
        """Get currently selected track."""
        rows = self.selectionModel().selectedRows()
        if rows:
            return self._get_track_at_row(rows[0].row())
        return None

    def get_all_tracks(self) -> List[Track]:
        """Get all tracks in the table."""
        return self._tracks.copy()

    def clear_tracks(self):
        """Remove all tracks from the table."""
        self._tracks.clear()
        self.setRowCount(0)

    def remove_selected_track(self):
        """Remove the currently selected track."""
        track = self.get_selected_track()
        if track:
            row = self._find_track_row(track)
            if row >= 0:
                self.removeRow(row)
                self._tracks.remove(track)

    def _show_context_menu(self, position):
        """Show context menu for track actions."""
        track = self.get_selected_track()
        if not track:
            return

        menu = QMenu(self)

        # Set as master
        master_action = QAction("Set as Master Track", self)
        master_action.triggered.connect(lambda: self.master_track_changed.emit(track))
        menu.addAction(master_action)

        # Play
        play_action = QAction("Play", self)
        play_action.triggered.connect(lambda: self.play_requested.emit(track))
        menu.addAction(play_action)

        # Reanalyze
        reanalyze_action = QAction("Reanalyze Track", self)
        reanalyze_action.triggered.connect(lambda: self.reanalyze_requested.emit(track))
        menu.addAction(reanalyze_action)

        menu.addSeparator()

        # BPM multiplier submenu
        bpm_menu = menu.addMenu("BPM Multiplier")

        # x0.5 option
        half_action = QAction("x0.5 (Half)", self)
        half_action.setCheckable(True)
        half_action.setChecked(track.bpm_multiplier == 0.5)
        half_action.triggered.connect(lambda: self._set_bpm_multiplier(track, 0.5))
        bpm_menu.addAction(half_action)

        # x1 option (original)
        normal_action = QAction("x1 (Original)", self)
        normal_action.setCheckable(True)
        normal_action.setChecked(track.bpm_multiplier == 1.0)
        normal_action.triggered.connect(lambda: self._set_bpm_multiplier(track, 1.0))
        bpm_menu.addAction(normal_action)

        # x2 option
        double_action = QAction("x2 (Double)", self)
        double_action.setCheckable(True)
        double_action.setChecked(track.bpm_multiplier == 2.0)
        double_action.triggered.connect(lambda: self._set_bpm_multiplier(track, 2.0))
        bpm_menu.addAction(double_action)

        # Show current effective BPM
        if track.bpm:
            bpm_menu.addSeparator()
            info_action = QAction(f"Original: {track.original_bpm_str} BPM", self)
            info_action.setEnabled(False)
            bpm_menu.addAction(info_action)

        menu.addSeparator()

        # Remove
        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(self.remove_selected_track)
        menu.addAction(remove_action)

        menu.exec_(self.viewport().mapToGlobal(position))

    def _set_bpm_multiplier(self, track: Track, multiplier: float):
        """Set BPM multiplier for a track and update display."""
        track.bpm_multiplier = multiplier
        self.update_track(track)
        self.bpm_multiplier_changed.emit(track)

    def _on_selection_changed(self):
        """Handle selection change."""
        track = self.get_selected_track()
        if track:
            self.track_selected.emit(track)

    def _on_double_click(self, index):
        """Handle double-click on row."""
        track = self._get_track_at_row(index.row())
        if track:
            self.play_requested.emit(track)

    def sort_by_compatibility(self, ascending: bool = None):
        """Sort tracks by compatibility score. If ascending is None, toggle current order."""
        if ascending is None:
            # Toggle current order
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_ascending = ascending

        order = Qt.AscendingOrder if self._sort_ascending else Qt.DescendingOrder
        self.sortByColumn(self.COL_COMPAT, order)
