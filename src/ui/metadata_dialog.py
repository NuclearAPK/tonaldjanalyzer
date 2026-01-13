"""
Dialog for editing MP3 metadata.
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QDialogButtonBox, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

from ..audio.metadata import get_metadata_handler
from ..core.localization import tr


class MetadataDialog(QDialog):
    """Dialog for viewing and editing MP3 metadata."""

    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self.file_path = Path(file_path)
        self._metadata_handler = get_metadata_handler()
        self._setup_ui()
        self._load_metadata()

    def _setup_ui(self):
        """Setup dialog UI."""
        self.setWindowTitle(tr('metadata_title', filename=self.file_path.name))
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Form layout for fields
        form = QFormLayout()

        # Title
        self._title_edit = QLineEdit()
        form.addRow(tr('title'), self._title_edit)

        # Artist
        self._artist_edit = QLineEdit()
        form.addRow(tr('artist'), self._artist_edit)

        # Album
        self._album_edit = QLineEdit()
        form.addRow(tr('album'), self._album_edit)

        # Genre
        self._genre_edit = QComboBox()
        self._genre_edit.setEditable(True)
        self._genre_edit.addItems([
            "", "Drum And Bass", "Dubstep", "House", "Techno", "Trance",
            "Hip Hop", "Trap", "Ambient", "Breakbeat", "Hardstyle",
            "Progressive", "Minimal", "Electro", "Jungle", "Garage"
        ])
        form.addRow(tr('genre'), self._genre_edit)

        # BPM
        self._bpm_edit = QLineEdit()
        self._bpm_edit.setPlaceholderText("e.g. 174")
        form.addRow("BPM:", self._bpm_edit)

        # Key
        self._key_edit = QComboBox()
        self._key_edit.setEditable(True)
        keys = [""]
        for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            keys.append(f"{note} major")
            keys.append(f"{note} minor")
        self._key_edit.addItems(keys)
        form.addRow(tr('key_label'), self._key_edit)

        layout.addLayout(form)

        # Info label
        info_label = QLabel(tr('metadata_note'))
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save_metadata)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_metadata(self):
        """Load current metadata from file."""
        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3

            if not self.file_path.suffix.lower() == '.mp3':
                QMessageBox.warning(self, "Not MP3", "Metadata editing is only supported for MP3 files.")
                return

            audio = MP3(str(self.file_path))
            if audio.tags is None:
                return

            # Title (TIT2)
            if 'TIT2' in audio.tags:
                self._title_edit.setText(str(audio.tags['TIT2']))

            # Artist (TPE1)
            if 'TPE1' in audio.tags:
                self._artist_edit.setText(str(audio.tags['TPE1']))

            # Album (TALB)
            if 'TALB' in audio.tags:
                self._album_edit.setText(str(audio.tags['TALB']))

            # Genre (TCON)
            if 'TCON' in audio.tags:
                genre = str(audio.tags['TCON'])
                idx = self._genre_edit.findText(genre)
                if idx >= 0:
                    self._genre_edit.setCurrentIndex(idx)
                else:
                    self._genre_edit.setCurrentText(genre)

            # BPM (TBPM)
            if 'TBPM' in audio.tags:
                self._bpm_edit.setText(str(audio.tags['TBPM']))

            # Key (TKEY)
            if 'TKEY' in audio.tags:
                key = str(audio.tags['TKEY'])
                idx = self._key_edit.findText(key)
                if idx >= 0:
                    self._key_edit.setCurrentIndex(idx)
                else:
                    self._key_edit.setCurrentText(key)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load metadata: {e}")

    def _save_metadata(self):
        """Save metadata to file."""
        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3, TIT2, TPE1, TALB, TCON, TBPM, TKEY

            audio = MP3(str(self.file_path))
            if audio.tags is None:
                audio.add_tags()

            # Title
            title = self._title_edit.text().strip()
            if title:
                audio.tags.delall('TIT2')
                audio.tags.add(TIT2(encoding=3, text=[title]))

            # Artist
            artist = self._artist_edit.text().strip()
            if artist:
                audio.tags.delall('TPE1')
                audio.tags.add(TPE1(encoding=3, text=[artist]))

            # Album
            album = self._album_edit.text().strip()
            if album:
                audio.tags.delall('TALB')
                audio.tags.add(TALB(encoding=3, text=[album]))

            # Genre
            genre = self._genre_edit.currentText().strip()
            if genre:
                audio.tags.delall('TCON')
                audio.tags.add(TCON(encoding=3, text=[genre]))

            # BPM
            bpm = self._bpm_edit.text().strip()
            if bpm:
                try:
                    bpm_val = int(float(bpm))
                    audio.tags.delall('TBPM')
                    audio.tags.add(TBPM(encoding=3, text=[str(bpm_val)]))
                except ValueError:
                    pass

            # Key
            key = self._key_edit.currentText().strip()
            if key:
                audio.tags.delall('TKEY')
                audio.tags.add(TKEY(encoding=3, text=[key]))

            audio.save()
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save metadata: {e}")
