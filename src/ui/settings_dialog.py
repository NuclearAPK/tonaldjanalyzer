"""
Settings dialog for application preferences.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
    QPushButton, QLabel, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt

from ..core.settings import get_settings
from ..core.localization import tr, LANGUAGES


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = get_settings()
        self._initial_language = self._settings.get_language()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Create dialog UI."""
        self.setWindowTitle(tr('settings_title'))
        self.setMinimumWidth(400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Metadata group
        metadata_group = QGroupBox(tr('mp3_metadata'))
        metadata_layout = QVBoxLayout(metadata_group)

        self._write_metadata_checkbox = QCheckBox(tr('write_metadata_option'))
        self._write_metadata_checkbox.setToolTip(tr('write_metadata_tooltip'))
        metadata_layout.addWidget(self._write_metadata_checkbox)

        info_label = QLabel(f"<i>{tr('write_metadata_note')}</i>")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 11px;")
        metadata_layout.addWidget(info_label)

        layout.addWidget(metadata_group)

        # Playback group
        playback_group = QGroupBox(tr('playback'))
        playback_layout = QVBoxLayout(playback_group)

        self._auto_play_checkbox = QCheckBox(tr('auto_play_next'))
        self._auto_play_checkbox.setToolTip(tr('auto_play_tooltip'))
        playback_layout.addWidget(self._auto_play_checkbox)

        layout.addWidget(playback_group)

        # Language group
        language_group = QGroupBox(tr('language'))
        language_layout = QVBoxLayout(language_group)

        lang_label = QLabel(tr('select_language'))
        language_layout.addWidget(lang_label)

        self._language_combo = QComboBox()
        for lang_code, lang_name in LANGUAGES.items():
            self._language_combo.addItem(lang_name, lang_code)
        language_layout.addWidget(self._language_combo)

        restart_label = QLabel(f"<i>{tr('restart_required')}</i>")
        restart_label.setStyleSheet("color: #888; font-size: 11px;")
        language_layout.addWidget(restart_label)

        layout.addWidget(language_group)

        # Logging group
        logging_group = QGroupBox(tr('logging'))
        logging_layout = QVBoxLayout(logging_group)

        self._logging_checkbox = QCheckBox(tr('enable_logging'))
        self._logging_checkbox.setToolTip(tr('enable_logging_tooltip'))
        logging_layout.addWidget(self._logging_checkbox)

        layout.addWidget(logging_group)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_button = QPushButton(tr('cancel'))
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        save_button = QPushButton(tr('save'))
        save_button.setDefault(True)
        save_button.clicked.connect(self._save_settings)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

    def _load_settings(self):
        """Load current settings into UI."""
        self._write_metadata_checkbox.setChecked(
            self._settings.get_write_metadata_enabled()
        )
        self._auto_play_checkbox.setChecked(
            self._settings.get_auto_play_next()
        )
        # Set language combo to current language
        current_lang = self._settings.get_language()
        index = self._language_combo.findData(current_lang)
        if index >= 0:
            self._language_combo.setCurrentIndex(index)
        # Set logging checkbox
        self._logging_checkbox.setChecked(
            self._settings.get_logging_enabled()
        )

    def _save_settings(self):
        """Save settings and close dialog."""
        self._settings.set_write_metadata_enabled(
            self._write_metadata_checkbox.isChecked()
        )
        self._settings.set_auto_play_next(
            self._auto_play_checkbox.isChecked()
        )
        # Save language setting
        selected_lang = self._language_combo.currentData()
        self._settings.set_language(selected_lang)
        # Save logging setting and apply
        logging_enabled = self._logging_checkbox.isChecked()
        self._settings.set_logging_enabled(logging_enabled)
        # Apply logging setting immediately
        from ..core.logger import setup_logger
        setup_logger(logging_enabled)
        self._settings.sync()
        self.accept()
