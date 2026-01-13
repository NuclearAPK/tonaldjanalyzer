"""
Application settings management using QSettings.
"""

from PyQt5.QtCore import QSettings


class AppSettings:
    """Manages application settings with persistent storage."""

    # Settings keys
    WRITE_METADATA_TO_FILES = "analysis/write_metadata_to_files"
    AUTO_PLAY_NEXT = "playback/auto_play_next"
    LANGUAGE = "general/language"
    ENABLE_LOGGING = "general/enable_logging"

    def __init__(self):
        """Initialize settings with application name."""
        self._settings = QSettings("TonalDJ", "TonalDJ")

    def get_write_metadata_enabled(self) -> bool:
        """
        Get whether writing metadata to MP3 files is enabled.
        Default is False (disabled).
        """
        return self._settings.value(self.WRITE_METADATA_TO_FILES, False, type=bool)

    def set_write_metadata_enabled(self, enabled: bool):
        """Set whether writing metadata to MP3 files is enabled."""
        self._settings.setValue(self.WRITE_METADATA_TO_FILES, enabled)

    def get_auto_play_next(self) -> bool:
        """
        Get whether auto-play next track is enabled.
        Default is False (disabled).
        """
        return self._settings.value(self.AUTO_PLAY_NEXT, False, type=bool)

    def set_auto_play_next(self, enabled: bool):
        """Set whether auto-play next track is enabled."""
        self._settings.setValue(self.AUTO_PLAY_NEXT, enabled)

    def get_language(self) -> str:
        """
        Get the interface language.
        Default is 'en' (English).
        """
        return self._settings.value(self.LANGUAGE, 'en', type=str)

    def set_language(self, lang_code: str):
        """Set the interface language."""
        self._settings.setValue(self.LANGUAGE, lang_code)

    def get_logging_enabled(self) -> bool:
        """
        Get whether error logging to file is enabled.
        Default is False (disabled).
        """
        return self._settings.value(self.ENABLE_LOGGING, False, type=bool)

    def set_logging_enabled(self, enabled: bool):
        """Set whether error logging to file is enabled."""
        self._settings.setValue(self.ENABLE_LOGGING, enabled)

    def sync(self):
        """Force sync settings to storage."""
        self._settings.sync()


# Global settings instance
_settings_instance: AppSettings = None


def get_settings() -> AppSettings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance
