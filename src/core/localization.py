"""
Localization module for multi-language support.
"""

from typing import Dict

# Available languages
LANGUAGES = {
    'en': 'English',
    'ru': 'Русский'
}

# Translation dictionaries
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    'en': {
        # Main window
        'app_title': 'Tonal DJ - Track Compatibility Analyzer',
        'load_tracks': 'Load Tracks',
        'set_master': 'Set Master',
        'sort_match': 'Sort Match',
        'analyze_ai': 'Analyze AI',
        'reanalyze_all': 'Reanalyze All',
        'export_csv': 'Export CSV',
        'clear_all': 'Clear All',
        'settings': 'Settings',
        'tracks_loaded': 'Tracks: {count}',
        'ready': 'Ready',
        'drop_files_here': 'Drop audio files here or click "Load Tracks"',

        # Track table columns
        'col_name': 'Name',
        'col_duration': 'Duration',
        'col_bpm': 'BPM',
        'col_key': 'Key',
        'col_camelot': 'Camelot',
        'col_ai': 'AI',
        'col_style': 'Style',
        'col_harmonic': 'Harmonic',
        'col_content': 'Content',
        'col_match': 'Match',

        # Context menu
        'set_as_master': 'Set as Master Track',
        'play': 'Play',
        'reanalyze_track': 'Reanalyze Track',
        'analyze_content_ai': 'Analyze Content (AI)',
        'reanalyze_content_ai': 'Re-analyze Content (AI)',
        'edit_metadata': 'Edit Metadata...',
        'bpm_multiplier': 'BPM Multiplier',
        'half': 'x0.5 (Half)',
        'original': 'x1 (Original)',
        'double': 'x2 (Double)',
        'original_bpm': 'Original: {bpm} BPM',
        'remove': 'Remove',

        # Settings dialog
        'settings_title': 'Settings',
        'mp3_metadata': 'MP3 Metadata',
        'write_metadata_option': 'Write analysis results to MP3 file metadata',
        'write_metadata_tooltip': 'When enabled, BPM and key information will be saved directly\nto MP3 files as ID3 tags. This modifies the original files.\nDisabled by default to protect your files.',
        'write_metadata_note': 'Note: When enabled, analysis results (BPM, key) will be written\nto MP3 files as standard ID3 tags, making them available in\nother DJ software. This modifies your original files.',
        'playback': 'Playback',
        'auto_play_next': 'Auto-play next track when current track ends',
        'auto_play_tooltip': 'When enabled, the next track in the list will automatically\nstart playing when the current track finishes.',
        'language': 'Language',
        'select_language': 'Interface language',
        'restart_required': 'Restart required to apply language change',
        'logging': 'Logging',
        'enable_logging': 'Enable error logging to file',
        'enable_logging_tooltip': 'When enabled, errors and warnings will be saved\nto a log file in the logs folder.',
        'cancel': 'Cancel',
        'save': 'Save',

        # Metadata dialog
        'metadata_title': 'Edit Metadata - {filename}',
        'title': 'Title:',
        'artist': 'Artist:',
        'album': 'Album:',
        'genre': 'Genre:',
        'key_label': 'Key:',
        'metadata_note': 'Note: Changes will be saved directly to the MP3 file.',

        # Player
        'no_track_loaded': 'No track loaded',
        'volume': 'Volume:',
        'failed_to_load': 'Failed to load track',

        # Status messages
        'analyzing': 'Analyzing: {current}/{total}',
        'analysis_complete': 'Analysis complete',
        'loading_ai_model': 'Loading AI model (first time may take a while)...',
        'analyzing_style': 'Analyzing style: {current}/{total}',
        'ai_analysis_complete': 'AI analysis complete',
        'ai_analysis_error': 'AI Analysis Error',
        'master_set': 'Master: {name}',
        'master_removed': 'Master track removed',
        'metadata_saved': 'Metadata saved for {filename}',
        'auto_playing': 'Auto-playing: {filename}',
        'exported_to': 'Exported to {filename}',
        'export_error': 'Export Error',
        'content_analysis_in_progress': 'Content analysis already in progress...',
        'analyzing_content': 'Analyzing content: {filename}...',

        # Dialogs
        'clear_all_confirm': 'Remove all tracks?',
        'failed_to_analyze': 'Failed to analyze content:',
        'select_audio_files': 'Select Audio Files',
        'export_csv_title': 'Export CSV',
    },

    'ru': {
        # Main window
        'app_title': 'Tonal DJ - Анализатор совместимости треков',
        'load_tracks': 'Загрузить треки',
        'set_master': 'Выбрать мастер',
        'sort_match': 'Сортировать',
        'analyze_ai': 'AI анализ',
        'reanalyze_all': 'Переанализ всех',
        'export_csv': 'Экспорт CSV',
        'clear_all': 'Очистить всё',
        'settings': 'Настройки',
        'tracks_loaded': 'Треков: {count}',
        'ready': 'Готово',
        'drop_files_here': 'Перетащите аудио файлы сюда или нажмите "Загрузить треки"',

        # Track table columns
        'col_name': 'Название',
        'col_duration': 'Время',
        'col_bpm': 'BPM',
        'col_key': 'Тональность',
        'col_camelot': 'Camelot',
        'col_ai': 'AI',
        'col_style': 'Стиль',
        'col_harmonic': 'Гармония',
        'col_content': 'Контент',
        'col_match': 'Совпад.',

        # Context menu
        'set_as_master': 'Установить как мастер-трек',
        'play': 'Воспроизвести',
        'reanalyze_track': 'Переанализировать трек',
        'analyze_content_ai': 'Анализ контента (AI)',
        'reanalyze_content_ai': 'Переанализ контента (AI)',
        'edit_metadata': 'Редактировать метаданные...',
        'bpm_multiplier': 'Множитель BPM',
        'half': 'x0.5 (Половина)',
        'original': 'x1 (Оригинал)',
        'double': 'x2 (Двойной)',
        'original_bpm': 'Оригинал: {bpm} BPM',
        'remove': 'Удалить',

        # Settings dialog
        'settings_title': 'Настройки',
        'mp3_metadata': 'Метаданные MP3',
        'write_metadata_option': 'Записывать результаты анализа в метаданные MP3',
        'write_metadata_tooltip': 'Когда включено, информация о BPM и тональности будет сохранена\nв MP3 файлы как ID3 теги. Это изменяет оригинальные файлы.\nПо умолчанию отключено для защиты ваших файлов.',
        'write_metadata_note': 'Примечание: При включении результаты анализа (BPM, тональность)\nбудут записаны в MP3 файлы как стандартные ID3 теги,\nчто сделает их доступными в других DJ программах.',
        'playback': 'Воспроизведение',
        'auto_play_next': 'Автоматически воспроизводить следующий трек',
        'auto_play_tooltip': 'Когда включено, следующий трек в списке автоматически\nначнёт воспроизводиться после завершения текущего.',
        'language': 'Язык',
        'select_language': 'Язык интерфейса',
        'restart_required': 'Требуется перезапуск для применения языка',
        'logging': 'Логирование',
        'enable_logging': 'Включить запись ошибок в файл',
        'enable_logging_tooltip': 'Когда включено, ошибки и предупреждения будут сохраняться\nв лог-файл в папке logs.',
        'cancel': 'Отмена',
        'save': 'Сохранить',

        # Metadata dialog
        'metadata_title': 'Редактирование метаданных - {filename}',
        'title': 'Название:',
        'artist': 'Исполнитель:',
        'album': 'Альбом:',
        'genre': 'Жанр:',
        'key_label': 'Тональность:',
        'metadata_note': 'Примечание: Изменения будут сохранены непосредственно в MP3 файл.',

        # Player
        'no_track_loaded': 'Трек не загружен',
        'volume': 'Громкость:',
        'failed_to_load': 'Не удалось загрузить трек',

        # Status messages
        'analyzing': 'Анализ: {current}/{total}',
        'analysis_complete': 'Анализ завершён',
        'loading_ai_model': 'Загрузка AI модели (первый раз может занять время)...',
        'analyzing_style': 'Анализ стиля: {current}/{total}',
        'ai_analysis_complete': 'AI анализ завершён',
        'ai_analysis_error': 'Ошибка AI анализа',
        'master_set': 'Мастер: {name}',
        'master_removed': 'Мастер-трек удалён',
        'metadata_saved': 'Метаданные сохранены для {filename}',
        'auto_playing': 'Авто-воспроизведение: {filename}',
        'exported_to': 'Экспортировано в {filename}',
        'export_error': 'Ошибка экспорта',
        'content_analysis_in_progress': 'Анализ контента уже выполняется...',
        'analyzing_content': 'Анализ контента: {filename}...',

        # Dialogs
        'clear_all_confirm': 'Удалить все треки?',
        'failed_to_analyze': 'Не удалось проанализировать контент:',
        'select_audio_files': 'Выберите аудио файлы',
        'export_csv_title': 'Экспорт CSV',
    }
}

# Current language
_current_language = 'en'


def set_language(lang_code: str):
    """Set the current language."""
    global _current_language
    if lang_code in TRANSLATIONS:
        _current_language = lang_code


def get_language() -> str:
    """Get the current language code."""
    return _current_language


def tr(key: str, **kwargs) -> str:
    """
    Get translated string for the given key.

    Args:
        key: Translation key
        **kwargs: Format arguments for the string

    Returns:
        Translated string or key if not found
    """
    translations = TRANSLATIONS.get(_current_language, TRANSLATIONS['en'])
    text = translations.get(key, TRANSLATIONS['en'].get(key, key))

    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    return text
