#!/usr/bin/env python3
"""
Tonal DJ Plugin - Track Compatibility Analyzer

Analyzes audio tracks and finds compatible matches based on
musical key (Camelot Wheel) and BPM for DJ mixing.
"""

import sys
import os

# Fix Windows DLL loading issue - must be done BEFORE any other imports
if sys.platform == 'win32':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    try:
        # Preload torch to ensure DLLs are loaded before PyQt5
        from pathlib import Path
        import importlib.util
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec and torch_spec.origin:
            torch_lib = Path(torch_spec.origin).parent / 'lib'
            if torch_lib.exists():
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(str(torch_lib))
                os.environ['PATH'] = str(torch_lib) + os.pathsep + os.environ.get('PATH', '')
        import torch  # Preload torch DLLs
        # Also preload transformers to avoid DLL conflicts with Qt
        import transformers
    except (ImportError, OSError):
        pass  # torch/transformers not available or DLL loading failed, skip

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from src.ui import MainWindow
from src.core.settings import get_settings
from src.core.localization import set_language
from src.core.logger import setup_logger


def main():
    """Application entry point."""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Tonal DJ")
    app.setApplicationVersion("1.1.0")

    # Initialize settings
    settings = get_settings()

    # Initialize logging from settings
    setup_logger(settings.get_logging_enabled())

    # Initialize language from settings
    set_language(settings.get_language())

    # Set application icon
    from pathlib import Path
    icon_path = Path(__file__).parent / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
