#!/usr/bin/env python3
"""
Tonal DJ Plugin - Track Compatibility Analyzer

Analyzes audio tracks and finds compatible matches based on
musical key (Camelot Wheel) and BPM for DJ mixing.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.ui import MainWindow


def main():
    """Application entry point."""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Tonal DJ")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
