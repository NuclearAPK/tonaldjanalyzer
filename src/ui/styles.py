"""
UI styles and color definitions.
"""

# Compatibility colors (RGB)
COLORS = {
    'excellent': (76, 175, 80),      # Green #4CAF50
    'good': (139, 195, 74),          # Yellow-Green #8BC34A
    'fair': (255, 235, 59),          # Yellow #FFEB3B
    'poor': (255, 152, 0),           # Orange #FF9800
    'bad': (244, 67, 54),            # Red #F44336
    'neutral': (128, 128, 128),      # Gray
    'master': (33, 150, 243),        # Blue #2196F3
}


def get_color_for_score(score: float | None) -> tuple[int, int, int]:
    """Get RGB color tuple for compatibility score."""
    if score is None:
        return COLORS['neutral']

    if score >= 80:
        return COLORS['excellent']
    elif score >= 60:
        return COLORS['good']
    elif score >= 40:
        return COLORS['fair']
    elif score >= 20:
        return COLORS['poor']
    else:
        return COLORS['bad']


def rgb_to_stylesheet(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to CSS color string."""
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


# Main application stylesheet
MAIN_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #2d2d2d;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}

QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 8px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #4d4d4d;
    border-color: #666666;
}

QPushButton:pressed {
    background-color: #2d2d2d;
}

QPushButton:disabled {
    background-color: #252525;
    color: #666666;
}

QPushButton#masterButton {
    background-color: #1976D2;
    border-color: #2196F3;
}

QPushButton#masterButton:hover {
    background-color: #2196F3;
}

QTableWidget {
    background-color: #1e1e1e;
    alternate-background-color: #252525;
    gridline-color: #3d3d3d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 8px;
}

QTableWidget::item:selected {
    background-color: #1976D2;
}

QHeaderView::section {
    background-color: #3d3d3d;
    padding: 8px;
    border: none;
    border-right: 1px solid #4d4d4d;
    border-bottom: 1px solid #4d4d4d;
    font-weight: bold;
}

QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    background-color: #252525;
}

QProgressBar::chunk {
    background-color: #4CAF50;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    border: 1px solid #3d3d3d;
    height: 8px;
    background: #252525;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #4CAF50;
    border: 1px solid #45a049;
    width: 16px;
    margin: -4px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #66BB6A;
}

QLabel#titleLabel {
    font-size: 14px;
    font-weight: bold;
}

QLabel#statusLabel {
    color: #888888;
    font-size: 11px;
}

QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
"""
