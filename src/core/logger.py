"""
Application logging module.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None
_is_enabled: bool = False


def get_log_file_path() -> Path:
    """Get the path to the log file."""
    # Store logs in user's app data or next to executable
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        base_path = Path(sys.executable).parent
    else:
        # Running as script
        base_path = Path(__file__).parent.parent.parent
    
    logs_dir = base_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Use date-based log file name
    date_str = datetime.now().strftime("%Y-%m-%d")
    return logs_dir / f"tonaldj_{date_str}.log"


def setup_logger(enabled: bool = False):
    """Initialize the logger with optional file logging."""
    global _logger, _file_handler, _is_enabled
    
    _is_enabled = enabled
    
    if _logger is None:
        _logger = logging.getLogger("TonalDJ")
        _logger.setLevel(logging.DEBUG)
        
        # Console handler for critical errors only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        _logger.addHandler(console_handler)
    
    # Add or remove file handler based on enabled state
    if enabled:
        if _file_handler is None:
            log_path = get_log_file_path()
            _file_handler = logging.FileHandler(log_path, encoding='utf-8')
            _file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            _file_handler.setFormatter(file_formatter)
            _logger.addHandler(_file_handler)
            _logger.info("Logging started")
    else:
        if _file_handler is not None:
            _logger.info("Logging stopped")
            _logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None


def get_logger() -> logging.Logger:
    """Get the application logger."""
    global _logger
    if _logger is None:
        setup_logger(False)
    return _logger


def log_error(message: str, exc_info: bool = True):
    """Log an error message."""
    logger = get_logger()
    if _is_enabled:
        logger.error(message, exc_info=exc_info)


def log_warning(message: str):
    """Log a warning message."""
    logger = get_logger()
    if _is_enabled:
        logger.warning(message)


def log_info(message: str):
    """Log an info message."""
    logger = get_logger()
    if _is_enabled:
        logger.info(message)


def log_debug(message: str):
    """Log a debug message."""
    logger = get_logger()
    if _is_enabled:
        logger.debug(message)


def is_logging_enabled() -> bool:
    """Check if file logging is enabled."""
    return _is_enabled
