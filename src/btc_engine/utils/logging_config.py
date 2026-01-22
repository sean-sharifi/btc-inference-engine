"""Structured logging configuration"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler

from btc_engine.utils.config_loader import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    name: str = "btc_engine"
) -> logging.Logger:
    """Setup structured logging with file and console handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=True
    )
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance
logger = setup_logging()
