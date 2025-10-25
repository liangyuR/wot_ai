"""
Logging utilities
"""

import sys
from pathlib import Path
from loguru import logger


def SetupLogger(log_dir: str = "logs", level: str = "INFO"):
    """
    Setup logger with file and console output
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "wot_ai_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="7 days",  # Keep logs for 7 days
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    return logger

