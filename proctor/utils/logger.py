from __future__ import annotations
"""
Proctor Logger

Centralized logging configuration.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        format_str: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if format_str is None:
            format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger
