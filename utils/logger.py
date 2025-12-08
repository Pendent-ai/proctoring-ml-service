"""
Logging Configuration
"""

import logging
import sys
from datetime import datetime


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from other libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProctoringLogger:
    """Specialized logger for proctoring events."""
    
    def __init__(self, participant_id: str | None = None):
        self.logger = get_logger("proctoring")
        self.participant_id = participant_id
    
    def log_alert(
        self,
        alert_type: str,
        severity: str,
        probability: float,
        details: dict | None = None,
    ):
        """Log an alert event."""
        self.logger.warning(
            f"ALERT | {self.participant_id} | {alert_type} | {severity} | prob={probability:.2f} | {details}"
        )
    
    def log_frame(self, frame_num: int, processing_time_ms: float):
        """Log frame processing."""
        self.logger.debug(
            f"FRAME | {self.participant_id} | #{frame_num} | {processing_time_ms:.1f}ms"
        )
    
    def log_connection(self, event: str, room_name: str):
        """Log connection events."""
        self.logger.info(
            f"CONNECTION | {event} | room={room_name}"
        )
