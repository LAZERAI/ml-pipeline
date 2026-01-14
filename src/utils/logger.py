"""
Logging Module
--------------
Centralized logging configuration for the ML Pipeline.
"""

import os
import logging
from datetime import datetime

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file with timestamp
LOG_FILE = os.path.join(LOGS_DIR, f"ml_pipeline_{datetime.now().strftime('%Y%m%d')}.log")

# Logging format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# Create a default logger for quick imports
logger = get_logger("ml_pipeline")


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    print(f"Log file: {LOG_FILE}")
