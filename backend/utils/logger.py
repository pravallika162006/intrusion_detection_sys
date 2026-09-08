"""
Execution logger for timing and progress logging across Phase 1 pipeline stages.
"""

import logging
import sys
import time

def setup_logger(name: str = "IDS_Phase1") -> logging.Logger:
    """Configures and returns a stream logger for console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

class Timer:
    """Context manager for measuring block execution time."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
