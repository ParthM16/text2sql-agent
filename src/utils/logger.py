import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """Gets or creates a logger configured to write to daily rotating log files."""
    logger = logging.getLogger(name)
    
    # Avoid attaching multiple handlers if logger is requested multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Daily rotating file: agent_YYYYMMDD.log
        log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d')}.log")
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(fh)
        
    return logger
