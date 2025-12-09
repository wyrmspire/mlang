import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        
        # File Handler
        log_file = LOGS_DIR / "app.log"
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        
        # Ensure immediate flush for file handler (often helpful in dev)
        # f_handler.flush = lambda: super(logging.FileHandler, f_handler).flush()
        
    return logger

# Configure Root logger to capture third party logs if needed
# But for now, just ensuring our named loggers work is usually enough.
