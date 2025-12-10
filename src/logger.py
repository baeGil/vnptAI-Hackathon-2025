import os
from datetime import datetime
from src.config import config

LOG_FILE = os.path.join(config.OUTPUT_DIR, "inference_detail.log")

def log(message: str):
    """Write message to log file instead of terminal."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
