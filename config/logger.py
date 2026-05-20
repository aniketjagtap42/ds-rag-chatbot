# config/logger.py
# ─────────────────────────────────────────────────────
# Central logging setup for the entire application.
# Every module imports get_logger() from here.
# Logs go to both terminal AND logs/app.log file.
# ─────────────────────────────────────────────────────

import logging
import os
from logging.handlers import RotatingFileHandler

# ── Log folder ───────────────────────────────────────
# Create logs/ folder automatically if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Log format ───────────────────────────────────────
# Every log line looks like:
# 2026-05-19 18:30:00 | INFO     | ingestion.loader | Loaded 1 sections
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for any module.
    Usage:
        from config.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # ── Avoid duplicate handlers ──────────────────────
    # Without this, every import adds new handlers
    # and every message prints multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Handler 1: Console ────────────────────────────
    # Shows logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Handler 2: Rotating File ──────────────────────
    # Saves to logs/app.log
    # When file hits 5MB → creates new file
    # Keeps last 3 files → disk never fills up
    file_handler = RotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "app.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger