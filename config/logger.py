# config/logger.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Central logging configuration for the entire application.
#          Every module gets its logger from here via get_logger().
#
# WHY CENTRALISED LOGGING?
#   If every file set up its own logging, you'd get:
#   - Inconsistent formats across modules
#   - Duplicate handlers being added on re-imports
#   - No single place to change log level or format
#   One get_logger() function = consistent logs everywhere.
#
# WHERE LOGS GO:
#   1. Terminal (stdout) → visible in Docker logs / kubectl logs
#   2. logs/app.log file → persists on disk, rotates at 5MB
#
# HOW TO USE IN ANY FILE:
#   from config.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Something happened")
#   logger.error("Something broke", exc_info=True)
#
# LOG LEVELS (lowest → highest severity):
#   DEBUG    → detailed dev info (disabled in production)
#   INFO     → normal operation events
#   WARNING  → something unexpected but non-fatal
#   ERROR    → something failed, needs attention
#   CRITICAL → startup failure, app cannot run
# ─────────────────────────────────────────────────────────────────────

import logging                                    # Python's built-in logging module
import os                                         # os.makedirs → create logs/ folder if missing
from logging.handlers import RotatingFileHandler  # rotates log file when it hits max size

# ── Create logs/ directory automatically ─────────────────────────────
# exist_ok=True → no error if the folder already exists
# WHY create here at import time? So the folder always exists
# before any handler tries to write to logs/app.log
LOGS_DIR = os.environ.get("LOG_DIR", "/tmp/logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Log format string ─────────────────────────────────────────────────
# Every log line will look like:
#   2026-05-19 18:30:00 | INFO     | ingestion.loader | Loaded 1 section
#
# %(asctime)s   → timestamp
# %(levelname)s → INFO / WARNING / ERROR etc.
# %-8s          → left-aligned, padded to 8 chars so columns line up
# %(name)s      → module name passed to get_logger(__name__)
# %(message)s   → the actual log message
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"   # human-readable timestamp format

# ── Read log level from environment ──────────────────────────────────
# WHY from environment? In production you want INFO.
# In development you might want DEBUG to see more detail.
# Setting LOG_LEVEL=DEBUG in .env enables verbose output
# without changing any code.
# os.getenv → reads env var, falls back to "INFO" if not set
_LOG_LEVEL = getattr(
    logging,
    os.getenv("LOG_LEVEL", "INFO").upper(),  # e.g. "DEBUG" → logging.DEBUG (= 10)
    logging.INFO                             # fallback if env var has invalid value
)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger for a module.

    Call this once at the top of every Python file:
        logger = get_logger(__name__)

    __name__ evaluates to the module path, e.g.:
        "ingestion.loader"
        "agents.router_agent"
        "api.main"

    This name appears in every log line so you instantly
    know which file produced each message.

    Args:
        name (str): module name — always pass __name__

    Returns:
        logging.Logger: configured logger with console + file handlers
    """
    logger = logging.getLogger(name)   # get existing or create new logger with this name

    # ── Avoid adding duplicate handlers ───────────────────────────────
    # WHY this check? Python's logging module is global.
    # If this module is imported multiple times (e.g. in tests),
    # without this check each import adds another handler.
    # Result: every log message prints 2x, 3x, 4x times.
    # Checking logger.handlers prevents adding handlers twice.
    if logger.handlers:
        return logger   # already configured — return as-is

    # ── Set the log level ─────────────────────────────────────────────
    # This controls the MINIMUM severity that gets logged.
    # INFO → logs INFO, WARNING, ERROR, CRITICAL (not DEBUG)
    # DEBUG → logs everything including DEBUG messages
    logger.setLevel(_LOG_LEVEL)

    # ── Prevent propagation to root logger ────────────────────────────
    # WHY? Python's root logger also has handlers by default.
    # Without this, every message propagates UP to root logger
    # and gets printed a second time by root's handler.
    # propagate=False keeps our logs clean and non-duplicated.
    logger.propagate = False

    # ── Shared formatter ──────────────────────────────────────────────
    # Both handlers use the same format for consistency.
    # One formatter instance shared by both handlers.
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── HANDLER 1: Console (stdout) ───────────────────────────────────
    # Writes log lines to the terminal.
    # In Docker → visible via: docker logs <container_name>
    # In Kubernetes → visible via: kubectl logs <pod_name>
    # In AWS ECS → visible in CloudWatch Logs automatically
    console_handler = logging.StreamHandler()          # streams to sys.stderr by default
    console_handler.setLevel(_LOG_LEVEL)               # same level as logger
    console_handler.setFormatter(formatter)            # apply shared format

    # ── HANDLER 2: Rotating File ──────────────────────────────────────
    # Writes log lines to logs/app.log on disk.
    # WHY rotating? Log files grow forever without rotation.
    # maxBytes=5MB → when app.log hits 5MB, it becomes app.log.1
    #                and a new empty app.log is created.
    # backupCount=3 → keeps app.log, app.log.1, app.log.2, app.log.3
    #                 older files are deleted automatically.
    # Total max disk usage: 4 files × 5MB = 20MB — safe for any server.
    file_handler = RotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "app.log"),  # logs/app.log
        maxBytes=5 * 1024 * 1024,                    # 5 MB per file (5 × 1024 × 1024 bytes)
        backupCount=3,                                # keep 3 old files before deleting
        encoding="utf-8"                              # handle unicode characters in log messages
    )
    file_handler.setLevel(_LOG_LEVEL)      # same level as logger
    file_handler.setFormatter(formatter)   # apply shared format

    # ── Attach both handlers to this logger ───────────────────────────
    # Every logger.info() / logger.error() call now goes to
    # BOTH the terminal AND the log file simultaneously.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger   # return configured logger to the calling module