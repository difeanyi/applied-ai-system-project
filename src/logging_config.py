"""Central logging setup: every module logs to system.log through one handler.

Modules get their logger with ``logging.getLogger("tunescout.<module>")`` so they
all inherit the handler set up here.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "tunescout"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Attach a file handler for ``log_file``. Safe to call repeatedly.

    Streamlit re-runs app.py on every interaction, so this function checks for
    an existing handler instead of adding a duplicate each time.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    log_path = str(Path(log_file).resolve())

    already_attached = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_path
        for h in logger.handlers
    )
    if not already_attached:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)

    # Keep our records out of the root logger (avoids duplicate console output).
    logger.propagate = False
    return logger
