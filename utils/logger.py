"""Consistent console and file logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logger(
    name: str,
    log_file: str | Path,
    level: int | str = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Create an idempotently configured logger.

    Existing handlers owned by this helper are replaced, avoiding duplicate
    output when experiments configure the same logger more than once.
    """

    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_project_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(DEFAULT_FORMAT)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._project_handler = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._project_handler = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    return logger
