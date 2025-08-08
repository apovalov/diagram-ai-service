from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging", "get_logger"]


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
