"""Structured logger configuration for the AI Quantum 2026 Pipeline."""

import logging
import sys


def setup_logger(name: str = "ai_quantum_2026", level: int = logging.INFO) -> logging.Logger:
    """Sets up a structured logger.

    Args:
        name: Name of the logger.
        level: Logger level (e.g. logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Create a default logger for package-wide use
logger = setup_logger()
