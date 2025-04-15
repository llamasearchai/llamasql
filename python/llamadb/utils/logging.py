"""
Logging Utilities

This module provides logging utilities for LlamaDB.
"""

import logging
import os
import sys
from typing import Optional

# Configure logging format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DETAILED_FORMAT = "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d %(message)s"

# Configure logging level from environment variable
DEFAULT_LEVEL = os.environ.get("LLAMADB_LOG_LEVEL", "INFO").upper()
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
LOG_LEVEL = DEFAULT_LEVEL if DEFAULT_LEVEL in VALID_LEVELS else "INFO"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Args:
        name: Name of the logger
        level: Optional logging level (defaults to LLAMADB_LOG_LEVEL env var or INFO)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Set level from parameter, environment variable, or default
    if level is not None and level.upper() in VALID_LEVELS:
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(getattr(logging, LOG_LEVEL))

    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(DETAILED_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def configure_root_logger(
    level: Optional[str] = None, log_file: Optional[str] = None
) -> None:
    """
    Configure the root logger.

    Args:
        level: Optional logging level (defaults to LLAMADB_LOG_LEVEL env var or INFO)
        log_file: Optional file to log to
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Set level from parameter, environment variable, or default
    if level is not None and level.upper() in VALID_LEVELS:
        root_logger.setLevel(getattr(logging, level.upper()))
    else:
        root_logger.setLevel(getattr(logging, LOG_LEVEL))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(DEFAULT_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(DETAILED_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


# Configure root logger with default settings
configure_root_logger()
