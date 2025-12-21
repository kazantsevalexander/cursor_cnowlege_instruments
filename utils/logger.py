"""
Logging configuration using loguru for the RAG Vector Demo project.
"""

import sys
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configure the logger with custom formatting.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()  # Remove default handler
    
    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler for errors
    logger.add(
        "logs/rag_demo_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="7 days",
        compression="zip"
    )


# Initialize logger on import
setup_logger()

