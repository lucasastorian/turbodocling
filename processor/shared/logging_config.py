"""
Shared logging configuration for all GPU processor services
"""
import sys
import logging
import threading


def setup_main_logging(level: int = logging.INFO):
    """
    Set up centralized logging for threaded services
    
    Args:
        level: Logging level
    """
    # Set up root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Clear any existing handlers
    root.handlers[:] = []
    
    # Create single StreamHandler for all logs
    stream = logging.StreamHandler(sys.stdout)
    # Minimal, readable format: timestamp + level + thread + message
    stream.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s:%(process)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    
    root.addHandler(stream)


def setup_worker_logging(service_name: str, level: int = logging.INFO):
    """
    Set up logging for worker threads (simplified - no cross-process complexity needed)
    
    Args:
        service_name: Name of the service (e.g., "GPUService", "PostprocessService")
        level: Logging level
    """
    # Set thread name for better identification
    threading.current_thread().name = service_name
    
    # Logging is already set up by main thread - this is mainly for thread naming
    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance - works in both main and worker processes
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
