"""
Shared constants for the GPU processor pipeline
"""

# Shutdown sentinel - used to signal worker processes to exit gracefully
SHUTDOWN = {"__shutdown__": True}

def is_shutdown(x) -> bool:
    """Check if a message is a shutdown sentinel"""
    return isinstance(x, dict) and x.get("__shutdown__") is True
