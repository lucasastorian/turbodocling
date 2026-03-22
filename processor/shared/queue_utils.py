"""
Utility functions for queue operations with graceful shutdown support
"""
from queue import Empty
from typing import List, Any
from .const import is_shutdown


def drain_queue(q, max_items: int, shutdown_event) -> List[Any]:
    """
    Drain items from a queue up to max_items, respecting shutdown sentinels
    
    Args:
        q: Queue to drain from
        max_items: Maximum number of items to retrieve
        shutdown_event: Event to set if shutdown sentinel is encountered
        
    Returns:
        List of items (excluding shutdown sentinels)
    """
    items = []
    while len(items) < max_items:
        try:
            item = q.get_nowait()
        except Empty:
            break
            
        # Check for shutdown sentinel
        if is_shutdown(item):
            # Put it back so other consumers can see it too
            try:
                q.put_nowait(item)
            except Exception:
                pass
            shutdown_event.set()
            break
            
        items.append(item)
    
    return items


def safe_queue_put(q, item, timeout=1.0) -> bool:
    """
    Safely put an item in a queue with timeout
    
    Args:
        q: Queue to put item into
        item: Item to put
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False if timeout or error
    """
    try:
        q.put(item, timeout=timeout)
        return True
    except Exception:
        return False


def safe_put(q, item, shutdown_event, timeout=1.0) -> bool:
    """
    Put an item in a queue, respecting shutdown events
    
    Args:
        q: Queue to put item into
        item: Item to put
        shutdown_event: Event to check for shutdown
        timeout: Timeout in seconds per attempt
        
    Returns:
        True if successful, False if shutdown was set
    """
    while not shutdown_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except Exception:
            # Queue full; loop until shutdown or success
            continue
    return False