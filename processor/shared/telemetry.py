import json
import logging
import time
import uuid
import os
import pickle
import threading
from contextlib import contextmanager
from typing import Dict, Any
from queue import Queue

log = logging.getLogger(__name__)

def now():
    """Get monotonic timestamp"""
    return time.perf_counter()

def ms(dt):
    """Convert time delta to milliseconds"""
    return round(dt * 1000.0, 3)

def new_batch_id():
    """Generate a short batch ID"""
    return uuid.uuid4().hex[:8]

@contextmanager
def span(stage: str, op: str, extra: dict):
    """Context manager for timing operations"""
    t0 = now()
    try:
        yield
    finally:
        log.info(json.dumps({
            "evt": "span",
            "stage": stage,
            "op": op,
            "lat_ms": ms(now() - t0),
            **extra
        }))

def mark(item: dict, key: str):
    """Attach a monotonic timestamp to an item without bloating payload"""
    tr = item.setdefault("_trace", {})
    tr[key] = now()

def trace_lat(item: dict, start: str, end: str) -> float:
    """Calculate latency between two marks"""
    tr = item.get("_trace", {})
    if start in tr and end in tr:
        return ms(tr[end] - tr[start])
    return -1.0

def proc_meta(**kw):
    """Get process metadata for logging"""
    base = {
        "pid": os.getpid(),
        "proc": os.getenv("PROC_NAME", "")
    }
    base.update(kw)
    return base

def log_event(evt: str, stage: str, **fields):
    """Log a structured event"""
    import os
    # Optional suppression of noisy per-page telemetry
    suppress_pages = os.getenv("TELEMETRY_PAGES", "0").lower() not in ("1", "true", "yes")
    if suppress_pages and evt in {"page_received", "page_complete", "sla_warn", "sla_error"}:
        return

    suppress_queue = os.getenv("TELEMETRY_QUEUE", "0").lower() not in ("1", "true", "yes")
    if evt == "queue_put":
        if suppress_queue:
            return
        log.info(f"→ {fields.get('q', '?')}: {fields.get('job_id', '?')[:4]}/p{fields.get('page_no', '?')}")
    elif evt == "queue_get":
        if suppress_queue:
            return
        qwait = fields.get('qwait_ms', 0)
        log.info(f"← {fields.get('q', '?')}: {fields.get('job_id', '?')[:4]}/p{fields.get('page_no', '?')} ({qwait:.0f}ms)")
    elif evt == "batch_done":
        n = fields.get('n', 0)
        log.info(f"batch done: {n} items")
    elif evt == "document_complete":
        log.info(f"DOC DONE: {fields.get('job_id', '?')[:4]} ({fields.get('n_pages', 0)} pages)")
    else:
        # Fall back to JSON for other events
        log.info(json.dumps({"evt": evt, "stage": stage, **fields}))

def benchmark_pickle(obj: Any, label: str = "object", *, force: bool = False, iters: int = 1, protocol: int = None) -> Dict[str, float]:
    """Benchmark pickle serialization/deserialization times"""
    # Only benchmark if enabled or forced
    if not force and os.getenv("PIPELINE_PROFILE", "").lower() not in ("1", "true", "yes"):
        return {}
    
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    
    # Serialize multiple times for better accuracy
    t0 = now()
    for _ in range(iters):
        buf = pickle.dumps(obj, protocol=protocol)
    ser_ms = ms(now() - t0)
    
    # Deserialize multiple times
    t0 = now()
    for _ in range(iters):
        pickle.loads(buf)
    de_ms = ms(now() - t0)
    
    size = len(buf)
    total = ser_ms + de_ms
    
    result = {
        "serialize_ms": ser_ms / iters,
        "deserialize_ms": de_ms / iters,
        "total_ms": total / iters,
        "size_bytes": size,
        "mb_per_sec": round((size / 1_048_576) / (total / iters / 1000), 2) if total > 0 else 0.0,
        "protocol": protocol,
        "iters": iters
    }
    
    # Simple, readable pickle benchmark log
    mb = result["size_bytes"] / 1_048_576
    total_ms = result["total_ms"]
    page_num = label.split('_p')[-1] if '_p' in label else '?'
    log.info(f"PICKLE p{page_num}: {mb:.1f}MB {total_ms:.0f}ms ({result['mb_per_sec']:.0f}MB/s)")
    return result


def benchmark_queue_roundtrip(ctx, payload: Any, label: str, *, iters: int = 1) -> Dict[str, float]:
    """Benchmark actual queue roundtrip time including pickle + pipe overhead"""
    if os.getenv("PIPELINE_PROFILE", "").lower() not in ("1", "true", "yes"):
        return {}
    
    try:
        q = ctx.Queue()
        
        def worker(q):
            for _ in range(iters):
                item = q.get()
                q.put(item)
        
        p = ctx.Process(target=worker, args=(q,), name="bench-worker")
        p.start()
        
        t0 = now()
        for _ in range(iters):
            q.put(payload)
            _ = q.get()
        total = ms(now() - t0)
        
        p.terminate()
        p.join(timeout=1)
        
        result = {"roundtrip_ms": total / iters}
        log_event("queue_roundtrip_benchmark", "dev", label=label, **result)
        return result
        
    except Exception as e:
        log.warning(f"Queue roundtrip benchmark failed: {e}")
        return {}

def start_queue_monitor(stage: str, queues: Dict[str, Queue], stop_event=None):
    """Start background thread to monitor queue depths.

    Environment controls:
      - QMON_ENABLED: 1/0 (default 0)
      - QMON_INTERVAL: seconds between checks (default 1.0)
      - QMON_ONLY_CHANGES: 1/0 log only on changes (default 1)
      - QMON_MIN_DEPTH: minimum depth to report (default 1)
    """
    import threading
    import os

    if stop_event is None:
        stop_event = threading.Event()

    enabled = os.getenv("QMON_ENABLED", "0").lower() in ("1", "true", "yes")
    interval = float(os.getenv("QMON_INTERVAL", "1.0"))
    only_changes = os.getenv("QMON_ONLY_CHANGES", "1").lower() in ("1", "true", "yes")
    min_depth = int(os.getenv("QMON_MIN_DEPTH", "1"))

    if not enabled:
        return None

    def run():
        last_snapshot = None
        while not stop_event.is_set():
            # Build snapshot of queues above threshold
            parts = []
            for k, q in queues.items():
                try:
                    qsize = q.qsize()
                    if qsize >= min_depth:
                        parts.append(f"{k}:{qsize}")
                except (NotImplementedError, AttributeError):
                    # qsize() not available on all platforms
                    pass

            snapshot = " ".join(parts)

            # Log only on changes (or always if disabled)
            if snapshot and (not only_changes or snapshot != last_snapshot):
                log.info(f"Q: {snapshot}")
                last_snapshot = snapshot

            # Use wait() to exit quickly on stop
            stop_event.wait(interval)

    t = threading.Thread(target=run, daemon=True, name=f"QueueMonitor-{stage}")
    t.start()
    return t

class SLAWatchdog:
    """Track page processing SLA and emit warnings/errors"""
    
    def __init__(self, stage: str, warn_ms: float = 3000, error_ms: float = 10000):
        self.stage = stage
        self.warn_ms = warn_ms
        self.error_ms = error_ms
        self.page_start_times = {}  # (job_id, page_no) -> start_time
        self.warned = set()  # Track which pages we've already warned about
        self._lock = threading.Lock()
        
        # Start watchdog thread
        t = threading.Thread(target=self._run_watchdog, daemon=True, name=f"SLAWatchdog-{stage}")
        t.start()
    
    def mark_page_start(self, job_id: str, page_no: int):
        """Mark when a page first enters the pipeline"""
        key = (job_id, page_no)
        with self._lock:
            if key not in self.page_start_times:
                self.page_start_times[key] = now()
    
    def mark_page_complete(self, job_id: str, page_no: int):
        """Mark when a page completes processing"""
        key = (job_id, page_no)
        with self._lock:
            start_time = self.page_start_times.pop(key, None)
            self.warned.discard(key)
            
        if start_time:
            total_ms = ms(now() - start_time)
            log_event("page_complete", self.stage, 
                     job_id=job_id, page_no=page_no, total_ms=total_ms)
    
    def _run_watchdog(self):
        """Background thread to check for SLA violations"""
        while True:
            current_time = now()
            
            with self._lock:
                for key, start_time in list(self.page_start_times.items()):
                    job_id, page_no = key
                    dur_ms = ms(current_time - start_time)
                    
                    # SLA warn/error logging disabled temporarily during tuning
                    # if dur_ms > self.error_ms:
                    #     log_event("sla_error", self.stage,
                    #              job_id=job_id, page_no=page_no, dur_ms=dur_ms)
                    #     self.warned.add(key)
                    # elif dur_ms > self.warn_ms and key not in self.warned:
                    #     log_event("sla_warn", self.stage,
                    #              job_id=job_id, page_no=page_no, dur_ms=dur_ms)
                    #     self.warned.add(key)
            
            time.sleep(1.0)
