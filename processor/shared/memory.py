from pathlib import Path
from typing import Optional


def _read_int(path: str) -> Optional[int]:
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _container_memory_limit_bytes() -> Optional[int]:
    limit = _read_int("/sys/fs/cgroup/memory.max")
    if limit:
        return limit
    return _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _container_memory_usage_bytes() -> Optional[int]:
    usage = _read_int("/sys/fs/cgroup/memory.current")
    if usage is not None:
        return usage
    return _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")


def memory_state() -> tuple[Optional[int], Optional[int], Optional[float]]:
    usage = _container_memory_usage_bytes()
    limit = _container_memory_limit_bytes()
    if usage is None or limit in (None, 0):
        return usage, limit, None
    return usage, limit, usage / limit
