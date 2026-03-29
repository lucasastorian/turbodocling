import time
import threading
from typing import Dict, Optional, Any

from processor.shared.logging_config import get_logger

logger = get_logger(__name__)


class JobRegistry:
    def __init__(self):
        self.task_tokens: Dict[str, str] = {}
        self.job_start_times: Dict[str, float] = {}
        self.pdf_cache: Dict[str, bytes] = {}
        self.final_docs: Dict[str, Dict[str, Any]] = {}

        self._stats_lock = threading.Lock()
        self._total_pages_completed = 0
        self._total_docs_completed = 0
        self._stats_start_time = time.monotonic()

    def active_job_count(self) -> int:
        active = set(self.task_tokens)
        active.update(self.pdf_cache)
        active.update(self.final_docs)
        return len(active)

    def register_job(self, job_id: str, token: str):
        self.task_tokens[job_id] = token
        self.job_start_times[job_id] = time.time()

    def pop_token(self, job_id: str) -> Optional[str]:
        self.job_start_times.pop(job_id, None)
        return self.task_tokens.pop(job_id, None)

    def record_completed(self, pages: int):
        with self._stats_lock:
            self._total_pages_completed += pages
            self._total_docs_completed += 1

    def stats(self) -> tuple[float, int, int]:
        with self._stats_lock:
            elapsed = time.monotonic() - self._stats_start_time
            return elapsed, self._total_pages_completed, self._total_docs_completed

    def fail_all_jobs(self, sfn_client, cause: str):
        for job_id, token in list(self.task_tokens.items()):
            try:
                sfn_client.send_task_failure(
                    taskToken=token,
                    error="GPUServiceCrash",
                    cause=cause,
                )
                logger.info(f"failed job={job_id[:4]}: {cause}")
            except Exception:
                pass
        self.task_tokens.clear()
        self.job_start_times.clear()
