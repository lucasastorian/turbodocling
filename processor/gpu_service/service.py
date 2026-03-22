import gc
import os
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import LayoutOptions, TableStructureOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2

from processor.shared.logging_config import get_logger
from processor.shared.telemetry import start_queue_monitor, mark
from processor.shared.queue_utils import drain_queue

logger = get_logger(__name__)


def get_device() -> str:
    """Get the device to use for inference. Supports cuda, mps, or cpu."""
    device = os.environ.get("DEVICE", "").lower()
    if device in ("cuda", "mps", "cpu"):
        return device

    # Auto-detect
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GPUService:

    def __init__(self, layout_queue: queue.Queue, postprocess_queue: queue.Queue,
                 table_queue: queue.Queue, table_postprocess_queue: queue.Queue,
                 shutdown_event: threading.Event, artifacts_path: Optional[Path] = None):
        from .layout_model import LayoutModel
        from .table_inference_model import TableInferenceModel

        self.layout_queue = layout_queue
        self.postprocess_queue = postprocess_queue
        self.table_queue = table_queue
        self.table_postprocess_queue = table_postprocess_queue
        self.shutdown_event = shutdown_event

        self.max_layout_batch = 128
        self.max_table_batch = 128  # Same limit for table processing
        self.artifacts_path = artifacts_path
        
        # Accumulate items by job_id to batch process entire documents
        self.pending_layout_jobs: Dict[str, Dict] = {}  # job_id -> {items: [], total_pages: int, received: int}
        self.pending_table_jobs: Dict[str, Dict] = {}  # job_id -> {items: [], total_pages: int, received: int}

        device = get_device()
        logger.info(f"Using device: {device}")

        self.layout_model = LayoutModel(
            artifacts_path=self.artifacts_path,
            accelerator_options=AcceleratorOptions(device=device),
            options=LayoutOptions(model_spec=DOCLING_LAYOUT_V2)
        )

        self.table_inference_model = TableInferenceModel(
            enabled=True,
            artifacts_path=self.artifacts_path,
            options=TableStructureOptions(),
            accelerator_options=AcceleratorOptions(device=device)
        )

        start_queue_monitor("gpu", {
            "layout": self.layout_queue,
            "postprocess": self.postprocess_queue,
            "table": self.table_queue,
            "table_postprocess": self.table_postprocess_queue
        }, self.shutdown_event)
        
        # Set torch threads (environment variables are set at process start)
        import torch
        torch.set_num_threads(1)

        logger.info(f"GPU service ready - device={device} layout_model=ok table_model=ok")

    def run(self):
        while not self.shutdown_event.is_set():
            layout_processed = self._process_layout_batch()
            tables_processed = self._process_table_batch()

            if not layout_processed and not tables_processed:
                time.sleep(0.01)

    def _process_layout_batch(self):
        """Accumulate layout items by job_id and process when complete or at 128 page limit"""
        if self.shutdown_event.is_set():
            return False
        
        # Drain whatever is currently available (non-blocking)
        batch_lists = drain_queue(self.layout_queue, self.max_layout_batch, self.shutdown_event)

        if not batch_lists:
            return False
            
        # Flatten micro-batches from MainService
        new_items = []
        for x in batch_lists:
            new_items.extend(x if isinstance(x, list) else [x])

        # Accumulate items by job_id
        for item in new_items:
            job_id = item["job_id"]
            total_pages = item.get("total_pages", 0)
            
            if job_id not in self.pending_layout_jobs:
                self.pending_layout_jobs[job_id] = {
                    "items": [],
                    "total_pages": total_pages,
                    "received": 0
                }
            
            job_data = self.pending_layout_jobs[job_id]
            job_data["items"].append(item)
            job_data["received"] += 1
            job_data["total_pages"] = max(job_data["total_pages"], total_pages)  # Handle race conditions

        # Check which jobs are ready to process
        jobs_to_process = []
        for job_id, job_data in list(self.pending_layout_jobs.items()):
            items_count = len(job_data["items"])
            should_process = (
                job_data["received"] >= job_data["total_pages"]  # All pages received
                or items_count >= self.max_layout_batch  # Hit batch limit (soft limit)
            )
            
            if should_process:
                jobs_to_process.append(job_id)

        if not jobs_to_process:
            return False

        # Process ready jobs
        all_items = []
        for job_id in jobs_to_process:
            job_data = self.pending_layout_jobs[job_id]
            all_items.extend(job_data["items"])
            logger.info(f"layout batch: job={job_id[:4]} pages={len(job_data['items'])}/{job_data['total_pages']}")

            if job_data["received"] >= job_data["total_pages"]:
                # All pages received, clean up tracking
                del self.pending_layout_jobs[job_id]
            else:
                # Batch limit hit but more pages expected — keep tracking
                job_data["items"] = []

        # Execute layout inference on accumulated batch
        return self._execute_layout_inference(all_items)

    def _execute_layout_inference(self, items):
        """Execute layout inference on a batch of items"""
        t0 = time.time()
        predictions = self.layout_model(page_batch=[item['page'] for item in items])
        t_layout = time.time()

        # Layout model timers + queue sizes
        lp = self.layout_model.layout_predictor
        def _qsz(q):
            try:
                return q.qsize()
            except Exception:
                return -1
        logger.info(
            f"layout: n={len(items)} wall={1000*(t_layout-t0):.0f}ms ql={_qsz(self.layout_queue)} qp={_qsz(self.postprocess_queue)}"
            f" | pre={lp._t_preprocess_ms:.0f}ms pred={lp._t_predict_ms:.0f}ms post={lp._t_postprocess_ms:.0f}ms"
        )

        for item, page_predictions in zip(items, predictions):
            out = {
                'job_id': item['job_id'],
                'user_id': item.get('user_id'),
                'page_no': item.get('page_no'),
                'page': item['page'],
                'predictions': page_predictions,
                'total_pages': item.get('total_pages'),
                '_trace': item.get('_trace', {}).copy(),
            }
            mark(out, 'lay')
            self.postprocess_queue.put(out)

        return True

    def _process_table_batch(self):
        """Accumulate table items by job_id and process when complete or at 128 page limit"""
        if self.shutdown_event.is_set():
            return False
        
        # Drain whatever is currently available (non-blocking)
        new_items = drain_queue(self.table_queue, 512, self.shutdown_event)

        if not new_items:
            return False

        # Accumulate items by job_id
        for item in new_items:
            job_id = item["job_id"]
            total_pages = item.get("total_pages", 0)
            
            if job_id not in self.pending_table_jobs:
                self.pending_table_jobs[job_id] = {
                    "items": [],
                    "total_pages": total_pages,
                    "received": 0
                }
            
            job_data = self.pending_table_jobs[job_id]
            job_data["items"].append(item)
            job_data["received"] += 1
            job_data["total_pages"] = max(job_data["total_pages"], total_pages)  # Handle race conditions

        # Check which jobs are ready to process
        jobs_to_process = []
        for job_id, job_data in list(self.pending_table_jobs.items()):
            items_count = len(job_data["items"])
            should_process = (
                job_data["received"] >= job_data["total_pages"]  # All pages received
                or items_count >= self.max_table_batch  # Hit batch limit
            )
            
            if should_process:
                jobs_to_process.append(job_id)

        if not jobs_to_process:
            return False

        # Process ready jobs
        all_items = []
        for job_id in jobs_to_process:
            job_data = self.pending_table_jobs[job_id]
            all_items.extend(job_data["items"])
            logger.info(f"table batch: job={job_id[:4]} pages={len(job_data['items'])}/{job_data['total_pages']}")

            if job_data["received"] >= job_data["total_pages"]:
                del self.pending_table_jobs[job_id]
            else:
                job_data["items"] = []

        # Execute table inference on accumulated batch
        return self._execute_table_inference(all_items)

    def _execute_table_inference(self, items):
        """Execute table inference on a batch of items"""
        table_args = [item['table_args'] for item in items]

        iocr_pages, table_bboxes, table_images, scale_factors = [], [], [], []

        for table_arg in table_args:
            iocr_pages.extend(table_arg['iocr_pages'])
            table_bboxes.extend(table_arg['table_bboxes'])
            table_images.extend(table_arg['table_images'])
            scale_factors.extend(table_arg['scale_factors'])

        total_tables = sum(len(a['table_bboxes']) for a in table_args)
        def _qsz(q):
            try:
                return q.qsize()
            except Exception:
                return -1

        logger.info(
            f"table: p={len(items)} t={total_tables} qt={_qsz(self.table_queue)} qtp={_qsz(self.table_postprocess_queue)}"
        )

        # No tables to process — pass items through with empty predictions
        if total_tables == 0:
            for item in items:
                out = {
                    "job_id": item["job_id"],
                    "user_id": item.get("user_id"),
                    "page_no": item.get("page_no"),
                    "page": item["page"],
                    "table_predictions": [],
                    "table_args": item['table_args'],
                    "total_pages": item.get("total_pages"),
                    "_trace": item.get("_trace", {}).copy(),
                }
                mark(out, 'tbl')
                self.table_postprocess_queue.put(out)
            return True

        t0_tbl = time.time()
        predictions = self.table_inference_model.predict(iocr_pages=iocr_pages, table_bboxes=table_bboxes,
                                                         table_images=table_images, scale_factors=scale_factors)
        t1_tbl = time.time()
        logger.info(f"table inference: tables={total_tables} wall={1000*(t1_tbl-t0_tbl):.0f}ms")
        n_tables = [len(item['table_args']['table_bboxes']) for item in items]

        prediction_chunks = []
        start_idx = 0

        for n in n_tables:
            end_idx = start_idx + n
            prediction_chunks.append(predictions[start_idx:end_idx])
            start_idx = end_idx

        for prediction_chunk, item in zip(prediction_chunks, items):
            out = {
                "job_id": item["job_id"],
                "user_id": item.get("user_id"),
                "page_no": item.get("page_no"),
                "page": item["page"],
                "table_predictions": prediction_chunk,
                "table_args": item['table_args'],
                "total_pages": item.get("total_pages"),
                "_trace": item.get("_trace", {}).copy(),
            }
            mark(out, 'tbl')
            self.table_postprocess_queue.put(out)

        return True

    @staticmethod
    def cleanup():
        """Clean up GPU resources on shutdown"""
        import torch
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
