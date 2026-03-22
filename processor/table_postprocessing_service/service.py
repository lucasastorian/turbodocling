"""Table Postprocessing Service - Handles table postprocessing and merging with pages"""
import queue
import threading

from .table_postprocessing_model import TablePostprocessingModel
from processor.shared.logging_config import get_logger
from processor.shared.telemetry import start_queue_monitor, mark

logger = get_logger(__name__)


class TablePostprocessingService:
    def __init__(self, input_queue: queue.Queue, final_queue: queue.Queue,
                 shutdown_event: threading.Event):
        self.input_queue = input_queue  # table_postprocess_queue from GPU service
        self.final_queue = final_queue
        self.shutdown_event = shutdown_event

        self.max_batch_size = 128

        self.table_postprocessing_model = TablePostprocessingModel()

        start_queue_monitor("table_postprocess", {
            "input": self.input_queue,
            "final": self.final_queue
        }, self.shutdown_event)
    
    def _put_with_shutdown(self, q: queue.Queue, item, shutdown_event: threading.Event) -> bool:
        """Block politely until space or shutdown - won't tear down the thread"""
        while not shutdown_event.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        while not self.shutdown_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Error containment - don't let bad pages kill the whole service
            try:
                import time as _time
                if not item['table_predictions']:
                    out = {
                        'job_id': item['job_id'],
                        'user_id': item['user_id'],
                        'page_no': item['page_no'],
                        'page': item['page'],
                        'total_pages': item['total_pages'],
                        '_trace': item.get('_trace', {}).copy(),
                    }
                    mark(out, 'tpp')
                    self._put_with_shutdown(self.final_queue, out, self.shutdown_event)
                    continue

                t0_tpp = _time.perf_counter()
                processed_pages = self.table_postprocessing_model.postprocess(
                    all_predictions=item['table_predictions'],
                    pages=[item['page']],
                    iocr_pages=item['table_args']['iocr_pages'],
                    table_bboxes=item['table_args']['table_bboxes'],
                    scale_factors=item['table_args']['scale_factors'],
                    page_clusters_list=item['table_args']['page_clusters_list'],
                    batched_page_indexes=item['table_args']['batched_page_indexes'],
                )
                t1_tpp = _time.perf_counter()
                logger.info(f"tbl_pp page={item.get('page_no')} tables={len(item['table_predictions'])} wall={1000*(t1_tpp-t0_tpp):.0f}ms")

                processed_page = processed_pages[0]

                out = {
                    'job_id': item['job_id'],
                    'user_id': item['user_id'],
                    'page_no': item['page_no'],
                    'page': processed_page,
                    'total_pages': item['total_pages'],
                    '_trace': item.get('_trace', {}).copy(),
                }
                mark(out, 'tpp')
                self._put_with_shutdown(self.final_queue, out, self.shutdown_event)
                
            except Exception as e:
                logger.exception(f"table postprocess error page={item.get('page_no')}: {e}")
                # Forward page to final queue to prevent job deadlock
                self._put_with_shutdown(self.final_queue, {
                    'job_id': item['job_id'],
                    'user_id': item.get('user_id'),
                    'page_no': item.get('page_no'),
                    'page': item['page'],
                    'total_pages': item.get('total_pages'),
                    '_trace': item.get('_trace', {}).copy(),
                }, self.shutdown_event)
                continue
