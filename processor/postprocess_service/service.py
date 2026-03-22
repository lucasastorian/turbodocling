
import queue
import threading
from typing import List, Dict
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox
from docling.datamodel.base_models import Page, Cluster, LayoutPrediction

from processor.postprocess_service.layout_postprocessor import (LayoutPostprocessor)
from processor.postprocess_service.table_structure_model import TableStructureModel
from processor.shared.telemetry import start_queue_monitor, mark
from processor.shared.logging_config import get_logger

logger = get_logger(__name__)

# Pre-built label cache: normalized string -> DocItemLabel (avoids enum construction per prediction)
_LABEL_CACHE = {member.value: member for member in DocItemLabel}


class PostprocessService:
    def __init__(self, input_queue: queue.Queue, table_queue: queue.Queue, final_queue: queue.Queue,
                 shutdown_event: threading.Event):
        self.input_queue = input_queue
        self.table_queue = table_queue
        self.final_queue = final_queue
        self.shutdown_event = shutdown_event

        self.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_V2)

        self.table_structure_model = TableStructureModel()

        start_queue_monitor("postprocess", {
            "input": self.input_queue,
            "table": self.table_queue,
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
        """Runs the service"""
        import time
        while not self.shutdown_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Error containment - don't let bad pages kill the whole service
            try:
                t0 = time.perf_counter()
                page = self._postprocess_page(page=item['page'], page_predictions=item['predictions'])
                t_layout_pp = time.perf_counter()

                # Mark layout postprocess completion (before table prep)
                mark(item, 'pp_lay')

                table_args = self.table_structure_model.preprocess(pages=[page])
                t_table_prep = time.perf_counter()

                logger.info(f"pp page={item.get('page_no')} layout_pp={1000*(t_layout_pp-t0):.0f}ms table_prep={1000*(t_table_prep-t_layout_pp):.0f}ms")

                # Route ALL pages to table queue for document-level batching
                # GPU service will handle pages with/without tables appropriately
                out = {
                    'job_id': item['job_id'],
                    'user_id': item['user_id'],
                    'page_no': item['page_no'],
                    'page': page,
                    'table_args': table_args,
                    'total_pages': item['total_pages'],
                    '_trace': item.get('_trace', {}).copy(),
                }
                mark(out, 'pp')
                self._put_with_shutdown(self.table_queue, out, self.shutdown_event)
                    
            except Exception as e:
                logger.exception(f"postprocess error page={item.get('page_no')}: {e}")
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

    def _postprocess_page(self, page: Page, page_predictions: List[Dict]) -> Page:
        """
        Postprocess layout predictions for a single page

        TODO: Copy the postprocessing logic from the commented section in layout_model.py
        """

        clusters = []
        for ix, pred_item in enumerate(page_predictions):
            label = _LABEL_CACHE[pred_item["label"].lower().replace(" ", "_").replace("-", "_")]
            cluster = Cluster.model_construct(
                id=ix,
                label=label,
                confidence=pred_item["confidence"],
                bbox=BoundingBox.model_construct(
                    l=pred_item["l"], t=pred_item["t"],
                    r=pred_item["r"], b=pred_item["b"],
                ),
                cells=[],
                children=[],
            )
            clusters.append(cluster)

        processed_clusters, _processed_cells = LayoutPostprocessor(
            page, clusters, self.layout_options
        ).postprocess()

        page.predictions.layout = LayoutPrediction(clusters=processed_clusters)

        return page
