# Set BLAS thread limits before any imports that might use them
import os
import json
import asyncio
import aiohttp
import threading
import queue
import boto3
import time
import msgpack
import numpy as np
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List, Dict, Optional, Any

from processor.shared.logging_config import setup_main_logging, get_logger
from processor.shared.telemetry import mark, now, ms
from processor.final_service.document_assembler import DocumentAssembler
from docling.datamodel.pipeline_options import PipelineOptions

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logger = get_logger(__name__)

# No longer need module aliases since we're using JSON instead of pickle




MAX_SQS_BATCH = 128
LAYOUT_ENQ_BATCH = int(os.getenv("LAYOUT_ENQ_BATCH", "32"))
MAX_LOCAL_PAGES = 512  # Stop pulling SQS when local queue exceeds this - enables clean shutdown


class MainService:
    def __init__(self, stage: str = "dev"):
        self.stage = stage
        self.bucket = os.environ["DOCUMENTS_BUCKET"]
        self.queue_url = os.environ["SQS_QUEUE_URL"]

        # Threads + queues
        self.layout_queue = queue.Queue(maxsize=4096)
        self.postprocess_queue = queue.Queue(maxsize=4096)
        self.table_queue = queue.Queue(maxsize=4096)
        self.table_postprocess_queue = queue.Queue(maxsize=4096)
        self.final_queue = queue.Queue(maxsize=4096)
        self.assembly_queue = queue.Queue()  # unbounded; one assembly worker drains it
        self.shutdown_event = threading.Event()

        setup_main_logging()

        region = os.environ.get("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client('s3', region_name=region)
        self.sqs_client = boto3.client(
            'sqs', region_name=region,
            config=Config(read_timeout=5, connect_timeout=3, retries={'max_attempts': 2, 'mode': 'standard'})
        )
        self.sfn_client = boto3.client('stepfunctions', region_name=region)
        self.task_tokens: Dict[str, str] = {}  # job_id -> Step Function task token
        self.job_start_times: Dict[str, float] = {}  # job_id -> wall-clock start

        connector = aiohttp.TCPConnector(limit=128, limit_per_host=128)
        self.http_session = aiohttp.ClientSession(connector=connector)

        # final assembly state (simple)
        self.final_docs: Dict[str, Dict[str, Any]] = {}
        self.pdf_cache: Dict[str, bytes] = {}  # job_id -> pdf_bytes
        self.assembler = DocumentAssembler(PipelineOptions())
        self._final_listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        self.child_threads: list[threading.Thread] = []

    async def run_async(self):
        logger.info("main: started")
        self._final_listener_task = asyncio.create_task(self._final_listener())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_sender())

        try:
            while not self.shutdown_event.is_set():
                # Only pull from SQS when local queue has capacity
                # Keeps local queue small for clean shutdown on SIGTERM
                try:
                    if self.layout_queue.qsize() >= MAX_LOCAL_PAGES:
                        await asyncio.sleep(0.1)
                        continue
                except Exception:
                    pass

                msgs = await asyncio.to_thread(self._poll_sqs_up_to, MAX_SQS_BATCH)
                if not msgs:
                    await asyncio.sleep(0.2)
                    continue

                logger.info(f"sqs: {len(msgs)}")

                # Simple sequential handler (I/O bound anyway)
                for m in msgs:
                    try:
                        await self._handle_message(m)
                    except Exception as e:
                        logger.exception(f"handle_message failed: {e}")
                        # keep going; no error queue
        finally:
            logger.info("main: stopping")
            await self._stop_background_tasks()

    async def _heartbeat_sender(self):
        """Send periodic heartbeats for all active Step Function tasks."""
        while not self.shutdown_event.is_set():
            await asyncio.sleep(20)
            for job_id, token in list(self.task_tokens.items()):
                try:
                    await asyncio.to_thread(
                        self.sfn_client.send_task_heartbeat,
                        taskToken=token,
                    )
                except Exception as e:
                    logger.warning(f"heartbeat failed: job={job_id[:4]} {e}")

    async def _stop_background_tasks(self):
        for task in (self._final_listener_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._final_listener_task = None
        self._heartbeat_task = None

    def _poll_sqs_up_to(self, n: int) -> List[Dict]:
        """Collect up to n messages. First call long-polls; subsequent are short."""
        out: List[Dict] = []
        first = True
        while len(out) < n and not self.shutdown_event.is_set():
            try:
                resp = self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=min(10, n - len(out)),
                    WaitTimeSeconds=2 if first else 0,
                    VisibilityTimeout=300
                )
            except ClientError as e:
                logger.error(f"SQS error: {e}")
                break

            msgs = resp.get('Messages', [])
            if not msgs:
                break
            first = False
            for msg in msgs:
                body = json.loads(msg['Body'])
                body['receipt_handle'] = msg['ReceiptHandle']
                out.append(body)
        return out

    def _unpack_cells_columnar(self, cell_dict: Dict) -> 'CellStore':
        """Unpack columnar cell data directly into CellStore - zero object creation."""
        from processor.shared.cell_store import CellStore, _CellColumns, _StringTable

        n = cell_dict.get('n', 0)
        if n == 0:
            # Empty store
            empty_cols = _CellColumns(
                r_x0=np.array([], np.float32), r_y0=np.array([], np.float32),
                r_x1=np.array([], np.float32), r_y1=np.array([], np.float32),
                r_x2=np.array([], np.float32), r_y2=np.array([], np.float32),
                r_x3=np.array([], np.float32), r_y3=np.array([], np.float32),
                index=np.array([], np.int32), rgba=np.array([], np.uint8).reshape(0, 4),
                conf=None, from_ocr=None, text_idx=np.array([], np.int32),
                orig_idx=None, rendering_mode=None, widget=None,
                font_key_idx=None, font_name_idx=None, text_dir=None
            )
            return CellStore(empty_cols, _StringTable([]), 0.0)

        # Load arrays directly from bytes - zero copy with np.frombuffer
        r_x0 = np.frombuffer(cell_dict['r_x0'], dtype=np.float32)
        r_y0 = np.frombuffer(cell_dict['r_y0'], dtype=np.float32)
        r_x1 = np.frombuffer(cell_dict['r_x1'], dtype=np.float32)
        r_y1 = np.frombuffer(cell_dict['r_y1'], dtype=np.float32)
        r_x2 = np.frombuffer(cell_dict['r_x2'], dtype=np.float32)
        r_y2 = np.frombuffer(cell_dict['r_y2'], dtype=np.float32)
        r_x3 = np.frombuffer(cell_dict['r_x3'], dtype=np.float32)
        r_y3 = np.frombuffer(cell_dict['r_y3'], dtype=np.float32)

        index = np.frombuffer(cell_dict['index'], dtype=np.int32)
        rgba_flat = np.frombuffer(cell_dict['rgba'], dtype=np.uint8)
        rgba = rgba_flat.reshape(-1, 4)
        conf = np.frombuffer(cell_dict['conf'], dtype=np.float32)
        text_idx = np.frombuffer(cell_dict['text_idx'], dtype=np.int32)
        orig_idx = np.frombuffer(cell_dict['orig_idx'], dtype=np.int32)

        # PDF-specific columns (optional)
        rendering_mode = np.frombuffer(cell_dict['rendering_mode'], dtype=np.int8) if 'rendering_mode' in cell_dict else None
        widget = np.frombuffer(cell_dict['widget'], dtype=np.uint8).astype(np.bool_) if 'widget' in cell_dict else None
        font_key_idx = np.frombuffer(cell_dict['font_key_idx'], dtype=np.int32) if 'font_key_idx' in cell_dict else None
        font_name_idx = np.frombuffer(cell_dict['font_name_idx'], dtype=np.int32) if 'font_name_idx' in cell_dict else None
        text_dir = np.frombuffer(cell_dict['text_dir'], dtype=np.uint8) if 'text_dir' in cell_dict else None

        strings = cell_dict.get('strings', [])
        is_topleft = cell_dict.get('topleft', True)

        cols = _CellColumns(
            r_x0=r_x0, r_y0=r_y0, r_x1=r_x1, r_y1=r_y1,
            r_x2=r_x2, r_y2=r_y2, r_x3=r_x3, r_y3=r_y3,
            index=index, rgba=rgba, conf=conf, from_ocr=None,
            text_idx=text_idx, orig_idx=orig_idx,
            rendering_mode=rendering_mode, widget=widget,
            font_key_idx=font_key_idx, font_name_idx=font_name_idx,
            text_dir=text_dir
        )

        return CellStore(cols, _StringTable(list(strings)), 0.0, bottomleft_origin=not is_topleft)

    def _reconstruct_page_from_dict(self, page_dict: Dict) -> 'Page':
        """Reconstruct a Page model from dictionary data sent by Lambda."""
        from docling.datamodel.base_models import Page
        from docling_core.types.doc.page import SegmentedPdfPage, PdfPageGeometry
        from docling_core.types.doc import Size, BoundingBox, CoordOrigin
        from processor.shared.cell_store import LazyCellList
        from docling_core.types.doc.page import TextCell, PdfTextCell, BoundingRectangle, ColorRGBA
        from docling_core.types.doc import TextDirection
        from docling_core.types.doc.page import PdfCellRenderingMode

        # Convert size dict to Size object
        size_dict = page_dict['size']
        size = Size(width=size_dict['w'], height=size_dict['h'])

        segmented_dict = page_dict['segmented']

        # Check for version 2 columnar format
        if segmented_dict.get('version') == 2:
            # New columnar format - zero object creation!
            dim_dict = segmented_dict['dimension']
            page_height = segmented_dict['page_height']

            # Create dimension - need to construct crop_bbox first, then use it for rect
            crop_bbox = BoundingBox(l=dim_dict['crop_bbox'][0], t=dim_dict['crop_bbox'][1],
                                     r=dim_dict['crop_bbox'][2], b=dim_dict['crop_bbox'][3],
                                     coord_origin=CoordOrigin.TOPLEFT)
            # Create rect from crop_bbox (PageGeometry base class requirement)
            rect = BoundingRectangle(
                r_x0=crop_bbox.l, r_y0=crop_bbox.t,
                r_x1=crop_bbox.r, r_y1=crop_bbox.t,
                r_x2=crop_bbox.r, r_y2=crop_bbox.b,
                r_x3=crop_bbox.l, r_y3=crop_bbox.b,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            dimension = PdfPageGeometry.model_construct(
                angle=dim_dict['angle'],
                rect=rect,
                boundary_type=dim_dict['boundary_type'],
                crop_bbox=crop_bbox,
                media_bbox=BoundingBox(l=dim_dict['media_bbox'][0], t=dim_dict['media_bbox'][1],
                                        r=dim_dict['media_bbox'][2], b=dim_dict['media_bbox'][3],
                                        coord_origin=CoordOrigin.TOPLEFT),
                art_bbox=BoundingBox(l=dim_dict['art_bbox'][0], t=dim_dict['art_bbox'][1],
                                      r=dim_dict['art_bbox'][2], b=dim_dict['art_bbox'][3],
                                      coord_origin=CoordOrigin.TOPLEFT),
                bleed_bbox=BoundingBox(l=dim_dict['bleed_bbox'][0], t=dim_dict['bleed_bbox'][1],
                                        r=dim_dict['bleed_bbox'][2], b=dim_dict['bleed_bbox'][3],
                                        coord_origin=CoordOrigin.TOPLEFT),
                trim_bbox=BoundingBox(l=dim_dict['trim_bbox'][0], t=dim_dict['trim_bbox'][1],
                                       r=dim_dict['trim_bbox'][2], b=dim_dict['trim_bbox'][3],
                                       coord_origin=CoordOrigin.TOPLEFT),
            )

            # Create cell stores from columnar data
            classes = (TextCell, PdfTextCell, BoundingRectangle, CoordOrigin, TextDirection,
                       ColorRGBA, PdfCellRenderingMode)

            char_store = self._unpack_cells_columnar(segmented_dict['char_cells'])
            char_store.page_height = page_height
            word_store = self._unpack_cells_columnar(segmented_dict['word_cells'])
            word_store.page_height = page_height
            textline_store = self._unpack_cells_columnar(segmented_dict['textline_cells'])
            textline_store.page_height = page_height

            # Create lazy cell lists
            char_cells = LazyCellList(char_store, classes)
            word_cells = LazyCellList(word_store, classes)
            textline_cells = LazyCellList(textline_store, classes)

            # Create SegmentedPdfPage with lazy cell lists
            parsed_page = SegmentedPdfPage.model_construct(
                dimension=dimension,
                char_cells=char_cells,
                word_cells=word_cells,
                textline_cells=textline_cells,
                has_chars=segmented_dict.get('has_chars', len(char_cells) > 0),
                has_words=segmented_dict.get('has_words', len(word_cells) > 0),
                has_lines=segmented_dict.get('has_lines', len(textline_cells) > 0),
            )

            # Attach stores for later access if needed
            parsed_page._char_store = char_store
            parsed_page._word_store = word_store
            parsed_page._line_store = textline_store

            logger.info(f"page {page_dict['page_index']}: columnar v2 - chars={len(char_cells)} words={len(word_cells)} lines={len(textline_cells)}")
        else:
            # Legacy format (version 1) - use model_validate
            parsed_page = SegmentedPdfPage.model_validate(segmented_dict)

        # Convert images dict to PIL Images (decode WebP bytes)
        from PIL import Image
        import io
        image_cache = {}
        for scale_str, img_bytes in page_dict['images'].items():
            scale = float(scale_str)
            pil_image = Image.open(io.BytesIO(img_bytes))
            image_cache[scale] = pil_image

        # Create the Page object
        page = Page(
            page_no=page_dict['page_index'],
            size=size,
            parsed_page=parsed_page,
        )
        page._image_cache = image_cache  # Pre-decoded PIL images

        return page

    async def _handle_message(self, message: Dict):
        """Download all batch parts, parse Pages, enqueue to layout, delete SQS. Only deletes after successful enqueue."""
        job_id = message['job_id']
        user_id = message['user_id']
        total_pages = message['total_pages']
        parts = message['parts']  # [{"batch_key": "...", "start_page": 0, "end_page": 9}, ...]

        # Store Step Function task token for completion callback
        task_token = message.get('task_token')
        if task_token:
            self.task_tokens[job_id] = task_token
            self.job_start_times[job_id] = time.time()

        # Download PDF and all parts concurrently 
        download_start = time.time()
        pdf_key = f"uploads/{user_id}/{job_id}/source.pdf"
        pdf_task = asyncio.create_task(self._download(pdf_key))
        part_tasks = [self._download(part["batch_key"]) for part in parts]
        part_data_list = await asyncio.gather(*part_tasks)
        pdf_bytes = await pdf_task
        self.pdf_cache[job_id] = pdf_bytes
        download_time = time.time() - download_start
        
        total_bytes = sum(len(data) for data in part_data_list)
        speed_mbps = (total_bytes / (1024 * 1024)) / download_time if download_time > 0 else 0
        logger.info(f"downloaded {len(parts)} parts: {total_bytes/1024/1024:.1f}MB in {download_time:.2f}s ({speed_mbps:.1f} MB/s)")

        # Fast rehydrate pages from dictionary format (new Lambda output)
        validation_start = time.time()
        all_pages = [None] * total_pages
        import gzip
        
        total_decomp_time = 0.0
        total_parse_time = 0.0
        
        for i, blob in enumerate(part_data_list):
            # Get the part metadata for this blob
            part = parts[i]
            start_page = part['start_page']  # 0-based
            
            # Measure decompression time
            t0 = time.time()
            decompressed = gzip.decompress(blob)
            t1 = time.time()

            # Parse msgpack (new format) or JSON (legacy fallback)
            try:
                page_dicts = msgpack.unpackb(decompressed, raw=False, strict_map_key=False)
            except Exception:
                # Fallback to JSON for legacy format
                page_dicts = json.loads(decompressed.decode('utf-8'))
            pages = [self._reconstruct_page_from_dict(page_dict) for page_dict in page_dicts]
            t2 = time.time()
            
            # Place pages directly at correct indices (no append + sort)
            for j, page in enumerate(pages):
                all_pages[start_page + j] = page
                
            total_decomp_time += (t1 - t0)
            total_parse_time += (t2 - t1)
            
        validation_time = time.time() - validation_start
        logger.info(f"decompress={total_decomp_time:.3f}s parse={total_parse_time:.3f}s total={validation_time:.3f}s for {len(all_pages)} pages")

        # Validate no holes — fail the job rather than silently drop pages
        if any(p is None for p in all_pages):
            missing = sum(1 for p in all_pages if p is None)
            logger.error(f"{missing}/{total_pages} pages missing after decompression")
            task_token = self.task_tokens.pop(job_id, None)
            if task_token:
                try:
                    self.sfn_client.send_task_failure(
                        taskToken=task_token,
                        error="MissingPages",
                        cause=f"{missing}/{total_pages} pages missing after decompression",
                    )
                except Exception:
                    pass
            self.pdf_cache.pop(job_id, None)
            return
        
        intake_time = time.time() - download_start
        logger.info(f"INTAKE job={job_id[:4]} pages={len(all_pages)} download={download_time:.3f}s decompress={total_decomp_time:.3f}s parse={total_parse_time:.3f}s total={intake_time:.3f}s")

        items = [{
            'job_id': job_id,
            'user_id': user_id,
            'page_no': p.page_no,
            'page': p,
            'total_pages': total_pages,
        } for p in all_pages]
        for item in items:
            mark(item, 'enq')

        for i in range(0, len(items), LAYOUT_ENQ_BATCH):
            batch = items[i:i + LAYOUT_ENQ_BATCH]
            self.layout_queue.put_nowait(batch)

        self._delete_message(message['receipt_handle'])

    async def _final_listener(self):
        """Collect finished pages; hand complete docs to the assembly thread."""
        while not self.shutdown_event.is_set():
            try:
                item = await asyncio.to_thread(self.final_queue.get, True, 0.5)
            except Exception:
                continue
            if item is None:
                continue

            try:
                job_id = item['job_id']
                page_no = item['page_no']
                page = item['page']
                total = item['total_pages']
                user_id = item['user_id']

                doc = self.final_docs.get(job_id)
                if doc is None:
                    doc = {'pages': {}, 'traces': {}, 'total_pages': total, 'user_id': user_id, 'count': 0}
                    self.final_docs[job_id] = doc

                if page_no not in doc['pages']:
                    doc['count'] += 1
                doc['pages'][page_no] = page
                doc['traces'][page_no] = item.get('_trace', {})
                if total is not None:
                    doc['total_pages'] = total
                if user_id is not None:
                    doc['user_id'] = user_id

                if doc['total_pages'] is not None and doc['count'] == doc['total_pages']:
                    pages = [doc['pages'][k] for k in sorted(doc['pages'].keys())]
                    pdf_bytes = self.pdf_cache.pop(job_id, None)
                    if pdf_bytes is None:
                        # Duplicate completion — already assembled this job
                        self.final_docs.pop(job_id, None)
                        continue
                    traces = doc['traces']
                    self.final_docs.pop(job_id, None)

                    self.assembly_queue.put_nowait({
                        'job_id': job_id,
                        'user_id': user_id,
                        'pages': pages,
                        'pdf_bytes': pdf_bytes,
                        'traces': traces,
                    })

            except Exception as e:
                logger.exception(f"final_listener error: {e}")
                if item and 'job_id' in item:
                    failed_job_id = item['job_id']
                    task_token = self.task_tokens.pop(failed_job_id, None)
                    if task_token:
                        try:
                            self.sfn_client.send_task_failure(
                                taskToken=task_token,
                                error="ProcessingError",
                                cause=str(e)[:256]
                            )
                        except Exception:
                            pass
                    self.final_docs.pop(failed_job_id, None)
                    self.pdf_cache.pop(failed_job_id, None)
                continue

    def _assembly_worker(self):
        """Dedicated thread: assemble document, upload to S3, notify Step Function."""
        from docling.datamodel.document import InputDocument
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.settings import DocumentLimits
        from docling.backend.noop_backend import NoOpBackend
        import io

        log = get_logger("AssemblyWorker")

        while not self.shutdown_event.is_set():
            try:
                job = self.assembly_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            job_id = job['job_id']
            try:
                user_id = job['user_id']
                pages = job['pages']
                pdf_bytes = job['pdf_bytes']
                traces = job['traces']

                input_doc = InputDocument(
                    path_or_stream=io.BytesIO(pdf_bytes),
                    format=InputFormat.PDF,
                    filename="document.pdf",
                    limits=DocumentLimits(),
                    backend=NoOpBackend,
                )

                t_asm_start = now()
                conv_res = self.assembler.assemble_document(input_doc, pages)
                t_asm_doc = now()
                md = conv_res.document.export_to_markdown()
                t_asm_md = now()
                structured = self.assembler.extract_structured_elements(conv_res)
                t_asm_struct = now()
                structured_json = json.dumps(structured, ensure_ascii=False)
                t_asm_end = now()

                log.info(
                    f"assembled: job={job_id[:4]} pages={len(pages)} md_len={len(md)} "
                    f"elements={sum(len(p['elements']) for p in structured['pages'])} | "
                    f"assemble={ms(t_asm_doc - t_asm_start):.0f}ms md={ms(t_asm_md - t_asm_doc):.0f}ms "
                    f"struct={ms(t_asm_struct - t_asm_md):.0f}ms json={ms(t_asm_end - t_asm_struct):.0f}ms"
                )

                # Upload MD + JSON to S3 (sequential — mostly GIL-released I/O wait)
                t_upload_start = now()
                md_key = f"processed/{user_id}/{job_id}/output.md"
                json_key = f"processed/{user_id}/{job_id}/elements.json"
                self.s3_client.put_object(
                    Bucket=self.bucket, Key=md_key,
                    Body=md.encode('utf-8'), ContentType='text/markdown'
                )
                self.s3_client.put_object(
                    Bucket=self.bucket, Key=json_key,
                    Body=structured_json.encode('utf-8'), ContentType='application/json'
                )
                t_upload_end = now()

                # Notify Step Function
                task_token = self.task_tokens.pop(job_id, None)
                if task_token:
                    try:
                        self.sfn_client.send_task_success(
                            taskToken=task_token,
                            output=json.dumps({
                                "status": "completed",
                                "md_key": md_key,
                                "json_key": json_key,
                                "total_pages": len(pages),
                            })
                        )
                    except Exception as e:
                        log.warning(f"send_task_success failed: {e}")

                # Per-stage timing summary
                elapsed = time.time() - self.job_start_times.pop(job_id, time.time())
                pps = len(pages) / elapsed if elapsed > 0 else 0
                trace_list = list(traces.values())
                if trace_list:
                    def _stat(key):
                        vals = [t.get(key) for t in trace_list if key in t]
                        return (min(vals), max(vals)) if vals else (None, None)
                    enq_min, _ = _stat('enq')
                    _, lay_max = _stat('lay')
                    _, pp_lay_max = _stat('pp_lay')
                    _, pp_max = _stat('pp')
                    _, tbl_max = _stat('tbl')
                    _, tpp_max = _stat('tpp')
                    parts = []
                    if enq_min and lay_max:
                        parts.append(f"layout={ms(lay_max - enq_min):.0f}ms")
                    if lay_max and pp_lay_max:
                        parts.append(f"pp_lay={ms(pp_lay_max - lay_max):.0f}ms")
                    if pp_lay_max and pp_max:
                        parts.append(f"pp_tbl={ms(pp_max - pp_lay_max):.0f}ms")
                    elif lay_max and pp_max:
                        parts.append(f"pp={ms(pp_max - lay_max):.0f}ms")
                    if pp_max and tbl_max:
                        parts.append(f"table={ms(tbl_max - pp_max):.0f}ms")
                    if tbl_max and tpp_max:
                        parts.append(f"tbl_pp={ms(tpp_max - tbl_max):.0f}ms")
                    parts.append(f"asm={ms(t_asm_end - t_asm_start):.0f}ms")
                    parts.append(f"upload={ms(t_upload_end - t_upload_start):.0f}ms")
                    log.info(f"DONE job={job_id[:4]} pages={len(pages)} wall={elapsed:.2f}s ({pps:.1f} p/s) | {' '.join(parts)}")
                else:
                    log.info(f"DONE job={job_id[:4]} pages={len(pages)} wall={elapsed:.2f}s ({pps:.1f} p/s)")

            except Exception as e:
                log.exception(f"assembly error: job={job_id[:4]} {e}")
                task_token = self.task_tokens.pop(job_id, None)
                if task_token:
                    try:
                        self.sfn_client.send_task_failure(
                            taskToken=task_token,
                            error="AssemblyError",
                            cause=str(e)[:256]
                        )
                    except Exception:
                        pass

    async def _download(self, key: str) -> bytes:
        url = self.s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': self.bucket, 'Key': key}, ExpiresIn=300
        )
        async with self.http_session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"S3 GET {key} failed: {resp.status}")
            return await resp.read()

    def _delete_message(self, receipt_handle: str):
        try:
            self.sqs_client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
        except ClientError as e:
            logger.warning(f"SQS delete failed: {e}")

    def spawn_child_threads(self):
        from processor.gpu_service.service import GPUService
        from processor.postprocess_service.service import PostprocessService
        from processor.table_postprocessing_service.service import TablePostprocessingService

        services = [
            ("GPUService", GPUService, (
            self.layout_queue, self.postprocess_queue, self.table_queue, self.table_postprocess_queue,
            self.shutdown_event)),
            ("PostprocessService", PostprocessService,
             (self.postprocess_queue, self.table_queue, self.final_queue, self.shutdown_event)),
            ("TablePostprocessingService", TablePostprocessingService,
             (self.table_postprocess_queue, self.final_queue, self.shutdown_event)),
        ]
        for name, cls, args in services:
            t = threading.Thread(target=self._run_service, args=(name, cls, args), name=name, daemon=True)
            t.start()
            self.child_threads.append(t)
            logger.info(f"thread started: {name}")

        # Assembly + upload thread (off the async loop)
        t = threading.Thread(target=self._run_service_fn, args=("AssemblyWorker", self._assembly_worker),
                             name="AssemblyWorker", daemon=True)
        t.start()
        self.child_threads.append(t)
        logger.info("thread started: AssemblyWorker")

    def _run_service(self, name: str, cls, args: tuple):
        logger = get_logger(name)
        try:
            svc = cls(*args)
            svc.run()
        except Exception as e:
            logger.exception(f"{name} crashed: {e}")
            self._fail_all_jobs(f"{name} crashed: {str(e)[:200]}")
            self.shutdown_event.set()

    def _run_service_fn(self, name: str, fn):
        """Run a plain function with the same crash-handling as _run_service."""
        try:
            fn()
        except Exception as e:
            get_logger(name).exception(f"{name} crashed: {e}")
            self._fail_all_jobs(f"{name} crashed: {str(e)[:200]}")
            self.shutdown_event.set()

    def _fail_all_jobs(self, cause: str):
        """Send task failure for all active step function tokens."""
        for job_id, token in list(self.task_tokens.items()):
            try:
                self.sfn_client.send_task_failure(
                    taskToken=token,
                    error="GPUServiceCrash",
                    cause=cause,
                )
                logger.info(f"failed job={job_id[:4]}: {cause}")
            except Exception:
                pass
        self.task_tokens.clear()
        self.job_start_times.clear()

    def shutdown(self):
        logger.info("shutdown: signaling")
        self.shutdown_event.set()
        # No sentinel needed - timeout-based gets will exit within 0.5s
        for t in self.child_threads:
            t.join(timeout=5)
        logger.info("shutdown: done")

    async def close(self):
        if self.http_session:
            await self.http_session.close()


async def main():
    import signal
    svc = MainService(stage=os.getenv("STAGE", "dev"))

    def _sig(*_):
        logger.info("signal: stopping")
        svc.shutdown()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        svc.spawn_child_threads()
        await svc.run_async()
    finally:
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
