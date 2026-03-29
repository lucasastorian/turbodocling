import io
import json
import time
import queue
import threading

from processor.shared.logging_config import get_logger
from processor.shared.telemetry import now, ms
from processor.final_service.document_assembler import DocumentAssembler
from processor.job_registry import JobRegistry

from docling.datamodel.document import InputDocument
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import DocumentLimits
from docling.datamodel.pipeline_options import PipelineOptions
from docling.backend.noop_backend import NoOpBackend

logger = get_logger(__name__)


class AssemblyService:
    def __init__(
        self,
        assembly_queue: queue.Queue,
        shutdown_event: threading.Event,
        s3_client,
        sfn_client,
        bucket: str,
        registry: JobRegistry,
    ):
        self.assembly_queue = assembly_queue
        self.shutdown_event = shutdown_event
        self.s3_client = s3_client
        self.sfn_client = sfn_client
        self.bucket = bucket
        self.registry = registry
        self.assembler = DocumentAssembler(PipelineOptions())

    def run(self):
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

                logger.info(
                    f"assembled: job={job_id[:4]} pages={len(pages)} md_len={len(md)} "
                    f"elements={sum(len(p['elements']) for p in structured['pages'])} | "
                    f"assemble={ms(t_asm_doc - t_asm_start):.0f}ms md={ms(t_asm_md - t_asm_doc):.0f}ms "
                    f"struct={ms(t_asm_struct - t_asm_md):.0f}ms json={ms(t_asm_end - t_asm_struct):.0f}ms"
                )

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

                task_token = self.registry.pop_token(job_id)
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
                        logger.warning(f"send_task_success failed: {e}")

                elapsed = time.time() - self.registry.job_start_times.pop(job_id, time.time())
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
                    logger.info(f"DONE job={job_id[:4]} pages={len(pages)} wall={elapsed:.2f}s ({pps:.1f} p/s) | {' '.join(parts)}")
                else:
                    logger.info(f"DONE job={job_id[:4]} pages={len(pages)} wall={elapsed:.2f}s ({pps:.1f} p/s)")

                self.registry.record_completed(len(pages))

            except Exception as e:
                logger.exception(f"assembly error: job={job_id[:4]} {e}")
                task_token = self.registry.pop_token(job_id)
                if task_token:
                    try:
                        self.sfn_client.send_task_failure(
                            taskToken=task_token,
                            error="AssemblyError",
                            cause=str(e)[:256]
                        )
                    except Exception:
                        pass
