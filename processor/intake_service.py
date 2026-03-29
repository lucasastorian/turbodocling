import json
import gzip
import time
import asyncio
import queue
import threading
import msgpack
from typing import List, Dict

from botocore.exceptions import ClientError

from processor.shared.logging_config import get_logger
from processor.shared.config import (
    MAX_SQS_BATCH, LAYOUT_ENQ_BATCH, MAX_LOCAL_PAGES,
    MAX_INFLIGHT_DOCS, MAX_MEMORY_UTILIZATION,
)
from processor.shared.memory import memory_state
from processor.shared.telemetry import mark
from processor.page_deserializer import reconstruct_page
from processor.job_registry import JobRegistry

logger = get_logger(__name__)


class IntakeService:
    def __init__(
        self,
        s3_client,
        sqs_client,
        sfn_client,
        http_session,
        bucket: str,
        queue_url: str,
        layout_queue: queue.Queue,
        shutdown_event: threading.Event,
        registry: JobRegistry,
    ):
        self.s3_client = s3_client
        self.sqs_client = sqs_client
        self.sfn_client = sfn_client
        self.http_session = http_session
        self.bucket = bucket
        self.queue_url = queue_url
        self.layout_queue = layout_queue
        self.shutdown_event = shutdown_event
        self.registry = registry
        self._last_mem_log = 0.0

    async def run(self):
        while not self.shutdown_event.is_set():
            try:
                layout_qsize = self.layout_queue.qsize()
                active_jobs = self.registry.active_job_count()
                usage_bytes, limit_bytes, usage_ratio = memory_state()
                memory_high = usage_ratio is not None and usage_ratio >= MAX_MEMORY_UTILIZATION

                if layout_qsize >= MAX_LOCAL_PAGES or active_jobs >= MAX_INFLIGHT_DOCS or memory_high:
                    if memory_high and time.monotonic() - self._last_mem_log >= 5.0:
                        logger.info(
                            "intake paused: memory %.1f%% (%0.1f/%0.1f GiB) active_jobs=%d ql=%d",
                            usage_ratio * 100.0,
                            usage_bytes / (1024 ** 3),
                            limit_bytes / (1024 ** 3),
                            active_jobs,
                            layout_qsize,
                        )
                        self._last_mem_log = time.monotonic()
                    await asyncio.sleep(0.1)
                    continue
            except Exception:
                pass

            fetch_limit = min(MAX_SQS_BATCH, max(1, MAX_INFLIGHT_DOCS - self.registry.active_job_count()))
            msgs = await asyncio.to_thread(self._poll_sqs, fetch_limit)
            if not msgs:
                await asyncio.sleep(0.2)
                continue

            logger.info(f"sqs: {len(msgs)}")
            for m in msgs:
                try:
                    await self._handle_message(m)
                except Exception as e:
                    logger.exception(f"handle_message failed: {e}")

    def _poll_sqs(self, n: int) -> List[Dict]:
        out: List[Dict] = []
        first = True
        while len(out) < n and not self.shutdown_event.is_set():
            try:
                resp = self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=min(10, n - len(out)),
                    WaitTimeSeconds=2 if first else 0,
                    VisibilityTimeout=300,
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

    async def _handle_message(self, message: Dict):
        job_id = message['job_id']
        user_id = message['user_id']
        total_pages = message['total_pages']
        parts = message['parts']

        task_token = message.get('task_token')
        if task_token:
            self.registry.register_job(job_id, task_token)

        download_start = time.time()
        pdf_key = f"uploads/{user_id}/{job_id}/source.pdf"
        pdf_task = asyncio.create_task(self._download(pdf_key))
        part_tasks = [self._download(part["batch_key"]) for part in parts]
        part_data_list = await asyncio.gather(*part_tasks)
        pdf_bytes = await pdf_task
        self.registry.pdf_cache[job_id] = pdf_bytes
        download_time = time.time() - download_start

        total_bytes = sum(len(data) for data in part_data_list)
        speed_mbps = (total_bytes / (1024 * 1024)) / download_time if download_time > 0 else 0
        logger.info(f"downloaded {len(parts)} parts: {total_bytes/1024/1024:.1f}MB in {download_time:.2f}s ({speed_mbps:.1f} MB/s)")

        validation_start = time.time()
        all_pages = [None] * total_pages
        total_decomp_time = 0.0
        total_parse_time = 0.0

        for i, blob in enumerate(part_data_list):
            part = parts[i]
            start_page = part['start_page']

            t0 = time.time()
            decompressed = gzip.decompress(blob)
            t1 = time.time()

            try:
                page_dicts = msgpack.unpackb(decompressed, raw=False, strict_map_key=False)
            except Exception:
                page_dicts = json.loads(decompressed.decode('utf-8'))

            pages = [reconstruct_page(page_dict) for page_dict in page_dicts]
            t2 = time.time()

            for j, page in enumerate(pages):
                all_pages[start_page + j] = page

            total_decomp_time += (t1 - t0)
            total_parse_time += (t2 - t1)

        validation_time = time.time() - validation_start
        logger.info(f"decompress={total_decomp_time:.3f}s parse={total_parse_time:.3f}s total={validation_time:.3f}s for {len(all_pages)} pages")

        if any(p is None for p in all_pages):
            missing = sum(1 for p in all_pages if p is None)
            logger.error(f"{missing}/{total_pages} pages missing after decompression")
            task_token = self.registry.pop_token(job_id)
            if task_token:
                try:
                    self.sfn_client.send_task_failure(
                        taskToken=task_token,
                        error="MissingPages",
                        cause=f"{missing}/{total_pages} pages missing after decompression",
                    )
                except Exception:
                    pass
            self.registry.pdf_cache.pop(job_id, None)
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
