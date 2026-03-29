import os
import asyncio
import aiohttp
import threading
import queue
import boto3
import time
import signal

from botocore.config import Config

from processor.shared.logging_config import setup_main_logging, get_logger
from processor.shared.memory import memory_state
from processor.job_registry import JobRegistry
from processor.intake_service import IntakeService
from processor.assembly_service import AssemblyService

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logger = get_logger(__name__)


class MainService:
    def __init__(self, stage: str = "dev"):
        self.stage = stage
        self.bucket = os.environ["DOCUMENTS_BUCKET"]
        self.queue_url = os.environ["SQS_QUEUE_URL"]

        self.layout_queue = queue.Queue(maxsize=4096)
        self.postprocess_queue = queue.Queue(maxsize=4096)
        self.table_queue = queue.Queue(maxsize=4096)
        self.table_postprocess_queue = queue.Queue(maxsize=4096)
        self.final_queue = queue.Queue(maxsize=4096)
        self.assembly_queue = queue.Queue()
        self.shutdown_event = threading.Event()

        setup_main_logging()

        region = os.environ.get("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client('s3', region_name=region)
        self.sqs_client = boto3.client(
            'sqs', region_name=region,
            config=Config(read_timeout=5, connect_timeout=3, retries={'max_attempts': 2, 'mode': 'standard'})
        )
        self.sfn_client = boto3.client('stepfunctions', region_name=region)

        connector = aiohttp.TCPConnector(limit=128, limit_per_host=128)
        self.http_session = aiohttp.ClientSession(connector=connector)

        self.registry = JobRegistry()
        self.intake = IntakeService(
            s3_client=self.s3_client,
            sqs_client=self.sqs_client,
            sfn_client=self.sfn_client,
            http_session=self.http_session,
            bucket=self.bucket,
            queue_url=self.queue_url,
            layout_queue=self.layout_queue,
            shutdown_event=self.shutdown_event,
            registry=self.registry,
        )

        self.child_threads: list[threading.Thread] = []

    async def run_async(self):
        logger.info("main: started")
        self._final_listener_task = asyncio.create_task(self._final_listener())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_sender())
        self._stats_task = asyncio.create_task(self._stats_reporter())

        try:
            await self.intake.run()
        finally:
            logger.info("main: stopping")
            for task in (self._final_listener_task, self._heartbeat_task, self._stats_task):
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def _final_listener(self):
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

                doc = self.registry.final_docs.get(job_id)
                if doc is None:
                    doc = {'pages': {}, 'traces': {}, 'total_pages': total, 'user_id': user_id, 'count': 0}
                    self.registry.final_docs[job_id] = doc

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
                    pdf_bytes = self.registry.pdf_cache.pop(job_id, None)
                    if pdf_bytes is None:
                        self.registry.final_docs.pop(job_id, None)
                        continue
                    traces = doc['traces']
                    self.registry.final_docs.pop(job_id, None)

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
                    task_token = self.registry.pop_token(failed_job_id)
                    if task_token:
                        try:
                            self.sfn_client.send_task_failure(
                                taskToken=task_token, error="ProcessingError", cause=str(e)[:256]
                            )
                        except Exception:
                            pass
                    self.registry.final_docs.pop(failed_job_id, None)
                    self.registry.pdf_cache.pop(failed_job_id, None)

    async def _heartbeat_sender(self):
        while not self.shutdown_event.is_set():
            await asyncio.sleep(20)
            for job_id, token in list(self.registry.task_tokens.items()):
                try:
                    await asyncio.to_thread(self.sfn_client.send_task_heartbeat, taskToken=token)
                except Exception as e:
                    logger.warning(f"heartbeat failed: job={job_id[:4]} {e}")

    async def _stats_reporter(self):
        while not self.shutdown_event.is_set():
            await asyncio.sleep(60)
            elapsed, pages, docs = self.registry.stats()
            usage_bytes, limit_bytes, usage_ratio = memory_state()
            mem_str = f"mem={usage_ratio*100:.0f}% ({usage_bytes/(1024**3):.1f}/{limit_bytes/(1024**3):.1f}GiB)" if usage_ratio else "mem=n/a"
            pps = pages / elapsed if elapsed > 0 else 0
            logger.info(
                f"STATS elapsed={elapsed:.0f}s docs={docs} pages={pages} "
                f"throughput={pps:.1f}p/s active={self.registry.active_job_count()} "
                f"ql={self.layout_queue.qsize()} {mem_str}"
            )

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
            ("AssemblyService", AssemblyService, (
                self.assembly_queue, self.shutdown_event, self.s3_client, self.sfn_client,
                self.bucket, self.registry)),
        ]
        for name, cls, args in services:
            t = threading.Thread(target=self._run_service, args=(name, cls, args), name=name, daemon=True)
            t.start()
            self.child_threads.append(t)
            logger.info(f"thread started: {name}")

    def _run_service(self, name: str, cls, args: tuple):
        try:
            svc = cls(*args)
            svc.run()
        except Exception as e:
            get_logger(name).exception(f"{name} crashed: {e}")
            self.registry.fail_all_jobs(self.sfn_client, f"{name} crashed: {str(e)[:200]}")
            self.shutdown_event.set()

    def shutdown(self):
        logger.info("shutdown: signaling")
        self.shutdown_event.set()
        for t in self.child_threads:
            t.join(timeout=5)
        logger.info("shutdown: done")

    async def close(self):
        if self.http_session:
            await self.http_session.close()


async def main():
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
