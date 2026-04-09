"""
Local runner for the Turbodocling pipeline.

Reuses the exact same compute path as the AWS pipeline (parse → layout → table
→ postprocess → assembly) but replaces S3/SQS/Step Functions with in-memory
queues. Produces identical output.md and elements.json.

Usage:
    from turbodocling.local_runner import run_local
    result = run_local("my_document.pdf", "output/")
"""

import io
import json
import os
import time
import queue
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Union

# Let PyTorch use all available cores for CPU inference.
# The GPU worker sets these to 1 because CPU threads fight the GPU there.
# Locally on CPU, we want full parallelism.
_N_THREADS = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)

import pypdfium2 as pdfium

from processor.page_deserializer import reconstruct_page
from processor.shared.telemetry import mark


@dataclass
class LocalRunResult:
    """Result of a local pipeline run."""
    md_path: Path
    elements_path: Path
    total_pages: int
    wall_time_s: float
    preprocess_time_s: float
    inference_time_s: float
    assembly_time_s: float
    device: str


def run_local(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "auto",
    workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> LocalRunResult:
    """
    Run the full Turbodocling pipeline locally on a single PDF.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to write output.md and elements.json.
        device: Inference device — "auto", "cuda", "mps", or "cpu".
        workers: Number of preprocessing workers (default: cpu_count).
        batch_size: Pages per preprocessing batch (default: auto).

    Returns:
        LocalRunResult with output paths and timing info.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device != "auto":
        os.environ["DEVICE"] = device

    pdf_bytes = pdf_path.read_bytes()
    total_pages = len(pdfium.PdfDocument(io.BytesIO(pdf_bytes)))

    print(f"Processing {pdf_path.name} ({total_pages} pages)")
    t_wall_start = time.perf_counter()

    # ── Start model loading in background while preprocessing runs ────────
    # Model init (loading weights + MPS warmup) takes ~8s and is independent
    # of preprocessing. Overlapping them saves most of the init cost.
    import threading as _threading

    pipeline_state = {}
    def _init_pipeline():
        pipeline_state['result'] = _create_inference_pipeline()

    init_thread = _threading.Thread(target=_init_pipeline, daemon=True)
    init_thread.start()

    # ── Phase 1: Preprocessing (parallel, CPU) ────────────────────────────
    t_pre_start = time.perf_counter()
    page_dicts = _preprocess_parallel(pdf_bytes, total_pages, workers, batch_size)
    t_pre_end = time.perf_counter()
    print(f"  Preprocessing: {t_pre_end - t_pre_start:.2f}s ({total_pages} pages)")

    # Reconstruct Page objects from packed dicts
    all_pages = [reconstruct_page(pd) for pd in page_dicts]

    # Wait for model loading to finish
    init_thread.join()

    # ── Phase 2: GPU Pipeline (threaded, single process) ──────────────────
    t_inf_start = time.perf_counter()
    processed_pages = _run_inference_pipeline(
        all_pages, total_pages, pipeline=pipeline_state['result']
    )
    t_inf_end = time.perf_counter()
    print(f"  Inference: {t_inf_end - t_inf_start:.2f}s")

    # ── Phase 3: Assembly ─────────────────────────────────────────────────
    t_asm_start = time.perf_counter()
    md_path, elements_path = _assemble_and_write(
        pdf_bytes, processed_pages, output_dir
    )
    t_asm_end = time.perf_counter()
    print(f"  Assembly: {t_asm_end - t_asm_start:.2f}s")

    t_wall_end = time.perf_counter()
    wall = t_wall_end - t_wall_start
    print(f"  Total: {wall:.2f}s ({total_pages / wall:.1f} pages/sec)")

    actual_device = os.environ.get("DEVICE", "auto")
    if actual_device == "auto":
        from processor.gpu_service.service import get_device
        actual_device = get_device()

    return LocalRunResult(
        md_path=md_path,
        elements_path=elements_path,
        total_pages=total_pages,
        wall_time_s=wall,
        preprocess_time_s=t_pre_end - t_pre_start,
        inference_time_s=t_inf_end - t_inf_start,
        assembly_time_s=t_asm_end - t_asm_start,
        device=actual_device,
    )


def _preprocess_parallel(
    pdf_bytes: bytes, total_pages: int, workers: Optional[int], batch_size: Optional[int]
) -> List[Dict]:
    """Run preprocessing across multiple worker processes."""
    from turbodocling.preprocessing import preprocess_pages, worker_init

    if workers is None:
        workers = min(os.cpu_count() or 4, total_pages)
    workers = min(workers, total_pages)

    if batch_size is None:
        # Same sizing rule as the Step Function: distribute evenly across workers
        batch_size = max(1, -(-total_pages // workers))  # ceil division

    # Build page ranges
    ranges = []
    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size - 1, total_pages - 1)
        ranges.append((start, end))

    if workers <= 1 or len(ranges) == 1:
        # Single-process path (simpler, avoids spawn overhead for small docs)
        worker_init()
        all_dicts = []
        for start, end in ranges:
            all_dicts.extend(preprocess_pages(pdf_bytes, start, end))
        return all_dicts

    # Multi-process path
    ctx = get_context("spawn")
    all_dicts = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=worker_init
    ) as executor:
        futures = [
            executor.submit(preprocess_pages, pdf_bytes, start, end)
            for start, end in ranges
        ]
        for future in futures:
            all_dicts.extend(future.result())

    # Sort by page_index to ensure consistent ordering
    all_dicts.sort(key=lambda d: d["page_index"])
    return all_dicts


def _create_inference_pipeline():
    """
    Create and initialize the inference pipeline (loads models, warmup).
    Can be called in a background thread to overlap with preprocessing.
    """
    import torch
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)

    from processor.gpu_service.service import GPUService
    from processor.postprocess_service.service import PostprocessService
    from processor.table_postprocessing_service.service import TablePostprocessingService

    layout_queue = queue.Queue(maxsize=4096)
    postprocess_queue = queue.Queue(maxsize=4096)
    table_queue = queue.Queue(maxsize=4096)
    table_postprocess_queue = queue.Queue(maxsize=4096)
    final_queue = queue.Queue(maxsize=4096)
    shutdown_event = threading.Event()

    services = [
        ("GPUService", GPUService, (
            layout_queue, postprocess_queue, table_queue,
            table_postprocess_queue, shutdown_event)),
        ("PostprocessService", PostprocessService, (
            postprocess_queue, table_queue, final_queue, shutdown_event)),
        ("TablePostprocessingService", TablePostprocessingService, (
            table_postprocess_queue, final_queue, shutdown_event)),
    ]

    threads = []
    for name, cls, args in services:
        t = threading.Thread(target=_run_service, args=(cls, args), name=name, daemon=True)
        t.start()
        threads.append(t)

    return {
        'layout_queue': layout_queue,
        'final_queue': final_queue,
        'shutdown_event': shutdown_event,
        'threads': threads,
    }


def _run_inference_pipeline(all_pages, total_pages: int, pipeline=None) -> List:
    """
    Run layout + table inference + postprocessing using the same threaded
    pipeline as the production GPU worker.
    """
    if pipeline is None:
        pipeline = _create_inference_pipeline()

    layout_queue = pipeline['layout_queue']
    final_queue = pipeline['final_queue']
    shutdown_event = pipeline['shutdown_event']
    threads = pipeline['threads']

    # Feed pages into the layout queue (same format as intake_service)
    job_id = str(uuid.uuid4())
    items = []
    for page in all_pages:
        item = {
            "job_id": job_id,
            "user_id": "local",
            "page_no": page.page_no,
            "page": page,
            "total_pages": total_pages,
            "_trace": {},
        }
        mark(item, "enq")
        items.append(item)

    # Enqueue in micro-batches of 32 (same as intake_service.LAYOUT_ENQ_BATCH)
    for i in range(0, len(items), 32):
        layout_queue.put(items[i : i + 32])

    # Collect results from final_queue
    collected = {}
    while len(collected) < total_pages:
        try:
            item = final_queue.get(timeout=300)
        except queue.Empty:
            raise TimeoutError(
                f"Inference timed out after 300s ({len(collected)}/{total_pages} pages collected)"
            )
        page_no = item["page_no"]
        collected[page_no] = item["page"]

    # Shutdown
    shutdown_event.set()
    for t in threads:
        t.join(timeout=5)

    return [collected[k] for k in sorted(collected.keys())]


def _run_service(cls, args):
    """Thread target for a pipeline service."""
    try:
        svc = cls(*args)
        svc.run()
    except Exception as e:
        import traceback
        traceback.print_exc()


def _assemble_and_write(
    pdf_bytes: bytes, pages: List, output_dir: Path
) -> tuple:
    """Assemble the document and write output files."""
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import DocumentLimits
    from docling.datamodel.pipeline_options import PipelineOptions
    from docling.backend.noop_backend import NoOpBackend
    from processor.final_service.document_assembler import DocumentAssembler

    input_doc = InputDocument(
        path_or_stream=io.BytesIO(pdf_bytes),
        format=InputFormat.PDF,
        filename="document.pdf",
        limits=DocumentLimits(),
        backend=NoOpBackend,
    )

    assembler = DocumentAssembler(PipelineOptions())
    conv_res = assembler.assemble_document(input_doc, pages)

    md = conv_res.document.export_to_markdown()
    structured = assembler.extract_structured_elements(conv_res)

    md_path = output_dir / "output.md"
    elements_path = output_dir / "elements.json"

    md_path.write_text(md, encoding="utf-8")
    elements_path.write_text(
        json.dumps(structured, ensure_ascii=False), encoding="utf-8"
    )

    n_elements = sum(len(p["elements"]) for p in structured["pages"])
    print(f"  Output: {len(md)} chars markdown, {n_elements} elements")

    return md_path, elements_path
