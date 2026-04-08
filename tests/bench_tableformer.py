#!/usr/bin/env python3
"""
TableFormer benchmark + golden regression harness.

Runs golden PDFs through preprocessing → layout → table prep → TableFormer inference
→ postprocessing, captures per-table OTSL sequences, bboxes, cell matches, and timing.

Usage:
    # Capture golden references from current path (CPU, threads=1 for determinism)
    python tests/bench_tableformer.py --update-golden

    # Benchmark and compare against golden
    python tests/bench_tableformer.py

    # Benchmark on MPS
    python tests/bench_tableformer.py --device mps

    # Benchmark specific PDF
    python tests/bench_tableformer.py --pdf nvidia_10k.pdf
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Deterministic baseline: single thread for golden captures
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
PDFS_DIR = GOLDEN_DIR / "pdfs"
TF_GOLDEN_DIR = GOLDEN_DIR / "tableformer"


@dataclass
class TableResult:
    """Per-table result for golden comparison."""
    page_no: int
    table_idx: int  # index within the page
    otsl_seq: List[str]
    num_rows: int
    num_cols: int
    n_cells: int
    cell_texts: List[str]  # ordered text content per cell
    # Timing (not compared in golden, just recorded)
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0


@dataclass
class BenchResult:
    """Full benchmark result for one PDF."""
    pdf_name: str
    total_pages: int
    total_tables: int
    tables: List[TableResult]
    # Timing
    preprocess_ms: float = 0.0
    layout_ms: float = 0.0
    table_prep_ms: float = 0.0
    table_inference_ms: float = 0.0
    table_postprocess_ms: float = 0.0
    total_ms: float = 0.0


def run_pdf(pdf_path: Path, device: str = "cpu", threads: int = 1) -> BenchResult:
    """Run the full pipeline on one PDF and capture per-table results."""
    import torch
    if device == "cpu":
        torch.set_num_threads(threads)

    if device != "cpu":
        os.environ["DEVICE"] = device

    from turbodocling.preprocessing import preprocess_pages, worker_init
    from processor.page_deserializer import reconstruct_page
    from processor.gpu_service.layout_model import LayoutModel
    from processor.postprocess_service.service import PostprocessService
    from processor.postprocess_service.table_structure_model import TableStructureModel
    from processor.gpu_service.table_inference_model import TableInferenceModel
    from processor.table_postprocessing_service.table_postprocessing_model import TablePostprocessingModel
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import (
        LayoutOptions, TableStructureOptions, TableFormerMode,
    )
    from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2
    from docling.datamodel.base_models import LayoutPrediction, Cluster
    from docling_core.types.doc import DocItemLabel
    from docling_core.types.doc.base import BoundingBox
    from processor.postprocess_service.layout_postprocessor import LayoutPostprocessor

    pdf_bytes = pdf_path.read_bytes()

    import pypdfium2 as pdfium
    total_pages = len(pdfium.PdfDocument(pdf_bytes))

    # ── Phase 1: Preprocess ──
    t0 = time.perf_counter()
    worker_init()
    page_dicts = preprocess_pages(pdf_bytes, 0, total_pages - 1)
    all_pages = [reconstruct_page(pd) for pd in page_dicts]
    t_preprocess = time.perf_counter()

    # ── Phase 2: Layout ──
    layout_model = LayoutModel(
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(device=device),
        options=LayoutOptions(model_spec=DOCLING_LAYOUT_V2),
    )

    _LABEL_CACHE = {member.value: member for member in DocItemLabel}
    layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_V2)

    all_predictions = layout_model(page_batch=all_pages)
    t_layout = time.perf_counter()

    # Apply layout predictions to pages (same as PostprocessService)
    for page, page_preds in zip(all_pages, all_predictions):
        clusters = []
        for ix, pred_item in enumerate(page_preds):
            label = _LABEL_CACHE[pred_item["label"].lower().replace(" ", "_").replace("-", "_")]
            cluster = Cluster.model_construct(
                id=ix, label=label, confidence=pred_item["confidence"],
                bbox=BoundingBox.model_construct(
                    l=pred_item["l"], t=pred_item["t"],
                    r=pred_item["r"], b=pred_item["b"],
                ),
                cells=[], children=[],
            )
            clusters.append(cluster)

        processed_clusters, _ = LayoutPostprocessor(
            page, clusters, layout_options
        ).postprocess()
        page.predictions.layout = LayoutPrediction(clusters=processed_clusters)

    # ── Phase 3: Table prep ──
    table_model = TableStructureModel()
    table_args = table_model.preprocess(pages=all_pages)
    t_table_prep = time.perf_counter()

    if not table_args.get('table_images'):
        return BenchResult(
            pdf_name=pdf_path.name,
            total_pages=total_pages,
            total_tables=0,
            tables=[],
            preprocess_ms=(t_preprocess - t0) * 1000,
            layout_ms=(t_layout - t_preprocess) * 1000,
            table_prep_ms=(t_table_prep - t_layout) * 1000,
        )

    # ── Phase 4: TableFormer inference ──
    tf_model = TableInferenceModel(
        enabled=True, artifacts_path=None,
        options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
        accelerator_options=AcceleratorOptions(device=device),
    )

    t_inf_start = time.perf_counter()
    predictions = tf_model.predict(
        iocr_pages=table_args['iocr_pages'],
        table_bboxes=table_args['table_bboxes'],
        table_images=table_args['table_images'],
        scale_factors=table_args['scale_factors'],
    )
    t_inf_end = time.perf_counter()

    # ── Phase 5: Table postprocessing ──
    tpp_model = TablePostprocessingModel()

    t_pp_start = time.perf_counter()
    processed_pages = tpp_model.postprocess(
        all_predictions=predictions,
        pages=all_pages,
        iocr_pages=table_args['iocr_pages'],
        table_bboxes=table_args['table_bboxes'],
        scale_factors=table_args['scale_factors'],
        page_clusters_list=table_args['page_clusters_list'],
        batched_page_indexes=table_args['batched_page_indexes'],
    )
    t_pp_end = time.perf_counter()

    # ── Capture per-table results ──
    table_results = []
    for page in processed_pages:
        if not page.predictions.tablestructure:
            continue
        for tbl_idx, (tbl_id, tbl) in enumerate(page.predictions.tablestructure.table_map.items()):
            # Extract cell texts in row-major order
            cell_texts = []
            for tc in sorted(tbl.table_cells, key=lambda c: (c.start_row_offset_idx, c.start_col_offset_idx)):
                text = ""
                for bbox_data in getattr(tc, 'text_cell_bboxes', []) or []:
                    tok = bbox_data.get("token", "") if isinstance(bbox_data, dict) else ""
                    if tok:
                        text += (" " + tok) if text else tok
                cell_texts.append(text)

            table_results.append(TableResult(
                page_no=page.page_no,
                table_idx=tbl_idx,
                otsl_seq=tbl.otsl_seq or [],
                num_rows=tbl.num_rows,
                num_cols=tbl.num_cols,
                n_cells=len(tbl.table_cells),
                cell_texts=cell_texts,
            ))

    total_ms = (t_pp_end - t0) * 1000
    return BenchResult(
        pdf_name=pdf_path.name,
        total_pages=total_pages,
        total_tables=len(table_results),
        tables=table_results,
        preprocess_ms=(t_preprocess - t0) * 1000,
        layout_ms=(t_layout - t_preprocess) * 1000,
        table_prep_ms=(t_table_prep - t_layout) * 1000,
        table_inference_ms=(t_inf_end - t_inf_start) * 1000,
        table_postprocess_ms=(t_pp_end - t_pp_start) * 1000,
        total_ms=total_ms,
    )


def compare_golden(golden: BenchResult, current: BenchResult) -> List[str]:
    """Compare current results against golden. Returns list of failure messages."""
    failures = []

    if golden.total_tables != current.total_tables:
        failures.append(
            f"Table count: golden={golden.total_tables} current={current.total_tables}"
        )

    golden_by_key = {(t.page_no, t.table_idx): t for t in golden.tables}
    current_by_key = {(t.page_no, t.table_idx): t for t in current.tables}

    for key, gt in golden_by_key.items():
        ct = current_by_key.get(key)
        if ct is None:
            failures.append(f"Page {key[0]} table {key[1]}: missing in current")
            continue

        if gt.otsl_seq != ct.otsl_seq:
            # Count row differences
            g_rows = gt.otsl_seq.count("nl")
            c_rows = ct.otsl_seq.count("nl")
            failures.append(
                f"Page {key[0]} table {key[1]}: OTSL mismatch "
                f"(golden={g_rows} rows, current={c_rows} rows)"
            )

        if gt.num_rows != ct.num_rows or gt.num_cols != ct.num_cols:
            failures.append(
                f"Page {key[0]} table {key[1]}: shape {gt.num_rows}x{gt.num_cols} -> {ct.num_rows}x{ct.num_cols}"
            )

        if gt.cell_texts != ct.cell_texts:
            diff_count = sum(1 for a, b in zip(gt.cell_texts, ct.cell_texts) if a != b)
            diff_count += abs(len(gt.cell_texts) - len(ct.cell_texts))
            failures.append(
                f"Page {key[0]} table {key[1]}: {diff_count} cells differ "
                f"(golden={len(gt.cell_texts)} current={len(ct.cell_texts)})"
            )

    for key in current_by_key:
        if key not in golden_by_key:
            failures.append(f"Page {key[0]} table {key[1]}: new (not in golden)")

    return failures


def main():
    parser = argparse.ArgumentParser(description="TableFormer benchmark + golden harness")
    parser.add_argument("--update-golden", action="store_true", help="Capture golden references")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default=1 for determinism)")
    parser.add_argument("--pdf", default=None, help="Run only this PDF filename")
    args = parser.parse_args()

    pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    if args.pdf:
        pdfs = [p for p in pdfs if p.name == args.pdf]
        if not pdfs:
            print(f"PDF not found: {args.pdf}")
            sys.exit(1)

    if not pdfs:
        print(f"No PDFs in {PDFS_DIR}")
        sys.exit(1)

    print(f"Device: {args.device} | Threads: {args.threads} | PDFs: {len(pdfs)}")
    print(f"{'='*70}")

    all_failures = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        golden_path = TF_GOLDEN_DIR / f"{stem}.json"

        print(f"\n{pdf_path.name}", flush=True)
        result = run_pdf(pdf_path, device=args.device, threads=args.threads)

        # Print timing
        print(f"  {result.total_pages} pages, {result.total_tables} tables")
        print(f"  preprocess:  {result.preprocess_ms:>8.0f} ms")
        print(f"  layout:      {result.layout_ms:>8.0f} ms")
        print(f"  table prep:  {result.table_prep_ms:>8.0f} ms")
        print(f"  TF inference:{result.table_inference_ms:>8.0f} ms")
        print(f"  TF postproc: {result.table_postprocess_ms:>8.0f} ms")
        print(f"  total:       {result.total_ms:>8.0f} ms")

        if result.total_tables > 0:
            per_table = result.table_inference_ms / result.total_tables
            print(f"  per-table:   {per_table:>8.1f} ms")

        if args.update_golden:
            TF_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            with open(golden_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  -> Saved golden: {golden_path.name}")
            continue

        if not golden_path.exists():
            print(f"  SKIP (no golden file — run with --update-golden)")
            continue

        with open(golden_path) as f:
            golden_data = json.load(f)

        golden = BenchResult(
            pdf_name=golden_data["pdf_name"],
            total_pages=golden_data["total_pages"],
            total_tables=golden_data["total_tables"],
            tables=[TableResult(**t) for t in golden_data["tables"]],
        )

        failures = compare_golden(golden, result)
        if failures:
            print(f"  FAIL ({len(failures)} regressions)")
            for f_msg in failures:
                print(f"    {f_msg}")
            all_failures.extend(failures)
        else:
            print(f"  PASS (matches golden)")

        # Print speed comparison if golden has timing
        g_inf = golden_data.get("table_inference_ms", 0)
        if g_inf > 0 and result.table_inference_ms > 0:
            ratio = g_inf / result.table_inference_ms
            print(f"  speed: {result.table_inference_ms:.0f}ms vs golden {g_inf:.0f}ms ({ratio:.2f}x)")

    print(f"\n{'='*70}")
    if args.update_golden:
        print(f"Golden files saved for {len(pdfs)} PDFs")
    elif all_failures:
        print(f"FAILED: {len(all_failures)} regressions")
        sys.exit(1)
    else:
        print(f"PASSED: all PDFs match golden")


if __name__ == "__main__":
    main()
