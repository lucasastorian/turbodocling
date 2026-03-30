#!/usr/bin/env python3
"""
Local end-to-end test: custom docling-parse → layout → our table model → our postprocessing.
Runs entirely on CPU, no AWS needed.

Usage:
    python tests/local_e2e.py [--pdf tests/golden/pdfs/nvidia_10q.pdf] [--page 14]
"""
import argparse, sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch


def extract_tokens(pdf_path, page_idx):
    """Extract tokens using our custom docling-parse v4 fork."""
    from docling_parse.docling_parse import pdf_parser_v2
    p = pdf_parser_v2(level="warning")
    p.load_document("d", pdf_path)
    page = p.parse_pdf_from_key_on_page("d", page_idx)

    cells = page.get("sanitized", {}).get("cells", {})
    header = cells.get("header", [])
    data = cells.get("data", [])

    # Find column indices from header
    cols = {name: idx for idx, name in enumerate(header)}

    tokens = []
    for i, row in enumerate(data):
        text = str(row[cols.get("text", 0)])
        if not text.strip():
            continue
        x0 = float(row[cols.get("x0", 1)])
        y0 = float(row[cols.get("y0", 2)])
        x1 = float(row[cols.get("x1", 3)])
        y1 = float(row[cols.get("y1", 4)])
        tokens.append({"id": i, "text": text, "bbox": {"l": x0, "t": y0, "r": x1, "b": y1}})

    p.unload_document("d")
    return tokens


def render_page(pdf_path, page_idx, scale=2.0):
    """Render page image with pypdfium2."""
    import pypdfium2
    pdf = pypdfium2.PdfDocument(pdf_path)
    page = pdf[page_idx]
    bm = page.render(scale=scale)
    img = bm.to_numpy()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def detect_tables(page_img, device="cpu"):
    """Run layout model to find tables."""
    from docling.models.utils.hf_model_download import download_hf_model
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

    layout_path = download_hf_model(repo_id="ds4sd/docling-layout-old", revision="main")
    layout = LayoutPredictor(str(layout_path), device=device)
    preds = list(layout.predict(page_img))

    tables = []
    for pred in preds:
        label = pred.get("label", "") if isinstance(pred, dict) else getattr(pred, "label", "")
        if label.lower() == "table":
            if isinstance(pred, dict):
                tables.append([pred["l"], pred["t"], pred["r"], pred["b"]])
            else:
                tables.append([pred.l, pred.t, pred.r, pred.b])
    return tables


def run_table_model(table_images, config_override=None, device="cpu"):
    """Run table model inference."""
    from docling.models.utils.hf_model_download import download_hf_model
    from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions, AcceleratorOptions
    from processor.gpu_service.table_inference_model import TableInferenceModel

    accel = AcceleratorOptions(device=device, num_threads=4)
    model = TableInferenceModel(
        enabled=True, artifacts_path=None,
        options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
        accelerator_options=accel,
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="tests/golden/pdfs/nvidia_10q.pdf")
    p.add_argument("--page", type=int, default=14, help="0-indexed page")
    args = p.parse_args()

    pdf_path = args.pdf
    page_idx = args.page

    # Step 1: Extract tokens with our custom parser
    print(f"[1/5] Extracting tokens (custom docling-parse)")
    t0 = time.time()
    all_tokens = extract_tokens(pdf_path, page_idx)
    print(f"  {len(all_tokens)} tokens in {time.time()-t0:.1f}s")

    # Step 2: Render page
    print(f"\n[2/5] Rendering page {page_idx}")
    page_img = render_page(pdf_path, page_idx, scale=2.0)
    H, W = page_img.shape[:2]
    print(f"  {W}x{H}")

    # Step 3: Detect tables
    print(f"\n[3/5] Layout detection")
    t0 = time.time()
    table_bboxes = detect_tables(page_img, device="cpu")
    print(f"  {len(table_bboxes)} tables in {time.time()-t0:.1f}s")
    for i, bb in enumerate(table_bboxes):
        print(f"  Table {i}: [{bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f}]")

    # Step 4: Prepare per-table inputs (matching our fixed pipeline)
    print(f"\n[4/5] Preprocessing + inference")
    t0 = time.time()

    scale_factor = 1024.0 / float(H)

    # Filter tokens per table bbox (our fix: per-table, not aggregated)
    def tokens_in_bbox(tokens, bbox, margin=5):
        """Filter tokens whose bbox overlaps the table bbox (in image coords)."""
        x1, y1, x2, y2 = bbox
        result = []
        for tok in tokens:
            tb = tok["bbox"]
            # Token bbox is in image coords (scaled by 2.0 from PDF coords)
            # Convert from ltrb dict format
            tl, tt, tr, tb_val = tb["l"] * 2.0, tb["t"] * 2.0, tb["r"] * 2.0, tb["b"] * 2.0
            # Check overlap with margin
            if tr < x1 - margin or tl > x2 + margin:
                continue
            if tb_val < y1 - margin or tt > y2 + margin:
                continue
            result.append(tok)
        return result

    iocr_pages = []
    scaled_bboxes = []
    table_images = []
    scale_factors = []

    for i, bbox in enumerate(table_bboxes):
        x1, y1, x2, y2 = bbox
        ix1 = max(0, int(round(x1)))
        iy1 = max(0, int(round(y1)))
        ix2 = min(W, int(round(x2)))
        iy2 = min(H, int(round(y2)))

        crop = page_img[iy1:iy2, ix1:ix2]
        tw = max(1, int(round((x2 - x1) * scale_factor)))
        th = max(1, int(round((y2 - y1) * scale_factor)))
        resized_crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

        # Per-table tokens (THE FIX)
        tbl_tokens = tokens_in_bbox(all_tokens, bbox)

        iocr_pages.append({
            "width": W,
            "height": H,
            "tokens": tbl_tokens,
        })
        scaled_bboxes.append([x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor])
        table_images.append(resized_crop)
        scale_factors.append(scale_factor)
        print(f"  Table {i}: {len(tbl_tokens)} tokens (filtered)")

    # Load model and run inference
    from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions, AcceleratorOptions
    from processor.gpu_service.table_inference_model import TableInferenceModel

    model = TableInferenceModel(
        enabled=True, artifacts_path=None,
        options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
    )

    predictions = model.predict(
        iocr_pages=iocr_pages,
        table_bboxes=scaled_bboxes,
        table_images=table_images,
        scale_factors=scale_factors,
    )
    print(f"  Inference done in {time.time()-t0:.1f}s")

    for i, pred in enumerate(predictions):
        rs = pred.get("rs_seq", [])
        print(f"  Table {i}: {' '.join(rs)}")

    # Step 5: Run postprocessing
    print(f"\n[5/5] Postprocessing")
    t0 = time.time()

    from processor.table_postprocessing_service.tf_cell_matcher import CellMatcher
    from processor.table_postprocessing_service.matching_post_processor import MatchingPostProcessor
    from processor.gpu_service.tf_predictor import TFPredictor

    cell_matcher = CellMatcher()
    post_processor = MatchingPostProcessor()

    for i, pred in enumerate(predictions):
        iocr_page = iocr_pages[i]
        sf = scale_factors[i]
        tbl_bbox = [
            scaled_bboxes[i][0] / sf,
            scaled_bboxes[i][1] / sf,
            scaled_bboxes[i][2] / sf,
            scaled_bboxes[i][3] / sf,
        ]

        if len(pred.get("bboxes", [])) == 0:
            continue

        matching_details = cell_matcher.match_cells(iocr_page, tbl_bbox, pred)
        if len(iocr_page.get("tokens", [])) > 0:
            matching_details = post_processor.process(matching_details)

        # Extract table content
        table_cells = matching_details["table_cells"]
        matches = matching_details["matches"]
        pdf_cells = matching_details["pdf_cells"]

        # Build simple text grid
        max_row = max((c["row_id"] for c in table_cells), default=0)
        max_col = max((c["column_id"] for c in table_cells), default=0)
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        pdf_by_id = {str(pc["id"]): pc for pc in pdf_cells}
        cell_by_id = {c["cell_id"]: c for c in table_cells}

        for pdf_id, match_list in matches.items():
            pdf_cell = pdf_by_id.get(str(pdf_id))
            if not pdf_cell:
                continue
            text = pdf_cell["text"]
            for m in match_list:
                tc = cell_by_id.get(m["table_cell_id"])
                if tc:
                    r, c = tc["row_id"], tc["column_id"]
                    if r <= max_row and c <= max_col:
                        if grid[r][c]:
                            grid[r][c] += " " + text
                        else:
                            grid[r][c] = text

        print(f"\n  === Table {i} ({max_row+1}x{max_col+1}) ===")
        for r, row in enumerate(grid):
            cells_str = " | ".join(f"{c:30s}" for c in row)
            print(f"  Row {r}: {cells_str}")

    print(f"\n  Postprocessing done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
