#!/usr/bin/env python3
"""
Full postprocessing debug harness for the "Other Assets" merge bug.

Runs model inference + full matching/postprocessing pipeline on a single table,
with instrumentation at every stage to trace where "Other" and "Total other assets" collapse.

Usage:
    python tests/table_postprocess_debug.py [--cpu] [--page=14] [--table-idx=0]
"""

import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Target PDF tokens to track through the pipeline
TARGET_TEXTS = {"Other", "632", "541", "Total other assets", "11,724", "6,425",
                "Total", "other", "assets", "$", "other assets"}

def is_target_token(text):
    """Check if a token text is one we're tracking."""
    return text.strip() in TARGET_TEXTS or any(t in text for t in ["Other", "Total other", "11,724", "6,425", "632", "541"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--pdf", default="tests/golden/pdfs/nvidia_10q.pdf")
    p.add_argument("--page", type=int, default=14, help="0-indexed page")
    p.add_argument("--table-idx", type=int, default=0)
    p.add_argument("--dump-json", default="/tmp/table_debug/postprocess_dump.json")
    return p.parse_args()


def get_page_tokens(pdf_path, page_idx):
    """Extract PDF text tokens with bboxes using docling-parse."""
    from docling_parse.docling_parse import pdf_parser_v2

    parser = pdf_parser_v2(level="warning")
    doc_key = f"debug-{page_idx}"
    success = parser.load_document(doc_key, pdf_path)
    if not success:
        raise RuntimeError(f"Failed to load PDF: {pdf_path}")

    num_pages = parser.number_of_pages(doc_key)
    print(f"[PARSE] Loaded PDF with {num_pages} pages")

    page_result = parser.parse_pdf_from_key_on_page(doc_key, page_idx)

    # Extract cells from all levels
    tokens = []
    token_id = 0

    if "sanitized" in page_result and "cells" in page_result["sanitized"]:
        cells_data = page_result["sanitized"]["cells"]
        if "data" in cells_data:
            for cell_row in cells_data["data"]:
                if len(cell_row) >= 5:
                    # Format: [text, x1, y1, x2, y2, ...]
                    text = str(cell_row[0]) if cell_row[0] is not None else ""
                    bbox = [float(cell_row[1]), float(cell_row[2]), float(cell_row[3]), float(cell_row[4])]
                    if text.strip():
                        tokens.append({"id": token_id, "text": text, "bbox": bbox})
                        token_id += 1

    if not tokens:
        # Try alternative extraction path
        print("[PARSE] No tokens from sanitized cells, trying raw extraction...")
        # Fall back to using docling directly
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        # This is a heavier approach but guaranteed to work
        print("[PARSE] WARNING: Using full docling conversion fallback")

    parser.unload_document(doc_key)
    return tokens


def get_page_tokens_via_docling(pdf_path, page_idx):
    """Alternative: use docling's full pipeline to get tokens for one page."""
    from docling_parse.docling_parse import pdf_parser_v2
    import pypdfium2

    parser = pdf_parser_v2(level="warning")
    doc_key = "debug"
    parser.load_document(doc_key, pdf_path)

    page = parser.parse_pdf_from_key_on_page(doc_key, page_idx)

    tokens = []
    token_id = 0

    # The page dict has nested structure - dig into it
    # Try multiple paths to find the cells
    def extract_from_dict(d, depth=0):
        nonlocal token_id, tokens
        if isinstance(d, dict):
            # Check if this looks like a cell
            if "text" in d and "bbox" in d:
                text = str(d["text"]).strip()
                if text:
                    bbox = d["bbox"]
                    if isinstance(bbox, dict):
                        bbox = [bbox.get("l", 0), bbox.get("t", 0), bbox.get("r", 0), bbox.get("b", 0)]
                    tokens.append({"id": token_id, "text": text, "bbox": list(bbox)})
                    token_id += 1
            for v in d.values():
                extract_from_dict(v, depth + 1)
        elif isinstance(d, (list, tuple)):
            for item in d:
                extract_from_dict(item, depth + 1)

    extract_from_dict(page)

    # If still no tokens, try the cells/data path more carefully
    if not tokens:
        print(f"[PARSE] Exploring page structure keys: {list(page.keys()) if isinstance(page, dict) else type(page)}")
        # Dump first few levels for debugging
        def show_structure(d, prefix="", depth=0):
            if depth > 3:
                return
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        print(f"{prefix}{k}: dict({len(v)} keys)")
                        show_structure(v, prefix + "  ", depth + 1)
                    elif isinstance(v, list):
                        print(f"{prefix}{k}: list({len(v)} items)")
                        if v and depth < 2:
                            show_structure(v[0], prefix + "  [0] ", depth + 1)
                    else:
                        val_str = str(v)[:80]
                        print(f"{prefix}{k}: {type(v).__name__} = {val_str}")
        show_structure(page)

    parser.unload_document(doc_key)
    return tokens, page


def build_iocr_page(page_img, tokens):
    """Build the iocr_page dict expected by the cell matcher."""
    H, W = page_img.shape[:2]
    return {
        "width": W,
        "height": H,
        "tokens": tokens,
        "image": page_img,
    }


def run_model(pdf_path, page_idx, table_idx, device):
    """Run layout + table model and return prediction + table bbox + page image."""
    from docling.models.utils.hf_model_download import download_hf_model
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    import docling_ibm_models.tableformer.common as c
    from processor.gpu_service.tablemodel04_rs import TableModel04_rs
    from safetensors.torch import load_model as load_safetensors
    import docling_ibm_models.tableformer.utils.utils as u
    import glob
    import pypdfium2

    # Render page
    pdf = pypdfium2.PdfDocument(pdf_path)
    page = pdf[page_idx]
    scale = 2.0
    bitmap = page.render(scale=scale)
    page_img = bitmap.to_numpy()[:, :, :3]
    print(f"[PAGE] {page_img.shape[1]}x{page_img.shape[0]}")

    # Layout detection
    layout_path = download_hf_model(repo_id="ds4sd/docling-layout-old", revision="main")
    layout = LayoutPredictor(str(layout_path), device=device)
    preds = list(layout.predict(page_img))
    table_bboxes = []
    for p in preds:
        label = p.get("label", "") if isinstance(p, dict) else getattr(p, "label", "")
        if label.lower() == "table":
            if isinstance(p, dict):
                table_bboxes.append([p["l"], p["t"], p["r"], p["b"]])
            else:
                table_bboxes.append([p.l, p.t, p.r, p.b])

    print(f"[LAYOUT] {len(table_bboxes)} tables found")
    target_bbox = table_bboxes[table_idx]
    print(f"[TARGET] Table {table_idx}: {[round(x,1) for x in target_bbox]}")

    # Load table model
    art_path = download_hf_model(repo_id="ds4sd/docling-models", revision="v2.2.0")
    art_path = os.path.join(str(art_path), "model_artifacts", "tableformer", "accurate")
    config = c.read_config(os.path.join(art_path, "tm_config.json"))
    config["model"]["save_dir"] = art_path

    word_map = config.get("dataset_wordmap")
    init_data = {"word_map": word_map}
    rev_word_map = {v: k for k, v in word_map["word_map_tag"].items()}

    model = TableModel04_rs(config, init_data, device)
    models_fn = sorted(glob.glob(f"{art_path}/tableformer_*.safetensors"))
    load_safetensors(model, models_fn[0], device=device)
    model.setup_for_inference()

    # Crop and preprocess table
    H, W = page_img.shape[:2]
    scale_factor = 1024.0 / float(H)
    x1, y1, x2, y2 = target_bbox
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(W, int(round(x2))), min(H, int(round(y2)))
    crop = page_img[iy1:iy2, ix1:ix2]
    tw = max(1, int(round((x2 - x1) * scale_factor)))
    th = max(1, int(round((y2 - y1) * scale_factor)))
    resized_crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    # Preprocess (stock path for purity)
    import docling_ibm_models.tableformer.data_management.transforms as T
    mean = config["dataset"]["image_normalization"]["mean"]
    std = config["dataset"]["image_normalization"]["std"]
    resized_size = config["dataset"]["resized_image"]
    normalize = T.Normalize(mean=mean, std=std)
    resize = T.Resize([resized_size, resized_size])
    img_n, _ = normalize(resized_crop, None)
    img_r, _ = resize(img_n, None)
    img_t = img_r.transpose(2, 1, 0)
    img_t = torch.FloatTensor(img_t / 255.0).unsqueeze(0).to(device)

    # Run model
    with torch.inference_mode():
        model.eval()
        results = model.predict(img_t, config["predict"]["max_steps"], config["predict"]["beam_size"])

    seq, cls_logits, coords = results[0]

    # Build prediction dict (same format as tf_predictor)
    tags = [rev_word_map.get(t, f"?{t}") for t in seq]
    rs_seq = [t for t in tags if t not in ("<start>", "<end>")]

    if torch.is_tensor(coords) and coords.numel() > 0:
        bbox_xyxy = u.box_cxcywh_to_xyxy(coords)
        bboxes = bbox_xyxy.cpu().tolist()
    else:
        bboxes = []

    if torch.is_tensor(cls_logits) and cls_logits.numel() > 0:
        classes = torch.argmax(cls_logits, dim=1).cpu().tolist()
    else:
        classes = []

    from docling_ibm_models.tableformer.otsl import otsl_to_html
    html_seq = otsl_to_html(rs_seq, False)

    prediction = {
        "tag_seq": seq,
        "rs_seq": rs_seq,
        "html_seq": html_seq,
        "bboxes": bboxes,
        "classes": classes,
    }

    print(f"[MODEL] rs_seq: {' '.join(rs_seq)}")
    print(f"[MODEL] {len(bboxes)} bboxes, {len(classes)} classes")

    return page_img, target_bbox, prediction, config


def run_postprocessing_instrumented(iocr_page, table_bbox, prediction):
    """Run full matching + postprocessing with instrumentation."""
    from processor.table_postprocessing_service.tf_cell_matcher import CellMatcher
    from processor.table_postprocessing_service.matching_post_processor import MatchingPostProcessor

    cell_matcher = CellMatcher()
    post_processor = MatchingPostProcessor()

    dump = {"stages": {}}

    # ============================================================
    # STAGE 0: Dump source tokens
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 0: Source PDF tokens")
    print("=" * 70)

    target_tokens = []
    for tok in iocr_page["tokens"]:
        if is_target_token(tok["text"]):
            target_tokens.append(tok)
            print(f"  id={tok['id']:3d} text={tok['text']!r:30s} bbox={[round(x,1) for x in tok['bbox']]}")

    dump["stages"]["0_source_tokens"] = target_tokens

    # ============================================================
    # STAGE 1: Initial matching (CellMatcher.match_cells)
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 1: CellMatcher.match_cells()")
    print("=" * 70)

    matching_details = cell_matcher.match_cells(iocr_page, table_bbox, prediction)

    # Dump predicted table cells (left column = column 0)
    print("\n  Predicted table cells (all):")
    for tc in matching_details["table_cells"]:
        print(f"    cell_id={tc['cell_id']:2d} row={tc['row_id']} col={tc['column_id']} "
              f"label={tc['label']:5s} bbox=[{tc['bbox'][0]:.1f}, {tc['bbox'][1]:.1f}, {tc['bbox'][2]:.1f}, {tc['bbox'][3]:.1f}]")

    dump["stages"]["1_table_cells"] = copy.deepcopy(matching_details["table_cells"])

    # Dump matches for target tokens
    print("\n  Initial matches for target tokens:")
    matches = matching_details["matches"]
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in matches:
            for m in matches[tok_id]:
                tc_id = m["table_cell_id"]
                # Find table cell
                tc = next((c for c in matching_details["table_cells"] if c["cell_id"] == tc_id), None)
                iopdf = m.get("iopdf", m.get("iou", 0))
                row = tc["row_id"] if tc else "?"
                col = tc["column_id"] if tc else "?"
                print(f"    token id={tok['id']} text={tok['text']!r:25s} -> table_cell={tc_id} row={row} col={col} iopdf={iopdf:.4f}")
        else:
            print(f"    token id={tok['id']} text={tok['text']!r:25s} -> NO MATCH")

    dump["stages"]["1_matches"] = {k: v for k, v in matches.items()
                                    if any(str(t["id"]) == k for t in target_tokens)}

    # ============================================================
    # STAGE 2: Post-processing (instrumented)
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 2: MatchingPostProcessor.process()")
    print("=" * 70)

    # We'll run the postprocessor but also intercept intermediate states
    # by running steps manually

    table_cells = matching_details["table_cells"]
    pdf_cells = post_processor._clear_pdf_cells(matching_details["pdf_cells"])
    initial_matches = matching_details["matches"]

    # Step 0: dimensions
    tab_columns, tab_rows, max_cell_id = post_processor._get_table_dimension(table_cells)
    print(f"\n  Table dimensions: {tab_columns} cols x {tab_rows} rows, max_cell_id={max_cell_id}")

    # Steps 1-4: Fix cells column by column
    fixed_table_cells = []
    for col in range(tab_columns):
        good, bad = post_processor._get_good_bad_cells_in_column(table_cells, col, initial_matches)
        alignment = post_processor._find_alignment_in_column(good)
        mx, my, mw, mh = post_processor._get_median_pos_size(good, alignment)
        new_bad = post_processor._move_cells_to_left_pos(bad, mx, False, mw, mh, alignment)
        fixed_table_cells.extend(good)
        fixed_table_cells.extend(new_bad)
        print(f"  Col {col}: good={len(good)} bad={len(bad)} alignment={alignment}")

    fixed_sorted = sorted(fixed_table_cells, key=lambda k: k["cell_id"])

    # Step 5: Re-run intersection match
    ioc_matches = post_processor._run_intersection_match(cell_matcher, fixed_sorted, pdf_cells)

    print("\n  After _run_intersection_match (step 5):")
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in ioc_matches:
            for m in ioc_matches[tok_id]:
                tc_id = m["table_cell_id"]
                tc = next((c for c in fixed_sorted if c["cell_id"] == tc_id), None)
                row = tc["row_id"] if tc else "?"
                col = tc["column_id"] if tc else "?"
                print(f"    token id={tok['id']} text={tok['text']!r:25s} -> cell={tc_id} row={row} col={col} iopdf={m.get('iopdf',0):.4f}")
        else:
            print(f"    token id={tok['id']} text={tok['text']!r:25s} -> NO MATCH")

    dump["stages"]["5_ioc_matches"] = {k: v for k, v in ioc_matches.items()
                                        if any(str(t["id"]) == k for t in target_tokens)}

    # Step 7: Deduplicate
    dedupl_cells, dedupl_matches, new_cols = post_processor._deduplicate_cells(
        tab_columns, fixed_sorted, initial_matches, ioc_matches)

    print(f"\n  After _deduplicate_cells (step 7): {new_cols} cols, {len(dedupl_cells)} cells")
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in dedupl_matches:
            for m in dedupl_matches[tok_id]:
                tc_id = m["table_cell_id"]
                tc = next((c for c in dedupl_cells if c["cell_id"] == tc_id), None)
                row = tc["row_id"] if tc else "?"
                col = tc["column_id"] if tc else "?"
                print(f"    token id={tok['id']} text={tok['text']!r:25s} -> cell={tc_id} row={row} col={col}")
        else:
            print(f"    token id={tok['id']} text={tok['text']!r:25s} -> NOT IN dedupl_matches")

    dump["stages"]["7_dedupl_matches"] = {k: v for k, v in dedupl_matches.items()
                                           if any(str(t["id"]) == k for t in target_tokens)}

    # Step 8: Final assignment
    final_matches = post_processor._do_final_asignment(dedupl_cells, initial_matches, dedupl_matches)

    print(f"\n  After _do_final_asignment (step 8):")
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in final_matches:
            for m in final_matches[tok_id]:
                tc_id = m["table_cell_id"]
                tc = next((c for c in dedupl_cells if c["cell_id"] == tc_id), None)
                row = tc["row_id"] if tc else "?"
                col = tc["column_id"] if tc else "?"
                print(f"    token id={tok['id']} text={tok['text']!r:25s} -> cell={tc_id} row={row} col={col}")
        else:
            print(f"    token id={tok['id']} text={tok['text']!r:25s} -> NOT IN final_matches")

    dump["stages"]["8_final_matches"] = {k: v for k, v in final_matches.items()
                                          if any(str(t["id"]) == k for t in target_tokens)}

    # Step 8a: Align to PDF
    dedupl_sorted = sorted(dedupl_cells, key=lambda k: k["cell_id"])
    aligned_cells = post_processor._align_table_cells_to_pdf(dedupl_sorted, pdf_cells, final_matches)

    print(f"\n  After _align_table_cells_to_pdf (step 8a): {len(aligned_cells)} cells")
    print("  Left-column cells (col=0) before/after alignment:")
    for ac in aligned_cells:
        if ac["column_id"] == 0:
            orig = next((c for c in dedupl_sorted if c["cell_id"] == ac["cell_id"]), None)
            orig_bbox = orig["bbox"] if orig else [0, 0, 0, 0]
            print(f"    cell_id={ac['cell_id']:2d} row={ac['row_id']} "
                  f"before=[{orig_bbox[0]:.1f},{orig_bbox[1]:.1f},{orig_bbox[2]:.1f},{orig_bbox[3]:.1f}] "
                  f"after=[{ac['bbox'][0]:.1f},{ac['bbox'][1]:.1f},{ac['bbox'][2]:.1f},{ac['bbox'][3]:.1f}]")

    dump["stages"]["8a_aligned_cells_col0"] = [
        {"cell_id": c["cell_id"], "row_id": c["row_id"], "bbox": c["bbox"]}
        for c in aligned_cells if c["column_id"] == 0
    ]

    # Step 9: Orphan cells
    orphan_matches, orphan_cells, _ = post_processor._pick_orphan_cells(
        tab_rows, tab_columns, max_cell_id, aligned_cells, pdf_cells, final_matches)

    print(f"\n  After _pick_orphan_cells (step 9):")
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in orphan_matches:
            for m in orphan_matches[tok_id]:
                tc_id = m["table_cell_id"]
                tc = next((c for c in orphan_cells if c["cell_id"] == tc_id), None)
                row = tc["row_id"] if tc else "?"
                col = tc["column_id"] if tc else "?"
                print(f"    token id={tok['id']} text={tok['text']!r:25s} -> cell={tc_id} row={row} col={col}")
        else:
            print(f"    token id={tok['id']} text={tok['text']!r:25s} -> NOT IN orphan_matches")

    dump["stages"]["9_orphan_matches"] = {k: v for k, v in orphan_matches.items()
                                           if any(str(t["id"]) == k for t in target_tokens)}

    # ============================================================
    # FINAL: Check if "Other" and "Total other assets" share a cell
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    other_cell = None
    total_cell = None
    for tok in target_tokens:
        tok_id = str(tok["id"])
        if tok_id in orphan_matches:
            tc_id = orphan_matches[tok_id][0]["table_cell_id"]
            if "Other" == tok["text"].strip() or tok["text"].strip() == "Other":
                other_cell = tc_id
            if "Total other assets" in tok["text"] or "Total" in tok["text"]:
                total_cell = tc_id

    if other_cell is not None and total_cell is not None:
        if other_cell == total_cell:
            print(f"  BUG CONFIRMED: 'Other' and 'Total other assets' map to SAME cell_id={other_cell}")
        else:
            print(f"  OK: 'Other' -> cell_id={other_cell}, 'Total other assets' -> cell_id={total_cell}")
    else:
        print(f"  Could not find both tokens. other_cell={other_cell}, total_cell={total_cell}")
        print("  Check the target token matching above for details.")

    # Save dump
    os.makedirs(os.path.dirname(args.dump_json), exist_ok=True)
    with open(args.dump_json, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\n[SAVE] Full dump: {args.dump_json}")

    return dump


def main():
    global args
    args = parse_args()
    device = "cpu" if args.cpu else "cuda"

    print("=" * 70)
    print("POSTPROCESSING DEBUG HARNESS")
    print(f"PDF: {args.pdf}, Page: {args.page}, Table: {args.table_idx}, Device: {device}")
    print("=" * 70)

    # Get PDF tokens
    print("\n[STEP 1] Extracting PDF tokens...")
    tokens, raw_page = get_page_tokens_via_docling(args.pdf, args.page)
    print(f"[PARSE] {len(tokens)} tokens extracted")

    if not tokens:
        print("[ERROR] No tokens found! Dumping raw page structure...")
        return

    # Run model
    print("\n[STEP 2] Running model inference...")
    page_img, table_bbox, prediction, config = run_model(
        args.pdf, args.page, args.table_idx, device)

    # Build iocr page
    iocr_page = build_iocr_page(page_img, tokens)

    # Run instrumented postprocessing
    print("\n[STEP 3] Running instrumented postprocessing...")
    dump = run_postprocessing_instrumented(iocr_page, table_bbox, prediction)


if __name__ == "__main__":
    main()
