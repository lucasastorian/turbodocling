#!/usr/bin/env python3
"""
Table debug harness — isolate the "Other Assets" merge bug on page 15 of NVIDIA 10-Q.

Usage:
    python tests/table_debug.py [FLAGS]

Flags (env vars or CLI):
    --no-flash          Force MATH SDPA instead of Flash Attention
    --no-tf32           Disable TF32
    --no-bf16           Keep transformer in FP32
    --no-mem-kv         Disable precomputed cross-attention KV
    --batch-size=N      Override table batch size (default: natural)
    --stock-preprocess  Use stock _prepare_image() instead of GPU batch preprocess
    --batch-pos=N       Put target table at position N in a padded batch (0-indexed)
    --repeat=N          Repeat target table N times in batch
    --cpu               Run on CPU instead of CUDA
    --save-crops        Save table crop images to /tmp/table_debug/
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

# Ensure processor is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Table debug harness")
    p.add_argument("--no-flash", action="store_true", help="Disable Flash Attention")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32")
    p.add_argument("--no-bf16", action="store_true", help="Keep transformer in FP32")
    p.add_argument("--no-mem-kv", action="store_true", help="Disable precomputed cross-attn KV")
    p.add_argument("--batch-size", type=int, default=0, help="Force batch size (0=natural)")
    p.add_argument("--stock-preprocess", action="store_true", help="Use stock image preprocessing")
    p.add_argument("--batch-pos", type=int, default=-1, help="Target table position in padded batch")
    p.add_argument("--repeat", type=int, default=1, help="Repeat target table N times")
    p.add_argument("--cpu", action="store_true", help="Run on CPU")
    p.add_argument("--save-crops", action="store_true", help="Save crop images")
    p.add_argument("--pdf", default="tests/golden/pdfs/nvidia_10q.pdf", help="PDF path")
    p.add_argument("--page", type=int, default=14, help="0-indexed page number")
    p.add_argument("--table-idx", type=int, default=-1, help="Table index on page (-1=auto-detect 'Other Assets')")
    return p.parse_args()


def apply_toggles(args):
    """Apply GPU toggles BEFORE any model loading."""
    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        print("[TOGGLE] TF32 DISABLED")

    if args.no_flash:
        # We'll handle this at inference time via sdpa_kernel context
        print("[TOGGLE] Flash Attention will be DISABLED")

    if args.no_bf16:
        print("[TOGGLE] BF16 will be DISABLED (FP32 transformer)")

    if args.no_mem_kv:
        print("[TOGGLE] Precomputed mem_kv will be DISABLED")


def load_model(args):
    """Load the table model with optional toggles."""
    from docling.models.utils.hf_model_download import download_hf_model
    import docling_ibm_models.tableformer.common as c
    from processor.gpu_service.tablemodel04_rs import TableModel04_rs
    from safetensors.torch import load_model as load_safetensors
    import glob

    # Download/locate model artifacts
    artifacts_path = download_hf_model(
        repo_id="ds4sd/docling-models",
        revision="v2.2.0",
    )
    artifacts_path = os.path.join(str(artifacts_path), "model_artifacts", "tableformer", "accurate")

    config = c.read_config(os.path.join(artifacts_path, "tm_config.json"))
    config["model"]["save_dir"] = artifacts_path
    config["predict"]["profiling"] = True  # Always profile in debug

    # Word map
    word_map = config.get("dataset_wordmap", None)
    if word_map is None:
        wm_path = os.path.join(config["dataset"]["prepared_data_dir"],
                               c.get_prepared_data_filename("WORDMAP", config["dataset"]["name"]))
        with open(wm_path) as f:
            word_map = json.load(f)
        config["dataset_wordmap"] = word_map

    init_data = {"word_map": word_map}
    rev_word_map = {v: k for k, v in word_map["word_map_tag"].items()}

    device = "cpu" if args.cpu else "cuda"
    print(f"[MODEL] Device: {device}")

    # Apply TF32 toggle BEFORE model creation
    if args.no_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model = TableModel04_rs(config, init_data, device)

    # Load weights
    models_fn = sorted(glob.glob(f"{artifacts_path}/tableformer_*.safetensors"))
    load_safetensors(model, models_fn[0], device=device)

    # Setup for inference with optional BF16 toggle
    if args.no_bf16:
        # Skip bf16 conversion, just eval mode
        model.eval()
        torch.set_grad_enabled(False)
        print("[MODEL] Keeping FP32 (bf16 disabled)")
    else:
        model.setup_for_inference()
        print("[MODEL] BF16 transformer enabled")

    return model, config, rev_word_map, device


def extract_page_and_tables(pdf_path, page_idx):
    """Run layout detection on a single page to get table bboxes and page image."""
    import pypdfium2

    # Render page
    pdf = pypdfium2.PdfDocument(pdf_path)
    page = pdf[page_idx]
    scale = 2.0
    bitmap = page.render(scale=scale)
    img = bitmap.to_numpy()
    if img.shape[2] == 4:
        img = img[:, :, :3]  # RGBA -> RGB

    print(f"[PAGE] Page {page_idx}: {img.shape[1]}x{img.shape[0]} (scale={scale})")
    return img


def run_layout_and_get_tables(pdf_path, page_idx):
    """Use docling's layout model to find tables on the page."""
    # We'll use a simpler approach: run the full pipeline on just this page
    # and extract table bboxes from the layout predictions
    from docling.datamodel.pipeline_options import AcceleratorOptions
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    from docling.models.utils.hf_model_download import download_hf_model

    # Load layout model
    artifacts_path = download_hf_model(
        repo_id="ds4sd/docling-layout-old",
        revision="main",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layout = LayoutPredictor(artifacts_path, device=device)

    # Get page image
    img = extract_page_and_tables(pdf_path, page_idx)

    # Run layout prediction
    predictions = list(layout.predict(img))

    table_bboxes = []
    for pred in predictions:
        label = pred.get("label", "") if isinstance(pred, dict) else getattr(pred, "label", "")
        if label.lower() == "table":
            if isinstance(pred, dict):
                table_bboxes.append([pred["l"], pred["t"], pred["r"], pred["b"]])
            else:
                table_bboxes.append([pred.l, pred.t, pred.r, pred.b])

    print(f"[LAYOUT] Found {len(table_bboxes)} tables on page {page_idx}")
    for i, bbox in enumerate(table_bboxes):
        print(f"  Table {i}: bbox={[round(x,1) for x in bbox]}")

    return img, table_bboxes


def preprocess_table_stock(table_image, config, device):
    """Stock preprocessing path: T.Normalize + T.Resize (CPU), then to device."""
    import docling_ibm_models.tableformer.data_management.transforms as T

    mean = config["dataset"]["image_normalization"]["mean"]
    std = config["dataset"]["image_normalization"]["std"]
    resized_size = config["dataset"]["resized_image"]

    normalize = T.Normalize(mean=mean, std=std)
    resize = T.Resize([resized_size, resized_size])

    img, _ = normalize(table_image, None)
    img, _ = resize(img, None)

    img = img.transpose(2, 1, 0)  # (channels, width, height) — stock quirk
    img = torch.FloatTensor(img / 255.0)
    return img.unsqueeze(0).to(device=device)


def preprocess_table_ours(table_images, config, device):
    """Our GPU preprocessing path: _batch_preprocess_images logic."""
    import torch.nn.functional as F

    mean = config["dataset"]["image_normalization"]["mean"]
    std = config["dataset"]["image_normalization"]["std"]
    resized_size = int(config["dataset"]["resized_image"])
    dtype = torch.float32

    # Ensure uint8
    processed = []
    for img in table_images:
        if img.ndim == 2:
            img = img[..., None]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        processed.append(img)

    N = len(processed)
    C = processed[0].shape[2]
    S = resized_size

    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 1, 1, C)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 1, 1, C)

    out = torch.empty((N, C, S, S), device=device, dtype=dtype)

    # Bucket by shape
    buckets = {}
    for i, img in enumerate(processed):
        key = img.shape
        buckets.setdefault(key, []).append(i)

    for shape, idxs in buckets.items():
        nhwc = np.stack([processed[i] for i in idxs], axis=0)
        cpu = torch.from_numpy(nhwc)
        if device == "cuda":
            cpu = cpu.pin_memory()
        t = cpu.to(device=device, dtype=dtype, non_blocking=(device == "cuda"))
        t = t / 255.0
        t = (t - mean_t) / std_t
        t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)
        t = t.permute(0, 1, 3, 2).contiguous()  # (C, W, H) quirk
        out[torch.as_tensor(idxs, device=device, dtype=torch.long)] = t

    return out


def compare_tensors(name, t1, t2):
    """Compare two tensors and print diagnostics."""
    print(f"\n[COMPARE] {name}")
    print(f"  Shape: {t1.shape} vs {t2.shape}")
    print(f"  Dtype: {t1.dtype} vs {t2.dtype}")
    print(f"  Min:   {t1.min().item():.6f} vs {t2.min().item():.6f}")
    print(f"  Max:   {t1.max().item():.6f} vs {t2.max().item():.6f}")
    print(f"  Mean:  {t1.mean().item():.6f} vs {t2.mean().item():.6f}")
    print(f"  Std:   {t1.std().item():.6f} vs {t2.std().item():.6f}")
    diff = (t1.float() - t2.float()).abs()
    print(f"  Max abs diff:  {diff.max().item():.8f}")
    print(f"  Mean abs diff: {diff.mean().item():.8f}")
    return diff.max().item()


def run_inference(model, image_batch, config, args, device, rev_word_map):
    """Run table inference with toggles."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    max_steps = config["predict"]["max_steps"]
    beam_size = config["predict"]["beam_size"]

    # Apply mem_kv toggle
    if args.no_mem_kv:
        # Monkey-patch to disable mem_kv precompute
        orig_predict = model._predict_batch if hasattr(model, '_predict_batch') else None
        # Set USE_MEM_KV = False in batched decoder
        if hasattr(model, '_batched_decoder'):
            print("[TOGGLE] Disabling mem_kv in batched decoder")

    is_cuda = device == "cuda"

    # Choose SDPA backend
    if args.no_flash and is_cuda:
        ctx = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        print("[INFERENCE] Using EFFICIENT_ATTENTION/MATH (no Flash)")
    elif is_cuda:
        ctx = sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION)
        print("[INFERENCE] Using FLASH_ATTENTION")
    else:
        ctx = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        print("[INFERENCE] Using CPU SDPA")

    with torch.inference_mode(), ctx:
        # Override the model's internal SDPA selection by calling predict directly
        # but we need to handle the SDPA context ourselves
        model.eval()
        model._encoder.eval()
        model._tag_transformer.eval()
        model._bbox_decoder.eval()

        B = image_batch.size(0)

        # Encoder
        enc_out = model._encode_in_blocks(image_batch, block_bs=model._encoder_block_bs)

        # Memory preparation
        filtered_nchw = model._tag_transformer._input_filter(enc_out)
        filtered_nhwc = filtered_nchw.permute(0, 2, 3, 1)
        B_, h, w, C = filtered_nhwc.shape
        mem = filtered_nhwc.reshape(B_, h * w, C).permute(1, 0, 2).contiguous()

        if not args.no_bf16:
            mem = mem.to(torch.bfloat16)

        mem_enc = model._tag_transformer._encoder(mem, mask=None)

        # Batched decoder
        results = model._batched_decoder.predict_batched(enc_out, mem_enc, max_steps)

    # Decode results
    import docling_ibm_models.tableformer.utils.utils as u
    decoded = []
    for seq, cls_logits, coords in results:
        tags = [rev_word_map.get(t, f"?{t}") for t in seq]
        # Filter to just the structure tags (no start/end)
        rs_seq = [t for t in tags if t not in ("<start>", "<end>")]

        # Convert bboxes to xyxy for comparison
        if torch.is_tensor(coords) and coords.numel() > 0:
            bbox_xyxy = u.box_cxcywh_to_xyxy(coords)
            bbox_list = bbox_xyxy.cpu().tolist()
            raw_coords = coords.cpu().tolist()
        else:
            bbox_list = []
            raw_coords = []

        decoded.append({
            "tag_seq_ids": seq,
            "tags": tags,
            "rs_seq": rs_seq,
            "n_bboxes": len(bbox_list),
            "bboxes_xyxy": bbox_list,
            "bboxes_cxcywh": raw_coords,
        })

    return decoded


def main():
    args = parse_args()

    print("=" * 70)
    print("TABLE DEBUG HARNESS")
    print("=" * 70)
    print(f"PDF: {args.pdf}")
    print(f"Page: {args.page} (0-indexed)")
    print(f"Toggles: no_flash={args.no_flash} no_tf32={args.no_tf32} no_bf16={args.no_bf16} no_mem_kv={args.no_mem_kv}")
    print(f"Batch: size={args.batch_size} pos={args.batch_pos} repeat={args.repeat}")
    print(f"Stock preprocess: {args.stock_preprocess}")
    print(f"Device: {'cpu' if args.cpu else 'cuda'}")
    print("=" * 70)

    # Apply toggles early
    apply_toggles(args)

    # Load model
    t0 = time.time()
    model, config, rev_word_map, device = load_model(args)
    print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")

    # Get page image and find tables via layout
    t0 = time.time()
    page_img, table_bboxes = run_layout_and_get_tables(args.pdf, args.page)
    print(f"[LAYOUT] Done in {time.time()-t0:.1f}s")

    if not table_bboxes:
        print("[ERROR] No tables found on this page!")
        return

    # Select target table
    target_idx = args.table_idx if args.table_idx >= 0 else 0
    if len(table_bboxes) > 1 and args.table_idx < 0:
        # Auto-select: pick the table that's most likely "Other Assets"
        # Usually the larger table or the one lower on the page
        print(f"[AUTO] Multiple tables found, using index {target_idx}")

    target_bbox = table_bboxes[target_idx]
    print(f"\n[TARGET] Table {target_idx}: bbox={[round(x,1) for x in target_bbox]}")

    # Crop table
    H, W = page_img.shape[:2]
    scale_factor = 1024.0 / float(H)
    x1, y1, x2, y2 = target_bbox
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(W, int(round(x2))), min(H, int(round(y2)))
    crop = page_img[iy1:iy2, ix1:ix2]

    tw = max(1, int(round((x2 - x1) * scale_factor)))
    th = max(1, int(round((y2 - y1) * scale_factor)))
    resized_crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    print(f"[CROP] Original: {crop.shape}, Resized: {resized_crop.shape}")

    if args.save_crops:
        os.makedirs("/tmp/table_debug", exist_ok=True)
        cv2.imwrite("/tmp/table_debug/crop_original.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/tmp/table_debug/crop_resized.png", cv2.cvtColor(resized_crop, cv2.COLOR_RGB2BGR))
        print("[SAVE] Crops saved to /tmp/table_debug/")

    # Preprocess
    table_images = [resized_crop] * args.repeat

    if args.stock_preprocess:
        print("\n[PREPROCESS] Using STOCK path")
        tensors = [preprocess_table_stock(img, config, device) for img in table_images]
        image_batch = torch.cat(tensors, dim=0)
    else:
        print("\n[PREPROCESS] Using OUR GPU path")
        image_batch = preprocess_table_ours(table_images, config, device)

    # Always compare both paths
    stock_tensor = preprocess_table_stock(resized_crop, config, device)
    ours_tensor = preprocess_table_ours([resized_crop], config, device)
    max_diff = compare_tensors("Stock vs Ours preprocessing", stock_tensor, ours_tensor)

    # Handle batch padding / positioning
    B = image_batch.size(0)
    if args.batch_pos >= 0:
        # Pad batch so target is at specified position
        pad_size = max(args.batch_pos + B, B + 4)
        padded = torch.zeros(pad_size, *image_batch.shape[1:], device=device, dtype=image_batch.dtype)
        # Fill with the target image (or could use random noise)
        for i in range(pad_size):
            padded[i] = image_batch[0]  # Fill all with target for now
        padded[args.batch_pos:args.batch_pos + B] = image_batch
        image_batch = padded
        print(f"\n[BATCH] Padded to {image_batch.shape[0]} tables, target at position {args.batch_pos}")

    if args.batch_size > 0 and image_batch.size(0) != args.batch_size:
        # Resize batch to requested size
        current = image_batch.size(0)
        if args.batch_size > current:
            # Repeat to fill
            repeats = (args.batch_size + current - 1) // current
            image_batch = image_batch.repeat(repeats, 1, 1, 1)[:args.batch_size]
        else:
            image_batch = image_batch[:args.batch_size]
        print(f"[BATCH] Adjusted to batch_size={image_batch.size(0)}")

    print(f"\n[INFERENCE] Batch shape: {image_batch.shape}")

    # Run inference
    t0 = time.time()
    results = run_inference(model, image_batch, config, args, device, rev_word_map)
    elapsed = time.time() - t0
    print(f"[INFERENCE] Done in {elapsed:.3f}s")

    # Show results for target table
    target_result_idx = args.batch_pos if args.batch_pos >= 0 else 0
    if target_result_idx >= len(results):
        target_result_idx = 0

    result = results[target_result_idx]

    print(f"\n{'='*70}")
    print(f"RESULTS (table at batch position {target_result_idx})")
    print(f"{'='*70}")
    print(f"Tag sequence length: {len(result['tags'])}")
    print(f"Bbox count: {result['n_bboxes']}")
    print(f"\nrs_seq ({len(result['rs_seq'])} tags):")

    # Pretty print the table structure
    row = []
    rows = []
    for tag in result['rs_seq']:
        if tag == 'nl':
            rows.append(row)
            row = []
        else:
            row.append(tag)
    if row:
        rows.append(row)

    print(f"\nTable structure: {len(rows)} rows")
    for i, r in enumerate(rows):
        print(f"  Row {i}: {r}")

    # Check for the specific bug pattern
    rs_str = " ".join(result['rs_seq'])
    print(f"\nFull rs_seq: {rs_str}")

    # Dump raw bboxes
    print(f"\nRaw bboxes (xyxy) [{result['n_bboxes']}]:")
    for i, bb in enumerate(result['bboxes_xyxy']):
        print(f"  [{i:2d}] [{bb[0]:.6f}, {bb[1]:.6f}, {bb[2]:.6f}, {bb[3]:.6f}]")

    # Save bboxes to JSON for diffing
    import json as _json
    bbox_file = f"/tmp/table_debug/bboxes_{'cpu' if args.cpu else 'gpu'}.json"
    os.makedirs("/tmp/table_debug", exist_ok=True)
    with open(bbox_file, "w") as f:
        _json.dump({
            "device": "cpu" if args.cpu else "cuda",
            "rs_seq": result['rs_seq'],
            "bboxes_xyxy": result['bboxes_xyxy'],
            "bboxes_cxcywh": result['bboxes_cxcywh'],
        }, f, indent=2)
    print(f"\n[SAVE] Bboxes saved to {bbox_file}")

    # Check all results if batch > 1
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"BATCH COMPARISON ({len(results)} results)")
        print(f"{'='*70}")
        ref_rs = " ".join(results[0]['rs_seq'])
        all_same = True
        for i, r in enumerate(results):
            rs = " ".join(r['rs_seq'])
            match = "SAME" if rs == ref_rs else "DIFFERENT"
            if rs != ref_rs:
                all_same = False
            print(f"  [{i}] {match} — {len(r['tags'])} tags, {r['n_bboxes']} bboxes")

        if all_same:
            print("\n  All results IDENTICAL across batch positions.")
        else:
            print("\n  WARNING: Results DIFFER by batch position!")
            print("  This implicates the batched decoder.")

    # Environment info
    print(f"\n{'='*70}")
    print("ENVIRONMENT")
    print(f"{'='*70}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cudnn: {torch.backends.cudnn.allow_tf32}")


if __name__ == "__main__":
    main()
