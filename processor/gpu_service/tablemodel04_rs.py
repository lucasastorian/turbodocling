#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import logging
import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel

from processor.shared.timers import _CPUTimer, _CudaTimer
from processor.gpu_service.encoder04_rs import Encoder04
from processor.gpu_service.bbox_decoder_rs import BBoxDecoder
from processor.gpu_service.transformer_rs import Tag_Transformer
from processor.gpu_service.batched_decoder import BatchedTableDecoder

LOG_LEVEL = logging.WARN


# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    TableNet04Model encoder, dual-decoder model with OTSL+ support
    """

    _prof: bool = False

    def __init__(self, config, init_data, device):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._device = device
        # Extract the word_map from the init_data
        word_map = init_data["word_map"]

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder = Encoder04(self._enc_image_size).to(device)
        # CRITICAL: use config hidden_dim (512) to match checkpoint bbox decoder shapes
        self._encoder_dim = config["model"]["hidden_dim"]  # 512, not actual encoder output (256)

        tag_vocab_size = len(word_map["word_map_tag"])

        td_encode = []
        for t in ["ecel", "fcel", "ched", "rhed", "srow"]:
            if t in word_map["word_map_tag"]:
                td_encode.append(word_map["word_map_tag"][t])
        self._log().debug("td_encode length: {}".format(len(td_encode)))
        self._log().debug("td_encode: {}".format(td_encode))

        self._tag_attention_dim = config["model"]["tag_attention_dim"]
        self._tag_embed_dim = config["model"]["tag_embed_dim"]
        self._tag_decoder_dim = config["model"]["tag_decoder_dim"]
        self._decoder_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]

        self._bbox = config["train"]["bbox"]
        self._bbox_attention_dim = config["model"]["bbox_attention_dim"]
        self._bbox_embed_dim = config["model"]["bbox_embed_dim"]
        self._bbox_decoder_dim = config["model"]["hidden_dim"]

        self._enc_layers = config["model"]["enc_layers"]
        self._dec_layers = config["model"]["dec_layers"]
        self._n_heads = config["model"]["nheads"]

        self._num_classes = config["model"]["bbox_classes"]
        self._enc_image_size = config["model"]["enc_image_size"]

        self._max_pred_len = config["predict"]["max_steps"]

        self._tag_transformer = Tag_Transformer(
            device,
            tag_vocab_size,
            td_encode,
            self._decoder_dim,
            self._enc_layers,
            self._dec_layers,
            self._enc_image_size,
            n_heads=self._n_heads,
        ).to(device)

        self._bbox_decoder = BBoxDecoder(
            device,
            self._bbox_attention_dim,
            self._bbox_embed_dim,
            self._tag_decoder_dim,
            self._bbox_decoder_dim,
            self._num_classes,
            self._encoder_dim,
            self._dropout,
        ).to(device)

        # Stage 2: Cache tag IDs as device tensors (avoid dict hits + reallocs in loop)
        wm_tag = word_map["word_map_tag"]
        self._ids = {k: torch.tensor(v, device=device, dtype=torch.long)
                     for k, v in wm_tag.items() if isinstance(v, int)}

        # Sets for quick membership tests in the loop
        _emit_names = ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]
        self._emit_ids = torch.stack([self._ids[n] for n in _emit_names if n in self._ids]) \
            if any(n in self._ids for n in _emit_names) else torch.empty(0, dtype=torch.long, device=device)

        _skip_names = ["nl", "ucel", "xcel"]
        self._skip_ids = torch.stack([self._ids[n] for n in _skip_names if n in self._ids]) \
            if any(n in self._ids for n in _skip_names) else torch.empty(0, dtype=torch.long, device=device)

        self._batched_decoder = BatchedTableDecoder(self, self._device)

        # Enable fast kernels where safe
        if device == 'cuda':
            # Don't enable benchmark globally - it interferes with CUDA Graphs!
            # torch.backends.cudnn.benchmark = True  # REMOVED - causes slowdown with graphs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")  # Ampere+ only

        # Optimization 8: Optionally disable gradients globally for inference
        if os.getenv("DISABLE_GRAD", "1") == "1":
            torch.set_grad_enabled(False)

        # Block size for encoder batching
        self._encoder_block_bs = int(os.getenv("ENCODER_BLOCK_BS", "32"))

        # Don't convert to bf16 here - wait until after checkpoint is loaded

    def setup_for_inference(self):
        """Call this AFTER loading checkpoint to prepare model for optimized inference"""
        self._convert_transformers_to_bf16()

        self.eval()
        torch.set_grad_enabled(False)

        return self

    def _convert_transformers_to_bf16(self):
        """Convert ONLY tag transformer to bf16 for Flash Attention; keep encoder and bbox decoder in FP32"""

        def _to_bf16(m):
            if isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.Embedding, nn.LayerNorm)):
                m.to(torch.bfloat16)
            return m

        # Apply bf16 ONLY to tag transformer, NOT the encoder or bbox decoder
        self._tag_transformer.apply(_to_bf16)

        # Ensure positional encoding is also bf16
        if hasattr(self._tag_transformer._positional_encoding, 'pe'):
            pe = self._tag_transformer._positional_encoding.pe
            self._tag_transformer._positional_encoding.pe = pe.to(torch.bfloat16)

        # Explicitly ensure bbox decoder stays in FP32 (in case it was accidentally converted)
        self._bbox_decoder.to(torch.float32)

    def _encode_in_blocks(self, imgs: torch.Tensor, block_bs: int = 32) -> torch.Tensor:
        """
        Simple batched encoding without graphs or compilation.
        Just ensures channels_last format and runs through encoder.
        """
        B0, C, H, W = imgs.shape
        device = imgs.device

        # Ensure channels_last format for optimal performance
        imgs_cl = imgs if imgs.is_contiguous(memory_format=torch.channels_last) \
            else imgs.contiguous(memory_format=torch.channels_last)

        # Simply run through encoder - no graphs, no compilation
        return self._encoder(imgs_cl)

    @torch.inference_mode()
    def predict(self, imgs, max_steps, k, return_attention=False):
        """
        Stage 3: batched encoder + batched AR decoder with dynamic batching.
        imgs: [B,3,448,448]
        returns: list of (seq, outputs_class, outputs_coord)
        """
        B = imgs.size(0)
        if B == 0:
            return []
        self.eval()  # Set entire model to eval mode
        self._encoder.eval()
        self._tag_transformer.eval()
        self._bbox_decoder.eval()

        # Dynamic batching with maximum batch size of 128
        MAX_BATCH_SIZE = 128
        
        # If batch size is within limit, process normally
        if B <= MAX_BATCH_SIZE:
            return self._predict_batch(imgs, max_steps, k, return_attention)
        
        # Otherwise, process in chunks
        all_results = []
        for i in range(0, B, MAX_BATCH_SIZE):
            batch_end = min(i + MAX_BATCH_SIZE, B)
            batch_imgs = imgs[i:batch_end]
            batch_results = self._predict_batch(batch_imgs, max_steps, k, return_attention)
            all_results.extend(batch_results)
        
        return all_results
    
    @torch.inference_mode()
    def _predict_batch(self, imgs, max_steps, k, return_attention=False):
        """
        Process a single batch of images.
        imgs: [B,3,448,448] where B <= MAX_BATCH_SIZE
        returns: list of (seq, outputs_class, outputs_coord)
        """
        B = imgs.size(0)
        
        # Use proper timer based on device
        is_cuda = str(self._device).startswith('cuda')
        timer = _CudaTimer() if is_cuda else _CPUTimer()

        # ===== ENCODER TIMING =====
        with timer.time_section('encoder_forward'):
            enc_out_batch = self._encode_in_blocks(imgs,
                                                   block_bs=self._encoder_block_bs)  # [B,C,H,W] - NCHW format, FP32

        # ===== MEMORY PREPARATION =====
        with timer.time_section('tag_input_filter'):
            # Keep in FP32 for CNN processing
            filtered_nchw = self._tag_transformer._input_filter(enc_out_batch)  # [B,C,h,w] NCHW, FP32

        # ===== CHECK MLX PATH =====
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # MLX decode: 1.3x faster TableFormer inference, 100% parity.
        # Set TURBODOCLING_MLX_DECODE=0 to disable.
        use_mlx = (not is_cuda
                   and os.environ.get("TURBODOCLING_MLX_DECODE", "1") == "1"
                   and str(self._device) in ("mps", "cpu"))
        if use_mlx:
            try:
                from processor.gpu_service.mlx_tag_decoder import MLXTagDecoder
                import mlx.core as mx
                import numpy as np
            except ImportError:
                use_mlx = False

        if use_mlx:
            if not hasattr(self, '_mlx_decoder'):
                self._mlx_decoder = MLXTagDecoder.from_torch_model(self)

            # MLX decode-only: PyTorch runs the transformer encoder,
            # MLX runs the AR tag decode. ~1.9x faster than PyTorch MPS
            # with 100% parity on the golden corpus.
            with timer.time_section('mlx_memory_reshape'):
                filtered_nhwc = filtered_nchw.permute(0, 2, 3, 1)
                B_, h, w, C = filtered_nhwc.shape
                mem = filtered_nhwc.reshape(B_, h * w, C).permute(1, 0, 2).contiguous()
                mem = mem.to(torch.bfloat16)

            with timer.time_section('mlx_tag_encoder'):
                mem_enc = self._tag_transformer._encoder(mem, mask=None)

            with timer.time_section('mlx_decode'):
                mlx_result = self._mlx_decoder.decode(mem_enc, max_steps=max_steps)
                mx.eval(mlx_result.tag_ids, mlx_result.tag_hidden)

            # Convert MLX result back to (seq, cls_logits, coords) tuples
            # matching BatchedTableDecoder.predict_batched's format.
            import numpy as np
            tag_ids_np = np.array(mlx_result.tag_ids.astype(mx.int32))
            end_id = self._mlx_decoder.end_id
            start_id = self._mlx_decoder.start_id

            results = []
            for b in range(B):
                seq = [start_id]
                for tid in tag_ids_np[b, 1:]:
                    t_int = int(tid)
                    seq.append(t_int)
                    if t_int == end_id:
                        break

                count = int(mlx_result.h_counts[b])
                assert count <= mlx_result.tag_hidden.shape[1], \
                    f"h_count {count} exceeds tag_hidden buffer {mlx_result.tag_hidden.shape[1]}"

                if self._bbox and count > 0:
                    tag_H_np = np.array(mlx_result.tag_hidden[b, :count].astype(mx.float32))
                    tag_H_tensor = torch.from_numpy(tag_H_np).to(dtype=torch.bfloat16, device=self._device)
                    enc_nchw = enc_out_batch[b:b+1]
                    cls_logits, coords = self._bbox_decoder.inference(enc_nchw, tag_H_tensor)

                    # Apply span merging (matches torch batched decoder)
                    span_s = mlx_result.span_starts[b]
                    span_e = mlx_result.span_ends[b]
                    if span_s and len(coords) > 0:
                        Tmax = max_steps
                        starts_t = torch.full((Tmax,), -1, dtype=torch.long, device=self._device)
                        ends_t = torch.full((Tmax,), -1, dtype=torch.long, device=self._device)
                        for si, (s, e) in enumerate(zip(span_s, span_e)):
                            if si < Tmax:
                                assert 0 <= s < count and 0 <= e < count, \
                                    f"span ({s}, {e}) out of range for count={count}"
                                starts_t[si] = s
                                ends_t[si] = e
                        count_t = torch.tensor(len(span_s), device=self._device)
                        cls_logits, coords = self._batched_decoder._merge_spans_gpu(
                            cls_logits, coords, starts_t, ends_t, count_t
                        )
                else:
                    cls_logits = torch.empty(0, device=self._device)
                    coords = torch.empty(0, device=self._device)

                results.append((seq, cls_logits, coords))
        else:
            # Torch path: run transformer encoder + decoder on PyTorch
            with timer.time_section('memory_reshape'):
                filtered_nhwc = filtered_nchw.permute(0, 2, 3, 1)
                B_, h, w, C = filtered_nhwc.shape
                mem = filtered_nhwc.reshape(B_, h * w, C).permute(1, 0, 2).contiguous()
                mem = mem.to(torch.bfloat16)

            with timer.time_section('tag_encoder'):
                mem_enc = self._tag_transformer._encoder(mem, mask=None)

            with timer.time_section('batched_ar_decoder'):
                if is_cuda:
                    with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
                        results = self._batched_decoder.predict_batched(enc_out_batch, mem_enc, max_steps)
                else:
                    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        results = self._batched_decoder.predict_batched(enc_out_batch, mem_enc, max_steps)

        # Finalize and print timing if profiling enabled
        if self._prof:
            timer.finalize()
            print(f"\n=== TableModel Timing (B={B}) ===")
            print(f"  encoder_forward:    {timer.get_time('encoder_forward'):7.1f} ms")
            print(f"  tag_input_filter:   {timer.get_time('tag_input_filter'):7.1f} ms")

            if use_mlx:
                print(f"  mlx_memory_reshape: {timer.get_time('mlx_memory_reshape'):7.1f} ms")
                print(f"  mlx_tag_encoder:    {timer.get_time('mlx_tag_encoder'):7.1f} ms")
                print(f"  mlx_decode:         {timer.get_time('mlx_decode'):7.1f} ms")
                total_time = (timer.get_time('encoder_forward') +
                              timer.get_time('tag_input_filter') +
                              timer.get_time('mlx_memory_reshape') +
                              timer.get_time('mlx_tag_encoder') +
                              timer.get_time('mlx_decode'))
            else:
                print(f"  memory_reshape:     {timer.get_time('memory_reshape'):7.1f} ms")
                print(f"  tag_encoder:        {timer.get_time('tag_encoder'):7.1f} ms")
                print(f"  batched_ar_decoder: {timer.get_time('batched_ar_decoder'):7.1f} ms")
                total_time = (timer.get_time('encoder_forward') +
                              timer.get_time('tag_input_filter') +
                              timer.get_time('memory_reshape') +
                              timer.get_time('tag_encoder') +
                              timer.get_time('batched_ar_decoder'))

                # Print decoder breakdown if available
                if timer.get_time('ar_loop') > 0:
                    print(f"\n  === Decoder Breakdown ===")
                    print(f"    ar_loop:          {timer.get_time('ar_loop'):7.1f} ms")
                    print(f"    bbox_decode:      {timer.get_time('bbox_decode'):7.1f} ms")

            print(f"  ---------------------------")
            print(f"  TOTAL:              {total_time:7.1f} ms\n")

        return results

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def _cxcywh_to_xyxy(self, b: torch.Tensor) -> torch.Tensor:
        """Convert from center format to corner format
        Args:
            b: [..., 4] tensor with (cx, cy, w, h)
        Returns:
            [..., 4] tensor with (x1, y1, x2, y2)
        """
        cx, cy, w, h = b.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _xyxy_to_cxcywh(self, b: torch.Tensor) -> torch.Tensor:
        """Convert from corner format to center format
        Args:
            b: [..., 4] tensor with (x1, y1, x2, y2)
        Returns:
            [..., 4] tensor with (cx, cy, w, h)
        """
        x1, y1, x2, y2 = b.unbind(-1)
        w = (x2 - x1).clamp_min(1e-6)  # Prevent zero/negative widths
        h = (y2 - y1).clamp_min(1e-6)  # Prevent zero/negative heights
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return torch.stack((cx, cy, w, h), dim=-1)

    def mergebboxes(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        """Merge two bboxes (order-agnostic union)"""
        # Convert to corner format for proper min/max
        a = self._cxcywh_to_xyxy(bbox1)
        b = self._cxcywh_to_xyxy(bbox2)

        # Compute union (elementwise min/max)
        x1 = torch.minimum(a[0], b[0])
        y1 = torch.minimum(a[1], b[1])
        x2 = torch.maximum(a[2], b[2])
        y2 = torch.maximum(a[3], b[3])

        # Convert back to center format
        return self._xyxy_to_cxcywh(torch.stack((x1, y1, x2, y2)))

    def mergebboxes_batch(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
        """Batched merge of bbox pairs (order-agnostic union)
        Args:
            bboxes1, bboxes2: [N, 4] tensors in cxcywh format
        Returns:
            merged: [N, 4] tensor in cxcywh format
        """
        # Convert to corner format for proper min/max
        a = self._cxcywh_to_xyxy(bboxes1)
        b = self._cxcywh_to_xyxy(bboxes2)

        # Compute union (elementwise min/max)
        x1 = torch.minimum(a[..., 0], b[..., 0])
        y1 = torch.minimum(a[..., 1], b[..., 1])
        x2 = torch.maximum(a[..., 2], b[..., 2])
        y2 = torch.maximum(a[..., 3], b[..., 3])

        # Stack and convert back to center format
        merged_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
        return self._xyxy_to_cxcywh(merged_xyxy)

    def _flatten_hw_to_sbc(self, hwc):
        """Utility: [B,H,W,C] -> [S,B,C] for transformer"""
        B, H, W, C = hwc.shape
        return hwc.flatten(1, 2).permute(1, 0, 2)  # no .contiguous() - avoid copy
