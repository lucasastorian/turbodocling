"""
MLX-native tag decoder for TableFormer.

Faithful port of the PyTorch batched AR decoder that keeps the entire
decode loop on-device. Same algorithm, same KV cache semantics, same
emit/skip logic — just no Python dispatch per step.

Usage:
    from processor.gpu_service.mlx_tag_decoder import MLXTagDecoder
    decoder = MLXTagDecoder.from_torch_model(torch_model, device='mps')
    tag_ids, tag_hidden, lengths = decoder.decode(encoder_memory, max_steps=200)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch


@dataclass
class MLXDecodeResult:
    """Structured result from the MLX tag decoder.

    Replaces the previous _last_* side-channel state on the decoder.
    """
    tag_ids: mx.array           # [B, T+1] with <start> at [:,0]
    tag_hidden: mx.array        # [B, max_emit, D] captured hidden states
    lengths: np.ndarray         # [B] real decoded length per sample
    h_counts: np.ndarray        # [B] number of hidden states captured per sample
    span_starts: List[List[int]]  # per-sample list of span start indices (into tag_hidden)
    span_ends: List[List[int]]    # per-sample list of span end indices


def _torch_to_mlx(t: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to MLX array via numpy."""
    return mx.array(t.detach().float().cpu().numpy())


def _torch_to_mlx_bf16(t: torch.Tensor) -> mx.array:
    """Convert to MLX bfloat16."""
    return mx.array(t.detach().float().cpu().numpy()).astype(mx.bfloat16)


def _mlx_to_torch(a: mx.array, device: str = "mps", dtype=torch.bfloat16) -> torch.Tensor:
    """Convert MLX array back to PyTorch tensor."""
    return torch.from_numpy(np.array(a, copy=False).astype(np.float32)).to(dtype=dtype, device=device)


class MLXDecoderLayer:
    """One transformer decoder layer with self-attn KV cache + cross-attn."""

    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Weights will be set by from_torch_layer()
        # Self-attention
        self.sa_qkv_weight = None  # [3E, E]
        self.sa_qkv_bias = None  # [3E]
        self.sa_out_weight = None  # [E, E]
        self.sa_out_bias = None  # [E]
        self.norm1_weight = None
        self.norm1_bias = None

        # Cross-attention
        self.ca_q_weight = None  # [E, E]
        self.ca_q_bias = None
        self.ca_out_weight = None  # [E, E]
        self.ca_out_bias = None
        self.norm2_weight = None
        self.norm2_bias = None

        # FFN
        self.ffn_w1 = None  # [4E, E]
        self.ffn_b1 = None
        self.ffn_w2 = None  # [E, 4E]
        self.ffn_b2 = None
        self.norm3_weight = None
        self.norm3_bias = None

    @classmethod
    def from_torch_layer(cls, layer, embed_dim: int, num_heads: int, use_fp32: bool = True) -> "MLXDecoderLayer":
        """Extract weights from a PyTorch TMTransformerDecoderLayer."""
        obj = cls(embed_dim, num_heads)
        E = embed_dim
        # Use fp32 for MLX decoder to match PyTorch MPS parity.
        # bf16 was tuned for CUDA Flash Attention which MLX doesn't have.
        convert = _torch_to_mlx if use_fp32 else _torch_to_mlx_bf16

        # Self-attention (fused QKV)
        mha = layer.self_attn
        obj.sa_qkv_weight = convert(mha.in_proj_weight)
        obj.sa_qkv_bias = convert(mha.in_proj_bias)
        obj.sa_out_weight = convert(mha.out_proj.weight)
        obj.sa_out_bias = convert(mha.out_proj.bias)

        # Norm1
        obj.norm1_weight = convert(layer.norm1.weight)
        obj.norm1_bias = convert(layer.norm1.bias)

        # Cross-attention
        ca = layer.multihead_attn
        W = ca.in_proj_weight
        b = ca.in_proj_bias
        obj.ca_q_weight = convert(W[:E, :])
        obj.ca_q_bias = convert(b[:E]) if b is not None else None
        obj.ca_k_weight = convert(W[E:2*E, :])
        obj.ca_k_bias = convert(b[E:2*E]) if b is not None else None
        obj.ca_v_weight = convert(W[2*E:, :])
        obj.ca_v_bias = convert(b[2*E:]) if b is not None else None
        obj.ca_out_weight = convert(ca.out_proj.weight)
        obj.ca_out_bias = convert(ca.out_proj.bias)

        # Norm2
        obj.norm2_weight = convert(layer.norm2.weight)
        obj.norm2_bias = convert(layer.norm2.bias)

        # FFN
        obj.ffn_w1 = convert(layer.linear1.weight)
        obj.ffn_b1 = convert(layer.linear1.bias)
        obj.ffn_w2 = convert(layer.linear2.weight)
        obj.ffn_b2 = convert(layer.linear2.bias)

        # Norm3
        obj.norm3_weight = convert(layer.norm3.weight)
        obj.norm3_bias = convert(layer.norm3.bias)

        return obj

    def __call__(
        self,
        x: mx.array,           # [B, D] current token
        sa_k_cache: mx.array,   # [B, H, T, Dh]
        sa_v_cache: mx.array,   # [B, H, T, Dh]
        t: int,                 # current step
        mem_k: mx.array,        # [B, H, S, Dh] precomputed
        mem_v: mx.array,        # [B, H, S, Dh] precomputed
    ):
        E = self.embed_dim
        H = self.num_heads
        Dh = self.head_dim
        B = x.shape[0]

        # ── Self-attention with KV cache ──
        qkv = x @ self.sa_qkv_weight.T + self.sa_qkv_bias  # [B, 3E]
        q, k, v = mx.split(qkv, 3, axis=-1)  # each [B, E]

        q = q.reshape(B, H, 1, Dh)
        k = k.reshape(B, H, 1, Dh)
        v = v.reshape(B, H, 1, Dh)

        # Write to cache at position t
        sa_k_cache[:, :, t:t+1, :] = k
        sa_v_cache[:, :, t:t+1, :] = v

        # Attend over cached keys/values [0..t]
        k_prefix = sa_k_cache[:, :, :t+1, :]  # [B, H, t+1, Dh]
        v_prefix = sa_v_cache[:, :, :t+1, :]

        scale = math.sqrt(Dh)
        attn = (q @ mx.transpose(k_prefix, (0, 1, 3, 2))) / scale  # [B, H, 1, t+1]
        attn = mx.softmax(attn, axis=-1)
        sa_out = attn @ v_prefix  # [B, H, 1, Dh]
        sa_out = sa_out.reshape(B, E)

        # Out projection + residual + norm
        sa_out = sa_out @ self.sa_out_weight.T + self.sa_out_bias
        x = _layer_norm(x + sa_out, self.norm1_weight, self.norm1_bias)

        # ── Cross-attention (precomputed K/V) ──
        q_ca = (x @ self.ca_q_weight.T + self.ca_q_bias).reshape(B, H, 1, Dh)

        attn_ca = (q_ca @ mx.transpose(mem_k, (0, 1, 3, 2))) / scale
        attn_ca = mx.softmax(attn_ca, axis=-1)
        ca_out = attn_ca @ mem_v  # [B, H, 1, Dh]
        ca_out = ca_out.reshape(B, E)

        ca_out = ca_out @ self.ca_out_weight.T + self.ca_out_bias
        x = _layer_norm(x + ca_out, self.norm2_weight, self.norm2_bias)

        # ── FFN ──
        ff = mx.maximum(x @ self.ffn_w1.T + self.ffn_b1, 0)  # ReLU
        ff = ff @ self.ffn_w2.T + self.ffn_b2
        x = _layer_norm(x + ff, self.norm3_weight, self.norm3_bias)

        return x, sa_k_cache, sa_v_cache


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    """LayerNorm matching PyTorch semantics."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias


class MLXTagDecoder:
    """
    MLX-native autoregressive tag decoder for TableFormer.

    Heavy math (attention, MLP, logits) runs on MLX.
    Control logic (structure corrections, emit bookkeeping, span tracking)
    stays on the host in numpy — it's tiny and easier to keep faithful
    to the PyTorch batched decoder.

    Experiments that did NOT help (removed from code, kept here for history):
    - Full MLX transformer encoder: marginally faster (~3%), parity-sensitive.
      The PyTorch transformer encoder is only ~5% of TableFormer time and
      parity is rock solid through that boundary.
    - mx.compile on layer __call__: slower (8.0s vs 6.0s for 44 tables)
      because the layer takes an integer `t` that changes every step, likely
      causing retrace. A proper compile would need stacked [L,...] caches,
      t as a tensor input, and mask-based attention. Not worth the refactor.
    """

    def __init__(
        self,
        layers: List[MLXDecoderLayer],
        embedding_weight: mx.array,
        pe: mx.array,
        fc_weight: mx.array,
        fc_bias: mx.array,
        word_map: Dict[str, int],
        num_heads: int,
        embed_dim: int,
    ):
        self.layers = layers
        self.embedding_weight = embedding_weight
        self.pe = pe
        self.fc_weight = fc_weight
        self.fc_bias = fc_bias
        self.word_map = word_map
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.n_layers = len(layers)

        # Token IDs
        self.start_id = word_map["<start>"]
        self.end_id = word_map["<end>"]
        self.xcel_id = word_map.get("xcel", -1)
        self.lcel_id = word_map.get("lcel", -1)
        self.fcel_id = word_map.get("fcel", -1)
        self.ucel_id = word_map.get("ucel", -1)
        self.nl_id = word_map.get("nl", -1)

        # Build emit/skip LUTs
        V = embedding_weight.shape[0]
        emit_names = ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]
        skip_names = ["nl", "ucel", "xcel"]
        self.emit_lut = np.zeros(V, dtype=np.bool_)
        self.skip_lut = np.zeros(V, dtype=np.bool_)
        for name in emit_names:
            if name in word_map:
                self.emit_lut[word_map[name]] = True
        for name in skip_names:
            if name in word_map:
                self.skip_lut[word_map[name]] = True

    @classmethod
    def from_torch_model(cls, model) -> "MLXTagDecoder":
        """Build from a PyTorch TableModel04_rs."""
        tt = model._tag_transformer
        wm = model._init_data["word_map"]["word_map_tag"]

        E = tt._decoder_dim
        H = tt._decoder.layers[0].self_attn.num_heads

        layers = []
        for torch_layer in tt._decoder.layers:
            layers.append(MLXDecoderLayer.from_torch_layer(torch_layer, E, H))

        # Embedding (fp32 for parity)
        emb_weight = _torch_to_mlx(tt._embedding.weight)

        # Positional encoding
        pe = _torch_to_mlx(tt._positional_encoding.pe.squeeze(1))  # [max_len, D]

        # FC head
        fc_weight = _torch_to_mlx(tt._fc.weight)
        fc_bias = _torch_to_mlx(tt._fc.bias)

        return cls(
            layers=layers,
            embedding_weight=emb_weight,
            pe=pe,
            fc_weight=fc_weight,
            fc_bias=fc_bias,
            word_map=wm,
            num_heads=H,
            embed_dim=E,
        )

    def _precompute_mem_kv_mlx(self, mem_enc: mx.array) -> List[Tuple[mx.array, mx.array]]:
        """
        Precompute cross-attention K/V from encoder memory (already on MLX).
        mem_enc: [S, B, D].
        Returns: list of (K_mem, V_mem) per layer, each [B, H, S, Dh].
        """
        # [S, B, D] -> [B, S, D]
        mem = mem_enc.transpose(1, 0, 2)
        B, S, D = mem.shape
        H = self.num_heads
        Dh = self.head_dim

        mem_flat = mem.reshape(B * S, D)
        mem_kv = []
        for layer in self.layers:
            K = (mem_flat @ layer.ca_k_weight.T + layer.ca_k_bias).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            V = (mem_flat @ layer.ca_v_weight.T + layer.ca_v_bias).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            mem_kv.append((K, V))

        mx.eval(*[k for pair in mem_kv for k in pair])
        return mem_kv

    def decode(
        self,
        mem_enc_torch: torch.Tensor,  # [S, B, D] encoder memory from PyTorch
        max_steps: int,
    ) -> MLXDecodeResult:
        """
        Decode-only path (encoder already run on PyTorch).
        Converts mem_enc once and runs the AR decode on MLX.
        """
        mem_enc = _torch_to_mlx(mem_enc_torch)
        return self._decode_from_mem(mem_enc, max_steps)

    def _decode_from_mem(
        self,
        mem_enc: mx.array,  # [S, B, D] already on MLX
        max_steps: int,
    ) -> MLXDecodeResult:
        """Core decode loop. mem_enc is already on MLX."""
        mem_kv = self._precompute_mem_kv_mlx(mem_enc)

        B = mem_kv[0][0].shape[0]
        H = self.num_heads
        Dh = self.head_dim
        D = self.embed_dim

        # Allocate all buffers
        tag_ids = mx.full((B, max_steps + 1), self.end_id, dtype=mx.int32)
        tag_ids[:, 0] = self.start_id

        # KV caches per layer: [B, H, max_steps, Dh] — fp32 for parity
        sa_k_caches = [mx.zeros((B, H, max_steps + 1, Dh), dtype=mx.float32) for _ in range(self.n_layers)]
        sa_v_caches = [mx.zeros((B, H, max_steps + 1, Dh), dtype=mx.float32) for _ in range(self.n_layers)]

        # Hidden state buffer for bbox emission
        tag_H_buffer = mx.zeros((B, max_steps, D), dtype=mx.float32)
        tag_H_counts = mx.zeros((B,), dtype=mx.int32)

        # All control state on host (numpy) — tiny, not worth device ops.
        # The heavy work (attention, MLP, logits) stays on MLX.
        finished = np.zeros(B, dtype=np.bool_)
        skip_next = np.ones(B, dtype=np.bool_)
        first_lcel = np.ones(B, dtype=np.bool_)
        prev_ucel = np.zeros(B, dtype=np.bool_)
        lengths = np.zeros(B, dtype=np.int32)
        h_counts = np.zeros(B, dtype=np.int32)  # host-side bbox emission counter

        # Span tracking for lcel merge (matches torch batched decoder)
        open_span_start = np.full(B, -1, dtype=np.int32)
        span_starts_list = [[] for _ in range(B)]
        span_ends_list = [[] for _ in range(B)]

        # Track token IDs on host too (tiny)
        tag_ids_np = np.full((B, max_steps + 1), self.end_id, dtype=np.int32)
        tag_ids_np[:, 0] = self.start_id

        # Decode loop
        for step in range(max_steps):
            # Embed + positional encoding (MLX)
            cur_tokens = mx.array(tag_ids_np[:, step])  # [B] from host
            emb = self.embedding_weight[cur_tokens]  # [B, D]
            pos = self.pe[step]  # [D]
            x = emb + pos

            # Run through decoder layers (MLX — the heavy work)
            for i in range(self.n_layers):
                x, sa_k_caches[i], sa_v_caches[i] = self.layers[i](
                    x, sa_k_caches[i], sa_v_caches[i], step,
                    mem_kv[i][0], mem_kv[i][1],
                )

            # Logits + argmax (MLX)
            logits = x @ self.fc_weight.T + self.fc_bias  # [B, V]
            new_tags_mlx = mx.argmax(logits, axis=-1)  # [B]

            # Single sync per step: pull token IDs to host
            mx.eval(new_tags_mlx)
            new_tags = np.array(new_tags_mlx, dtype=np.int32)

            # ── Structure corrections (numpy, tiny) ──
            if self.xcel_id >= 0 and self.lcel_id >= 0:
                new_tags[new_tags == self.xcel_id] = self.lcel_id

            if self.ucel_id >= 0 and self.lcel_id >= 0 and self.fcel_id >= 0:
                mask = prev_ucel & (new_tags == self.lcel_id)
                new_tags[mask] = self.fcel_id

            new_tags[finished] = self.end_id

            # Write to host output
            t = step + 1
            tag_ids_np[:, t] = new_tags

            # Update lengths for active sequences
            active = ~finished
            lengths[active] += 1

            # Bbox emission decisions (numpy — tiny boolean ops)
            m_emit = (~skip_next) & self.emit_lut[new_tags] & (~finished)
            m_is_lcel = (new_tags == self.lcel_id) if self.lcel_id >= 0 else np.zeros(B, dtype=np.bool_)
            m_first_lcel = first_lcel & m_is_lcel & (~finished)
            append_mask = m_emit | m_first_lcel

            # Store hidden states for bbox + track spans
            emit_indices = np.where(append_mask)[0]
            if len(emit_indices) > 0:
                for b_idx in emit_indices:
                    c = int(h_counts[b_idx])
                    bi = int(b_idx)
                    if c < max_steps:
                        tag_H_buffer = tag_H_buffer.at[bi, c, :].add(x[bi])

                        # Span tracking (matches torch batched decoder)
                        is_first_lcel_here = bool(m_first_lcel[b_idx])
                        is_emit_here = bool(m_emit[b_idx])

                        if is_first_lcel_here:
                            open_span_start[b_idx] = c
                        elif is_emit_here and not m_is_lcel[b_idx] and open_span_start[b_idx] >= 0:
                            span_starts_list[b_idx].append(int(open_span_start[b_idx]))
                            span_ends_list[b_idx].append(c)
                            open_span_start[b_idx] = -1

                        h_counts[b_idx] = c + 1

            # Update flags
            first_lcel = np.where(m_is_lcel, False, True)
            skip_next = self.skip_lut[new_tags]
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id >= 0 else np.zeros(B, dtype=np.bool_)

            # Check finished
            newly_finished = (new_tags == self.end_id)
            finished = finished | newly_finished
            if finished.all():
                break

        # Bounds assertion: captured hidden count should never exceed buffer size
        assert (h_counts <= max_steps).all(), \
            f"h_counts overflow: max={h_counts.max()} > {max_steps}"

        return MLXDecodeResult(
            tag_ids=mx.array(tag_ids_np),
            tag_hidden=tag_H_buffer,
            lengths=lengths,
            h_counts=h_counts,
            span_starts=span_starts_list,
            span_ends=span_ends_list,
        )
