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
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch


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
    def from_torch_layer(cls, layer, embed_dim: int, num_heads: int) -> "MLXDecoderLayer":
        """Extract weights from a PyTorch TMTransformerDecoderLayer."""
        obj = cls(embed_dim, num_heads)
        E = embed_dim

        # Self-attention (fused QKV)
        mha = layer.self_attn
        obj.sa_qkv_weight = _torch_to_mlx_bf16(mha.in_proj_weight)  # [3E, E]
        obj.sa_qkv_bias = _torch_to_mlx_bf16(mha.in_proj_bias)
        obj.sa_out_weight = _torch_to_mlx_bf16(mha.out_proj.weight)
        obj.sa_out_bias = _torch_to_mlx_bf16(mha.out_proj.bias)

        # Norm1
        obj.norm1_weight = _torch_to_mlx_bf16(layer.norm1.weight)
        obj.norm1_bias = _torch_to_mlx_bf16(layer.norm1.bias)

        # Cross-attention (store K/V projection weights too for precompute)
        ca = layer.multihead_attn
        W = ca.in_proj_weight
        b = ca.in_proj_bias
        obj.ca_q_weight = _torch_to_mlx_bf16(W[:E, :])
        obj.ca_q_bias = _torch_to_mlx_bf16(b[:E]) if b is not None else None
        obj.ca_k_weight = _torch_to_mlx_bf16(W[E:2*E, :])
        obj.ca_k_bias = _torch_to_mlx_bf16(b[E:2*E]) if b is not None else None
        obj.ca_v_weight = _torch_to_mlx_bf16(W[2*E:, :])
        obj.ca_v_bias = _torch_to_mlx_bf16(b[2*E:]) if b is not None else None
        obj.ca_out_weight = _torch_to_mlx_bf16(ca.out_proj.weight)
        obj.ca_out_bias = _torch_to_mlx_bf16(ca.out_proj.bias)

        # Norm2
        obj.norm2_weight = _torch_to_mlx_bf16(layer.norm2.weight)
        obj.norm2_bias = _torch_to_mlx_bf16(layer.norm2.bias)

        # FFN
        obj.ffn_w1 = _torch_to_mlx_bf16(layer.linear1.weight)
        obj.ffn_b1 = _torch_to_mlx_bf16(layer.linear1.bias)
        obj.ffn_w2 = _torch_to_mlx_bf16(layer.linear2.weight)
        obj.ffn_b2 = _torch_to_mlx_bf16(layer.linear2.bias)

        # Norm3
        obj.norm3_weight = _torch_to_mlx_bf16(layer.norm3.weight)
        obj.norm3_bias = _torch_to_mlx_bf16(layer.norm3.bias)

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
    """Full MLX tag decoder with device-native decode loop."""

    def __init__(
        self,
        layers: List[MLXDecoderLayer],
        embedding_weight: mx.array,   # [V, D]
        pe: mx.array,                 # [max_len, D]
        fc_weight: mx.array,          # [V, D]
        fc_bias: mx.array,            # [V]
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

        # Extract decoder layers
        layers = []
        E = tt._decoder_dim
        H = tt._decoder.layers[0].self_attn.num_heads
        for torch_layer in tt._decoder.layers:
            layers.append(MLXDecoderLayer.from_torch_layer(torch_layer, E, H))

        # Embedding
        emb_weight = _torch_to_mlx_bf16(tt._embedding.weight)

        # Positional encoding
        pe = _torch_to_mlx_bf16(tt._positional_encoding.pe.squeeze(1))  # [max_len, D]

        # FC head
        fc_weight = _torch_to_mlx_bf16(tt._fc.weight)
        fc_bias = _torch_to_mlx_bf16(tt._fc.bias)

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

    def precompute_mem_kv(self, mem_enc_torch: torch.Tensor) -> List[Tuple[mx.array, mx.array]]:
        """
        Precompute cross-attention K/V from encoder memory using MLX.
        mem_enc_torch: [S, B, D] from PyTorch encoder.
        Returns: list of (K_mem, V_mem) per layer, each [B, H, S, Dh].
        """
        # Convert [S, B, D] -> [B, S, D]
        mem = _torch_to_mlx_bf16(mem_enc_torch.transpose(0, 1))
        B, S, D = mem.shape
        H = self.num_heads
        Dh = self.head_dim

        mem_kv = []
        for layer in self.layers:
            K = (mem @ layer.ca_k_weight.T + layer.ca_k_bias).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            V = (mem @ layer.ca_v_weight.T + layer.ca_v_bias).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            mem_kv.append((K, V))  # each [B, H, S, Dh]

        mx.eval(*[k for pair in mem_kv for k in pair])  # Force compute
        return mem_kv

    def decode(
        self,
        mem_enc_torch: torch.Tensor,  # [S, B, D] encoder memory from PyTorch
        max_steps: int,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Run the full decode loop on MLX.

        Returns:
            tag_ids: [B, T] decoded token IDs
            tag_hidden: [B, max_emit, D] hidden states for bbox emission
            lengths: [B] actual sequence lengths
        """
        mem_kv = self.precompute_mem_kv(mem_enc_torch)

        B = mem_kv[0][0].shape[0]
        H = self.num_heads
        Dh = self.head_dim
        D = self.embed_dim

        # Allocate all buffers
        tag_ids = mx.full((B, max_steps + 1), self.end_id, dtype=mx.int32)
        tag_ids[:, 0] = self.start_id

        # KV caches per layer: [B, H, max_steps, Dh]
        sa_k_caches = [mx.zeros((B, H, max_steps + 1, Dh), dtype=mx.bfloat16) for _ in range(self.n_layers)]
        sa_v_caches = [mx.zeros((B, H, max_steps + 1, Dh), dtype=mx.bfloat16) for _ in range(self.n_layers)]

        # Hidden state buffer for bbox emission
        tag_H_buffer = mx.zeros((B, max_steps, D), dtype=mx.bfloat16)
        tag_H_counts = mx.zeros((B,), dtype=mx.int32)

        # All control state on host (numpy) — tiny, not worth device ops.
        # The heavy work (attention, MLP, logits) stays on MLX.
        finished = np.zeros(B, dtype=np.bool_)
        skip_next = np.ones(B, dtype=np.bool_)
        first_lcel = np.ones(B, dtype=np.bool_)
        prev_ucel = np.zeros(B, dtype=np.bool_)
        lengths = np.zeros(B, dtype=np.int32)
        h_counts = np.zeros(B, dtype=np.int32)  # host-side bbox emission counter

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
            for i, layer in enumerate(self.layers):
                x, sa_k_caches[i], sa_v_caches[i] = layer(
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

            # Store hidden states for bbox
            emit_indices = np.where(append_mask)[0]
            if len(emit_indices) > 0:
                for b_idx in emit_indices:
                    c = int(h_counts[b_idx])
                    bi = int(b_idx)
                    if c < max_steps:
                        tag_H_buffer = tag_H_buffer.at[bi, c, :].add(x[bi])
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

        return mx.array(tag_ids_np), tag_H_buffer, mx.array(lengths)
