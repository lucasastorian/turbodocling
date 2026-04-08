# TableFormer Optimization Log

Branch: `tableformer-mlx-decode`

## Baseline (MPS, BF16 tag transformer, FP32 encoder/bbox)

**NVIDIA 10-Q (48 pages, 44 tables)**
- Total: 26,250ms
- Per-table: 155ms

**Per-stage breakdown (pages 14-15, 6 tables):**
| Stage | Time |
|---|---|
| Encoder (ResNet-18, FP32) | 41ms |
| Tag input filter | 14ms |
| Memory reshape + bf16 cast | 2ms |
| Tag encoder (transformer, BF16) | 16ms |
| **Batched AR decoder (BF16)** | **961ms** |

**Bottleneck**: AR decoder is 92% of model time (excluding encoder).

**Full pipeline on attention.pdf (15 pages):**
| Config | Inference |
|---|---|
| CPU, 1 thread | 283s |
| CPU, all threads | 100s |
| MPS layout + CPU table | 37.6s |
| MPS layout + MPS table | 19.6s |

**Golden parity**: MPS produces identical OTSL + cell texts to CPU on all tested documents.

---

## Experiment 1: torch.compile on decoder

**Status**: Previously attempted by user. No improvement on MPS. Skipped.

---

## Experiment 2: FP32 decoder on MPS (skip bf16 conversion)

**Rationale**: MPS may have hidden dtype conversion overhead at bf16/fp32 boundaries. The decoder is also known to be precision-sensitive (fp16 causes regressions).

**Result**: Python crashes / hangs when running FP32 tag transformer on MPS. BF16 baseline completes fine (604ms decoder for 3 tables), but FP32 run never returns. Likely MPS driver bug — the SDPA kernel path may differ for fp32 vs bf16.

**Conclusion**: Cannot use FP32 decoder on MPS. BF16 stays. This is fine — the bf16 choice was deliberate for the decoder anyway.

**Note from user**: "The decoder is super sensitive. If you use fp16, it regresses." FP32 encoder is intentional because it's a vision model. BF16 was specifically chosen for the decoder to enable Flash Attention on CUDA — and apparently MPS also prefers it.

---

## Experiment 3: MLX tag decoder (hybrid — heavy math on MLX, control on host)

**Rationale**: Replace the PyTorch MPS decode path with MLX for the expensive per-step operations (embedding, self-attn with KV cache, cross-attn, MLP, logits). Keep control logic (structure corrections, emit/skip flags) on numpy since it's tiny.

**Result**: 3 tables on page 14 of NVIDIA 10-Q:
- PyTorch MPS: 720ms
- **MLX hybrid: 205ms — 3.5x faster**
- **All 3 tables produce identical OTSL sequences**

This is the naive version with per-step `mx.eval()` + numpy sync for argmax. The speedup comes purely from eliminating PyTorch MPS kernel dispatch overhead. MLX attention/MLP kernels are also faster for the small q_len=1 incremental decode pattern.

---

## Next experiments to try

1. **FP32 decoder on MPS** (properly instrumented) — does removing bf16 help or hurt on MPS?
2. **Precompute cross-attention once per batch** — already done (`precompute_mem_kv`), verify it's actually used
3. **Reduce Python overhead in decode loop** — profile Python-side vs GPU-side time per step
4. **torch.mps.synchronize() placement** — check if MPS is hiding latency behind async dispatch
5. **Batch size sensitivity** — does batching more tables together improve MPS utilization?
6. **MLX tag decoder** — replace the Python decode loop with device-native MLX step function
