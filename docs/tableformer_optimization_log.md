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

**Full pipeline integration**: When wired into the actual model (encoder on MPS → MLX decode → bbox on MPS), the torch↔MLX conversion overhead at boundaries (~250ms for mem_enc and tag_H) eats the decode speedup. Warm PyTorch MPS: 322ms decode vs MLX hybrid: 580ms (including conversions).

**Key insight**: The MLX decode kernel is genuinely faster (205ms vs 720ms cold), but the data conversion between PyTorch and MLX tensors is expensive. To realize the gain, either:
1. Move the encoder + transformer encoder to MLX too (eliminates mem_enc conversion)
2. Or find a zero-copy path between MPS tensors and MLX arrays (DLPack?)

## Experiment 4: MLX transformer encoder + decoder (full MLX path)

**Rationale**: Eliminate the torch→MLX conversion boundary by running both the transformer encoder and tag decoder on MLX. Only the ResNet encoder + input_filter stays on PyTorch MPS.

**Result (3 tables, warm)**:
- PyTorch MPS full path: 336ms
- **MLX encoder+decoder: 168ms — 2x faster**

**Result (44 tables, full 10-Q)**:
- PyTorch MPS: 7.8s (177ms/table)
- **MLX encoder+decoder: 4.5s (109ms/table) — 1.7x faster**
- **But: 93 regressions against golden**

The regressions are structural (row drops: 19→18, 23→22 etc.) caused by MLX transformer encoder producing slightly different attention outputs than PyTorch. Greedy autoregressive decode amplifies these tiny differences into token flips.

**FP32 encoder test**: Still 93 regressions. The drift isn't from bf16 precision alone — it's from the attention implementation differences (softmax accumulation order, LayerNorm reductions).

**Key finding**: The "safe" MLX boundary is at `mem_enc`, not at `filtered_nchw`. Moving the transformer encoder into MLX crosses a parity threshold that the autoregressive decoder cannot tolerate.

**End-to-end local pipeline**: 18.8s for 48 pages (vs 24s PyTorch MPS). 20% faster overall but with quality regressions.

---

## Summary of findings

| Path | Speed | Parity | Verdict |
|---|---|---|---|
| MLX decode only | 3.5x faster decode | 100% | ✅ Concept proven |
| Hybrid torch→MLX→torch | Slower than baseline | 100% | ❌ Conversion tax kills it |
| MLX encoder+decoder | 2x faster | 93 regressions | ⚠️ Fast but unstable |

**The bottleneck is dispatch overhead, not compute.** MLX proves this. But the encoder parity problem means the integrated MLX path needs more work.

---

## Next steps (prioritized)

1. **fp32 parity triage** — per-layer encoder diff to find where drift starts. Half-day experiment for go/no-go on MLX encoder.
2. **Torch/CUDA state machine regularization** — apply MLX lessons: tensorize flags, fixed-step masked loop, store all hidden states + compact after decode. No framework boundary tax.
3. **Zero-copy interop** — test DLPack for MPS↔MLX to see if decode-only hybrid becomes viable.


1. **FP32 decoder on MPS** (properly instrumented) — does removing bf16 help or hurt on MPS?
2. **Precompute cross-attention once per batch** — already done (`precompute_mem_kv`), verify it's actually used
3. **Reduce Python overhead in decode loop** — profile Python-side vs GPU-side time per step
4. **torch.mps.synchronize() placement** — check if MPS is hiding latency behind async dispatch
5. **Batch size sensitivity** — does batching more tables together improve MPS utilization?
6. **MLX tag decoder** — replace the Python decode loop with device-native MLX step function
