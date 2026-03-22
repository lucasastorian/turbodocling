<p align="center">
  <img src="docs/images/logo.png" alt="Turbodocling" width="480">
</p>

<h1 align="center">Turbodocling</h1>

<p align="center">
  <strong>PDF-to-structured-data, fast enough to ship in real products.</strong><br>
  A re-architected <a href="https://github.com/DS4SD/docling">Docling</a> pipeline that decouples CPU-bound parsing from GPU-bound inference, letting both scale independently.
</p>

---

## Why This Exists

[Docling](https://github.com/DS4SD/docling) is one of the best open-source PDF extraction pipelines — layout detection, table structure recognition, reading order, structured output. But it's too slow to actually use.

An 80-page 10-K takes **~67 seconds** on an A10G. A user uploads a document and stares at a spinner for a full minute. That's a product-killing latency. And for batch workloads — processing thousands of filings per day — the cost math doesn't work either.

The root cause is architectural: Docling processes pages serially, with CPU-heavy work (PDF parsing, image rendering, coordinate transforms) running on the same GPU instance as GPU-bound inference. You're paying A10G rates for work a Lambda function could do.

## Results

Turbodocling processes an 80-page 10-K in **11 seconds** and costs **~6x less per page** than stock Docling.

### Latency

| Document | Pages | Turbodocling | Stock Docling | Speedup |
|---|---|---|---|---|
| Docling paper | 8 | **3.0s** | 8.0s | 2.7x |
| Apple 10-Q (Q3 2025) | 29 | **5.5s** | 28.0s | 5.1x |
| Apple 10-K (FY2025) | 80 | **11.3s** | 66.6s | **5.9x** |

Small docs finish in 3-5 seconds. Longer documents benefit more from parallelism — 40-100 page docs see 5-6x speedups.

### Cost Efficiency

| Metric | Stock Docling | Turbodocling |
|---|---|---|
| Pages/sec/GPU | ~1.2 | ~20+ |
| Cost per page | ~$0.00023 | ~$0.00004 |
| GPU utilization | Low (waiting on CPU) | High (inference only) |

A single A10G now processes **20+ pages/second** because it only runs inference — all CPU work runs on Lambda (~$0.000023/page, essentially a rounding error). That's a **~6x cost reduction** at ~$1/hr GPU pricing, and the gap widens with real-world overhead.

---

## Getting Started

### Prerequisites

- AWS account with CDK bootstrapped
- Python 3.13+
- An A10G (or similar) GPU instance running the GPU worker container (ECS)

### Deploy

```bash
git clone https://github.com/lucasastorian/turbodocling.git
cd turbodocling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the C++ parser
python shared/docling_parse/build.py
pip install -e shared/docling_parse

# Deploy to AWS
cdk deploy
```

### Process a PDF

```python
import boto3, json, time, uuid
import pypdfium2 as pdfium

cfn = boto3.client("cloudformation")
s3 = boto3.client("s3")
sfn = boto3.client("stepfunctions")

# Get stack outputs
outputs = {o["OutputKey"]: o["OutputValue"]
           for o in cfn.describe_stacks(StackName="turbodocling-turbo-dev")["Stacks"][0]["Outputs"]}

# Upload PDF
job_id = str(uuid.uuid4())
s3.upload_file("my_document.pdf", outputs["DocumentsBucketName"],
               f"uploads/user/{job_id}/source.pdf")

# Start processing — Step Function computes batch_size automatically
total_pages = len(pdfium.PdfDocument("my_document.pdf"))
sfn.start_execution(
    stateMachineArn=outputs["StateMachineArn"],
    name=f"job-{job_id[:8]}",
    input=json.dumps({
        "job_id": job_id,
        "user_id": "user",
        "total_pages": total_pages,
    })
)

# Results land in S3:
#   processed/user/{job_id}/output.md       — structured markdown
#   processed/user/{job_id}/elements.json   — elements with bboxes per page
```

---

## Core Idea

**Decouple CPU-bound work from GPU-bound work. Let them scale independently.**

<p align="center">
  <img src="docs/images/turbodocling_architecture.png" alt="Stock Docling (coupled) vs Turbodocling (decoupled)" width="900">
</p>

Stock Docling processes everything in a single-threaded loop — parse, render, infer, assemble — one page at a time. The GPU sits mostly idle, waiting on CPU work.

Turbodocling splits the pipeline at the CPU/GPU boundary. Up to 40 parallel Lambda functions handle all CPU work (parsing, rendering), feeding pre-processed pages through S3+SQS to a GPU worker that runs inference at full utilization. Parsing 80 pages in parallel takes the same wall-clock time as parsing 2.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step Function                                                        │
│                                                                      │
│  ┌─────────┐     ┌──────────────────────────────────────────┐        │
│  │ Compute  │     │ Map State (max_concurrency=40)           │        │
│  │ Batch    │────→│                                          │        │
│  │ Size     │     │  Lambda 1: pages [0, batch_size)         │        │
│  │          │     │  Lambda 2: pages [batch_size, 2*bs)      │──→ S3  │
│  │ ceil(N   │     │  ...                                     │ batch  │
│  │  / 40)   │     │  Lambda K: pages [(K-1)*bs, N)           │ files  │
│  └─────────┘     └──────────────────────────────────────────┘        │
│                                          │                           │
│                                   SQS (batch refs)                   │
│                                          │                           │
│                                          ▼                           │
│                              ┌───────────────────┐                   │
│                              │  GPU Worker (A10G) │                   │
│                              │                    │                   │
│                              │  Layout inference  │                   │
│                              │  Table inference   │──→ S3 (output)   │
│                              │  Postprocessing    │    elements.json  │
│                              │  Assembly          │    output.md      │
│                              └───────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

- **Lambda (ARM64, 1769 MB, SnapStart):** Downloads PDF from S3, parses pages with a custom C++ docling-parse fork, renders page images with pdfium, packs everything into a columnar msgpack+gzip binary, uploads to S3. Typical runtime: 1-3 seconds per batch.
- **GPU Worker (A10G, ECS):** Polls SQS, downloads batches, runs layout detection (RT-DETR) and table structure recognition (TableFormer), assembles the final structured document (markdown + elements JSON). Multi-threaded internally — inference, postprocessing, and assembly overlap via bounded queues.
- **Batch sizing:** `ceil(total_pages / 40)` pages per Lambda, so an 80-page doc fans out across 40 parallel invocations.

---

## Build & Deploy

```bash
source .venv/bin/activate

# Rebuild C++ parser after changes
python shared/docling_parse/build.py
pip install -e shared/docling_parse

# Deploy
cdk deploy

# End-to-end smoke test (deploys and runs against live infrastructure)
python tests/smoke_test.py tests/golden/pdfs/attention.pdf

# Local parser benchmark
python tests/bench_parse.py --pdf apple_10k.pdf
python tests/bench_parse.py --pdf attention.pdf --check    # compare against baseline
python tests/bench_parse.py --update-golden                 # save new baseline

# Trace last execution (Lambda + GPU timing breakdown)
python tests/trace_job.py --latest 1
```

### Golden Test PDFs

Located in `tests/golden/pdfs/`: Docling paper (8p), Attention Is All You Need (15p), Berkshire Hathaway letter (20p), AlphaGo Zero (42p), NVIDIA investor roadshow (15p), Apple 10-K (80p).

---

## What's Next: Batch Processing

The current architecture processes one document at a time per GPU worker. The next step is a dedicated batch stack for bulk ingestion:

- **Horizontal CPU scaling:** ECS Spot cluster for preprocessing — hundreds of documents parsed in parallel, independent of GPU availability.
- **GPU worker pool:** Auto-scaling A10G pool pulling from SQS, fully saturated on inference. Spot instances work here too (interruptible jobs just re-queue).
- **Target cost:** At 20+ pages/sec/GPU and ~$1/hr, a single A10G processes ~72,000 pages/hour — **~$0.00001/page** for GPU compute.

---

## Optimization Deep Dives

Everything above tells you *what* Turbodocling does. This section explains *how* — the specific optimizations that get from ~1.2 pages/sec to 20+. If you're evaluating the project, you can stop here. If you want to understand the engineering, keep reading.

### CPU/GPU Decoupling

The single highest-impact change. Stock Docling runs parse → render → infer → postprocess → assemble in a single-threaded loop per page.

Turbodocling splits this into two stages:
- **Lambda stage (CPU):** PDF parsing (custom C++ docling-parse), image rendering (pdfium), coordinate transforms, cell extraction. Up to 40 parallel ARM64 Lambdas with SnapStart. Wall-clock for 80 pages: ~5 seconds (bounded by the slowest Lambda).
- **GPU stage:** Layout inference (RT-DETR), table structure inference (TableFormer), postprocessing, document assembly. Receives pre-parsed pages — no CPU-bound work to stall the pipeline.

The GPU worker itself is multi-threaded: dedicated threads for inference, layout postprocessing, table cell matching, and assembly — all communicating via bounded queues with backpressure.

### Compact Page Transport Format

Pages cross the Lambda → GPU boundary as binary blobs. Every byte matters at 40x fan-out.

- **Columnar layout:** Instead of per-cell JSON objects, all x0 values are packed together, all y0 together, etc. ~3x better gzip compression, enables `np.frombuffer()` zero-copy deserialization.
- **String interning:** Font names, text, and repeated strings are deduplicated into a string table. Each cell stores a 2-byte index. Cuts payload ~40% on typical documents.
- **Lazy cell materialization:** Cells stay columnar (`CellStore` backed by numpy arrays) and are materialized on-demand. Avoids creating tens of thousands of Python objects that would immediately be GC'd.
- **Wire format:** msgpack → gzip (level 6). An 80-page batch compresses to ~2-3 MB.

### C++ Parser Optimizations

Custom fork of [docling-parse](https://github.com/DS4SD/docling-parse):

- **Path operator skipping:** Skip `PATH_CONSTRUCTION` and `PATH_PAINTING` operators entirely — downstream never uses `page_lines`. On pages with dense vector graphics, this cuts parse time from ~3.6s to ~1.2s per page.
- **Const-ref operator dispatch:** Pass operator arguments by const reference instead of by value, eliminating copy constructors on every call.
- **Header lookup elimination:** Build a one-time column-index map instead of calling `header.index("x0")` per field per row.
- **Shared char-data construction:** Centralize `model_dump()` payload construction into `_build_char_data()` and share between word and line creation.

### TableFormer Redesign — KV-Cached Batched Decoding

The most significant code change. Stock Docling's TableFormer processes tables one at a time with a naive autoregressive decode loop that reallocates tensors every step.

**Stage decoupling:** Split the monolithic 800-line `TFPredictor` into three independent stages — inference only (`TFPredictor`), vectorized cell matching (`TFCellMatcher`), and row/column assignment (`MatchingPostProcessor`). Table inference runs on the GPU thread while matching runs on a dedicated CPU thread, overlapping compute.

**KV-cached autoregressive decode:** Stock TableFormer re-attends to all previous tokens every step (O(T²) memory). Fork preallocates KV buffers to `max_pred_len` and writes in-place via `copy_()` — O(T) memory, zero allocations after init. Cross-attention K/V are precomputed once from encoder output.

**Fused QKV projection:** Three separate linear projections per attention head fused into a single `F.linear()` call.

**Batched decode:** All tables from a document decoded simultaneously with preallocated output buffers, LUT-based token classification (replaces `torch.isin()`), and GPU-resident span tracking — no CPU sync during the decode loop.

**Streaming softmax for bbox inference:** Two-pass streaming softmax with configurable tile size keeps peak memory constant regardless of image resolution, avoiding the full `[num_cells × num_pixels × attention_dim]` intermediate tensor.

**Mixed precision:** Transformer layers in BF16 for fast attention. Bbox decoder stays FP32 (sensitive to precision). ResNet-18 encoder in FP32 with TF32 matmul enabled.

**Encoder optimizations:** Conv+BN fusion, optional CUDA graph capture, channels-last memory format.

### Layout Inference

Same RT-DETR checkpoint, same detection heads. Execution strategy optimizations only:

- **Static fixed-shape batching:** All images resized/padded to 640×640, processed in batches of 32. Fixed shapes enable CUDA graph capture.
- **GPU preprocessing pipeline:** Normalization, resize, and padding on GPU with persistent CUDA streams. H2D transfers overlap with compute.
- **Channels-last memory format** and **TF32 matmul** enabled globally.
- **Optional `torch.compile`:** `TORCH_COMPILE=1` enables Inductor JIT — marginal gains on current model size, adds ~60s to cold start.

### Layout Postprocessing

- **Precomputed geometry:** Valid-cell lists, bounding boxes, and areas computed once in `__init__()`. Stock rebuilds these per page — 154ms × 80 pages = 12 seconds wasted on Apple 10-K.
- **Grid-based spatial indexing:** O(1) cell-to-cluster lookup via fixed-size grid bins instead of O(N×M) brute-force intersection.
- **Vectorized overlap resolution:** Numpy broadcasting for IoU computation instead of per-pair Python loops.

### Table Postprocessing

- **Vectorized cell matching:** Numpy broadcasting (`[T,1,4] × [1,P,4]`) computes all intersections in one operation. Stock uses nested Python loops.
- **Stage isolation:** Matching runs on a dedicated CPU thread, overlapping with the next GPU inference batch.

### Assembly

- **Serializer reuse:** `MarkdownDocSerializer` instantiated once, reused across all tables. Stock constructs a new serializer per table.
- **Data URI fast-path:** Detects `data:image/png;base64,...` URIs and extracts the base64 string directly, skipping redundant PNG re-encoding (~400ms saved on image-heavy documents).
- **Targeted picture iteration:** Iterates `conv_res.document.pictures` directly instead of traversing all document nodes.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Turbodocling is built on the excellent work of the [Docling](https://github.com/DS4SD/docling) team at IBM Research, whose original code is MIT-licensed. We are grateful for their contributions to open-source document understanding.
