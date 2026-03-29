<p align="center">
  <img src="docs/images/logo.png" alt="Turbodocling" width="480">
</p>

<h1 align="center">Turbodocling</h1>

<p align="center">
  <a href="https://github.com/DS4SD/docling">Docling</a>, re-architected from the ground up for performance, throughput, and cost. Deploy to AWS in minutes, process documents in seconds.
</p>

---

## The Problem

A lot of production applications need fast, high-quality PDF-to-Markdown conversion. A user uploads a document and you want to feed it into an LLM. Mistral OCR takes 20-30 seconds. vLLM-based pipelines are slow. Traditional parsers might not meet your quality bar.

Or you need to backfill a million pages and don't want to pay $0.015/page for AWS Textract or $0.01/page for Mistral OCR. At 10M pages, that's $100-150K just for extraction. If you're building an MVP, that math kills you.

[Docling](https://github.com/DS4SD/docling) solves the quality problem. It properly extracts headers, paragraphs, tables, and images from PDFs without hallucinating values. But an 80-page 10-K takes **~67 seconds** on an A10G, with the GPU mostly idle. That makes it unusable for anything user-facing, and because it's slow, it's effectively expensive. Scaling it means spinning up a cluster of GPU instances and managing the infra yourself.

## What We Built

Turbodocling is Docling, re-architected for AWS. `cdk deploy` and you're running in minutes. Process a document by calling a Step Function. Scale up for a batch job, scale back down when you're done.

- **Fast.** 10 seconds for a 93-page 10-K. 2.5 seconds for a short document.
- **Cheap at scale.** Orders of magnitude less than Textract ($0.015/page) or Mistral OCR ($0.01/page). GPU processes 20+ pages/second and can be shared across concurrent jobs.
- **Easy to deploy.** One `cdk deploy`. Lambda, SQS, Step Functions, ECS.
- **Easy to integrate.** Just invoke a Step Function from your existing services.
- **Easy to scale.** Fan-out is automatic. Bump the GPU worker count for batch jobs, scale back to zero when idle.

| Document | Pages | Turbodocling | Stock Docling | Speedup |
|---|---|---|---|---|
| Docling paper | 8 | **2.5s** | 8.1s | 3.2x |
| Attention paper | 15 | **4.6s** | 12.6s | 2.7x |
| NVIDIA 10-Q (Q3 FY2026) | 48 | **6.0s** | 40.7s | **6.8x** |
| NVIDIA 10-K (FY2026) | 93 | **9.5s** | 71.0s | **7.5x** |

### Throughput (20x NVIDIA 10-K concurrent, 1860 pages)

| Metric | Stock Docling | Turbodocling |
|---|---|---|
| Total wall time | ~1420s (serial) | **111.6s** |
| Throughput | 1.3 pages/sec | **16.7 pages/sec** |
| Concurrent docs | 1 (single-threaded) | 4+ (multi-threaded pipeline) |
| Peak memory | N/A | 49% (13.7/28.0 GiB) |

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

### Experimental PDFium Variant

The default Lambda image installs upstream `pypdfium2==5.6.0`. You can opt into the patched PDFium build at deploy time by passing CDK context for the Lambda Docker build:

```bash
cdk deploy \
  -c pdfium_variant=experimental \
  -c pypdfium2_wheel_url=https://github.com/lucasastorian/pdfium/releases/download/YOUR_TAG/pypdfium2-EXPERIMENTAL.whl
```

Notes:

- `pdfium_variant=upstream` remains the default.
- `pypdfium2_wheel_url` is required only for `experimental`.
- This keeps the Python code path unchanged and swaps only the bundled PDFium binary in the Lambda image.
- Source for the experimental renderer lives in [`lucasastorian/pdfium`](https://github.com/lucasastorian/pdfium) on the `codex-clip-mask-experiment` branch.

### Process a PDF

```python
import boto3, json, uuid
import pypdfium2 as pdfium

STEP_FUNCTION_ARN = "arn:aws:states:us-east-1:123456789:stateMachine:..."
BUCKET = "turbodocling-turbo-dev-documentsbucket-..."

s3 = boto3.client("s3")
sfn = boto3.client("stepfunctions")

# Upload PDF
job_id = str(uuid.uuid4())
s3.upload_file("my_document.pdf", BUCKET, f"uploads/user/{job_id}/source.pdf")

# Start processing
total_pages = len(pdfium.PdfDocument("my_document.pdf"))
sfn.start_execution(
    stateMachineArn=STEP_FUNCTION_ARN,
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

## How Docling Works (and Why It's Slow)

To understand what we changed, you need to understand the pipeline. Docling processes pages through seven stages:

```
  1. Parse         →  C++ PDF parser extracts text cells, fonts, coordinates
  2. Render        →  pdfium renders a page image for the vision models
  3. Layout Model  →  RT-DETR detects tables, figures, headers, text blocks
  4. Layout Post   →  NMS, reading order, spatial clustering
  5. Table Prep    →  Crop detected tables, match PDF cells to table regions
  6. TableFormer   →  Autoregressive transformer predicts table structure
  7. Assembly      →  Stitch everything into final Markdown + structured JSON
```

Docling does batch pages in small chunks (default 4) for the model stages, but all stages still run on the same machine. CPU-bound parsing and GPU-bound inference are coupled together, so the GPU sits idle during parsing and vice versa.

We optimized at every layer. Three changes account for most of the speedup.

### 1. CPU/GPU Decoupling

Stock Docling couples CPU and GPU work on the same machine. PDF parsing, image rendering, and page image creation all happen inline on the GPU instance. Image rendering in particular used to live inside the layout processor itself. You're paying A10G rates (~$1/hr) for pure CPU work that doesn't touch the GPU.

We moved every piece of CPU work we could off the GPU and into Lambda. Parsing, rendering, and page image creation all happen in **up to 40 parallel Lambda functions** (ARM64, 1769 MB, SnapStart) via a Step Function fan-out. The GPU receives fully pre-processed pages and only runs inference. An 80-page doc that took ~40 seconds to preprocess serially now finishes in ~5 seconds wall-clock, bounded by the slowest Lambda rather than the sum.

```
                    ┌─ Lambda 1:  pages 1-2   [parse + render] ─┐
                    ├─ Lambda 2:  pages 3-4   [parse + render] ─┤
PDF → Step Function ├─ Lambda 3:  pages 5-6   [parse + render] ─┼→ SQS → GPU Worker
                    ├─ ...                                      │   (inference only)
                    └─ Lambda 40: pages 79-80 [parse + render] ─┘
```

The GPU worker receives pre-parsed pages and runs **only** inference and postprocessing. No CPU-bound work to stall the pipeline.

### 2. TableFormer Rewrite

This is where stock Docling is most wasteful. TableFormer is an autoregressive transformer that predicts table structure token by token, like a language model. Stock Docling's implementation has three compounding problems:

- **No batching.** Tables are processed one at a time. A page with 5 tables runs 5 separate inference passes.
- **No KV cache.** Every decode step re-attends to all previous tokens from scratch. This is the O(T²) pattern that the LLM world solved years ago with KV caching.
- **Inline execution.** Stock Docling already had separate helper classes for matching and postprocessing, but executed them inline in the same call path, so the GPU blocks on CPU work between tables.

We rewrote the entire inference path:

**KV-cached decoding:** We preallocate key/value buffers to `max_pred_len` and write in-place each step. O(T) memory, zero allocations after init. Cross-attention K/V are precomputed once from the encoder output and reused across all decode steps.

**Batched inference:** All tables from a document are decoded simultaneously. Table crops are grouped by shape, stacked into uniform batches, and processed in a single forward pass with preallocated output buffers and GPU-resident span tracking. No CPU sync during the decode loop.

**Stage decoupling:** The redesign makes `TFPredictor` inference-only. It returns predictions and the GPU thread immediately continues to the next batch. Matching and postprocessing run on a dedicated CPU thread, overlapping with the next round of GPU inference.

**Additional optimizations:** Fused QKV projections (one GEMM instead of three per attention head), streaming softmax for bbox inference (constant peak memory regardless of image resolution), BF16 for transformer layers with FP32 retained for precision-sensitive bbox regression, and Conv+BN fusion with optional CUDA graph capture for the ResNet-18 encoder.

### 3. Aggressive Optimization Across the Pipeline

Beyond the two architectural changes above, we rewrote layout postprocessing, table postprocessing, and table preprocessing. Hundreds of optimizations across every stage, but the common themes:

- **Eliminate redundant work.** Precompute geometry, spatial indices, and valid-cell lists once instead of rebuilding per page. Stock Docling's layout postprocessing alone wastes ~12 seconds on the Apple 10-K recomputing the same data 80 times.
- **Vectorize with numpy.** Replace nested Python loops with broadcasting operations for overlap detection, cell matching, IoU computation, and coordinate transforms.
- **Reduce object churn.** Columnar data layouts, string interning, lazy materialization. Avoid creating tens of thousands of short-lived Python objects that immediately get GC'd.
- **Kill hot loops.** Grid-based spatial indexing for O(1) lookups instead of brute-force intersection. LUT-based token classification instead of `torch.isin()`. One-time column-index maps instead of per-row `header.index()` calls in C++.
- **C++ parser patches.** Custom fork of [docling-parse](https://github.com/DS4SD/docling-parse): skip path operators the downstream pipeline never uses (3x speedup on graphics-heavy pages), const-ref operator dispatch, eliminated per-row header lookups.
- **Spatial cluster and interval indexing.** Reading order and element clustering use interval trees and grid indices instead of O(N*M) pairwise comparisons. This was one of the bigger wins in postprocessing.
- **Multi-threaded GPU pipeline.** Inference, layout postprocessing, table postprocessing, and assembly run on dedicated threads communicating via bounded queues with backpressure. Stages overlap instead of running sequentially.
- **Pinned staging cache and shape bucketing.** GPU preprocessing reuses pinned memory buffers and buckets images by shape to minimize allocations and H2D transfer overhead.
- **Render-scale cap and dual-resolution images.** Lambda generates both a layout-resolution and a table-resolution image per page, capping render scale for large pages. Avoids redundant re-rendering on the GPU side.
- **Intake backpressure and concurrent downloads.** The GPU worker gates how fast it pulls batches from S3 based on downstream queue depth, and downloads PDF chunks concurrently rather than sequentially.
- **Layout model tuning.** Stock Docling already batches layout inference in small dynamic batches (typically 4). We run fixed-size 640x640 batches of 32, channels-last + TF32 on Ampere, with GPU preprocessing and persistent streams to overlap H2D and compute.

### GPU Pipeline Timing

Here's the GPU worker's time breakdown on the Apple 10-K (80 pages), after receiving pre-parsed pages from Lambda:

| Stage | Time |
|---|---|
| Intake (S3 download + decompress + deserialize) | 543ms |
| Layout inference (RT-DETR, 80 pages) | 1,996ms |
| Layout postprocessing | 1,042ms |
| Table inference (TableFormer, batched) | 1,211ms |
| Table postprocessing | 547ms |
| Assembly | 492ms |
| Upload | 112ms |

The GPU worker is internally multi-threaded: dedicated threads for inference, layout postprocessing, table postprocessing, and assembly communicate via bounded queues with backpressure, so stages overlap wherever possible.

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

**Lambda (ARM64, 1769 MB, SnapStart):** Downloads PDF from S3, parses with custom C++ docling-parse fork, renders page images, packs into columnar msgpack+gzip, uploads to S3. Runtime: 1-3 seconds per batch.

**GPU Worker (A10G, ECS):** Polls SQS, downloads batches, runs layout + table inference, assembles structured output (markdown + elements JSON).

**Batch sizing:** `ceil(total_pages / 40)` pages per Lambda, so an 80-page doc fans out across 40 parallel invocations.

### GPU Memory and Concurrency

The GPU worker holds entire documents in memory during processing — page images, cell stores, and the source PDF. A 93-page 10-K uses ~2.5 GB resident. The default configuration (g5.2xlarge, 28 GB task memory) comfortably handles 4 concurrent large documents at ~50% memory utilization.

Memory-based backpressure automatically throttles intake when utilization exceeds 50%, so the worker never OOMs — it just queues. For very large documents (400+ pages) or high-concurrency batch workloads, consider upgrading to a larger instance (g5.4xlarge/g5.8xlarge) for more system RAM. The GPU (A10G, 24 GB VRAM) is the same across all g5 sizes.

### Cost

At 16.7 pages/second sustained throughput, processing costs break down to:

| Component | Cost per 10K pages |
|---|---|
| Lambda (ARM64, ~2s/batch) | ~$0.06 |
| Step Functions | ~$0.015 |
| GPU (A10G, g5.2xlarge @ $1.21/hr) | ~$0.20 |
| **Total** | **~$0.28** |

For comparison: AWS Textract costs $15 per 10K pages. Mistral OCR costs $10 per 10K pages. Turbodocling is **50-60x cheaper** at scale.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Turbodocling is built on the excellent work of the [Docling](https://github.com/DS4SD/docling) team at IBM Research, whose original code is MIT-licensed. We are grateful for their contributions to open-source document understanding.
