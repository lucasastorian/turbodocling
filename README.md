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

[Docling](https://github.com/DS4SD/docling) is one of the best open-source PDF extraction pipelines: layout detection, table structure recognition, reading order, structured output. But it's too slow to actually use.

An 80-page 10-K takes **~67 seconds** on an A10G. That's unusable for anything user-facing. For batch workloads processing thousands of filings per day, the cost math doesn't work either.

Turbodocling processes the same document in **12 seconds** and costs **~6x less per page**.

| Document | Pages | Turbodocling | Stock Docling | Speedup |
|---|---|---|---|---|
| Docling paper | 8 | **3.5s** | 8.0s | 2.3x |
| Apple 10-Q (Q3 2025) | 29 | **5.7s** | 28.0s | 4.9x |
| Apple 10-K (FY2025) | 80 | **12.4s** | 66.6s | **5.4x** |

| Metric | Stock Docling | Turbodocling |
|---|---|---|
| Pages/sec/GPU | ~1.2 | ~20+ |
| Cost per page | ~$0.00023 | ~$0.00004 |
| GPU utilization | Low (waiting on CPU) | High (inference only) |

Small docs finish in 3-5 seconds. Longer documents benefit more since parallelism amortizes fixed costs; 40-100 page docs see 5-6x speedups.

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

## How Docling Works (and Why It's Slow)

To understand what we changed, you need to understand the pipeline. Docling processes every page through seven stages, **serially**, one page at a time, start to finish, before touching the next:

```
For each page:
  1. Parse         →  C++ PDF parser extracts text cells, fonts, coordinates
  2. Render        →  pdfium renders a page image for the vision models
  3. Layout Model  →  RT-DETR detects tables, figures, headers, text blocks
  4. Layout Post   →  NMS, reading order, spatial clustering
  5. Table Prep    →  Crop detected tables, match PDF cells to table regions
  6. TableFormer   →  Autoregressive transformer predicts table structure
  7. Assembly      →  Stitch everything into final Markdown + structured JSON
```

Every stage runs on the same machine. CPU-bound parsing and GPU-bound inference share a single thread. The GPU sits idle while the CPU parses, and vice versa. Time scales linearly with page count.

We made significant optimizations at **every layer** of this pipeline. Here's what was wrong and what we did about it.

### 1-2. Parse & Render — Wrong Machine, Wrong Architecture

**The problem:** PDF parsing and image rendering are pure CPU work (string processing, font decoding, pixel rendering). Stock Docling runs all of this on the GPU instance, serially, one page at a time. You're paying A10G rates (~$1/hr) for work that doesn't touch the GPU.

**What we did:** We moved all CPU work off the GPU entirely. A Step Function fans out to **up to 40 parallel Lambda functions** (ARM64, 1769 MB, SnapStart), each parsing and rendering a batch of pages simultaneously. An 80-page doc that took ~40 seconds to parse serially now finishes in ~5 seconds wall-clock, bounded by the slowest Lambda rather than the sum.

```
                    ┌─ Lambda 1:  pages 1-2   [parse + render] ─┐
                    ├─ Lambda 2:  pages 3-4   [parse + render] ─┤
PDF → Step Function ├─ Lambda 3:  pages 5-6   [parse + render] ─┼→ SQS → GPU Worker
                    ├─ ...                                      │   (steps 3-7 only)
                    └─ Lambda 40: pages 79-80 [parse + render] ─┘
```

The GPU worker receives pre-parsed pages and runs **only** inference and postprocessing. No CPU-bound work to stall the pipeline.

We also optimized the parser itself. Our custom fork of [docling-parse](https://github.com/DS4SD/docling-parse) includes targeted C++ patches: skipping path operators the downstream pipeline never uses (3x speedup on graphics-heavy pages), eliminating per-row header lookups in hot loops, and passing operator arguments by const reference instead of by value.

### Transport — Getting Pages to the GPU Efficiently

Pages cross the Lambda → GPU boundary as binary blobs. At 40x fan-out, every byte matters.

Instead of shipping per-cell JSON objects, we pack all coordinates into columnar arrays (all x0 values together, all y0 together, etc.), which compresses ~3x better under gzip and enables zero-copy `np.frombuffer()` deserialization on the GPU side. Repeated strings (font names, text content) are deduplicated into a string table; each cell stores a 2-byte index instead of the full string. On the GPU side, cells stay columnar and are materialized into Python objects only on demand, avoiding tens of thousands of throwaway allocations.

Wire format is msgpack → gzip. An 80-page batch compresses to ~2-3 MB.

### 3. Layout Model — Same Model, Better Execution

**The problem:** The RT-DETR layout model is good; the issue is how it's called. Stock Docling runs inference one page at a time with dynamic input shapes, which prevents the GPU from batching efficiently or leveraging hardware-level optimizations.

**What we did:** Same checkpoint, same detection heads, no model changes. We batch all pages into fixed 640×640 tensors (batches of 32), enable TF32 matmul and channels-last memory format for Ampere GPUs, and run image preprocessing (normalization, resize, padding) on GPU with persistent CUDA streams so H2D transfers overlap with compute.

### 4. Layout Postprocessing — Death by Python Loop

**The problem:** Layout postprocessing converts raw predictions into structured page elements (coordinate transforms, NMS, reading order, spatial clustering). Stock Docling rebuilds spatial indices **from scratch on every page**. On the Apple 10-K, it reconstructs the valid-cell list 80 times at 154ms each, wasting 12 seconds on redundant work.

**What we did:** Precompute all geometry once. Valid-cell lists, bounding boxes, and areas are built in `__init__()` and reused across every page. Cell-to-cluster assignment uses a grid index for O(1) spatial lookup instead of brute-force intersection testing. Overlap detection uses numpy broadcasting instead of per-pair Python loops.

### 5-6. TableFormer — The Biggest Win

**The problem:** This is where stock Docling is most wasteful. TableFormer is an autoregressive transformer that predicts table structure token by token, like a language model. Stock Docling's implementation has three compounding problems:

- **No batching.** Tables are processed one at a time. A page with 5 tables runs 5 separate inference passes.
- **No KV cache.** Every decode step re-attends to all previous tokens from scratch. This is the O(T²) pattern that the LLM world solved years ago with KV caching.
- **Inline execution.** Stock Docling already had separate helper classes for matching and postprocessing, but executed them inline in the same call path, so the GPU blocks on CPU work between tables.

**What we did:** We rewrote the entire TableFormer inference path.

**KV-cached decoding:** We preallocate key/value buffers to `max_pred_len` and write in-place each step. O(T) memory, zero allocations after init. Cross-attention K/V are precomputed once from the encoder output and reused across all decode steps.

**Batched inference:** All tables from a document are decoded simultaneously. Table crops are grouped by shape, stacked into uniform batches, and processed in a single forward pass with preallocated output buffers and GPU-resident span tracking. No CPU sync during the decode loop.

**Stage decoupling:** The redesign makes `TFPredictor` inference-only. It returns predictions and the GPU thread immediately continues to the next batch. Matching and postprocessing run on a dedicated CPU thread, overlapping with the next round of GPU inference.

**Additional optimizations:** Fused QKV projections (one GEMM instead of three per attention head), streaming softmax for bbox inference (constant peak memory regardless of image resolution), BF16 for transformer layers with FP32 retained for precision-sensitive bbox regression, and Conv+BN fusion with optional CUDA graph capture for the ResNet-18 encoder.

### 7. Assembly — Small Wins That Add Up

**The problem:** Stock Docling instantiates a new Markdown serializer per table (re-scanning the full document tree each time) and redundantly re-encodes images that are already stored as base64 data URIs.

**What we did:** Serializer reuse across all tables, direct base64 extraction from data URIs (skipping ~400ms of redundant PNG re-encoding on image-heavy docs), and targeted iteration over `document.pictures` instead of full tree traversal.

### The GPU Pipeline — All Together

These optimizations compound. Here's the GPU worker's time breakdown on the Apple 10-K (80 pages), after receiving pre-parsed pages from Lambda:

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

---

## Build & Deploy

```bash
source .venv/bin/activate

# Rebuild C++ parser after changes
python shared/docling_parse/build.py
pip install -e shared/docling_parse

# Deploy
cdk deploy

# End-to-end smoke test
python tests/smoke_test.py tests/golden/pdfs/attention.pdf

# Local parser benchmark
python tests/bench_parse.py --pdf apple_10k.pdf
python tests/bench_parse.py --pdf attention.pdf --check    # compare against baseline
python tests/bench_parse.py --update-golden                 # save new baseline

# Trace last execution (Lambda + GPU timing breakdown)
python tests/trace_job.py --latest 1
```

**Golden test PDFs** in `tests/golden/pdfs/`: Docling paper (8p), Attention Is All You Need (15p), Berkshire Hathaway letter (20p), AlphaGo Zero (42p), NVIDIA investor roadshow (15p), Apple 10-K (80p).

---

## What's Next: Batch Processing

The current architecture processes one document at a time per GPU worker. The next step is a dedicated batch stack: ECS Spot cluster for CPU preprocessing (hundreds of documents in parallel), feeding an auto-scaling pool of A10G workers pulling from SQS. At 20+ pages/sec/GPU, a single A10G processes ~72,000 pages/hour for roughly **~$0.00001/page** in GPU compute.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Turbodocling is built on the excellent work of the [Docling](https://github.com/DS4SD/docling) team at IBM Research, whose original code is MIT-licensed. We are grateful for their contributions to open-source document understanding.
