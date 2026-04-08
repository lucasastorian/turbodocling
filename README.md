<p align="center">
  <img src="docs/images/logo.png" alt="Turbodocling" width="480">
</p>
<h1 align="center">Turbodocling</h1>
<p align="center">
  <strong>Docling, but actually fast enough for real apps.</strong><br>
  5–6× faster. 40–200× cheaper than Textract or Mistral OCR.<br>
  <code>cdk deploy</code> and it's running in minutes.
</p>

---

Stock Docling is great at quality. Terrible at speed.
A 93-page 10-K takes **71 seconds** on an A10G. That's unusable if a user just dropped a PDF in your chat.

Turbodocling fixes it. Same quality, but the same 10-K now finishes in **13 seconds**.
That's fast enough to do PDF → Markdown → LLM response inline. No background jobs. No "we'll email you later."

### Real numbers

| Document          | Pages | Turbodocling | Stock Docling | Speedup |
|-------------------|-------|--------------|---------------|---------|
| Docling paper     | 8     | **3.0 s**    | 8.1 s         | 2.7×    |
| NVIDIA 10-Q       | 48    | **6.5 s**    | 40.7 s        | 6.3×    |
| NVIDIA 10-K       | 93    | **12.8 s**   | 71.0 s        | 5.5×    |

**20 concurrent 93-page 10-Ks (1 860 pages total):**
**16.7 pages/sec** vs ~1.3 pages/sec for stock Docling.

### Cost that actually makes sense

**Turbodocling cost per 10 000 pages** (Lambda + Step Functions + GPU)

| GPU utilization | Turbodocling | vs Textract ($150) | vs Mistral ($30) |
|-----------------|--------------|--------------------|------------------|
| **100%** (batch/backfill) | **$0.28** | 536× cheaper | 107× cheaper |
| **30%** (real-time / bursty) | **$0.74** | 203× cheaper | 41× cheaper |

- The **$0.28** figure is at full utilization (16.7 pages/sec sustained). That's the number for overnight batch jobs or backfills.
- At 30% utilization (typical for real-time chat/RAG traffic) the effective cost rises to **$0.74** because you're still paying for the GPU when it's idle.
- Spot instances cut the GPU component by ~70% ($0.36/hr vs $1.21/hr on-demand).
- One always-on A10G is ~$870/mo on-demand. Still **orders of magnitude** cheaper than API services while handling real traffic.

Bottom line: whether you're doing batch or real-time, this is the difference between "we can't afford OCR" and "it just works."

---

## Architecture (the part that actually matters)

```
PDF → Step Function
     ↓
┌─ Lambda 1: pages 1-2  [parse + render] ─┐
├─ Lambda 2: pages 3-4  [parse + render] ─┤
├─ ...                                     │
└─ Lambda 40: pages 79-80 [parse + render] ┘
     ↓ (SQS)
GPU Worker (A10G ECS) → only inference + post-processing
     ↓
S3 (output.md + elements.json)
```

Everything CPU-heavy (parsing, rendering) runs in parallel Lambdas.
The GPU only does the expensive model work. That single change is most of the speedup.

### The three big changes we made

1. **Split CPU and GPU work**
   Stock Docling runs everything on the same machine. You pay A10G prices for parsing and image rendering. We moved that to cheap parallel Lambdas.

2. **Rewrote TableFormer from scratch**
   No more one-table-at-a-time. No more recomputing attention every token. Batched + KV-cached + multi-threaded. Huge win.

3. **Everything else optimized to death**
   Vectorized numpy everywhere, spatial indexes instead of O(N²) loops, C++ parser patches, zero-copy data paths, multi-threaded pipeline with backpressure. The boring stuff that adds up.

---

## Getting Started

```bash
git clone https://github.com/lucasastorian/turbodocling.git
cd turbodocling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the optimized C++ parser
python shared/docling_parse/build.py
pip install -e shared/docling_parse
```

### Run locally (no AWS needed)

```bash
python -m turbodocling my_document.pdf -o output/
```

That's it. Outputs `output/output.md` and `output/elements.json`. Works on CPU, CUDA, or Apple Silicon (MPS).

Options:
- `--device auto|cuda|mps|cpu` — inference device (default: auto-detect)
- `--workers N` — preprocessing parallelism (default: CPU count)

On Apple Silicon, both layout and table inference run on the GPU via MPS. On a MacBook Pro M1 Max, a 48-page NVIDIA 10-Q processes in ~24 seconds.

#### Python API

```python
from turbodocling.local_runner import run_local

result = run_local("my_document.pdf", "output/", device="auto")
# result.md_path, result.elements_path, result.wall_time_s, ...
```

### Deploy to AWS (for speed + scale)

For production throughput, deploy the full pipeline to AWS:

```bash
cdk deploy
```

Then process PDFs via Step Functions:

```python
import boto3, json, uuid
import pypdfium2 as pdfium

STEP_FUNCTION_ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:..."
BUCKET = "turbodocling-...-documentsbucket-..."

s3 = boto3.client("s3")
sfn = boto3.client("stepfunctions")

job_id = str(uuid.uuid4())
s3.upload_file("my_document.pdf", BUCKET, f"uploads/user/{job_id}/source.pdf")

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
```

Results land in S3:
- `processed/user/{job_id}/output.md` → clean markdown
- `processed/user/{job_id}/elements.json` → the good stuff (see below)

### Working with elements.json (this is what you'll actually use in production)

```json
{
  "pages": [
    {
      "page_no": 1,
      "width": 612,
      "height": 792,
      "elements": [
        {
          "type": "section_header",
          "content": "1. Introduction",
          "bbox": { "l": 72, "t": 100, "r": 400, "b": 120 },
          "level": 1
        },
        {
          "type": "table",
          "content": "| Col1 | Col2 |\n|------|------|",
          "bbox": { "l": 72, "t": 200, "r": 540, "b": 400 }
        },
        {
          "type": "picture",
          "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
          "bbox": { "l": 72, "t": 450, "r": 400, "b": 650 }
        }
      ]
    }
  ]
}
```

Every element has bounding boxes, page numbers, and image crops ready for multimodal LLMs. Perfect for RAG chunking, provenance tracking, or custom filtering.

---

## Limitations (being honest)

- **No OCR yet** (native text layer only)
- **Equations come out as image crops only**
- **TableFormer still occasionally struggles with weird layouts** (we added fallback reconciliation to prevent silent data loss, but test your docs)
- **MPS (Apple Silicon) support is experimental.** Layout and table inference both run on MPS and produce correct output on our test corpus, but performance varies by table complexity. Long or structurally dense tables may be slower than expected because the autoregressive decode path is inherently sequential. Set `TURBODOCLING_TABLE_MPS=0` to fall back to CPU for table inference if you hit issues.

---

## License

Apache 2.0. Built on top of the excellent [Docling](https://github.com/DS4SD/docling) work from IBM Research (MIT licensed).
