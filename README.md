<p align="center">
  <img src="docs/images/logo.png" alt="Turbodocling" width="480">
</p>
<h1 align="center">Turbodocling</h1>
<p align="center">
  <strong>Docling, but actually fast enough for real apps.</strong><br>
  5–6× faster on GPU &bull; runs locally on your laptop in seconds<br>
  Zero AWS required to start. Deploy to AWS only when you need scale.
</p>

---

Stock Docling is great at quality. Terrible at speed.
A 93-page 10-K takes **71 seconds** on an A10G.

Turbodocling is the same high-quality pipeline, completely re-architected.
Same 10-K now finishes in **13 seconds** on AWS.
On a MacBook Pro M1 Max it processes a 48-page 10-Q in **~24 seconds** via MPS. Still optimizing — target is sub-20s.

That's fast enough to drop a PDF into your chat UI and get structured Markdown + elements back inline. No background jobs. No "we'll email you later."

### Real numbers (AWS GPU)

| Document          | Pages | Turbodocling | Stock Docling | Speedup |
|-------------------|-------|--------------|---------------|---------|
| Docling paper     | 8     | **3.0 s**    | 8.1 s         | 2.7×    |
| NVIDIA 10-Q       | 48    | **6.5 s**    | 40.7 s        | 6.3×    |
| NVIDIA 10-K       | 93    | **12.8 s**   | 71.0 s        | 5.5×    |

**20 concurrent 93-page 10-Ks (1 860 pages):** **16.7 pages/sec** vs ~1.3 pages/sec stock.

### Local performance (Apple Silicon)

- 48-page 10-Q → **~24 s** on M1 Max (MPS)
- Works out of the box on CPU / CUDA / MPS
- No Docker, no cloud, no credentials

---

## Getting Started (local first)

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

One command. Outputs:
- `output/output.md` — clean Markdown
- `output/elements.json` — structured elements with bboxes + image crops

**Options:**
- `--device auto|cuda|mps|cpu` (default: auto-detect)
- `--workers N` (default: CPU core count)

**Python API:**

```python
from turbodocling.local_runner import run_local

result = run_local("my_document.pdf", output_dir="output/", device="auto")
print(result.wall_time_s, result.md_path, result.elements_path)
```

---

### Deploy to AWS (only if you need production scale)

```bash
cdk deploy
```

Then call the Step Function:

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

---

## Cost that actually makes sense (AWS only)

**Turbodocling cost per 10 000 pages**

| GPU utilization       | Cost      | vs Textract ($150) | vs Mistral ($30) |
|-----------------------|-----------|--------------------|------------------|
| 100% (batch/backfill) | **$0.28** | 536× cheaper       | 107× cheaper     |
| 30% (real-time)       | **$0.74** | 203× cheaper       | 41× cheaper      |

Spot instances drop the GPU portion another ~70%.
One always-on A10G is ~$870/mo on-demand — still orders of magnitude cheaper than API services.

---

## Architecture (the part that actually matters)

**Local** → single-process, multi-threaded + MPS/CUDA/CPU
**AWS** → Step Function → 40 parallel Lambdas (parse + render) → A10G worker (inference only)

Everything CPU-heavy is off the critical path. TableFormer is fully batched + KV-cached. The rest is hundreds of vectorized + zero-copy optimizations.

### The three big changes we made

1. **Split CPU and GPU work**
   Stock Docling runs everything on the same machine. You pay A10G prices for parsing and image rendering. We moved that to cheap parallel Lambdas.

2. **Rewrote TableFormer from scratch**
   No more one-table-at-a-time. No more recomputing attention every token. Batched + KV-cached + multi-threaded. Huge win.

3. **Everything else optimized to death**
   Vectorized numpy everywhere, spatial indexes instead of O(N²) loops, C++ parser patches, zero-copy data paths, multi-threaded pipeline with backpressure. The boring stuff that adds up.

---

## Working with elements.json (what you'll actually use)

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

Every element includes bounding boxes and (for pictures) ready-to-use base64 crops. Perfect for RAG, multimodal prompts, provenance, or custom filtering.

---

## Limitations (being honest)

- **No OCR yet** (native text layer only)
- **Equations come out as image crops only**
- **TableFormer occasionally struggles with very weird layouts** (fallback reconciliation prevents data loss, but test your docs)
- **MPS (Apple Silicon) support is experimental.** Layout and table inference both run on MPS and produce correct output on our test corpus, but performance varies by table complexity. Long or structurally dense tables may be slower because the autoregressive decode path is inherently sequential. Set `TURBODOCLING_TABLE_MPS=0` to fall back to CPU for table inference if needed.

---

## License

Apache 2.0. Built on the excellent [Docling](https://github.com/DS4SD/docling) work from IBM Research (MIT licensed).

---

**Local is the default.** Try it in 30 seconds. Deploy to AWS only when you need massive throughput.
We're still pushing local performance harder — next target: sub-20s on M1/M2/M3 for a 48-page 10-Q.
