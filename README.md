<p align="center">
  <img src="docs/images/logo.png" alt="Turbodocling" width="480">
</p>

<h1 align="center">Turbodocling</h1>

<p align="center">
  <strong>Docling, re-architected for speed, cost, and production use on AWS.</strong><br>
  Deploy in minutes with CDK. Process documents in seconds.
</p>

---

**5–6× faster &bull; 100–500× cheaper than Textract/Mistral OCR &bull; Production ready**

Turbodocling is a ground-up rewrite of [Docling](https://github.com/DS4SD/docling) optimized for high throughput and low cost on AWS.

Stock Docling delivers excellent extraction quality but is too slow for production — 71 seconds for a 93-page 10-K. Commercial APIs are fast but expensive ($30–150 per 10K pages). Turbodocling makes high-quality PDF → Markdown + structured extraction usable and affordable.

### Performance

| Document             | Pages | Turbodocling | Stock Docling | Speedup |
|----------------------|-------|--------------|---------------|---------|
| Docling paper        | 8     | **3.0s**     | 8.1s          | 2.7×    |
| NVIDIA 10-Q          | 48    | **6.5s**     | 40.7s         | 6.3×    |
| NVIDIA 10-K          | 93    | **12.8s**    | 71.0s         | 5.5×    |

**Throughput** (20× concurrent 93-page 10-Ks, 1860 pages): **16.7 pages/sec** vs ~1.3 pages/sec stock. Zero OOMs.

### Cost

**~$0.28 per 10,000 pages** (Lambda + Step Functions + A10G).

| Service        | Cost per 10K pages |
|----------------|-------------------|
| AWS Textract   | **$150**          |
| Mistral OCR    | **$30**           |
| Turbodocling   | **$0.28**         |

**100–500× cheaper** at scale, with better structure and provenance.

---

## Architecture

```
                    ┌─ Lambda 1:  pages 1-2   [parse + render] ─┐
                    ├─ Lambda 2:  pages 3-4   [parse + render] ─┤
PDF → Step Function ├─ Lambda 3:  pages 5-6   [parse + render] ─┼→ SQS → GPU Worker
                    ├─ ...                                      │   (inference only)
                    └─ Lambda 40: pages 79-80 [parse + render] ─┘
```

**Lambda (ARM64, SnapStart):** Parses PDF pages with a custom C++ [docling-parse](https://github.com/DS4SD/docling-parse) fork, renders page images, packs into columnar msgpack batches, uploads to S3. 1–3 seconds per batch.

**GPU Worker (A10G, ECS):** Polls SQS, downloads batches, runs layout + table inference, postprocesses, assembles structured output. Internally multi-threaded — inference, postprocessing, and assembly overlap via bounded queues with backpressure.

**Batch sizing:** Stepped rule in the Step Function — 1 page/batch up to 40 pages, 2 up to 80, etc. An 80-page doc fans out across 40 parallel Lambdas; larger docs cap at 6 pages per batch.

---

## What We Changed

Three changes account for most of the speedup.

### 1. CPU/GPU Decoupling

Stock Docling couples CPU and GPU work on the same machine. PDF parsing, image rendering, and page image creation all happen inline on the GPU instance — you're paying A10G rates for pure CPU work.

We moved every piece of CPU work off the GPU and into **up to 40 parallel Lambda functions** via Step Function fan-out. The GPU receives pre-parsed pages and focuses exclusively on inference and assembly. An 80-page doc that took ~40s to preprocess serially now finishes in ~5s wall-clock.

### 2. TableFormer Rewrite

Stock Docling's TableFormer — the autoregressive transformer that predicts table structure — has no batching (tables processed one at a time), no KV cache (re-attends to all previous tokens every step), and inline execution (GPU blocks on CPU work between tables).

We rewrote the entire inference path: preallocated KV caches, batched decoding across all tables simultaneously, fused QKV projections, BF16 for transformer layers, and decoupled matching/postprocessing onto a dedicated CPU thread that overlaps with the next GPU batch.

### 3. Aggressive Optimization Everywhere Else

Hundreds of optimizations across every stage. Common themes:

- **Vectorize with numpy.** Replace nested Python loops with broadcasting for overlap detection, cell matching, IoU computation, coordinate transforms.
- **Spatial indexing.** Grid-based O(1) lookups and interval trees instead of brute-force O(N×M) pairwise comparisons for reading order and element clustering.
- **Reduce object churn.** Columnar data layouts, string interning, lazy materialization instead of tens of thousands of short-lived Python objects.
- **C++ parser patches.** Skip path operators the downstream pipeline never uses (3× on graphics-heavy pages), const-ref dispatch, eliminated per-row header lookups.
- **Layout model tuning.** Fixed-size 640×640 batches of 32, channels-last + TF32 on Ampere, GPU preprocessing, persistent streams to overlap H2D and compute.
- **Multi-threaded GPU pipeline.** Inference, postprocessing, and assembly run on dedicated threads with bounded queues and backpressure.

---

## Getting Started

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
import boto3, json, uuid
import pypdfium2 as pdfium

STEP_FUNCTION_ARN = "arn:aws:states:us-east-1:123456789:stateMachine:..."
BUCKET = "turbodocling-turbo-dev-documentsbucket-..."

s3 = boto3.client("s3")
sfn = boto3.client("stepfunctions")

# Upload and process
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

# Results land in S3:
#   processed/user/{job_id}/output.md       — structured markdown
#   processed/user/{job_id}/elements.json   — elements with bboxes per page
```

### Working with Elements

`output.md` is a flat Markdown rendering for quick inspection. For production use, `elements.json` is the primary output — pages with elements in reading order:

```
elements.json
└── pages[]
    ├── page_no, width, height
    └── elements[]
        ├── type          — text, section_header, table, picture, list_item, ...
        ├── content       — extracted text or markdown
        ├── bbox          — { l, t, r, b } in PDF points (provenance)
        ├── image_base64  — PNG crop for pictures (feed directly to multimodal LLMs)
        └── level         — heading depth for section_header elements
```

Every element traces back to a specific region on a specific page. Use it to chunk by page for RAG, feed images to multimodal LLMs, track provenance, or filter by element type.

---

## Limitations

- **No OCR yet.** Native text layer only — scanned/image-only PDFs produce no output.
- **No equation → LaTeX yet.** Equations are extracted as image crops only.
- **Table structure is model-dependent.** TableFormer occasionally drops rows or misaligns columns on unusual layouts. We include reconciliation for unmatched tokens, but verify on your document types.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Built on the excellent work of the [Docling](https://github.com/DS4SD/docling) team at IBM Research (MIT-licensed).
