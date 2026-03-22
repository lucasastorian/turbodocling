"""
Integration test for the turbodocling PDF processing pipeline.

Usage:
    # First run: generates golden files from pipeline output
    pytest tests/integration/test_pipeline.py --update-golden -v

    # Regression check against golden files
    pytest tests/integration/test_pipeline.py -v

    # Run a single PDF
    pytest tests/integration/test_pipeline.py -k apple_10k -v

Requires a deployed stack. Reads stack outputs from CloudFormation.
"""
import json
import os
import time
import uuid
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Tuple

import boto3
import pytest

GOLDEN_DIR = Path(__file__).parent.parent / "golden"
PDFS_DIR = GOLDEN_DIR / "pdfs"
EXPECTED_DIR = GOLDEN_DIR / "expected"

STACK_NAME = os.environ.get("STACK_NAME", "turbodocling-turbo-dev")
REGION = os.environ.get("AWS_REGION", "us-east-1")

GOLDEN_PDFS = [
    ("apple_10k", "apple_10k.pdf"),
    ("berkshire_letter", "berkshire_letter.pdf"),
    ("attention", "attention.pdf"),
    ("nvidia_investor", "nvidia_investor.pdf"),
]

# Thresholds — fail below these, warn between these and 1.0
LABEL_F1_FAIL = 0.90
LABEL_F1_WARN = 0.95
BBOX_IOU_FAIL = 0.80
BBOX_IOU_WARN = 0.90
TEXT_SIM_FAIL = 0.85
TEXT_SIM_WARN = 0.92
PERF_REGRESSION_PCT = 30  # fail if wall time regresses >30%



@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="session")
def stack_outputs():
    cfn = boto3.client("cloudformation", region_name=REGION)
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = {}
    for o in resp["Stacks"][0]["Outputs"]:
        outputs[o["OutputKey"]] = o["OutputValue"]
    return outputs


@pytest.fixture(scope="session")
def s3_client():
    return boto3.client("s3", region_name=REGION)


@pytest.fixture(scope="session")
def sfn_client():
    return boto3.client("stepfunctions", region_name=REGION)


def _count_pdf_pages(pdf_path: Path) -> int:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def _run_pipeline(pdf_path: Path, stack_outputs: dict, s3_client, sfn_client) -> dict:
    bucket = stack_outputs["DocumentsBucketName"]
    state_machine_arn = stack_outputs["StateMachineArn"]

    job_id = str(uuid.uuid4())
    user_id = "test-user"
    total_pages = _count_pdf_pages(pdf_path)

    s3_key = f"uploads/{user_id}/{job_id}/source.pdf"
    s3_client.upload_file(str(pdf_path), bucket, s3_key)

    t_start = time.time()
    execution = sfn_client.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"test-{job_id[:8]}",
        input=json.dumps({
            "job_id": job_id,
            "user_id": user_id,
            "total_pages": total_pages,
            "batch_size": 1,
        }),
    )
    execution_arn = execution["executionArn"]

    deadline = time.time() + 300
    while time.time() < deadline:
        status = sfn_client.describe_execution(executionArn=execution_arn)
        state = status["status"]

        if state == "SUCCEEDED":
            wall_time = time.time() - t_start
            output = json.loads(status["output"])

            json_key = output["json_key"]
            resp = s3_client.get_object(Bucket=bucket, Key=json_key)
            elements = json.loads(resp["Body"].read().decode("utf-8"))

            md_key = output["md_key"]
            resp = s3_client.get_object(Bucket=bucket, Key=md_key)
            markdown = resp["Body"].read().decode("utf-8")

            return {
                "elements": elements,
                "markdown": markdown,
                "total_pages": output["total_pages"],
                "wall_time": wall_time,
            }
        elif state in ("FAILED", "TIMED_OUT", "ABORTED"):
            error = status.get("error", "unknown")
            cause = status.get("cause", "unknown")
            pytest.fail(f"Step Function {state}: {error} — {cause}")

        time.sleep(2)

    pytest.fail(f"Step Function timed out after 5 minutes")


def _bbox_iou(a: dict, b: dict) -> float:
    """Intersection-over-union for two bounding boxes {l, t, r, b}."""
    x1 = max(a["l"], b["l"])
    y1 = max(a["t"], b["t"])
    x2 = min(a["r"], b["r"])
    y2 = min(a["b"], b["b"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a["r"] - a["l"]) * (a["b"] - a["t"])
    area_b = (b["r"] - b["l"]) * (b["b"] - b["t"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _label_f1(actual_elements: List[dict], expected_elements: List[dict]) -> float:
    """F1 score for element type labels across a page."""
    actual_counts = Counter(e["type"] for e in actual_elements)
    expected_counts = Counter(e["type"] for e in expected_elements)

    all_labels = set(actual_counts) | set(expected_counts)
    if not all_labels:
        return 1.0

    tp = sum(min(actual_counts[l], expected_counts[l]) for l in all_labels)
    fp = sum(max(0, actual_counts[l] - expected_counts[l]) for l in all_labels)
    fn = sum(max(0, expected_counts[l] - actual_counts[l]) for l in all_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _match_elements_greedy(
    actual: List[dict], expected: List[dict]
) -> List[Tuple[dict, dict]]:
    """Greedily match actual elements to expected elements by bbox IoU.

    Returns list of (actual, expected) pairs. Unmatched elements are skipped.
    """
    if not actual or not expected:
        return []

    used_expected = set()
    pairs = []

    for a in actual:
        a_bbox = a.get("bbox")
        if not a_bbox:
            continue

        best_iou = 0.0
        best_idx = -1
        for j, e in enumerate(expected):
            if j in used_expected:
                continue
            e_bbox = e.get("bbox")
            if not e_bbox:
                continue
            iou = _bbox_iou(a_bbox, e_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0 and best_iou > 0.1:
            pairs.append((a, expected[best_idx]))
            used_expected.add(best_idx)

    return pairs


def _page_bbox_iou(actual_elements: List[dict], expected_elements: List[dict]) -> float:
    """Average bbox IoU across matched elements on a page."""
    pairs = _match_elements_greedy(actual_elements, expected_elements)
    if not pairs:
        return 1.0  # no bboxes to compare (e.g. notes)

    ious = [_bbox_iou(a["bbox"], e["bbox"]) for a, e in pairs if a.get("bbox") and e.get("bbox")]
    return sum(ious) / len(ious) if ious else 1.0


def _text_similarity(actual_md: str, expected_md: str) -> float:
    """Normalized text similarity using SequenceMatcher (fast, good enough)."""
    if not actual_md and not expected_md:
        return 1.0
    return SequenceMatcher(None, actual_md, expected_md).ratio()


def _compare_strict(actual: dict, expected: dict, pdf_name: str) -> List[str]:
    """Strict checks that must pass. Returns list of failure messages."""
    failures = []
    actual_pages = actual["elements"]["pages"]
    expected_pages = expected["elements"]["pages"]

    if len(actual_pages) != len(expected_pages):
        failures.append(
            f"page count: got {len(actual_pages)}, expected {len(expected_pages)}"
        )

    # Schema: every element must have type + content
    for i, page in enumerate(actual_pages):
        for j, elem in enumerate(page.get("elements", [])):
            if "type" not in elem:
                failures.append(f"page {i} element {j}: missing 'type'")
            if "content" not in elem:
                failures.append(f"page {i} element {j}: missing 'content'")

    return failures


def _compare_tolerant(actual: dict, expected: dict, pdf_name: str) -> dict:
    """Tolerant quality metrics. Returns dict of metric name → value."""
    actual_pages = actual["elements"]["pages"]
    expected_pages = expected["elements"]["pages"]

    n_pages = min(len(actual_pages), len(expected_pages))
    if n_pages == 0:
        return {"label_f1": 0.0, "bbox_iou": 0.0, "text_sim": 0.0}

    page_label_f1s = []
    page_bbox_ious = []

    for i in range(n_pages):
        act_elems = actual_pages[i].get("elements", [])
        exp_elems = expected_pages[i].get("elements", [])

        page_label_f1s.append(_label_f1(act_elems, exp_elems))
        page_bbox_ious.append(_page_bbox_iou(act_elems, exp_elems))

    text_sim = _text_similarity(
        actual.get("markdown", ""),
        expected.get("markdown", ""),
    )

    return {
        "label_f1": sum(page_label_f1s) / len(page_label_f1s),
        "bbox_iou": sum(page_bbox_ious) / len(page_bbox_ious),
        "text_sim": text_sim,
    }


def _format_report(
    pdf_name: str, strict_failures: List[str], metrics: dict,
    wall_time: float, expected_wall_time: float | None,
) -> str:
    """Format a human-readable test report."""
    lines = [f"\n{'=' * 60}", f"  {pdf_name}", f"{'=' * 60}"]

    # Strict
    if strict_failures:
        lines.append("  STRICT FAILURES:")
        for f in strict_failures:
            lines.append(f"    ✗ {f}")
    else:
        lines.append("  Strict checks: PASS")

    # Tolerant metrics
    lines.append("  Quality metrics:")
    for name, value in metrics.items():
        status = "PASS"
        fail_thresh = {"label_f1": LABEL_F1_FAIL, "bbox_iou": BBOX_IOU_FAIL, "text_sim": TEXT_SIM_FAIL}
        warn_thresh = {"label_f1": LABEL_F1_WARN, "bbox_iou": BBOX_IOU_WARN, "text_sim": TEXT_SIM_WARN}
        if value < fail_thresh.get(name, 0):
            status = "FAIL"
        elif value < warn_thresh.get(name, 0):
            status = "WARN"
        lines.append(f"    {name}: {value:.4f}  [{status}]")

    # Performance
    lines.append(f"  Wall time: {wall_time:.1f}s")
    if expected_wall_time:
        pct_change = (wall_time - expected_wall_time) / expected_wall_time * 100
        perf_status = "FAIL" if pct_change > PERF_REGRESSION_PCT else "OK"
        lines.append(f"  vs golden:  {pct_change:+.1f}%  [{perf_status}]")

    return "\n".join(lines)


@pytest.mark.parametrize("pdf_name,pdf_file", GOLDEN_PDFS)
def test_pipeline(pdf_name, pdf_file, stack_outputs, s3_client, sfn_client, update_golden):
    pdf_path = PDFS_DIR / pdf_file
    assert pdf_path.exists(), f"Golden PDF not found: {pdf_path}"

    golden_path = EXPECTED_DIR / f"{pdf_name}.json"

    print(f"\nProcessing {pdf_name}...")
    result = _run_pipeline(pdf_path, stack_outputs, s3_client, sfn_client)
    print(f"  Completed in {result['wall_time']:.1f}s — {result['total_pages']} pages")

    if update_golden:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

        # Compute diff summary if previous golden exists
        diff_summary = None
        if golden_path.exists():
            with open(golden_path) as f:
                old = json.load(f)
            old_pages = len(old["elements"]["pages"])
            old_elems = sum(len(p["elements"]) for p in old["elements"]["pages"])
            new_pages = len(result["elements"]["pages"])
            new_elems = sum(len(p["elements"]) for p in result["elements"]["pages"])
            diff_summary = (
                f"pages: {old_pages} → {new_pages}, "
                f"elements: {old_elems} → {new_elems}, "
                f"md_len: {old.get('markdown_length', '?')} → {len(result['markdown'])}"
            )

        golden_data = {
            "elements": result["elements"],
            "markdown": result["markdown"],
            "markdown_length": len(result["markdown"]),
            "total_pages": result["total_pages"],
            "wall_time": result["wall_time"],
        }
        with open(golden_path, "w") as f:
            json.dump(golden_data, f, indent=2, ensure_ascii=False)

        msg = f"  Updated golden: {golden_path.name}"
        if diff_summary:
            msg += f"\n  Diff: {diff_summary}"
        print(msg)
        return

    # Regression mode
    assert golden_path.exists(), (
        f"Golden file not found: {golden_path}\n"
        f"Run with --update-golden to create it."
    )
    with open(golden_path) as f:
        expected = json.load(f)

    # Strict checks
    strict_failures = _compare_strict(result, expected, pdf_name)

    # Tolerant metrics
    metrics = _compare_tolerant(
        {**result, "markdown": result["markdown"]},
        {**expected, "markdown": expected.get("markdown", "")},
        pdf_name,
    )

    expected_wall = expected.get("wall_time")
    report = _format_report(pdf_name, strict_failures, metrics, result["wall_time"], expected_wall)
    print(report)

    # Assert strict
    assert not strict_failures, f"{pdf_name} strict failures:\n" + "\n".join(strict_failures)

    # Assert tolerant thresholds
    assert metrics["label_f1"] >= LABEL_F1_FAIL, (
        f"{pdf_name}: label F1 {metrics['label_f1']:.4f} < {LABEL_F1_FAIL}"
    )
    assert metrics["bbox_iou"] >= BBOX_IOU_FAIL, (
        f"{pdf_name}: bbox IoU {metrics['bbox_iou']:.4f} < {BBOX_IOU_FAIL}"
    )
    assert metrics["text_sim"] >= TEXT_SIM_FAIL, (
        f"{pdf_name}: text similarity {metrics['text_sim']:.4f} < {TEXT_SIM_FAIL}"
    )

    # Performance regression
    if expected_wall:
        pct_change = (result["wall_time"] - expected_wall) / expected_wall * 100
        assert pct_change <= PERF_REGRESSION_PCT, (
            f"{pdf_name}: wall time regression {pct_change:+.1f}% "
            f"({result['wall_time']:.1f}s vs {expected_wall:.1f}s golden)"
        )

    # Warn-level output
    warnings = []
    if metrics["label_f1"] < LABEL_F1_WARN:
        warnings.append(f"label F1 drifting: {metrics['label_f1']:.4f}")
    if metrics["bbox_iou"] < BBOX_IOU_WARN:
        warnings.append(f"bbox IoU drifting: {metrics['bbox_iou']:.4f}")
    if metrics["text_sim"] < TEXT_SIM_WARN:
        warnings.append(f"text similarity drifting: {metrics['text_sim']:.4f}")
    if warnings:
        print(f"  WARNINGS: {'; '.join(warnings)}")
