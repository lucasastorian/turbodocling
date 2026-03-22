"""Golden file regression tests.

Runs all PDFs through the pipeline, compares output against golden expected files.
Catches regressions in table structure, element extraction, and markdown output.

Usage:
    python tests/test_golden.py                  # Compare against golden files
    python tests/test_golden.py --update-golden   # Regenerate golden files from current output
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import boto3
import pypdfium2 as pdfium

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
PDFS_DIR = GOLDEN_DIR / "pdfs"
EXPECTED_DIR = GOLDEN_DIR / "expected"

DEFAULT_STACK = "turbodocling-turbo-dev"
DEFAULT_REGION = "us-east-1"


def get_stack_resources(stack_name: str, region: str):
    cfn = boto3.client("cloudformation", region_name=region)
    resp = cfn.describe_stacks(StackName=stack_name)
    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
    return outputs["DocumentsBucketName"], outputs["StateMachineArn"]


def run_pdf(pdf_path: Path, bucket: str, state_machine_arn: str, region: str):
    s3 = boto3.client("s3", region_name=region)
    sfn = boto3.client("stepfunctions", region_name=region)

    doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    job_id = str(uuid.uuid4())
    user_id = "test-user"
    s3_key = f"uploads/{user_id}/{job_id}/source.pdf"
    s3.upload_file(str(pdf_path), bucket, s3_key)

    import math
    batch_size = max(1, math.ceil(total_pages / 40))

    t_start = time.time()
    execution = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"golden-{job_id[:8]}",
        input=json.dumps({
            "job_id": job_id,
            "user_id": user_id,
            "total_pages": total_pages,
            "batch_size": batch_size,
        }),
    )

    deadline = time.time() + 300
    while time.time() < deadline:
        status = sfn.describe_execution(executionArn=execution["executionArn"])
        state = status["status"]

        if state == "SUCCEEDED":
            output = json.loads(status["output"])
            wall_time = time.time() - t_start

            json_key = output["json_key"]
            resp = s3.get_object(Bucket=bucket, Key=json_key)
            elements = json.loads(resp["Body"].read().decode("utf-8"))

            md_key = output["md_key"]
            resp = s3.get_object(Bucket=bucket, Key=md_key)
            markdown = resp["Body"].read().decode("utf-8")

            return {
                "elements": elements,
                "markdown": markdown,
                "markdown_length": len(markdown),
                "total_pages": total_pages,
                "wall_time": wall_time,
            }

        elif state in ("FAILED", "TIMED_OUT", "ABORTED"):
            raise RuntimeError(f"{pdf_path.name}: {state} - {status.get('error', 'unknown')}")

        time.sleep(3)

    raise TimeoutError(f"{pdf_path.name}: timed out after 5 minutes")


def compare_tables(golden_elements, current_elements, pdf_name: str) -> list:
    failures = []
    golden_pages = {p["page_no"]: p for p in golden_elements["pages"]}
    current_pages = {p["page_no"]: p for p in current_elements["pages"]}

    for page_no, gp in golden_pages.items():
        cp = current_pages.get(page_no)
        if not cp:
            failures.append(f"  {pdf_name} page {page_no}: missing in current output")
            continue

        g_tables = [e for e in gp["elements"] if e.get("type") == "table"]
        c_tables = [e for e in cp["elements"] if e.get("type") == "table"]

        if len(g_tables) != len(c_tables):
            failures.append(
                f"  {pdf_name} page {page_no}: table count {len(g_tables)} -> {len(c_tables)}"
            )

        for i, gt in enumerate(g_tables):
            if i >= len(c_tables):
                failures.append(f"  {pdf_name} page {page_no} table {i}: missing")
                continue
            ct = c_tables[i]
            if gt["content"] != ct["content"]:
                g_lines = gt["content"].split("\n")
                c_lines = ct["content"].split("\n")
                diff_count = sum(1 for a, b in zip(g_lines, c_lines) if a != b)
                diff_count += abs(len(g_lines) - len(c_lines))
                failures.append(
                    f"  {pdf_name} page {page_no} table {i}: "
                    f"{diff_count} lines differ (golden={len(g_lines)} current={len(c_lines)})"
                )

    return failures


def compare_elements(golden_elements, current_elements, pdf_name: str) -> list:
    failures = []
    golden_pages = {p["page_no"]: p for p in golden_elements["pages"]}
    current_pages = {p["page_no"]: p for p in current_elements["pages"]}

    if len(golden_pages) != len(current_pages):
        failures.append(
            f"  {pdf_name}: page count {len(golden_pages)} -> {len(current_pages)}"
        )

    for page_no, gp in golden_pages.items():
        cp = current_pages.get(page_no)
        if not cp:
            continue

        g_count = len(gp["elements"])
        c_count = len(cp["elements"])
        if g_count != c_count:
            failures.append(
                f"  {pdf_name} page {page_no}: element count {g_count} -> {c_count}"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Golden file regression tests")
    parser.add_argument("--update-golden", action="store_true", help="Regenerate golden files")
    parser.add_argument("--stack", default=DEFAULT_STACK)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--pdf", default=None, help="Run only this PDF (filename, not path)")
    args = parser.parse_args()

    pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    if args.pdf:
        pdfs = [p for p in pdfs if p.name == args.pdf]
        if not pdfs:
            print(f"PDF not found: {args.pdf}")
            sys.exit(1)

    if not pdfs:
        print(f"No PDFs found in {PDFS_DIR}")
        sys.exit(1)

    bucket, state_machine_arn = get_stack_resources(args.stack, args.region)
    print(f"Stack: {args.stack} | Bucket: {bucket}")
    print(f"PDFs: {', '.join(p.name for p in pdfs)}\n")

    all_failures = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        golden_path = EXPECTED_DIR / f"{stem}.json"

        print(f"Running {pdf_path.name}...", end=" ", flush=True)
        result = run_pdf(pdf_path, bucket, state_machine_arn, args.region)
        print(f"{result['total_pages']} pages, {result['wall_time']:.1f}s")

        if args.update_golden:
            EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
            with open(golden_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Updated {golden_path.name}")
            continue

        if not golden_path.exists():
            print(f"  SKIP - no golden file (run with --update-golden)")
            continue

        with open(golden_path) as f:
            golden = json.load(f)

        # Compare tables (strictest check)
        table_failures = compare_tables(golden["elements"], result["elements"], pdf_path.name)

        # Compare element counts
        element_failures = compare_elements(golden["elements"], result["elements"], pdf_path.name)

        failures = table_failures + element_failures
        if failures:
            print(f"  FAIL ({len(failures)} issues)")
            for f in failures:
                print(f)
            all_failures.extend(failures)
        else:
            print(f"  PASS")

    if args.update_golden:
        print(f"\nGolden files updated for {len(pdfs)} PDFs")
        return

    print(f"\n{'='*60}")
    if all_failures:
        print(f"FAILED: {len(all_failures)} regressions across {len(pdfs)} PDFs")
        sys.exit(1)
    else:
        print(f"PASSED: all {len(pdfs)} PDFs match golden files")


if __name__ == "__main__":
    main()
