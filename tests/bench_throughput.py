"""Throughput benchmark: submit N copies of a PDF simultaneously to the pipeline."""
import argparse
import json
import math
import time
import uuid
from pathlib import Path

import boto3
import pypdfium2 as pdfium

parser = argparse.ArgumentParser(description="Throughput benchmark for turbodocling pipeline")
parser.add_argument("pdf", help="Path to PDF")
parser.add_argument("--copies", type=int, default=20, help="Number of concurrent copies (default: 20)")
parser.add_argument("--stack", default="turbodocling-turbo-dev")
parser.add_argument("--region", default="us-east-1")
args = parser.parse_args()

PDF_PATH = Path(args.pdf)
N = args.copies

cfn = boto3.client("cloudformation", region_name=args.region)
s3 = boto3.client("s3", region_name=args.region)
sfn = boto3.client("stepfunctions", region_name=args.region)

# Stack outputs
resp = cfn.describe_stacks(StackName=args.stack)
outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
bucket = outputs["DocumentsBucketName"]
state_machine_arn = outputs["StateMachineArn"]

# Count pages
doc = pdfium.PdfDocument(str(PDF_PATH))
total_pages = len(doc)
doc.close()
batch_size = max(1, math.ceil(total_pages / 40))

print(f"=== Turbodocling Throughput Benchmark ===")
print(f"Document: {PDF_PATH.name} ({total_pages} pages)")
print(f"Copies: {N} ({N * total_pages} total pages)")
print(f"Batch size: {batch_size}")
print()

# Upload N copies and start N executions
executions = []
t_start = time.time()

for i in range(N):
    job_id = str(uuid.uuid4())
    user_id = "test-user"
    s3_key = f"uploads/{user_id}/{job_id}/source.pdf"
    s3.upload_file(str(PDF_PATH), bucket, s3_key)

    execution = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"bench-{job_id[:8]}",
        input=json.dumps({
            "job_id": job_id,
            "user_id": user_id,
            "total_pages": total_pages,
            "batch_size": batch_size,
        }),
    )
    executions.append((i, execution["executionArn"]))

t_uploaded = time.time()
print(f"All {N} jobs submitted in {t_uploaded - t_start:.1f}s")
print(f"Polling for completion...", flush=True)

# Poll all executions
completed = {}
deadline = time.time() + 600

while len(completed) < N and time.time() < deadline:
    for i, arn in executions:
        if i in completed:
            continue
        status = sfn.describe_execution(executionArn=arn)
        state = status["status"]
        if state == "SUCCEEDED":
            sfn_elapsed = (status["stopDate"] - status["startDate"]).total_seconds()
            wall_elapsed = time.time() - t_start
            completed[i] = sfn_elapsed
            print(f"  Job {i+1:2d}/{N}: {sfn_elapsed:.1f}s (wall: {wall_elapsed:.1f}s)", flush=True)
        elif state in ("FAILED", "TIMED_OUT", "ABORTED"):
            completed[i] = -1
            print(f"  Job {i+1:2d}/{N}: {state}", flush=True)
    if len(completed) < N:
        time.sleep(2)

t_total = time.time() - t_start
pages_total = N * total_pages
successful = [v for v in completed.values() if v > 0]

print(f"\n=== RESULTS ===")
print(f"Total wall time: {t_total:.1f}s")
print(f"Total pages processed: {pages_total}")
print(f"Throughput: {pages_total / t_total:.1f} pages/second")
print(f"Successful: {len(successful)}/{N}")
if successful:
    print(f"Avg per-job time: {sum(successful) / len(successful):.1f}s")
    print(f"Min/Max per-job: {min(successful):.1f}s / {max(successful):.1f}s")
