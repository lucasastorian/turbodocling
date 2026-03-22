"""Quick smoke test: upload a PDF, start the Step Function, poll for result."""
import argparse
import json
import time
import uuid
from pathlib import Path

import boto3
import pypdfium2 as pdfium

parser = argparse.ArgumentParser(description="Run a PDF through the turbodocling pipeline")
parser.add_argument("pdf", nargs="?", default=None, help="Path to PDF (default: berkshire_letter.pdf)")
parser.add_argument("--stack", default="turbodocling-turbo-dev", help="CloudFormation stack name")
parser.add_argument("--region", default="us-east-1")
args = parser.parse_args()

PDF_PATH = Path(args.pdf) if args.pdf else Path(__file__).parent / "golden" / "pdfs" / "berkshire_letter.pdf"

cfn = boto3.client("cloudformation", region_name=args.region)
s3 = boto3.client("s3", region_name=args.region)
sfn = boto3.client("stepfunctions", region_name=args.region)

# Get stack outputs
resp = cfn.describe_stacks(StackName=args.stack)
outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
bucket = outputs["DocumentsBucketName"]
state_machine_arn = outputs["StateMachineArn"]

print(f"Bucket: {bucket}")
print(f"State Machine: {state_machine_arn}")

# Count pages
doc = pdfium.PdfDocument(str(PDF_PATH))
total_pages = len(doc)
doc.close()
print(f"PDF: {PDF_PATH.name} ({total_pages} pages, {PDF_PATH.stat().st_size / 1024:.0f}KB)")

# Upload
job_id = str(uuid.uuid4())
user_id = "test-user"
s3_key = f"uploads/{user_id}/{job_id}/source.pdf"
s3.upload_file(str(PDF_PATH), bucket, s3_key)
print(f"Uploaded to s3://{bucket}/{s3_key}")

# Compute batch_size to fit within one Map wave (max 40 concurrency)
import math
batch_size = max(1, math.ceil(total_pages / 40))
print(f"batch_size={batch_size} ({math.ceil(total_pages / batch_size)} Lambda invocations)")

# Start execution
t_start = time.time()
execution = sfn.start_execution(
    stateMachineArn=state_machine_arn,
    name=f"smoke-{job_id[:8]}",
    input=json.dumps({
        "job_id": job_id,
        "user_id": user_id,
        "total_pages": total_pages,
        "batch_size": batch_size,
    }),
)
execution_arn = execution["executionArn"]
print(f"Started execution: {execution_arn}")

# Poll
deadline = time.time() + 300
while time.time() < deadline:
    status = sfn.describe_execution(executionArn=execution_arn)
    state = status["status"]
    elapsed = time.time() - t_start

    if state == "SUCCEEDED":
        output = json.loads(status["output"])
        print(f"\n=== SUCCESS in {elapsed:.1f}s ===")
        print(f"Output: {json.dumps(output, indent=2)}")

        # Download and inspect results
        json_key = output["json_key"]
        resp = s3.get_object(Bucket=bucket, Key=json_key)
        elements = json.loads(resp["Body"].read().decode("utf-8"))
        n_pages = len(elements.get("pages", []))
        n_elements = sum(len(p.get("elements", [])) for p in elements.get("pages", []))

        md_key = output["md_key"]
        resp = s3.get_object(Bucket=bucket, Key=md_key)
        markdown = resp["Body"].read().decode("utf-8")

        print(f"\nResults:")
        print(f"  Pages: {n_pages}")
        print(f"  Elements: {n_elements}")
        print(f"  Markdown length: {len(markdown)} chars")
        print(f"  First 500 chars of markdown:")
        print(f"  {markdown[:500]}")
        break

    elif state in ("FAILED", "TIMED_OUT", "ABORTED"):
        error = status.get("error", "unknown")
        cause = status.get("cause", "unknown")
        print(f"\n=== {state} after {elapsed:.1f}s ===")
        print(f"Error: {error}")
        print(f"Cause: {cause}")
        break

    print(f"  [{elapsed:.0f}s] {state}...")
    time.sleep(3)
else:
    print(f"\n=== TIMED OUT after 5 minutes ===")
