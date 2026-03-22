import time
from typing import Dict, Any

from preprocessing_pipeline import PreprocessingPipeline


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda handler for PDF page batch processing."""
    t0 = time.time()
    print("Processing batch:", event)

    try:
        job_id, user_id = event["job_id"], event["user_id"]
        start_page, total_pages = event["start_page"], event["total_pages"]
    except KeyError as e:
        raise ValueError(f"Missing required field: {e}")

    end_page = min(start_page + event.get("batch_size", 2) - 1, total_pages - 1)
    print(f"Processing pages {start_page}-{end_page} of {total_pages}")

    result = PreprocessingPipeline(user_id).process_batch(
        job_id=job_id,
        start_page=start_page,
        end_page=end_page,
        total_pages=total_pages,
    )

    print(f"TOTAL REQUEST TIME: {time.time() - t0:.2f}s")
    return result
