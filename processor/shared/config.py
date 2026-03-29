import os

MAX_SQS_BATCH = int(os.getenv("MAX_SQS_BATCH", "4"))
LAYOUT_ENQ_BATCH = int(os.getenv("LAYOUT_ENQ_BATCH", "32"))
MAX_LOCAL_PAGES = int(os.getenv("MAX_LOCAL_PAGES", "256"))
MAX_INFLIGHT_PAGES = int(os.getenv("MAX_INFLIGHT_PAGES", "512"))
MAX_INFLIGHT_DOCS = int(os.getenv("MAX_INFLIGHT_DOCS", "0"))
MAX_MEMORY_UTILIZATION = float(os.getenv("MAX_MEMORY_UTILIZATION", "0.50"))
