FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    HF_HUB_OFFLINE=1

# --- Layer 1: docling-core (pure Python) ---
COPY shared/docling_core /app/shared/docling_core
RUN pip install --no-cache-dir -e /app/shared/docling_core

# --- Layer 2: docling fork (pulls in docling-ibm-models, scipy, etc.) ---
# docling-parse is commented out in fork deps — GPU service doesn't need it
COPY shared/docling /app/shared/docling
RUN pip install --no-cache-dir -e /app/shared/docling

# --- Layer 3: remaining GPU service deps ---
RUN pip install --no-cache-dir \
    aiohttp \
    boto3 \
    msgpack \
    opencv-python-headless \
    safetensors \
    torchvision \
    transformers

# --- Layer 3.5: bake HuggingFace models into image (no internet at runtime) ---
RUN HF_HUB_OFFLINE=0 python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ds4sd/docling-layout-old', revision='main', cache_dir='/opt/huggingface/hub', local_files_only=False); \
snapshot_download('ds4sd/docling-models', revision='v2.2.0', cache_dir='/opt/huggingface/hub', local_files_only=False); \
print('Models cached successfully')"

# --- Layer 4: service code (changes most often) ---
COPY processor /app/processor

ENV PYTHONPATH=/app
ENV DEVICE=cuda

CMD ["python", "-m", "processor.main"]
