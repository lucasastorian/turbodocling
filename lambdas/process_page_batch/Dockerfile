FROM public.ecr.aws/lambda/python:3.13

ARG PDFIUM_VARIANT=upstream
ARG PYPDFIUM2_WHEEL_URL=
ARG PYPDFIUM2_SPEC=pypdfium2==5.6.0

# Install system dependencies for building
RUN microdnf update -y && microdnf install -y \
    gcc-c++ \
    cmake \
    make \
    zlib-devel \
    git \
    && microdnf clean all

# Copy Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN python3.13 -m pip install --no-cache-dir -r requirements.txt
RUN set -eux; \
    case "$PDFIUM_VARIANT" in \
      upstream) \
        python3.13 -m pip install --no-cache-dir "$PYPDFIUM2_SPEC" \
        ;; \
      experimental) \
        test -n "$PYPDFIUM2_WHEEL_URL"; \
        python3.13 -m pip install --no-cache-dir "$PYPDFIUM2_WHEEL_URL" \
        ;; \
      *) \
        echo "Unsupported PDFIUM_VARIANT: $PDFIUM_VARIANT" >&2; \
        exit 1 \
        ;; \
    esac

# Copy our custom libraries (nested structure)
COPY ../../shared/docling_core/docling_core/ ${LAMBDA_TASK_ROOT}/docling_core/
COPY ../../shared/docling_parse/ ${LAMBDA_TASK_ROOT}/docling_parse/

# Build docling-parse from source
WORKDIR ${LAMBDA_TASK_ROOT}/docling_parse
RUN rm -rf build/ && python3.13 build.py

# Install the built docling-parse
RUN python3.13 -m pip install --no-cache-dir --no-deps .

# Shared serialization module
COPY ../../shared/__init__.py ${LAMBDA_TASK_ROOT}/shared/__init__.py
COPY ../../shared/page_serialization.py ${LAMBDA_TASK_ROOT}/shared/page_serialization.py

# Copy Lambda function code
WORKDIR ${LAMBDA_TASK_ROOT}
COPY handler.py preprocessing_pipeline.py ./

# Clean up build artifacts to reduce image size
RUN find ${LAMBDA_TASK_ROOT} -name "*.pyc" -delete
RUN find ${LAMBDA_TASK_ROOT} -name "__pycache__" -type d -exec rm -rf {} + || true
RUN rm -rf ${LAMBDA_TASK_ROOT}/docling_parse/build/

CMD ["handler.lambda_handler"]
