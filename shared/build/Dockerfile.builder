FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.13

# Tooling needed for docling-parse native build
RUN microdnf update -y && microdnf install -y \
      gcc-c++ make cmake zlib-devel git \
  && microdnf clean all

# Python build helpers
RUN python3.13 -m pip install --no-cache-dir \
      "pybind11>=2.13.6" "cmake>=3.27.0" \
      "setuptools>=77.0.3" "wheel>=0.43.0" \
      "build>=1.2.1"

WORKDIR /builder

# Override entrypoint to allow running bash commands
ENTRYPOINT []