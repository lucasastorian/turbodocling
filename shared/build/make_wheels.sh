#!/usr/bin/env bash
set -euxo pipefail

# Where your repo is mounted inside the builder
WORK=/work
WHEEL_DIR="$WORK/shared/wheels/arm64-py313"
mkdir -p "$WHEEL_DIR"

echo ">>> Building docling_core wheel"
cd "$WORK/shared/docling_core"
rm -rf build/ dist/
python3.13 -m build --wheel --outdir "$WHEEL_DIR"

echo ">>> Building docling_parse native .so + wheel"
cd "$WORK/shared/docling_parse"
rm -rf build/ dist/
# Build the native module into the tree first
python3.13 build.py
# Then package a wheel that includes the .so
python3.13 -m build --wheel --outdir "$WHEEL_DIR"

echo ">>> Wheels in $WHEEL_DIR:"
ls -lah "$WHEEL_DIR"