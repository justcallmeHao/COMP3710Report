#!/bin/bash
set -e

echo "Starting pipeline"

# Prefer Python's cuDNN over the server's incompatible system libraries.
CUDNN_LIB_DIR=$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu13").locate_file("nvidia/cudnn/lib"))')
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python main.py

echo "Finish running pipeline"
