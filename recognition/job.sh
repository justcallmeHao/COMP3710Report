#!/bin/bash
set -e

echo "Starting project"

# Prefer Python's cuDNN over the server's incompatible system libraries.
CUDNN_LIB_DIR=$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu13").locate_file("nvidia/cudnn/lib"))')
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Preprocess all files for lighter workload"

python preloader.py

echo "Done preprocess"

echo "Train 3D improved unet"

python main3d.py

echo "Done 3D improved"

echo "Train 2D unet"

python main3d.py

echo "Done 2D unet"

echo "Train 3D normal unet"

python main3dnormal.py

echo "Done 3D unet"

echo "Finish running pipeline"
