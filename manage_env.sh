#!/bin/bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-microscopy_index}"
CONTAINER_NAME="${CONTAINER_NAME:-microscopy_index}"

# Host paths (override by exporting env vars before running)
DATA_DIR="${DATA_DIR:-/mnt/nvme8tb/microscopy_index}"
HF_HUB_DIR="${HF_HUB_DIR:-/mnt/nvme8tb/huggingface_cache/hub}"
MODELS_DIR="${MODELS_DIR:-/mnt/nvme8tb/microscopy_index_models}"

mkdir -p "$DATA_DIR" "$HF_HUB_DIR" "$MODELS_DIR"

echo "🏗️  Building image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "🚀 Launching container: $CONTAINER_NAME"
docker run --gpus all -it \
  --name "$CONTAINER_NAME" \
  --ipc=host \
  --shm-size=16gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -v "$DATA_DIR:/app" \
  -v "$HF_HUB_DIR:/root/.cache/huggingface/hub" \
  -v "$MODELS_DIR:/models" \
  "$IMAGE_NAME"
