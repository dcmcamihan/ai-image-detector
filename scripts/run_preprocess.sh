#!/usr/bin/env bash
set -euo pipefail

# Optional: allow overriding image size via IMG_SIZE env (default 224)
: "${IMG_SIZE:=224}"

echo "[info] Preprocessing images (IMG_SIZE=${IMG_SIZE})"
python -m src.preprocess_images --image_size "${IMG_SIZE}"
