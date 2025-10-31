#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-models/regularized_training/model_swin_t_regularized.pth}
DATA_ROOT=${DATA_ROOT:-data/standardized_jpg/general}

# Guard: ensure the model exists (common when using Git LFS or fresh clone)
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[error] Model not found at $MODEL_PATH" >&2
  echo "Hint: If this repo uses Git LFS, run: git lfs pull" >&2
  echo "Or set MODEL_PATH=/abs/path/to/your_model.pth" >&2
  exit 1
fi

python -m src.evaluation.evaluate_model \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT"
