#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-models/regularized_training/regularized_general.pth}
DATA_ROOT=${DATA_ROOT:-data/standardized_jpg/general}

python -m src.evaluation.evaluate_model \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT"
