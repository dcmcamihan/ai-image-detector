#!/usr/bin/env bash
set -euo pipefail
mkdir -p models/regularized_training
curl -L -o models/regularized_training/model_swin_t_regularized.pth \
  "https://github.com/<your-username>/<your-repo>/releases/download/v1.0.0/model_swin_t_regularized.pth"
echo "Downloaded to models/regularized_training/model_swin_t_regularized.pth"
