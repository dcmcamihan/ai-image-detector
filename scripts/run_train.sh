#!/usr/bin/env bash
set -euo pipefail

# Train with regularization (dropout, weight decay, label smoothing)
python -m src.training.train_regularized --config config/training_regularized.yaml
