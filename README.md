# AI Image Detector

Detect AI-generated images vs natural images across multiple generators (ADM, BigGAN, GLIDE, Midjourney, SDv5, VQDM, Wukong).

## Folder structure

- data/
  - raw/: original datasets (not tracked)
  - processed/: preprocessed 224x224 images (not tracked)
  - previews/: dataset preview PNGs (not tracked)
- src/
  - preprocess_images.py: preprocess raw → processed
  - dataset_loader.py: build PyTorch DataLoaders
  - models/: ResNet/ViT definitions and factory
  - training/: training entry + utils + config
  - training/config.yaml: legacy config placeholder (deprecated; use config/training.yaml)
  - evaluation/: evaluation scripts and plots
  - xai/: Grad-CAM/attention visualizations
  - api/: optional FastAPI inference service
- models/
  - saved/: trained weights/checkpoints
- results/
  - logs/ and metrics/plots per run
- notebooks/
  - 01_dataset_exploration.ipynb
  - 02_training_analysis.ipynb
- config/
  - training.yaml, evaluation.yaml, api.yaml
- scripts/
  - run_preprocess.sh, run_train.sh, run_eval.sh, export_model.py

## Quickstart

1) Create venv and install deps:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2) Preprocess data:

```bash
./scripts/run_preprocess.sh
```

3) Train:

```bash
./scripts/run_train.sh
```

4) Evaluate:

```bash
./scripts/run_eval.sh
```

## Notes
- Heavy folders are git-ignored: `data/`, `models/saved/`, `results/`.
- Configure experiments via files in `config/`.

## Phase 3 — Training Procedure (Swin-T)

1. Load & Prepare

```python
import timm, torch.nn as nn
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=2)
model.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.head.in_features, 2),
)
```

2. Stage 1: Train Classifier Head
- Freeze backbone (`for p in model.parameters(): p.requires_grad=False` except head)
- Train 2–3 epochs; monitor `val_loss`↓ and `val_acc`↑

3. Stage 2: Fine-Tune Entire Model
- Unfreeze gradually; apply smaller LR to backbone than head
- Use early stopping and best-model checkpointing

4. Logging
- Save per-epoch `train_loss`, `val_loss`, `train_acc`, `val_acc`
- Track F1, precision, recall; optionally log via TensorBoard or W&B

5. Model Outputs
- Save final weights as `model_swin_t_regularized.pth`
- Optionally export embeddings for generator head retraining

Use `config/training_regularized.yaml` to control warm-up, discriminative LRs, and AMP.

---

## Overview

This repository contains a complete pipeline to detect AI-generated images vs natural (real) images across multiple generators. It includes:

- Data preprocessing and standardization to 224×224
- Model training (Swin-Tiny `swin_tiny_patch4_window7_224`) with regularization
- Evaluation and threshold calibration (global and per-generator)
- Explainability (XAI) via Eigen-CAM / Grad-CAM, aligned with the app policy
- Optional FastAPI server + simple web app for interactive inference

## Data

The dataset is not included in this repo due to size. Download it here:

Link: https://drive.google.com/drive/folders/1aw8bOds6N6vxrYEUhAyMBdyR6hEJSrr3?usp=sharing

Expected layout after extraction (standardized 224×224 JPEGs):

```
data/
  standardized_jpg/
    <generator>/                 # adm, biggan, glide, midjourney, sdv5, vqdm, wukong, general, etc.
      train/
        ai/                      # AI-generated images
        nature/                  # real images
      val/
        ai/
        nature/
      test/
        ai/
        nature/
```

Notes:

- Some splits may be missing for certain generators depending on the source.
- The `general` cohort mixes multiple sources and is useful for cross-generator evaluation.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model weights

The app uses a single deployed checkpoint by default:

```
models/regularized_training/model_swin_t_regularized.pth
```

This file is large (~315 MB). Recommended approaches:

- Git LFS (tracked file): if you cloned via LFS, run `git lfs pull`.
- Or publish/download via a Release asset and place at the path above.

Verify checksum (optional, for reproducibility):

```bash
python - <<'PY'
import hashlib, pathlib
p = pathlib.Path('models/regularized_training/model_swin_t_regularized.pth')
h = hashlib.sha256(p.read_bytes()).hexdigest()
print({'sha256': h, 'sha256_12': h[:12], 'size_bytes': p.stat().st_size})
PY
```

Threshold and temperature metadata (committed, tiny):

- `results/thresholds_swin_tiny_patch4_window7_224.json` (best-accuracy thr=0.54, best-F1 thr=0.44; plus per-generator)
- `results/temperature_swin_tiny_patch4_window7_224.txt` (if temperature scaling is used)

## File structure (key paths)

```
src/
  api/                 # FastAPI server (loads model, thresholds, optional generator head)
  xai/                 # gradcam_quick.py for Eigen-CAM / Grad-CAM
  models/              # model factory and definitions
  training/            # training code & utilities
  evaluation/          # evaluation utilities and plotting
  preprocess_images.py # dataset preprocessing
scripts/
  run_preprocess.sh    # preprocess images (IMG_SIZE env supported)
  run_train.sh         # train regularized Swin-T
  run_eval.sh          # evaluate a checkpoint; guards for missing model
  reporting/           # report-generation helpers
results/
  report_assets/       # curated figures for the paper/report (commit small PNGs)
  thresholds_swin_tiny_patch4_window7_224.json
  temperature_swin_tiny_patch4_window7_224.txt
reports/
  run_*/               # evaluation artifacts (git-ignored)
web_app/               # simple UI for the API
```

## Quickstart (end-to-end)

1) Preprocess data

```bash
./scripts/run_preprocess.sh                  # uses data/standardized_jpg or generates from raw
```

2) Train Swin‑Tiny (regularized)

```bash
./scripts/run_train.sh                       # reads config/training_regularized.yaml
```

3) Evaluate a model (uses the app’s default checkpoint by default)

```bash
./scripts/run_eval.sh                        # set MODEL_PATH or run `git lfs pull`
```

4) Run API locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5001 --reload
# then POST /predict with an image file or open the web front-end if wired
```

5) XAI: Eigen‑CAM / Grad‑CAM (policy‑aligned)

```bash
# Explain using app policy (best-accuracy threshold 0.54)
python3 src/xai/gradcam_quick.py --method eigen \
  --images "/abs/path/to/image.jpg" \
  --policy best_acc

# Counterfactual evidence map (force specific label)
python3 src/xai/gradcam_quick.py --method eigen \
  --images "/abs/path/to/image.jpg" \
  --policy best_acc --target_label ai    # or nature
```

Output includes: `pred_label`, `prob_ai`, `prob_nature`, chosen `policy`, `threshold`, and `target` class for the CAM.

## Threshold sensitivity figures

We provide scripts to generate:

- `results/report_assets/threshold_sensitivity.png` (accuracy, precision, recall, F1 vs threshold)
- `results/report_assets/threshold_best_points.png` (global best points)
- `results/report_assets/per_generator_thresholds.png` (per-generator bests)

These illustrate that best-accuracy is at 0.54 and best-F1 at 0.44, with examples like Wukong benefiting from ≈0.34.

## Recommended figures (XAI composites)

Triptych per sample: Original + Real evidence (class 1) + AI evidence (class 0). Save under `results/report_assets/`.

Example filenames used in the report:

- `image-14_orig_real_ai_triptych.png` (real)
- `image-12_orig_real_ai_triptych.png` (real)
- `image-15_orig_real_ai_triptych.png` (AI)

## Git guidance

This repo ignores heavy artifacts by default (`data/`, most of `models/`, raw `reports/`). If you need to keep a single checkpoint:

- Use Git LFS and track only `models/regularized_training/model_swin_t_regularized.pth`
- Collaborators should run `git lfs install && git lfs pull` after cloning

## Troubleshooting

- Missing model error when evaluating
  - Run `git lfs pull` (if using LFS) or place your checkpoint at `models/regularized_training/model_swin_t_regularized.pth`.
  - Or export `MODEL_PATH=/abs/path/to/model.pth`.

- XAI script says `No module named 'src'`
  - Ensure you run from project root and Python can import `src/`. The script now prepends project root to `sys.path`.

- Different decisions between CLI vs app
  - The XAI CLI now mirrors the app’s threshold policy (best_acc or best_f1) and temperature scaling. Use `--policy` accordingly.

- Large files blocked by GitHub
  - Use Git LFS for the single deploy checkpoint, or publish it as a Release asset and provide a download script.

