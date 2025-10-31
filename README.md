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
