#!/usr/bin/env python3
"""
Calibrate model probabilities with temperature scaling using the validation split.
Saves the learned temperature to results/temperature_<model>.txt and an optional calibrated metrics CSV.

Usage:
  python scripts/calibrate_temperature.py \
    --model_path models/regularized_training/model_swin_t_regularized.pth \
    --model_name swin_tiny_patch4_window7_224 \
    --data_root data/standardized_jpg/general \
    --out results/calibrated_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from src.models.model_factory import create_model
from src.dataset_loader import get_dataloaders
from src.evaluation.temperature_scaling import ModelWithTemperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_root", default="data/standardized_jpg/general")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
    )

    # Load val loader for calibration
    _, val_loader, class_names = get_dataloaders(data_root=args.data_root, batch_size=64)

    # Build model and load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model_state_dict"]
    # infer dropout head shape for resnet if needed
    inferred_dropout = 0.5 if any(k.startswith("fc.1.") for k in state.keys()) else 0.0
    model = create_model(args.model_name, num_classes=len(class_names), pretrained=False, dropout=inferred_dropout)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Collect logits/labels
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collect logits"):
            images = images.to(device)
            out = model(images)
            logits_list.append(out.cpu())
            labels_list.append(labels.cpu())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Fit temperature
    wrapper = ModelWithTemperature(model)
    wrapper.to(device)
    T = wrapper.set_temperature(logits.to(device), labels.to(device))

    # Save temperature
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_path = out_dir / f"temperature_{args.model_name}.txt"
    temp_path.write_text(f"{T:.6f}\n")
    print(f"Saved calibrated temperature to {temp_path}")

    # Optionally compute calibrated metrics on val
    if args.out:
        with torch.no_grad():
            probs = torch.softmax(logits / max(T, 1e-3), dim=1)[:, 1]
            preds = (probs >= 0.5).long()
        from src.evaluation.metrics_utils import compute_metrics
        m = compute_metrics(labels, preds, average="binary", y_score=probs)
        pd.DataFrame([m]).to_csv(args.out, index=False)
        print(f"Saved calibrated metrics to {args.out}")


if __name__ == "__main__":
    main()
