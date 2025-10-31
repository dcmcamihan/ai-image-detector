from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from src.dataset_loader import get_dataloaders
from src.models.model_factory import create_model
from .trainer_utils import (
    resolve_device,
    set_seed,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


def train(config_path: str = "config/training_augmented.yaml") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Load config values
    model_name = cfg.get("model_name", "resnet18")
    epochs = int(cfg.get("epochs", 12))
    lr = float(cfg.get("learning_rate", 2.5e-4))
    batch_size = int(cfg.get("batch_size", 32))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    image_size = int(cfg.get("image_size", 224))
    save_dir = Path(cfg.get("save_dir", "models/augmented_training"))
    data_root = Path(cfg.get("data_root", "data/standardized_jpg"))

    # Validation: ensure standardized dataset is being used
    if "standardized" not in str(data_root):
        raise ValueError(
            f"Config data_root is set to '{data_root}'. "
            f"This is NOT the standardized dataset. Please update config/training_augmented.yaml "
            f"to point to 'data/standardized_jpg'."
        )
    if not data_root.exists():
        raise FileNotFoundError(
            f"The dataset path '{data_root}' does not exist. "
            f"Make sure you’ve run scripts/standardize_dataset.py first."
        )

    print(f"Using standardized dataset at: {data_root}")

    # Device setup
    device = resolve_device()
    print(f"Using device: {device}")

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Load data (augmented transforms enabled) with quality balancing and optional upsampling
    enable_quality_balance = bool(cfg.get("enable_quality_balance", True))
    p_sharp_ai = float(cfg.get("p_sharp_ai", 0.10))
    p_blur_real = float(cfg.get("p_blur_real", 0.15))
    sharp_boost_generators = tuple(cfg.get("sharp_boost_generators", ["sdv5", "wukong"]))
    upsample_hard_cases_factor = cfg.get("upsample_hard_cases_factor", None)
    hard_cases_dir = cfg.get("hard_cases_dir", None)
    # Optimizer/precision flags
    use_adamw = bool(cfg.get("use_adamw", False))
    use_amp = bool(cfg.get("use_amp", False))

    train_loader, val_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        augmented=True,
        hard_cases_dir=hard_cases_dir,
        enable_quality_balance=enable_quality_balance,
        p_sharp_ai=p_sharp_ai,
        p_blur_real=p_blur_real,
        sharp_boost_generators=sharp_boost_generators,
        upsample_hard_cases_factor=upsample_hard_cases_factor,
    )
    print(f"Classes: {class_names}")

    # Model setup (use config flags for dropout/pretrained/freeze_backbone)
    pretrained = bool(cfg.get("pretrained", True))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    dropout = float(cfg.get("dropout", 0.5))

    model = create_model(
        model_name,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    model.to(device)

    # Optimization setup
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if use_adamw else
        optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    metrics_log = []
    patience = int(cfg.get("early_stopping_patience", 3))
    no_improve_epochs = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)

        scheduler.step()

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        metrics_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, save_dir)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if patience > 0 and no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = save_dir / f"training_log_{timestamp}.json"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Save last model state
    last_path = save_dir / "last_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }, last_path)
    print(f"Last model checkpoint saved to {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training_augmented.yaml")
    args = parser.parse_args()
    train(args.config)
