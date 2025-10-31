from __future__ import annotations

import argparse
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


def fine_tune(config_path: str = "config/fine_tuning.yaml") -> None:
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Read config values
    model_name = cfg.get("model_name", "resnet18")
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("learning_rate", 1e-5))
    batch_size = int(cfg.get("batch_size", 32))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    image_size = int(cfg.get("image_size", 224))
    save_dir = Path(cfg.get("save_dir", "models/fine_tuned_general"))
    data_root = Path(cfg.get("data_root", "data/standardized_jpg/general"))
    patience = int(cfg.get("early_stopping_patience", 2))
    seed = int(cfg.get("seed", 42))

    # Dataset checks (must be standardized general dataset)
    if "standardized" not in str(data_root):
        raise ValueError(
            f"Config data_root is set to '{data_root}'. This is NOT the standardized dataset. "
            f"Please point to 'data/standardized_jpg/general'."
        )
    if not data_root.exists():
        raise FileNotFoundError(
            f"The dataset path '{data_root}' does not exist. Make sure your standardized dataset is prepared."
        )

    print(f"Using standardized dataset at: {data_root}")

    # Device and seed
    device = resolve_device()
    print(f"Using device: {device}")
    set_seed(seed)

    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
    )
    print(f"Classes: {class_names}")

    # Model (create uninitialized weights; we'll load checkpoint next)
    model = create_model(model_name, num_classes=len(class_names), pretrained=False)
    model.to(device)

    # Load checkpoint
    checkpoint_path = Path("models/saved/last_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train a base model first (see config/training.yaml)."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Freeze feature extractor except top block and head (allow limited adaptation)
    for name, param in model.named_parameters():
        if not ("layer4" in name or "fc" in name):
            param.requires_grad = False

    # Training setup: only train trainable params
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    no_improve_epochs = 0

    # Train for a few epochs with early stopping
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
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

    # Save final fine-tuned weights
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "fine_tuned_general.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": best_val_acc,
    }, out_path)
    print(f"Fine-tuned model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/fine_tuning.yaml")
    args = parser.parse_args()
    fine_tune(args.config)
