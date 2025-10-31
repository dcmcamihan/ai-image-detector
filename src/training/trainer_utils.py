from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_acc: float, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"best_model_epoch{epoch}_acc{val_acc:.4f}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )
    return path


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc = 0.0, 0.0
    amp_enabled = use_amp and torch.cuda.is_available()
    # Use new torch.amp GradScaler API (backend specified as 'cuda')
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        running_acc += compute_accuracy(outputs, labels)
    return running_loss / len(dataloader), running_acc / len(dataloader)


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    amp_enabled = use_amp and torch.cuda.is_available()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, labels)
    return val_loss / len(dataloader), val_acc / len(dataloader)


