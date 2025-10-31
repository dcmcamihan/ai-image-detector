import torch
import torch.nn as nn
import torch.optim as optim
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (self.alpha * (1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class Lookahead(optim.Optimizer):
    def __init__(self, optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step = 0
        self.slow_params = [p.clone().detach() for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        for p in self.slow_params:
            p.requires_grad = False

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            idx = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad and p.data is not None:
                        sp = self.slow_params[idx]
                        sp.data.add_(self.alpha * (p.data - sp.data))
                        p.data.copy_(sp.data)
                        idx += 1
        return loss

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    # Delegate state dict ops to the wrapped optimizer to avoid missing hook attributes
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import re
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import yaml
from tqdm import tqdm
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_

from src.dataset_loader import get_dataloaders
from src.models.model_factory import create_model
from .trainer_utils import (
    resolve_device,
    set_seed,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


def _load_checkpoint_safely(model: nn.Module, checkpoint_path: Path, device: torch.device):
    """
    Load a checkpoint into the model. If classifier head shape mismatches (e.g., dropout vs linear),
    load only backbone weights and leave head randomly initialized.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict: Dict[str, torch.Tensor] = ckpt["model_state_dict"]

    try:
        model.load_state_dict(state_dict, strict=False)
        print("Loaded weights from checkpoint (backbone matched; head may be re-initialized).")
        return ckpt
    except RuntimeError as e:
        print(f"Full load failed due to shape mismatch: {e}\nFalling back to backbone-only load (excluding 'fc' keys).")

    # Filter out classifier head parameters and load the rest
    backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"Backbone weights loaded. Missing keys (expected for new head): {missing}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")


def train(config_path: str = "config/training_regularized.yaml") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Config
    model_name = cfg.get("model_name", "resnet18")
    epochs = int(cfg.get("epochs", 8))
    lr = float(cfg.get("learning_rate", 2.5e-4))
    batch_size = int(cfg.get("batch_size", 32))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    min_lr = float(cfg.get("min_lr", 1e-6))
    image_size = int(cfg.get("image_size", 224))
    save_dir = Path(cfg.get("save_dir", "models/regularized_training"))
    data_root = Path(cfg.get("data_root", "data/standardized_jpg"))
    augmented = bool(cfg.get("augmented", True))
    use_class_weights = bool(cfg.get("use_class_weights", False))
    hard_cases_dir = cfg.get("hard_cases_dir", None)

    # Regularization knobs
    pretrained = bool(cfg.get("pretrained", True))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    dropout = float(cfg.get("dropout", 0.3))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))

    # Advanced training knobs
    head_warmup_epochs = int(cfg.get("head_warmup_epochs", 0))
    head_warmup_lr = float(cfg.get("head_warmup_lr", 1e-3))
    backbone_lr = float(cfg.get("backbone_lr", 1e-5))
    head_lr = float(cfg.get("head_lr", max(lr, 1e-4)))
    use_adamw = bool(cfg.get("use_adamw", True))
    use_lookahead = bool(cfg.get("use_lookahead", False))
    use_amp = bool(cfg.get("use_amp", True))
    ema_decay = float(cfg.get("ema_decay", 0.0))
    clip_grad = float(cfg.get("clip_grad", 1.0))
    scheduler_kind = str(cfg.get("scheduler", "cosine")).lower()
    reduce_on_plateau = bool(cfg.get("reduce_on_plateau", False))
    reduce_factor = float(cfg.get("reduce_factor", 0.5))
    reduce_patience = int(cfg.get("reduce_patience", 4))
    reduce_min_lr = float(cfg.get("reduce_min_lr", 1e-6))
    # SWA options
    use_swa = bool(cfg.get("use_swa", False))
    swa_start_epoch = int(cfg.get("swa_start_epoch", max(1, int(0.7 * epochs))))
    swa_lr = float(cfg.get("swa_lr", max(1e-5, lr * 0.2)))

    checkpoint_path_str = cfg.get("checkpoint_path", None)
    checkpoint_path = Path(checkpoint_path_str) if checkpoint_path_str else None
    init_from_path_str = cfg.get("init_from_path", None)
    init_from_path = Path(init_from_path_str) if init_from_path_str else None

    # Validations
    if "standardized" not in str(data_root):
        raise ValueError(
            f"Config data_root is set to '{data_root}'. This is NOT the standardized dataset. "
            f"Please point to 'data/standardized_jpg'."
        )
    if not data_root.exists():
        raise FileNotFoundError(
            f"The dataset path '{data_root}' does not exist. Make sure the standardized dataset is prepared."
        )

    print(f"Using standardized dataset at: {data_root}")

    # Device & seed
    device = resolve_device()
    print(f"Using device: {device}")
    set_seed(int(cfg.get("seed", 42)))

    # Data (optionally augmented)
    # Prefer fewer workers on Mac MPS to avoid dataloader overhead/hangs
    workers_override = cfg.get("num_workers", None)
    effective_workers = (
        int(workers_override)
        if workers_override is not None
        else (0 if device.type == "mps" else 4)
    )

    # Read label-aware quality balance and upsampling knobs (Phase 2.5)
    enable_quality_balance = bool(cfg.get("enable_quality_balance", True))
    p_sharp_ai = float(cfg.get("p_sharp_ai", 0.10))
    p_blur_real = float(cfg.get("p_blur_real", 0.10))
    sharp_boost_generators = tuple(cfg.get("sharp_boost_generators", ["sdv5", "wukong"]))
    upsample_hard_cases_factor = cfg.get("upsample_hard_cases_factor", None)

    dl_args = dict(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        augmented=augmented,
        return_class_weights=use_class_weights,
        hard_cases_dir=hard_cases_dir,
        num_workers=effective_workers,
        enable_quality_balance=enable_quality_balance,
        p_sharp_ai=p_sharp_ai,
        p_blur_real=p_blur_real,
        sharp_boost_generators=sharp_boost_generators,
        upsample_hard_cases_factor=upsample_hard_cases_factor,
    )
    if use_class_weights:
        train_loader, val_loader, class_names, class_weights_tensor = get_dataloaders(**dl_args)
    else:
        train_loader, val_loader, class_names = get_dataloaders(**dl_args)
    print(f"Classes: {class_names}")
    # Quick visibility into epoch sizes
    try:
        print(f"Batches — train: {len(train_loader)}, val: {len(val_loader)}")
    except Exception:
        pass

    # Model with requested regularization (dropout)
    model = create_model(
        model_name,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    model.to(device)

    # Helper: scan for latest checkpoint in save_dir
    def _find_latest_checkpoint(dir_path: Path, device_for_map: torch.device) -> Tuple[Optional[Path], Optional[int]]:
        best_path: Optional[Path] = None
        best_epoch: int = -1
        if not dir_path.exists():
            return None, None
        for p in dir_path.glob("*.pth"):
            epoch_val: Optional[int] = None
            # Try reading epoch from file contents
            try:
                ck = torch.load(p, map_location=device_for_map)
                if isinstance(ck, dict) and "epoch" in ck:
                    epoch_val = int(ck.get("epoch", -1))
            except Exception:
                epoch_val = None
            # Fallback: parse from filename best_model_epoch{E}_acc*.pth
            if epoch_val is None:
                m = re.search(r"best_model_epoch(\d+)_", p.name)
                if m:
                    try:
                        epoch_val = int(m.group(1))
                    except Exception:
                        epoch_val = None
            if epoch_val is not None and epoch_val > best_epoch:
                best_epoch = epoch_val
                best_path = p
        return (best_path, best_epoch if best_path is not None else None)

    # Resume flags
    resume_training = bool(cfg.get("resume_training", False))
    start_epoch = 1
    loaded_ckpt = None
    # Auto-resume: prefer newest checkpoint in save_dir when resume_training is True
    if resume_training:
        auto_path, auto_epoch = _find_latest_checkpoint(save_dir, device)
        if auto_path is not None and auto_epoch is not None:
            loaded_ckpt = _load_checkpoint_safely(model, auto_path, device)
            start_epoch = int(auto_epoch) + 1
            print(f"Auto-resume: Loaded latest checkpoint (epoch {auto_epoch}) from {save_dir}/")
        elif checkpoint_path and checkpoint_path.exists():
            loaded_ckpt = _load_checkpoint_safely(model, checkpoint_path, device)
            if isinstance(loaded_ckpt, dict):
                start_epoch = int(loaded_ckpt.get("epoch", 0)) + 1
            print(f"Resuming training from epoch {start_epoch} using checkpoint_path: {checkpoint_path}")
        elif checkpoint_path:
            print(f"Warning: checkpoint_path does not exist: {checkpoint_path}")
    else:
        # Fresh run but allow initialization from provided weights (backbone/head safe load)
        if init_from_path and init_from_path.exists():
            _ = _load_checkpoint_safely(model, init_from_path, device)
            print(f"Initialized model weights from init_from_path: {init_from_path}")

    # Loss (CrossEntropy or FocalLoss)
    use_focal = bool(cfg.get("use_focal_loss", False))
    focal_gamma = float(cfg.get("focal_gamma", 1.5))
    focal_alpha = float(cfg.get("focal_alpha", 0.75))
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        if use_class_weights:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device), label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_autocast = use_amp and device.type == "cuda"
    # Use new torch.amp.GradScaler API to avoid deprecation warnings on non-CUDA devices
    if device.type == "cuda":
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler("cuda", enabled=False)
    use_autocast = use_amp and device.type == "cuda"
    ema_shadow: Dict[str, torch.Tensor] | None = None

    def _update_ema():
        nonlocal ema_shadow
        if ema_decay <= 0.0:
            return
        if ema_shadow is None:
            ema_shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_shadow[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)

    def _apply_ema_weights(fn):
        if ema_shadow is None:
            return fn()
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_shadow, strict=False)
        try:
            return fn()
        finally:
            model.load_state_dict(backup, strict=False)

    # Helper to collect head vs backbone params
    head_names = [n for n, _ in model.named_parameters() if any(h in n for h in ("classifier", "head", "fc"))]
    head_params = [p for n, p in model.named_parameters() if n in head_names and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if n not in head_names and p.requires_grad]

    # Phase 1: optional head warm-up (skip if resuming past warm-up)
    current_epoch = 0
    if head_warmup_epochs > 0 and start_epoch <= head_warmup_epochs:
        for p in model.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
        if use_adamw:
            optimizer = optim.AdamW(head_params, lr=head_warmup_lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(head_params, lr=head_warmup_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, head_warmup_epochs))

        for epoch in range(1, head_warmup_epochs + 1):
            model.train()
            running_loss, running_acc = 0.0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                if use_autocast:
                    with autocast():
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
                _, preds = torch.max(outputs, 1)
                running_acc += (preds == labels).float().mean().item()
                _update_ema()
            train_loss = running_loss / max(1, len(train_loader))
            train_acc = running_acc / max(1, len(train_loader))
            print("Validating...", flush=True)
            val_loss, val_acc = _apply_ema_weights(lambda: validate_one_epoch(model, val_loader, criterion, device))
            last_val_acc = val_acc
            scheduler.step()
            print(
                f"Warmup Epoch [{epoch}/{head_warmup_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        for p in model.parameters():
            p.requires_grad = True
        current_epoch = head_warmup_epochs
    else:
        # No warm-up run (either disabled or resuming past it)
        for p in model.parameters():
            p.requires_grad = True
        current_epoch = head_warmup_epochs

    # Phase 2: full fine-tune with discriminative LRs
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if use_adamw:
        base_opt = optim.AdamW(param_groups if param_groups else model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        base_opt = optim.SGD(param_groups if param_groups else model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = Lookahead(base_opt) if use_lookahead else base_opt
    # If resuming, restore optimizer state and set scheduler to correct last_epoch
    if resume_training and isinstance(loaded_ckpt, dict) and "optimizer_state_dict" in loaded_ckpt:
        try:
            optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")
    if scheduler_kind == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(10, epochs // 5), T_mult=2, eta_min=min_lr)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch - 2 if start_epoch > 1 else -1)
    rop = ReduceLROnPlateau(optimizer, mode='min', factor=reduce_factor, patience=reduce_patience, min_lr=reduce_min_lr) if reduce_on_plateau else None
    # Prepare SWA if enabled
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    else:
        swa_model = None
        swa_scheduler = None

    best_val_acc = 0.0
    metrics_log = []
    patience = int(cfg.get("early_stopping_patience", 3))
    no_improve_epochs = 0
    # Initialize safe epoch and last_val_acc for cases where loop doesn't execute (e.g., resume past epochs)
    epoch = max(0, start_epoch - 1)
    last_val_acc = float(loaded_ckpt.get("val_acc", 0.0)) if isinstance(loaded_ckpt, dict) else 0.0

    
    # Optional caps to get quicker feedback on slow backends (e.g., MPS)
    max_train_batches_first = cfg.get("max_train_batches_first_epoch", None)
    max_train_batches = cfg.get("max_train_batches", None)

    # Optional late upsampling schedule
    late_upsample_epoch = int(cfg.get("late_upsample_epoch", 0))
    late_upsample_factor = cfg.get("late_upsample_factor", None)

    # Training loop
    train_start_epoch = max(1, start_epoch if start_epoch > head_warmup_epochs else 1)
    try:
        for epoch in range(train_start_epoch, epochs + 1):
            # Increase hard_cases_web sampling later in training if configured
            if late_upsample_epoch and late_upsample_factor is not None and epoch == int(late_upsample_epoch):
                print(f"Applying late upsampling at epoch {epoch}: factor -> {late_upsample_factor}")
                dl_args["upsample_hard_cases_factor"] = late_upsample_factor
                if use_class_weights:
                    train_loader, val_loader, class_names, class_weights_tensor = get_dataloaders(**dl_args)
                else:
                    train_loader, val_loader, class_names = get_dataloaders(**dl_args)
                try:
                    print(f"Batches — train: {len(train_loader)}, val: {len(val_loader)}")
                except Exception:
                    pass
            model.train()
            running_loss, running_acc = 0.0, 0.0
            pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
            seen = 0
            if use_autocast:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        # MixUp/CutMix
                        apply_cutmix = bool(cfg.get("use_cutmix", False))
                        apply_mixup = bool(cfg.get("use_mixup", False))
                        lam = 1.0
                        y_a, y_b = labels, labels
                        if apply_cutmix or apply_mixup:
                            use_cm = apply_cutmix and (not apply_mixup or random.random() < 0.5)
                            if use_cm:
                                alpha = float(cfg.get("cutmix_alpha", 1.0))
                                lam = np.random.beta(alpha, alpha)
                                batch_size_curr = images.size()[0]
                                index = torch.randperm(batch_size_curr, device=images.device)
                                bbx1 = int(images.size(2) * random.random() * lam)
                                bby1 = int(images.size(3) * random.random() * lam)
                                images[:, :, :bbx1, :bby1] = images[index, :, :bbx1, :bby1]
                                y_a, y_b = labels, labels[index]
                            else:
                                alpha = float(cfg.get("mixup_alpha", 0.2))
                                lam = np.random.beta(alpha, alpha)
                                batch_size_curr = images.size()[0]
                                index = torch.randperm(batch_size_curr, device=images.device)
                                mixed = lam * images + (1 - lam) * images[index, :]
                                images = mixed
                                y_a, y_b = labels, labels[index]
                        outputs = model(images)
                        if (apply_cutmix or apply_mixup) and lam < 0.999:
                            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                        else:
                            loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    if clip_grad and clip_grad > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    if use_swa and epoch >= swa_start_epoch:
                        swa_model.update_parameters(model)
                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    running_acc += (preds == labels).float().mean().item()
                    _update_ema()
                    pbar.set_postfix({"loss": f"{running_loss/(pbar.n or 1):.4f}"})
                    seen += 1
                    limit = max_train_batches_first if (epoch == 1 and max_train_batches_first is not None) else max_train_batches
                    if limit is not None and seen >= int(limit):
                        break
            else:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    # MixUp/CutMix (non-AMP branch)
                    apply_cutmix = bool(cfg.get("use_cutmix", False))
                    apply_mixup = bool(cfg.get("use_mixup", False))
                    lam = 1.0
                    y_a, y_b = labels, labels
                    if apply_cutmix or apply_mixup:
                        use_cm = apply_cutmix and (not apply_mixup or random.random() < 0.5)
                        if use_cm:
                            alpha = float(cfg.get("cutmix_alpha", 1.0))
                            lam = np.random.beta(alpha, alpha)
                            batch_size_curr = images.size()[0]
                            index = torch.randperm(batch_size_curr, device=images.device)
                            bbx1 = int(images.size(2) * random.random() * lam)
                            bby1 = int(images.size(3) * random.random() * lam)
                            images[:, :, :bbx1, :bby1] = images[index, :, :bbx1, :bby1]
                            y_a, y_b = labels, labels[index]
                        else:
                            alpha = float(cfg.get("mixup_alpha", 0.2))
                            lam = np.random.beta(alpha, alpha)
                            batch_size_curr = images.size()[0]
                            index = torch.randperm(batch_size_curr, device=images.device)
                            mixed = lam * images + (1 - lam) * images[index, :]
                            images = mixed
                            y_a, y_b = labels, labels[index]
                    outputs = model(images)
                    if (apply_cutmix or apply_mixup) and lam < 0.999:
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    if clip_grad and clip_grad > 0:
                        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optimizer.step()
                    if use_swa and epoch >= swa_start_epoch:
                        swa_model.update_parameters(model)
                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    running_acc += (preds == labels).float().mean().item()
                    _update_ema()
                    pbar.set_postfix({"loss": f"{running_loss/(pbar.n or 1):.4f}"})
                    seen += 1
                    limit = max_train_batches_first if (epoch == 1 and max_train_batches_first is not None) else max_train_batches
                    if limit is not None and seen >= int(limit):
                        break

            # Use the actual number of processed batches (seen) to compute averages,
            # especially important when using max_train_batches* caps.
            train_loss = running_loss / max(1, seen)
            train_acc = running_acc / max(1, seen)

            val_loss, val_acc = _apply_ema_weights(lambda: validate_one_epoch(model, val_loader, criterion, device))

            # Step appropriate scheduler
            if use_swa and epoch >= swa_start_epoch and swa_scheduler is not None:
                swa_scheduler.step()
            else:
                if isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + (seen / max(1, len(train_loader))))
                else:
                    scheduler.step()
                if rop is not None:
                    rop.step(val_loss)

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

            # Always update the named resume checkpoint each epoch
            named_path = save_dir / "model_swin_t_regularized.pth"
            try:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, named_path)
            except Exception as e:
                print(f"Warning: could not save named checkpoint during epoch {epoch}: {e}")
    except KeyboardInterrupt:
        # Save a last checkpoint on interrupt for safe resume
        print("\nKeyboardInterrupt: saving checkpoint for resume...")
        save_checkpoint(model, optimizer, epoch if 'epoch' in locals() else 0, best_val_acc, save_dir)
        # Also persist the named resume checkpoint
        try:
            named_path = save_dir / "model_swin_t_regularized.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch if 'epoch' in locals() else 0,
                "val_acc": best_val_acc,
            }, named_path)
        except Exception as e:
            print(f"Warning: could not save named checkpoint on interrupt: {e}")
        raise

    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = save_dir / f"training_log_{timestamp}.json"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # If SWA was used, update BN stats and save SWA model
    if use_swa and swa_model is not None:
        try:
            update_bn(train_loader, swa_model, device=device)
        except TypeError:
            # older torch versions don't accept device kwarg
            update_bn(train_loader, swa_model)
        swa_path = save_dir / "model_swin_t_regularized_swa.pth"
        torch.save({
            "model_state_dict": swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": last_val_acc,
        }, swa_path)
        print(f"SWA model checkpoint saved to {swa_path}")

    # Save last model state
    last_path = save_dir / "regularized_general.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": last_val_acc,
    }, last_path)
    print(f"Regularized model checkpoint saved to {last_path}")

    # Save an additional named checkpoint for Swin-T runs
    named_path = save_dir / "model_swin_t_regularized.pth"
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": last_val_acc,
        }, named_path)
        print(f"Named model checkpoint saved to {named_path}")
    except Exception as e:
        print(f"Warning: could not save named checkpoint: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training_regularized.yaml")
    args = parser.parse_args()
    train(args.config)
