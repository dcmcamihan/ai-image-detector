#!/usr/bin/env python3
"""
Train a generator classification linear head compatible with the current backbone.

This script:
- Loads the backbone (e.g., Swin-T) from a checkpoint.
- Extracts penultimate features for images organized by generator label.
- Trains a multinomial logistic regression (linear layer) in PyTorch.
- Saves a lightweight head file with keys: 'weight' [N,D], 'bias' [N].

Example:
PYTHONPATH=. python scripts/train_generator_head.py \
  --model_path models/regularized_training/model_swin_t_regularized.pth \
  --model_name swin_tiny_patch4_window7_224 \
  --data_root data/standardized_jpg \
  --labels adm biggan glide midjourney sdv5 vqdm wukong \
  --out gen_head_swin_t.pth

Data expectation:
- For each label L in --labels, images should be under one of:
  - {data_root}/{L}/train and/or {data_root}/{L}/val
  - or {data_root}/{L} (flat)
We will scan both train/val and flat directory forms.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models.model_factory import create_model
from src.preprocess_images import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class ImageList(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], transform):
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y


def collect_paths(root: Path, labels: List[str], max_per_label: int | None = None) -> List[Tuple[Path, int]]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    items: List[Tuple[Path, int]] = []
    for i, lab in enumerate(labels):
        count = 0
        # Search train/val and flat
        candidates = [root / lab / "train", root / lab / "val", root / lab]
        seen: set[Path] = set()
        for c in candidates:
            if c.exists() and c.is_dir():
                for p in c.rglob("*"):
                    if p.is_file() and p.suffix.lower() in exts and p not in seen:
                        items.append((p, i))
                        seen.add(p)
                        count += 1
                        if max_per_label and count >= max_per_label:
                            break
            if max_per_label and count >= max_per_label:
                break
        if count == 0:
            print(f"[warn] No images found for label '{lab}' under {root}")
    return items


def hook_penultimate_features(model: nn.Module):
    buf: dict[str, torch.Tensor] = {}
    def _hook(module, inputs, output):
        if inputs and isinstance(inputs[0], torch.Tensor):
            buf["feat"] = inputs[0].detach()
    hooked = False
    # ResNet style
    if hasattr(model, "fc") and model.fc is not None:
        target = model.fc
        if isinstance(target, nn.Sequential) and len(target) > 0:
            last = list(target.modules())[-1]
            (last if isinstance(last, nn.Linear) else target).register_forward_hook(_hook)
        else:
            target.register_forward_hook(_hook)
        hooked = True
    # ViT/Swin style
    if not hooked and hasattr(model, "head") and model.head is not None:
        target = model.head
        if isinstance(target, nn.Sequential) and len(target) > 0:
            last = list(target.modules())[-1]
            (last if isinstance(last, nn.Linear) else target).register_forward_hook(_hook)
        else:
            target.register_forward_hook(_hook)
        hooked = True
    # Fallback to last Linear
    if not hooked:
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            linear_layers[-1].register_forward_hook(_hook)
    return buf


def _to_2d(feat: torch.Tensor) -> torch.Tensor:
    """Convert captured feature to [N, D].
    - If [N, C, H, W] -> GAP over H,W to [N, C]
    - If [N, T, D]    -> mean over tokens to [N, D]
    - If [N, D]       -> return as-is
    """
    if feat.dim() == 4:
        # Allow both NCHW and NHWC; detect by channel position
        # Common for timm backbones to produce [N, C, H, W]
        if feat.shape[1] in (256, 384, 512, 768, 1024):
            return feat.mean(dim=(2, 3))
        # If NHWC, move channels and pool
        return feat.permute(0, 3, 1, 2).mean(dim=(2, 3))
    if feat.dim() == 3:
        # sequence: average tokens (or choose CLS if you prefer)
        return feat.mean(dim=1)
    if feat.dim() == 2:
        return feat
    raise RuntimeError(f"Unsupported feature shape {tuple(feat.shape)}; expected 2D/3D/4D")


def extract_features(model: nn.Module, loader: DataLoader, buf: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extract feats"):
            x = x.to(DEVICE)
            _ = model(x)
            f = buf.get("feat")
            if f is None:
                raise RuntimeError("Hook did not capture features. Check backbone/head mapping.")
            f2 = _to_2d(f)
            feats.append(f2.detach().cpu())
            labels.append(y)
    X = torch.cat(feats, dim=0)
    Y = torch.cat(labels, dim=0)
    return X, Y


def train_linear_head(X: torch.Tensor, Y: torch.Tensor, n_classes: int, max_iter: int = 300) -> nn.Linear:
    D = X.shape[1]
    head = nn.Linear(D, n_classes, bias=True).to(DEVICE)
    # LBFGS on cross-entropy
    opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    Xd = X.to(DEVICE)
    Yd = Y.to(DEVICE)

    def closure():
        opt.zero_grad()
        logits = head(Xd)
        loss = ce(logits, Yd)
        loss.backward()
        return loss

    opt.step(closure)
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True, help="Root containing per-generator folders")
    ap.add_argument("--labels", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_per_label", type=int, default=2000, help="Cap images per label to limit runtime")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    labels = args.labels

    # Build model and load checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Infer if head is Sequential(Dropout, Linear)
    use_seq_head = any(k.startswith("fc.1.") for k in state_dict.keys())
    dropout = 0.5 if use_seq_head else 0.0
    model = create_model(args.model_name, num_classes=2, pretrained=False, dropout=dropout)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    # Hook
    feat_buf = hook_penultimate_features(model)

    # Data and loader
    _, val_tfms = get_transforms(image_size=args.image_size)
    items = collect_paths(data_root, labels, max_per_label=args.max_per_label)
    if not items:
        raise FileNotFoundError(f"No images found under {data_root} for labels: {labels}")
    ds = ImageList(items, val_tfms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Extract features
    X, Y = extract_features(model, dl, feat_buf)
    print(f"Features: X={tuple(X.shape)}, Y={tuple(Y.shape)}")

    # Train linear head
    head = train_linear_head(X, Y, n_classes=len(labels))

    # Save lightweight head
    with torch.no_grad():
        weight = head.weight.detach().cpu().clone()
        bias = head.bias.detach().cpu().clone()
    torch.save({"weight": weight, "bias": bias, "labels": labels}, out_path)
    print(f"Saved generator head to {out_path} with weight {tuple(weight.shape)} and bias {tuple(bias.shape)}")


if __name__ == "__main__":
    main()
