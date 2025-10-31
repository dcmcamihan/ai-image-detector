#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional
import json
import sys
from pathlib import Path

import torch
import timm
import numpy as np
from PIL import Image

# Ensure project root is importable when running as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.model_factory import create_model
from src.preprocess_images import get_transforms

# pip install pytorch-grad-cam
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Notes (for report):
# This script produces quick Grad-CAM/Eigen-CAM heatmaps for the deployed Swin-Tiny model.
# It highlights regions that most influence the model's decision.
# Eigen-CAM is often robust for transformers (no gradients through attention map projections).
# We run on CPU and save outputs under reports/gradcam_examples/.


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_as_tensor(img_path: str, img_size: int = 224) -> (torch.Tensor, np.ndarray):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    # Save original in [0,1] RGB for overlay
    rgb_float = img_np.copy()
    # Normalize for model input
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # 1x3xHxW, float32
    return tensor, rgb_float


def load_image_with_app_transform(img_path: str, img_size: int = 224) -> (torch.Tensor, np.ndarray):
    img = Image.open(img_path).convert("RGB")
    _, val_tf = get_transforms(img_size)
    tensor = val_tf(img).unsqueeze(0)
    rgb_float = np.array(img.resize((img_size, img_size), Image.BICUBIC)).astype(np.float32) / 255.0
    return tensor.float(), rgb_float


def build_swin_tiny(num_classes: int = 2, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    model = create_model('swin_tiny_patch4_window7_224', num_classes=num_classes, pretrained=False, dropout=0.0)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        use_seq_head = any(k.startswith('fc.1.') for k in state.keys())
        if use_seq_head:
            model = create_model('swin_tiny_patch4_window7_224', num_classes=num_classes, pretrained=False, dropout=0.5)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def swin_reshape_transform(tensor: torch.Tensor, h: int = 7, w: int = 7) -> torch.Tensor:
    """
    Reshape Swin block outputs from [B, L, C] to [B, C, H, W] for CAM.
    For 224x224 inputs, final stage spatial dims are 7x7.
    """
    if tensor.dim() == 3:  # [B, L, C]
        B, L, C = tensor.shape
        assert L == h * w, f"Unexpected token length {L}, expected {h*w}"
        x = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        return x
    elif tensor.dim() == 4:  # already [B, C, H, W]
        return tensor
    else:
        raise ValueError(f"Unsupported tensor shape for reshape: {tensor.shape}")


def get_swin_target_layers(model: torch.nn.Module):
    # Use the last block's norm as a reasonable activation target for CAM on Swin.
    # This works with Eigen-CAM without extra gradients logic.
    try:
        last_block = model.layers[-1].blocks[-1]
        # Use attention projection as target; tends to give cleaner CAM for Swin
        target_layer = last_block.attn.proj
        return [target_layer]
    except Exception:
        # Fallback: try model.norm
        if hasattr(model, 'norm'):
            return [model.norm]
        raise


def run_cam_on_images(
    image_paths: List[str],
    out_dir: str,
    checkpoint_path: str = 'models/regularized_training/model_swin_t_regularized.pth',
    method: str = 'eigen',
    target_class: Optional[int] = None,
    thresholds_json: str = 'results/thresholds_swin_tiny_patch4_window7_224.json',
    temperature_path: str = 'results/temperature_swin_tiny_patch4_window7_224.txt',
    policy: str = 'best_acc',  # argmax|best_acc|best_f1
    target_label: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cpu')

    model = build_swin_tiny(num_classes=2, checkpoint_path=checkpoint_path).to(device)
    target_layers = get_swin_target_layers(model)

    if method.lower() == 'eigen':
        cam = EigenCAM(model=model, target_layers=target_layers, reshape_transform=lambda x: swin_reshape_transform(x, 7, 7))
    else:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=lambda x: swin_reshape_transform(x, 7, 7))

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[skip] Not found: {img_path}")
            continue
        input_tensor, rgb_float = load_image_with_app_transform(img_path, img_size=224)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            if os.path.exists(temperature_path):
                try:
                    T = float(open(temperature_path, 'r').read().strip())
                    if T > 0:
                        logits = logits / T
                except Exception:
                    pass
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_ai = float(probs[0])
        prob_nature = float(probs[1])
        argmax_idx = int(probs.argmax())

        thr_used = None
        app_idx = argmax_idx
        try:
            if os.path.exists(thresholds_json) and policy in ('best_acc', 'best_f1'):
                th = json.load(open(thresholds_json))
                thr_used = float(th['global']['best_acc']['threshold']) if policy == 'best_acc' else float(th['global']['best_f1']['threshold'])
                app_idx = 0 if prob_ai >= thr_used else 1
        except Exception:
            pass

        if target_label is not None:
            class_idx = 0 if target_label.lower() == 'ai' else 1
        elif target_class is not None:
            class_idx = int(target_class)
        else:
            class_idx = app_idx if policy in ('best_acc', 'best_f1') else argmax_idx

        grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)[0]
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_png = os.path.join(out_dir, f"{base}_cam_{method}_cls{class_idx}.png")
        Image.fromarray(visualization).save(out_png)
        pred_label = 'ai' if (app_idx if policy in ('best_acc','best_f1') else argmax_idx) == 0 else 'nature'
        print(f"[ok] Saved: {out_png} (pred_label={pred_label}, prob_ai={prob_ai:.3f}, prob_nature={prob_nature:.3f}, policy={policy}, threshold={thr_used if thr_used is not None else 'argmax'}, target={class_idx})")


def main():
    parser = argparse.ArgumentParser(description="Quick Grad-CAM/Eigen-CAM for Swin-Tiny (CPU-only)")
    parser.add_argument('--images', nargs='*', default=[], help='Image paths to process')
    parser.add_argument('--out_dir', default='reports/gradcam_examples/', help='Output directory for heatmaps')
    parser.add_argument('--checkpoint', default='models/regularized_training/model_swin_t_regularized.pth', help='Model checkpoint path')
    parser.add_argument('--method', default='eigen', choices=['eigen', 'gradcam'], help='CAM method')
    parser.add_argument('--target_class', type=int, default=None, help='Class index override')
    parser.add_argument('--target_label', type=str, default=None, choices=['ai','nature'], help='Label override')
    parser.add_argument('--thresholds_json', default='results/thresholds_swin_tiny_patch4_window7_224.json')
    parser.add_argument('--temperature_path', default='results/temperature_swin_tiny_patch4_window7_224.txt')
    parser.add_argument('--policy', default='best_acc', choices=['argmax','best_acc','best_f1'])
    args = parser.parse_args()

    # If no images provided, try a couple of likely example paths (optional convenience)
    default_candidates = [
        'val/ai/midjourney_001.jpg',
        'val/real/pexels_001.jpg',
    ]
    image_list = args.images if args.images else default_candidates

    run_cam_on_images(
        image_paths=image_list,
        out_dir=args.out_dir,
        checkpoint_path=args.checkpoint,
        method=args.method,
        target_class=args.target_class,
        thresholds_json=args.thresholds_json,
        temperature_path=args.temperature_path,
        policy=args.policy,
        target_label=args.target_label,
    )


if __name__ == '__main__':
    main()
