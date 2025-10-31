from typing import Literal

import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit(
    variant: Literal["vit_b_16"] = "vit_b_16",
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create a ViT model with a custom classification head.
    Currently supports vit_b_16.
    """
    if variant != "vit_b_16":
        raise ValueError(f"Unsupported ViT variant: {variant}")

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    net = vit_b_16(weights=weights)

    # Replace head
    in_features = net.heads.head.in_features  # type: ignore[attr-defined]
    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    if dropout > 0:
        net.heads.head = nn.Sequential(  # type: ignore[attr-defined]
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        net.heads.head = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]

    return net


