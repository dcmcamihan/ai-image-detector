from typing import Literal, Optional

import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


def build_resnet(
    model: Literal["resnet18", "resnet50"] = "resnet18",
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create a ResNet model with an updated classification head.

    Args:
        model: which resnet variant to use
        num_classes: number of output classes
        pretrained: load ImageNet weights
        freeze_backbone: if True, freeze all layers except the final head
        dropout: optional dropout before the final linear layer

    Returns:
        nn.Module
    """
    if model == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        in_features = net.fc.in_features
    elif model == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        in_features = net.fc.in_features
    else:
        raise ValueError(f"Unsupported ResNet model: {model}")

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    # Replace classification head
    if dropout > 0:
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        net.fc = nn.Linear(in_features, num_classes)

    return net


