from __future__ import annotations

from typing import Optional

import torch.nn as nn
import timm


def build_timm(
    model_name: str,
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    # use timm to create model with correct classifier
    # drop_rate applies dropout before classifier in many timm models
    net = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )

    if freeze_backbone:
        for name, param in net.named_parameters():
            if any(h in name for h in ("classifier", "head", "fc")):
                continue
            param.requires_grad = False

    return net
