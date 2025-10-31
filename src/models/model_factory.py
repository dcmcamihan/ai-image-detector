from typing import Any, Dict

import torch.nn as nn

from .resnet_model import build_resnet
from .vit_model import build_vit  # type: ignore
from .timm_model import build_timm


def create_model(name: str, **kwargs: Dict[str, Any]) -> nn.Module:
    """
    Factory to create models by name.

    Supported names:
      - "resnet18"
      - "resnet50"
      - "vit_b_16"
      - "swin_t" / "swin_tiny_patch4_window7_224"
      - "efficientnet_b4"

    Common kwargs:
      - num_classes (int)
      - pretrained (bool)
      - freeze_backbone (bool)
      - dropout (float)
    """
    name = name.lower()
    if name in {"resnet18", "resnet50"}:
        return build_resnet(model=name, **kwargs)
    if name == "vit_b_16":
        return build_vit(variant=name, **kwargs)
    if name in {"swin_t", "swin_tiny_patch4_window7_224"}:
        # normalize alias
        model_name = "swin_tiny_patch4_window7_224"
        return build_timm(model_name, **kwargs)
    if name == "efficientnet_b4":
        return build_timm(name, **kwargs)
    raise ValueError(f"Unsupported model name: {name}")


