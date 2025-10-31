from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import LBFGS


class ModelWithTemperature(nn.Module):
    """
    Wrap a classification model and calibrate its softmax probabilities by optimizing
    a single temperature parameter on validation logits/labels.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def set_temperature(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        logits = logits.detach()
        labels = labels.detach()

        def _eval():
            optimizer.zero_grad()
            loss = nll_criterion(self._scaled_logits(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self.temperature.item()

    def _scaled_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._scaled_logits(logits), dim=1)
