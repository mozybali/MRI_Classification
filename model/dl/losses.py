#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
losses.py
---------
Derin öğrenme kayıp fonksiyonları: Focal Loss ve class weight hesaplama.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - sınıf dengesizliğine karşı etkili kayıp fonksiyonu.

    Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        alpha_factor = None

        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            if alpha.ndim == 0:
                alpha_factor = torch.full_like(focal_loss, fill_value=alpha.item())
            else:
                alpha_factor = alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_factor

        if self.reduction == "mean":
            if alpha_factor is not None and alpha_factor.ndim > 0:
                return focal_loss.sum() / alpha_factor.sum().clamp_min(1e-12)
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels, num_classes: int = 4) -> torch.Tensor:
    """Ters-frekans tabanlı sınıf ağırlıkları hesapla."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
