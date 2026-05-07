#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
losses.py
---------
Derin öğrenme kayıp fonksiyonları: Focal Loss ve class weight hesaplama.
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_REDUCTIONS = ("mean", "sum", "none")


class FocalLoss(nn.Module):
    """
    Focal Loss - sınıf dengesizliğine karşı etkili kayıp fonksiyonu.

    Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Not: gamma=0 ve alpha verildiğinde bu modül "örnek bazlı ağırlıklı CE
    ortalaması" döner (alpha[t] * CE örnek başına, sonra .mean()).
    PyTorch'un ``nn.CrossEntropyLoss(weight=alpha)`` davranışı ise ağırlıklı
    toplamı hedef sınıf ağırlıkları toplamına böler. Bu nedenle iki kaybın
    ham değerleri (örneğin HPO karşılaştırmalarında) birebir aynı ölçekte
    olmayabilir.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"FocalLoss.reduction gecersiz: {reduction!r}. "
                f"Beklenen: {_VALID_REDUCTIONS}"
            )
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            raise TypeError(
                f"FocalLoss.alpha torch.Tensor olmali, alindi: {type(alpha).__name__}"
            )
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "alpha",
            alpha.detach().clone() if alpha is not None else None,
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            if alpha.ndim == 0:
                alpha_factor = torch.full_like(focal_loss, fill_value=alpha.item())
            else:
                if alpha.ndim != 1:
                    raise ValueError(
                        f"FocalLoss.alpha 1D tensor (veya skaler) olmali; "
                        f"alinan shape: {tuple(alpha.shape)}"
                    )
                num_classes = inputs.shape[1]
                if alpha.numel() != num_classes:
                    raise ValueError(
                        f"FocalLoss.alpha uzunlugu ({alpha.numel()}) sinif sayisi "
                        f"({num_classes}) ile eslesmiyor."
                    )
                alpha_factor = alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_factor

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    num_classes: int = 4,
) -> torch.Tensor:
    """Ters-frekans tabanlı sınıf ağırlıkları hesapla."""
    if num_classes <= 0:
        raise ValueError(f"num_classes pozitif olmali, alindi: {num_classes}")

    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    if labels_arr.ndim != 1:
        raise ValueError(
            f"compute_class_weights 1D etiket dizisi bekliyor; alinan shape: "
            f"{labels_arr.shape}"
        )

    if labels_arr.size == 0:
        raise ValueError(
            "compute_class_weights bos etiket listesi ile cagrildi; "
            "agirlik hesaplamak icin en az bir ornek gerekli."
        )

    if not np.issubdtype(labels_arr.dtype, np.integer):
        if np.issubdtype(labels_arr.dtype, np.floating) and np.all(
            labels_arr == labels_arr.astype(np.int64)
        ):
            labels_arr = labels_arr.astype(np.int64)
        else:
            raise ValueError(
                f"compute_class_weights tamsayi etiket bekliyor; alinan dtype: "
                f"{labels_arr.dtype}"
            )

    label_min = int(labels_arr.min())
    label_max = int(labels_arr.max())
    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            f"Etiket araligi [0, {num_classes - 1}] disinda deger iceriyor: "
            f"min={label_min}, max={label_max}, num_classes={num_classes}."
        )

    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
