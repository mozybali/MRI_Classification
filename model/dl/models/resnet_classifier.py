#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resnet_classifier.py
--------------------
Pretrained ResNet18 tabanli MRI siniflandirma modeli.
"""

import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """ResNet18 tabanli siniflandirici."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 0 ile 1 arasinda olmali (alinan: {dropout}).")
        if pretrained:
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as exc:
                raise RuntimeError(
                    "ResNet18 icin pretrained agirliklar istendi ama yuklenemedi. "
                    "Baglanti/cache durumunu kontrol edin veya --pretrained olmadan calistirin."
                ) from exc
        else:
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
