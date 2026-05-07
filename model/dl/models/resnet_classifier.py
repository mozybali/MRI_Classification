#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resnet_classifier.py
--------------------
ResNet18 tabanli MRI siniflandirma modeli.
"""

from urllib.error import URLError

import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """ResNet18 tabanli siniflandirici."""

    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.5):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(
                f"num_classes en az 2 olan bir tamsayi olmali (alinan: {num_classes!r})."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 0 ile 1 arasinda olmali (alinan: {dropout}).")
        if pretrained:
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except (URLError, OSError, TimeoutError) as exc:
                raise RuntimeError(
                    "ResNet18 icin pretrained agirliklar istendi ama indirilemedi. "
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
