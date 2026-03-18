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

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights_enum = getattr(models, "ResNet18_Weights", None)

        if pretrained:
            try:
                if weights_enum is not None:
                    self.backbone = models.resnet18(weights=weights_enum.DEFAULT)
                else:
                    self.backbone = models.resnet18(pretrained=True)
            except Exception as exc:
                raise RuntimeError(
                    "ResNet18 icin pretrained agirliklar istendi ama yuklenemedi. "
                    "Baglanti/cache durumunu kontrol edin veya --pretrained olmadan calistirin."
                ) from exc
        else:
            if weights_enum is not None:
                self.backbone = models.resnet18(weights=None)
            else:
                self.backbone = models.resnet18(pretrained=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
