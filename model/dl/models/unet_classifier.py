#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unet_classifier.py
------------------
U-Net encoder + global average pooling + classification head.
Segmentasyon mask'ı olmadığında sınıflandırma için kullanılır.
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """İki ardışık Conv-BN-ReLU bloğu."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetClassifier(nn.Module):
    """
    U-Net encoder + classification head.

    Segmentasyon mask'ı yoksa decoder kullanılmaz; bunun yerine
    encoder çıktısı global average pooling ile sıkıştırılıp
    FC katmanlarıyla sınıflandırılır.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(512, 1024)

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        x = self.bottleneck(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
