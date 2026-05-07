#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
engine.py
---------
Egitim ve degerlendirme donguleri, early stopping mekanizmasi.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..ayarlar import VARSAYILAN_EARLY_STOPPING_SABIR


def _batch_loss_sum(loss: torch.Tensor, criterion: nn.Module, batch_size: int) -> float:
    """Criterion'un reduction moduna gore batch icindeki toplam kaybi dondurur.

    reduction="none" -> elemanlari topla; "sum" -> oldugu gibi; "mean" -> batch ile carp.
    """
    detached = loss.detach()
    if detached.dim() > 0:
        return float(detached.sum().item())
    reduction = getattr(criterion, "reduction", "mean")
    if reduction == "sum":
        return float(detached.item())
    return float(detached.item()) * batch_size


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Tek epoch egitim dongusu."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += _batch_loss_sum(loss, criterion, images.size(0))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    if n == 0:
        raise RuntimeError("train_one_epoch: bos veri yukleyici alindi, egitim adimi atilamiyor.")
    return {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Degerlendirme dongusu. Metrikler ve tahmin/prob dizi bilgisi dondurur."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)

        running_loss += _batch_loss_sum(loss, criterion, images.size(0))
        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    n = len(all_labels)
    if n == 0:
        raise RuntimeError("evaluate: bos veri yukleyici alindi, metrik hesaplanamiyor.")
    probs_arr = np.array(all_probs, dtype=np.float32)
    confidences = probs_arr.max(axis=1) if probs_arr.size else np.array([], dtype=np.float32)
    return {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": probs_arr,
        "confidences": confidences,
    }


class EarlyStopping:
    """Overfitting'i onlemek icin early stopping mekanizmasi."""

    def __init__(self, patience: int = VARSAYILAN_EARLY_STOPPING_SABIR, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        # NaN/Inf gelirse karsilastirmalar False doner ve best_score bozulur;
        # bu durumu acik bir bozulma adimi olarak say.
        if not math.isfinite(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop

        score = -val_loss
        if self.best_score is None:
            # Ilk gecerli skor baseline'i kurar; onceki NaN/Inf adimlarinda
            # artmis olabilecek counter'i sifirla.
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop
