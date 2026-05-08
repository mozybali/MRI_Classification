#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
engine.py
---------
Egitim ve degerlendirme donguleri, early stopping mekanizmasi.
"""

import contextlib
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


def _scalar_metrics_from_arrays(
    labels: np.ndarray,
    preds: np.ndarray,
    *,
    loss: float,
) -> Dict[str, float]:
    """Tek precision_recall_fscore_support cagrisi ile skaler metrikleri uret."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _autocast_context(device: torch.device, use_amp: bool):
    """AMP autocast context'i; CUDA'da fp16, MPS/CPU'da no-op."""
    if not use_amp:
        return contextlib.nullcontext()
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> Dict[str, float]:
    """Tek epoch egitim dongusu.

    ``use_amp=True`` ve CUDA mevcutsa forward/backward autocast altinda calisir;
    fp16 iken ``scaler`` ile gradient scaling uygulanir. ``scaler`` her epoch
    icin yeniden olusturulmamali, training_runner'da bir kez yaratilip burada
    paylasilmali (PyTorch best practice).
    """
    model.train()
    running_loss = 0.0
    pred_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    use_scaler = bool(use_amp and scaler is not None and device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += _batch_loss_sum(loss, criterion, images.size(0))
        pred_chunks.append(outputs.detach().argmax(dim=1).cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())

    if not label_chunks:
        raise RuntimeError("train_one_epoch: bos veri yukleyici alindi, egitim adimi atilamiyor.")

    all_labels = np.concatenate(label_chunks)
    all_preds = np.concatenate(pred_chunks)
    n = int(all_labels.shape[0])
    return _scalar_metrics_from_arrays(all_labels, all_preds, loss=running_loss / n)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> Dict[str, object]:
    """Degerlendirme dongusu. Metrikler ve tahmin/prob dizi bilgisi dondurur.

    ``inference_mode`` no_grad'den biraz daha hizli (autograd metadata'sini da
    devre disi birakir). ``use_amp=True`` iken CUDA'da autocast altinda forward
    yapilir; logits softmax oncesi float32'ye cevrilir, prob/array'lerin tipi
    deterministik kalir.
    """
    model.eval()
    running_loss = 0.0
    pred_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    prob_chunks: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Softmax ve metrikler her zaman float32 uzerinden hesaplansin.
        probs = torch.softmax(outputs.float(), dim=1)

        running_loss += _batch_loss_sum(loss, criterion, images.size(0))
        pred_chunks.append(probs.argmax(dim=1).detach().cpu())
        label_chunks.append(labels.detach().cpu())
        prob_chunks.append(probs.detach().cpu())

    if not label_chunks:
        raise RuntimeError("evaluate: bos veri yukleyici alindi, metrik hesaplanamiyor.")

    all_labels = torch.cat(label_chunks).numpy()
    all_preds = torch.cat(pred_chunks).numpy()
    probs_arr = torch.cat(prob_chunks).numpy().astype(np.float32, copy=False)

    n = int(all_labels.shape[0])
    metrics = _scalar_metrics_from_arrays(all_labels, all_preds, loss=running_loss / n)
    confidences = probs_arr.max(axis=1) if probs_arr.size else np.array([], dtype=np.float32)
    metrics.update(
        {
            "preds": all_preds,
            "labels": all_labels,
            "probs": probs_arr,
            "confidences": confidences,
        }
    )
    return metrics


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
