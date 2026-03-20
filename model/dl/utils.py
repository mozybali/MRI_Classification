#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
--------
Yardimci fonksiyonlar: seed, device, confusion matrix ve egitim egrileri.
"""

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik icin tum random seed'leri ayarla."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(verbose: bool = True) -> torch.device:
    """GPU varsa CUDA, yoksa CPU dondur."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[OK] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[UYARI] GPU bulunamadi, CPU kullaniliyor")
    return device


def load_checkpoint(path: Path, map_location: torch.device | str):
    """Checkpoint'i pickle calistirmadan guvenli modda yukle."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint guvenli modda yuklenemedi. Dosyanin bu proje tarafindan "
            "olusturulan standart bir checkpoint oldugundan emin olun."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint formati gecersiz: sozluk bekleniyordu.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint formati gecersiz: 'model_state_dict' anahtari eksik.")

    return checkpoint


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: Path,
):
    """Confusion matrix gorseli olustur ve kaydet."""
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(labels, preds, labels=class_ids)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Confusion matrix kaydedildi: {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path,
):
    """Loss ve accuracy egitim egrilerini ciz ve kaydet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Egrisi")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Egrisi")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Egitim egrileri kaydedildi: {save_path}")
