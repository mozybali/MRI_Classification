#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
--------
Yardimci fonksiyonlar: seed, device ve detayli degerlendirme gorselleri.
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
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


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
    """CUDA > MPS > CPU oncelik sirasi ile uygun device dondur."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[OK] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[OK] Apple MPS GPU kullaniliyor")
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


def _prepare_save_path(save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: Path,
    *,
    normalize: bool = False,
):
    """Confusion matrix gorseli olustur ve kaydet."""
    save_path = _prepare_save_path(save_path)
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(labels, preds, labels=class_ids)

    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        annot = np.vectorize(lambda value: f"{value:.2f}")(cm)
        fmt = ""
        cmap = "Blues"
        title = "Normalize Confusion Matrix"
    else:
        annot = True
        fmt = "d"
        cmap = "Blues"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Confusion matrix kaydedildi: {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float] | None,
    train_accs: List[float],
    val_accs: List[float] | None,
    save_path: Path,
    *,
    train_precisions: List[float] | None = None,
    val_precisions: List[float] | None = None,
    train_recalls: List[float] | None = None,
    val_recalls: List[float] | None = None,
    train_f1s: List[float] | None = None,
    val_f1s: List[float] | None = None,
    best_epoch: int | None = None,
):
    """Loss ve temel performans trendlerini tek dashboard'ta ciz."""
    save_path = _prepare_save_path(save_path)
    epochs = np.arange(1, len(train_losses) + 1)
    has_val_curves = bool(val_losses) and bool(val_accs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_loss, ax_acc, ax_f1, ax_pr = axes.flatten()

    ax_loss.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    if has_val_curves:
        ax_loss.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Egrisi")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, train_accs, label="Train Acc", linewidth=2)
    if has_val_curves:
        ax_acc.plot(epochs, val_accs, label="Val Acc", linewidth=2)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Egrisi")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    if train_f1s is not None and val_f1s is not None and has_val_curves:
        ax_f1.plot(epochs, train_f1s, label="Train F1", linewidth=2)
        ax_f1.plot(epochs, val_f1s, label="Val F1", linewidth=2)
    elif train_f1s is not None:
        ax_f1.plot(epochs, train_f1s, label="Train F1", linewidth=2)
    elif has_val_curves:
        gap = np.array(train_accs) - np.array(val_accs)
        ax_f1.plot(epochs, gap, label="Acc Gap", linewidth=2, color="#d62728")
    else:
        ax_f1.plot(epochs, train_accs, label="Train Acc", linewidth=2, color="#d62728")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Skor")
    ax_f1.set_title("F1 / Genel Ayrisım")
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)

    plotted_pr = False
    if train_precisions is not None:
        ax_pr.plot(epochs, train_precisions, label="Train Precision", linewidth=2)
        plotted_pr = True
    if train_precisions is not None and val_precisions is not None and has_val_curves:
        ax_pr.plot(epochs, val_precisions, label="Val Precision", linewidth=2)
    if train_recalls is not None:
        ax_pr.plot(epochs, train_recalls, label="Train Recall", linewidth=2, linestyle="--")
        plotted_pr = True
    if train_recalls is not None and val_recalls is not None and has_val_curves:
        ax_pr.plot(epochs, val_recalls, label="Val Recall", linewidth=2, linestyle="--")
    if not plotted_pr:
        if has_val_curves:
            ax_pr.plot(epochs, np.array(train_losses) - np.array(val_losses), label="Loss Gap", linewidth=2)
        else:
            ax_pr.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylabel("Skor")
    ax_pr.set_title("Precision / Recall")
    ax_pr.legend()
    ax_pr.grid(True, alpha=0.3)

    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        for ax in axes.flatten():
            ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Egitim dashboard'u kaydedildi: {save_path}")


def plot_classification_summary(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: Path,
):
    """Sinif bazli precision/recall/F1 ve support ozet grafigi olustur."""
    save_path = _prepare_save_path(save_path)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    x = np.arange(len(class_names))
    width = 0.22
    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.bar(x - width, precision, width=width, label="Precision", color="#4C78A8")
    ax1.bar(x, recall, width=width, label="Recall", color="#F58518")
    ax1.bar(x + width, f1, width=width, label="F1", color="#54A24B")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Skor")
    ax1.set_xlabel("Sinif")
    ax1.set_title("Sinif Bazli Performans Ozeti")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, support, color="#B279A2", marker="o", linewidth=2, label="Support")
    ax2.set_ylabel("Ornek Sayisi")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Sinif bazli performans ozeti kaydedildi: {save_path}")


def plot_prediction_confidence(
    confidences: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
):
    """Dogru ve yanlis tahminlerin guven dagilimini goster."""
    save_path = _prepare_save_path(save_path)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)

    if labels.shape != preds.shape:
        raise ValueError("labels ve preds ayni sekilde olmali.")

    if confidences.size != labels.size:
        for ax in axes:
            ax.axis("off")
            ax.text(
                0.5,
                0.58,
                "Confidence verisi mevcut degil",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.42,
                f"Ornek sayisi: {labels.size}",
                ha="center",
                va="center",
                fontsize=10,
            )
        fig.suptitle("Tahmin Guveni")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Tahmin guven grafikleri kaydedildi: {save_path}")
        return

    correctness = labels == preds

    bins = np.linspace(0.0, 1.0, 16)
    axes[0].hist(confidences[correctness], bins=bins, alpha=0.7, label="Dogru", color="#54A24B")
    axes[0].hist(confidences[~correctness], bins=bins, alpha=0.7, label="Yanlis", color="#E45756")
    axes[0].set_xlabel("Tahmin Guveni")
    axes[0].set_ylabel("Ornek Sayisi")
    axes[0].set_title("Dogru/Yanlis Tahmin Guveni")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(
        [
            confidences[correctness] if np.any(correctness) else np.array([0.0]),
            confidences[~correctness] if np.any(~correctness) else np.array([0.0]),
        ],
        labels=["Dogru", "Yanlis"],
        patch_artist=True,
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Tahmin Guveni")
    axes[1].set_title("Guven Dagilimi Ozet")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Tahmin guven grafikleri kaydedildi: {save_path}")


def plot_multiclass_roc_pr_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_path: Path,
):
    """Cok sinifli one-vs-rest ROC ve PR egirilerini ciz."""
    save_path = _prepare_save_path(save_path)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    classes = np.arange(len(class_names))
    labels_bin = label_binarize(labels, classes=classes)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))
    valid_class_count = 0

    for class_idx, class_name in enumerate(class_names):
        positives = labels_bin[:, class_idx]
        if positives.max() == 0 or positives.min() == 1:
            continue

        valid_class_count += 1
        fpr, tpr, _ = roc_curve(positives, probs[:, class_idx])
        precision, recall, _ = precision_recall_curve(positives, probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(positives, probs[:, class_idx])

        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        ax_pr.plot(recall, precision, linewidth=2, label=f"{class_name} (AP={avg_precision:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("One-vs-Rest ROC Egrileri")
    ax_roc.grid(True, alpha=0.3)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("One-vs-Rest Precision-Recall Egrileri")
    ax_pr.grid(True, alpha=0.3)

    if valid_class_count > 0:
        ax_roc.legend(fontsize=9, loc="lower right")
        ax_pr.legend(fontsize=9, loc="lower left")
    else:
        ax_roc.text(0.5, 0.5, "ROC icin yeterli sinif cesitliligi yok", ha="center", va="center")
        ax_pr.text(0.5, 0.5, "PR icin yeterli sinif cesitliligi yok", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] ROC/PR egirileri kaydedildi: {save_path}")
