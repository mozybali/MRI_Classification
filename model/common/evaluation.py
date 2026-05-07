#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation.py
-------------
DL ve SL pipeline'lari tarafindan ortaklasa kullanilan
metrik hesaplama ve rapor yardimci fonksiyonlari.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

_PROB_TOL = 1e-6


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Ham metrik sozlugunden standart skaler metrikleri cikar."""
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def compute_valid_multiclass_auc_ap(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> tuple[float | None, float | None]:
    """Yalnizca gecerli siniflari kullanarak macro ROC-AUC ve AP hesapla.

    Caller (`build_detailed_eval_report`) sekil ve aralik validasyonunu yaptigi
    icin burada manuel one-hot ikili/cok sinifli ayrimi yapmadan calisir.
    """
    n_classes = len(class_names)
    labels_bin = np.eye(n_classes, dtype=int)[np.asarray(labels, dtype=int)]
    auc_scores: list[float] = []
    ap_scores: list[float] = []

    for class_idx in range(n_classes):
        positives = labels_bin[:, class_idx]
        if positives.size == 0 or positives.max() == 0 or positives.min() == 1:
            continue

        class_probs = probs[:, class_idx]
        try:
            auc_scores.append(float(roc_auc_score(positives, class_probs)))
        except ValueError:
            pass
        try:
            ap_scores.append(float(average_precision_score(positives, class_probs)))
        except ValueError:
            pass

    macro_auc_ovr = round_or_none(float(np.mean(auc_scores)), 4) if auc_scores else None
    macro_average_precision = round_or_none(float(np.mean(ap_scores)), 4) if ap_scores else None
    return macro_auc_ovr, macro_average_precision


def build_detailed_eval_report(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None,
    class_names: list[str],
) -> dict[str, Any]:
    """Derinlemesine degerlendirme metriklerini JSON uyumlu sekilde ozetle."""
    n_classes = len(class_names)
    if n_classes <= 0:
        raise ValueError("class_names bos olamaz.")

    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    if labels.shape != preds.shape:
        raise ValueError(
            f"labels ve preds sekilleri uyusmuyor: {labels.shape} vs {preds.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels 1D olmali, gelen sekil: {labels.shape}")
    if labels.size:
        for arr, name in ((labels, "labels"), (preds, "preds")):
            if arr.min() < 0 or arr.max() >= n_classes:
                raise ValueError(
                    f"{name} class_names araligi disinda: min={int(arr.min())}, "
                    f"max={int(arr.max())}, n_classes={n_classes}"
                )

    if probs is None:
        probs = np.empty((labels.shape[0], 0), dtype=np.float32)
    else:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.size and not np.isfinite(probs).all():
            raise ValueError("probs NaN/Inf icermemeli.")
        if probs.ndim == 1:
            if n_classes != 2:
                raise ValueError(
                    f"1D probs yalnizca ikili siniflandirmada kabul edilir "
                    f"(n_classes={n_classes})."
                )
            if probs.size and (
                probs.min() < -_PROB_TOL or probs.max() > 1 + _PROB_TOL
            ):
                raise ValueError("1D probs [0, 1] araliginda olmali.")
            probs = np.column_stack([1.0 - probs, probs]).astype(np.float32)
        if probs.shape != (labels.shape[0], n_classes):
            raise ValueError(
                f"probs sekli {probs.shape}, beklenen "
                f"{(labels.shape[0], n_classes)}"
            )

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(class_names)
    }

    confidences = probs.max(axis=1) if probs.size else np.array([], dtype=np.float32)
    correct_mask = labels == preds
    confidence_summary = {
        "mean_confidence": round_or_none(float(confidences.mean()), 4) if confidences.size else None,
        "mean_confidence_correct": (
            round_or_none(float(confidences[correct_mask].mean()), 4)
            if confidences.size and np.any(correct_mask)
            else None
        ),
        "mean_confidence_incorrect": (
            round_or_none(float(confidences[~correct_mask].mean()), 4)
            if confidences.size and np.any(~correct_mask)
            else None
        ),
        "high_confidence_error_count": (
            int(np.sum((~correct_mask) & (confidences >= 0.9)))
            if confidences.size
            else 0
        ),
    }

    summary = {
        "per_class": per_class,
        "confidence": confidence_summary,
        "macro_auc_ovr": None,
        "macro_average_precision": None,
    }

    if probs.size:
        (
            summary["macro_auc_ovr"],
            summary["macro_average_precision"],
        ) = compute_valid_multiclass_auc_ap(labels, probs, class_names)

    return summary
