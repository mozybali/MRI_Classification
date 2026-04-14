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
from sklearn.preprocessing import label_binarize


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
    """Yalnizca gecerli siniflari kullanarak macro ROC-AUC ve AP hesapla."""
    labels_bin = label_binarize(labels, classes=np.arange(len(class_names)))
    auc_scores: list[float] = []
    ap_scores: list[float] = []

    for class_idx in range(len(class_names)):
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
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs) if probs is not None else np.empty((len(labels), 0), dtype=np.float32)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
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
