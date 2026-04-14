#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py  (SL – Shallow Learning)
--------------------------------------------
XGBoost tabanli sig ogrenme egitim akisi.
DL training_runner ile ayni donuş şemasini kullanir.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)

from ..ayarlar import (
    CIKTI_KLASORU,
    GORSELLER_KLASORU,
    MODELS_KLASORU,
    RASTGELE_TOHUM,
    RAPORLAR_KLASORU,
    TEST_VERI_DIZINI,
    TRAINVAL_VERI_DIZINI,
)
from ..common.evaluation import (
    build_detailed_eval_report,
    compute_valid_multiclass_auc_ap,
    round_or_none,
)
from ..dl.dataset import (
    SINIF_ISIMLERI,
    _group_stratified_train_val_split,
    _stratified_train_val_split,
    _summarize_grouping,
    _augmentasyon_kopyasi_mi,
    kaynak_id_belirle,
)
from ..dl.utils import (
    plot_classification_summary,
    plot_confusion_matrix,
    plot_multiclass_roc_pr_curves,
    plot_prediction_confidence,
)
from .dataset import build_feature_matrix
from .xgb_classifier import build_xgb_classifier, save_xgb_model, load_xgb_model_with_meta

SUPPORTED_SELECTION_METRICS = {"loss", "accuracy", "precision", "recall", "f1"}


@dataclass(slots=True)
class SLTrainingConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    image_size: int = 224
    trainval_dir: Path | str | None = None
    test_dir: Path | str | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = RASTGELE_TOHUM
    feature_cache: Path | str | None = None


def sl_config_to_dict(config: SLTrainingConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("trainval_dir", "test_dir", "feature_cache"):
        if data[key] is not None:
            data[key] = str(Path(data[key]))
    return data


# ==================== Veri dizinleri ====================

def _contains_class_dirs(data_dir: Path) -> bool:
    return any((data_dir / class_name).is_dir() for class_name in SINIF_ISIMLERI)


def _resolve_split_subdir(data_dir: Path, split_name: str) -> Path | None:
    data_dir = Path(data_dir)
    split_dir = data_dir / split_name
    if split_dir.exists() and _contains_class_dirs(split_dir):
        return split_dir
    if data_dir.exists() and _contains_class_dirs(data_dir):
        return data_dir
    return None


def resolve_sl_data_dirs(
    config: SLTrainingConfig,
) -> tuple[Path, Path | None]:
    if config.trainval_dir:
        explicit = Path(config.trainval_dir)
        trainval_dir = _resolve_split_subdir(explicit, "trainval") or explicit
    else:
        trainval_dir = TRAINVAL_VERI_DIZINI

    if config.test_dir:
        explicit = Path(config.test_dir)
        test_dir = _resolve_split_subdir(explicit, "test") or explicit
    else:
        if TEST_VERI_DIZINI.exists() and _contains_class_dirs(TEST_VERI_DIZINI):
            if TEST_VERI_DIZINI.resolve() != trainval_dir.resolve():
                test_dir = TEST_VERI_DIZINI
            else:
                test_dir = None
        else:
            test_dir = None
    return trainval_dir, test_dir


def validate_sl_config(
    config: SLTrainingConfig,
    *,
    require_test_dir: bool = True,
    full_trainval: bool = False,
) -> None:
    if config.n_estimators < 1:
        raise ValueError("--xgb-n-estimators en az 1 olmali.")
    if config.max_depth < 1:
        raise ValueError("--xgb-max-depth en az 1 olmali.")
    if config.learning_rate <= 0:
        raise ValueError("--xgb-learning-rate pozitif olmali.")
    if not 0.0 < config.subsample <= 1.0:
        raise ValueError("--xgb-subsample 0 ile 1 arasinda olmali (0 haric, 1 dahil).")
    if not 0.0 < config.colsample_bytree <= 1.0:
        raise ValueError("--xgb-colsample-bytree 0 ile 1 arasinda olmali (0 haric, 1 dahil).")
    if config.reg_lambda < 0:
        raise ValueError("--xgb-reg-lambda negatif olamaz.")
    if config.min_child_weight < 0:
        raise ValueError("--xgb-min-child-weight negatif olamaz.")
    if config.image_size < 32:
        raise ValueError("--image-size en az 32 olmali.")
    if not full_trainval and not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali.")
    if config.test_ratio < 0.0 or config.test_ratio >= 1.0:
        raise ValueError("--test-ratio 0 ile 1 arasinda olmali.")

    trainval_dir, test_dir = resolve_sl_data_dirs(config)
    if not full_trainval and test_dir is None and config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio 1'den kucuk olmali.")
    if require_test_dir and test_dir is None and config.test_ratio <= 0.0:
        raise ValueError("Harici test dizini yoksa --test-ratio pozitif olmali.")
    if not trainval_dir.exists():
        raise FileNotFoundError(f"TrainVal veri dizini bulunamadi: {trainval_dir}")
    if test_dir is not None and not test_dir.exists():
        raise FileNotFoundError(f"Test veri dizini bulunamadi: {test_dir}")
    if full_trainval and require_test_dir:
        if test_dir is None:
            raise ValueError(
                "Full-trainval final egitim icin harici test dizini gerekli."
            )
        if test_dir.resolve() == trainval_dir.resolve():
            raise ValueError(
                "Full-trainval final egitim icin test dizini trainval'den farkli olmali."
            )


# ==================== Split ====================

def _group_split_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Ozellik matrisini grup-bazli train/val olarak bol."""
    num_classes = len(SINIF_ISIMLERI)
    augmented_flags = [_augmentasyon_kopyasi_mi(g.split("::")[-1]) for g in groups]
    group_stats = _summarize_grouping(groups, augmented_flags)

    use_group_split = group_stats["grouping_reliable"]
    split_warnings: list[str] = []
    if not use_group_split and group_stats["augmented_samples"] > 0:
        split_warnings.append(
            "TrainVal dosya adlarindan tekrarli kaynak grup cikarilamadi; "
            "stratified split kullanildi."
        )
        warnings.warn(
            "Augmentasyon kopyalari tespit edildi ancak grup-bazli split yapilamadi. "
            "Stratified split kullanildi — augmentasyon kopyalari farkli split'lere "
            "dusebilir ve veri sizintisina yol acabilir.",
            stacklevel=2,
        )

    labels_list = y.tolist()
    if use_group_split:
        train_idxs, val_idxs = _group_stratified_train_val_split(
            labels=labels_list,
            groups=groups,
            val_ratio=val_ratio,
            seed=seed,
            num_classes=num_classes,
        )
    else:
        train_idxs, val_idxs = _stratified_train_val_split(
            labels=labels_list,
            val_ratio=val_ratio,
            seed=seed,
            num_classes=num_classes,
        )

    strategy = "group_stratified" if use_group_split else "stratified_without_groups"
    train_groups_set = {groups[i] for i in train_idxs}
    val_groups_set = {groups[i] for i in val_idxs}

    info = {
        "num_classes": num_classes,
        "train_size": len(train_idxs),
        "val_size": len(val_idxs),
        "train_groups": len(train_groups_set),
        "val_groups": len(val_groups_set),
        "split_strategy": strategy,
        "split_warnings": split_warnings,
        "trainval_grouping": group_stats,
    }
    return (
        X[train_idxs],
        y[train_idxs],
        X[val_idxs],
        y[val_idxs],
        info,
    )


# ==================== Cikti dizinleri ====================

def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    if output_root.resolve() == CIKTI_KLASORU.resolve():
        models_dir = MODELS_KLASORU
        reports_dir = RAPORLAR_KLASORU
        visuals_dir = GORSELLER_KLASORU
    else:
        models_dir = output_root / "modeller"
        reports_dir = output_root / "raporlar"
        visuals_dir = output_root / "gorseller"
    for d in (models_dir, reports_dir, visuals_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "models": models_dir, "reports": reports_dir, "visuals": visuals_dir}


# ==================== Egitim egrisi ====================

def _plot_xgb_training_curves(
    evals_result: dict[str, dict[str, list[float]]],
    save_path: Path,
    *,
    best_iteration: int | None = None,
) -> None:
    """XGBoost evals_result çıktısından train/val logloss eğrisi çiz."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            ax.plot(range(1, len(values) + 1), values, label=f"{label} {metric_name}", linewidth=2)

    if best_iteration is not None:
        ax.axvline(best_iteration, color="gray", linestyle=":", alpha=0.7, label=f"Best iter={best_iteration}")

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Log Loss")
    ax.set_title("XGBoost Egitim Egrisi")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] XGBoost egitim egrisi kaydedildi: {save_path}")


# ==================== Ana egitim fonksiyonu ====================

def _selection_mode_for_metric(metric: str) -> Literal["minimize", "maximize"]:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Gecersiz selection metric: {metric}")
    return "minimize" if metric == "loss" else "maximize"


def run_sl_training(
    config: SLTrainingConfig,
    *,
    output_root: Path | None = None,
    artifact_tag: str | None = None,
    save_artifacts: bool = True,
    evaluate_test_set: bool = True,
    full_trainval: bool = False,
    verbose: bool = True,
    selection_metric: str = "f1",
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """XGBoost ile tek bir egitim deneyimi calistir."""

    validate_sl_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=full_trainval,
    )

    np.random.seed(config.seed)
    trainval_dir, test_dir = resolve_sl_data_dirs(config)
    if not evaluate_test_set:
        test_dir = None

    if verbose:
        print("\n[INFO] XGBoost egitimi basliyor")
        print(f"  Veri dizini     : {trainval_dir}")
        print(f"  Test dizini     : {test_dir}")

    # Ozellik cikartimi
    tv_cache = Path(config.feature_cache) / f"trainval_img{config.image_size}.npz" if config.feature_cache else None
    X_tv, y_tv, groups_tv, paths_tv = build_feature_matrix(
        trainval_dir, image_size=config.image_size, cache_path=tv_cache,
    )

    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_grouping: dict[str, Any] | None = None
    if test_dir is not None:
        te_cache = Path(config.feature_cache) / f"test_img{config.image_size}.npz" if config.feature_cache else None
        X_test, y_test, groups_te, paths_te = build_feature_matrix(
            test_dir, image_size=config.image_size, cache_path=te_cache,
        )
        aug_flags_te = [_augmentasyon_kopyasi_mi(g.split("::")[-1]) for g in groups_te]
        test_grouping = _summarize_grouping(groups_te, aug_flags_te)

    num_classes = len(SINIF_ISIMLERI)

    # Split
    split_info: dict[str, Any]
    if full_trainval:
        X_train, y_train = X_tv, y_tv
        X_val, y_val = None, None
        aug_flags_tv = [_augmentasyon_kopyasi_mi(g.split("::")[-1]) for g in groups_tv]
        tv_grouping = _summarize_grouping(groups_tv, aug_flags_tv)
        split_info = {
            "num_classes": num_classes,
            "train_size": len(y_tv),
            "val_size": 0,
            "train_groups": tv_grouping["unique_groups"],
            "val_groups": 0,
            "split_strategy": "full_trainval",
            "split_warnings": [],
            "trainval_grouping": tv_grouping,
        }
    else:
        X_train, y_train, X_val, y_val, split_info = _group_split_feature_matrix(
            X_tv, y_tv, groups_tv, val_ratio=config.val_ratio, seed=config.seed,
        )

    if verbose:
        print(f"  Train: {split_info['train_size']}, Val: {split_info['val_size']}")
        print(f"  Test : {len(y_test) if y_test is not None else 0}")
        print(f"  Ozellik boyutu: {X_train.shape[1]}")

    # Model olustur
    xgb_params: dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "min_child_weight": config.min_child_weight,
        "random_state": config.seed,
    }
    model = build_xgb_classifier(num_classes, xgb_params)

    # Fit
    fit_params: dict[str, Any] = {"verbose": verbose}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        early_stopping_rounds = max(10, config.n_estimators // 10)
        model.set_params(early_stopping_rounds=early_stopping_rounds)
    else:
        fit_params["eval_set"] = [(X_train, y_train)]

    model.fit(X_train, y_train, **fit_params)

    best_iteration = getattr(model, "best_iteration", None)
    evals_result = model.evals_result()

    # Val metrikleri
    best_val_metrics: dict[str, float] | None = None
    best_val_detailed: dict[str, Any] | None = None
    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, labels=list(range(num_classes)), zero_division=0, average="macro",
        )
        # val logloss from evals_result — best_iteration varsa onu kullan
        val_logloss_list = list(evals_result.get("validation_1", {}).get("mlogloss", []))
        if val_logloss_list:
            bi = best_iteration if best_iteration is not None and best_iteration < len(val_logloss_list) else len(val_logloss_list) - 1
            val_loss = val_logloss_list[bi]
        else:
            val_loss = 0.0
        best_val_metrics = {
            "loss": float(val_loss),
            "accuracy": val_acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        best_val_detailed = build_detailed_eval_report(y_val, val_preds, val_probs, SINIF_ISIMLERI)

    # Train metrikleri
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)
    train_acc = float(accuracy_score(y_train, train_preds))
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
        y_train, train_preds, labels=list(range(num_classes)), zero_division=0, average="macro",
    )
    train_logloss_list = list(evals_result.get("validation_0", {}).get("mlogloss", []))
    train_loss = train_logloss_list[-1] if train_logloss_list else 0.0
    best_train_metrics = {
        "loss": float(train_loss),
        "accuracy": train_acc,
        "precision": float(t_prec),
        "recall": float(t_rec),
        "f1": float(t_f1),
    }

    # Artifact kaydi
    artifact_stem = artifact_tag or "xgboost"
    output_dirs = None
    checkpoint_path = None
    if save_artifacts:
        output_dirs = _build_output_dirs(output_root or CIKTI_KLASORU)
        checkpoint_path = output_dirs["models"] / f"best_{artifact_stem}.json"
        save_xgb_model(
            model,
            checkpoint_path,
            image_size=config.image_size,
            class_names=SINIF_ISIMLERI,
            seed=config.seed,
        )
        if verbose:
            print(f"[OK] XGBoost model kaydedildi: {checkpoint_path}")

    # Test degerlendirmesi
    test_metrics: dict[str, float] | None = None
    test_detailed: dict[str, Any] | None = None
    test_size = 0
    if evaluate_test_set and X_test is not None and y_test is not None:
        if verbose:
            print(f"\n{'=' * 70}")
            print("TEST DEGERLENDIRMESI")
            print(f"{'=' * 70}\n")

        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        test_size = len(y_test)
        te_acc = float(accuracy_score(y_test, test_preds))
        te_prec, te_rec, te_f1, _ = precision_recall_fscore_support(
            y_test, test_preds, labels=list(range(num_classes)), zero_division=0, average="macro",
        )
        test_metrics = {
            "loss": float(log_loss(y_test, test_probs, labels=list(range(num_classes)))),
            "accuracy": te_acc,
            "precision": float(te_prec),
            "recall": float(te_rec),
            "f1": float(te_f1),
        }
        test_detailed = build_detailed_eval_report(y_test, test_preds, test_probs, SINIF_ISIMLERI)

        if verbose:
            print(f"  Accuracy : {te_acc:.4f}")
            print(f"  Precision: {float(te_prec):.4f}")
            print(f"  Recall   : {float(te_rec):.4f}")
            print(f"  F1 (macro): {float(te_f1):.4f}")
            if test_detailed["macro_auc_ovr"] is not None:
                print(f"  ROC-AUC (macro OVR): {test_detailed['macro_auc_ovr']:.4f}")

        # Gorseller
        if output_dirs is not None:
            plot_confusion_matrix(
                y_test, test_preds, SINIF_ISIMLERI,
                output_dirs["visuals"] / f"confusion_matrix_{artifact_stem}.png",
            )
            plot_confusion_matrix(
                y_test, test_preds, SINIF_ISIMLERI,
                output_dirs["visuals"] / f"confusion_matrix_normalized_{artifact_stem}.png",
                normalize=True,
            )
            plot_classification_summary(
                y_test, test_preds, SINIF_ISIMLERI,
                output_dirs["visuals"] / f"classification_summary_{artifact_stem}.png",
            )
            test_confidences = test_probs.max(axis=1)
            plot_prediction_confidence(
                test_confidences, y_test, test_preds,
                output_dirs["visuals"] / f"prediction_confidence_{artifact_stem}.png",
            )
            if test_probs.size:
                plot_multiclass_roc_pr_curves(
                    y_test, test_probs, SINIF_ISIMLERI,
                    output_dirs["visuals"] / f"roc_pr_curves_{artifact_stem}.png",
                )

    # Egitim egrisi gorseli
    if output_dirs is not None and evals_result:
        _plot_xgb_training_curves(
            evals_result,
            output_dirs["visuals"] / f"training_curves_{artifact_stem}.png",
            best_iteration=best_iteration,
        )

    # Rapor
    report_path = None
    # best_selection_value hesapla (rapor ve return dict icin)
    best_selection_value: float | None = None
    if best_val_metrics is not None:
        best_selection_value = best_val_metrics.get(selection_metric)
    elif best_train_metrics is not None:
        best_selection_value = best_train_metrics.get(selection_metric)

    if output_dirs is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dirs["reports"] / f"rapor_{artifact_stem}_{timestamp}.json"

        # logloss listelerini history olarak kaydet
        history: dict[str, list[float]] = {}
        for set_name, set_metrics in evals_result.items():
            for metric_name, values in set_metrics.items():
                history[f"{set_name}_{metric_name}"] = [float(v) for v in values]

        selection_mode = "fixed_epoch_full_trainval" if full_trainval else _selection_mode_for_metric(selection_metric)

        report: dict[str, Any] = {
            "model": "xgboost",
            "timestamp": timestamp,
            "n_estimators_trained": best_iteration or config.n_estimators,
            "best_iteration": best_iteration,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_value": round(best_selection_value, 6) if best_selection_value is not None else None,
            "best_val_metrics": (
                {
                    "accuracy": round(best_val_metrics["accuracy"], 4),
                    "precision": round(best_val_metrics["precision"], 4),
                    "recall": round(best_val_metrics["recall"], 4),
                    "f1_macro": round(best_val_metrics["f1"], 4),
                }
                if best_val_metrics is not None else None
            ),
            "best_train_metrics": (
                {
                    "accuracy": round(best_train_metrics["accuracy"], 4),
                    "precision": round(best_train_metrics["precision"], 4),
                    "recall": round(best_train_metrics["recall"], 4),
                    "f1_macro": round(best_train_metrics["f1"], 4),
                }
                if best_train_metrics is not None else None
            ),
            "best_val_detailed_metrics": best_val_detailed,
            "test_metrics": (
                {
                    "accuracy": round(test_metrics["accuracy"], 4),
                    "precision": round(test_metrics["precision"], 4),
                    "recall": round(test_metrics["recall"], 4),
                    "f1_macro": round(test_metrics["f1"], 4),
                }
                if test_metrics is not None else None
            ),
            "test_detailed_metrics": test_detailed,
            "data_split": {
                "strategy": split_info["split_strategy"],
                "trainval_dir": str(trainval_dir),
                "test_dir": str(test_dir) if test_dir else None,
                "val_ratio": config.val_ratio if not full_trainval else None,
                "test_ratio": config.test_ratio,
                "train_size": split_info["train_size"],
                "val_size": split_info["val_size"],
                "test_size": test_size,
                "trainval_grouping": split_info["trainval_grouping"],
                "test_grouping": test_grouping,
                "warnings": split_info["split_warnings"],
                "full_trainval_run": full_trainval,
            },
            "config": sl_config_to_dict(config),
            "history": history,
            "artifacts": {
                "training_dashboard": str(
                    output_dirs["visuals"] / f"training_curves_{artifact_stem}.png"
                ),
                "confusion_matrix": (
                    str(output_dirs["visuals"] / f"confusion_matrix_{artifact_stem}.png")
                    if test_metrics else None
                ),
                "confusion_matrix_normalized": (
                    str(output_dirs["visuals"] / f"confusion_matrix_normalized_{artifact_stem}.png")
                    if test_metrics else None
                ),
                "classification_summary": (
                    str(output_dirs["visuals"] / f"classification_summary_{artifact_stem}.png")
                    if test_metrics else None
                ),
                "prediction_confidence": (
                    str(output_dirs["visuals"] / f"prediction_confidence_{artifact_stem}.png")
                    if test_metrics else None
                ),
                "roc_pr_curves": (
                    str(output_dirs["visuals"] / f"roc_pr_curves_{artifact_stem}.png")
                    if test_metrics else None
                ),
            },
        }
        if extra_report:
            report["extra"] = extra_report

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n[OK] Rapor kaydedildi: {report_path}")
            print(f"\n{'=' * 70}")
            print("EGITIM TAMAMLANDI")
            print(f"{'=' * 70}\n")

    return {
        "best_iteration": best_iteration,
        "best_val_loss": best_val_metrics["loss"] if best_val_metrics else None,
        "lowest_val_loss": best_val_metrics["loss"] if best_val_metrics else None,
        "best_val_metrics": best_val_metrics,
        "best_train_metrics": best_train_metrics,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": "fixed_epoch_full_trainval" if full_trainval else _selection_mode_for_metric(selection_metric),
        "test_metrics": test_metrics,
        "history": history if output_dirs else {},
        "data_info": {
            **split_info,
            "test_size": test_size,
            "test_grouping": test_grouping,
        },
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
        "output_root": output_dirs["root"] if output_dirs else None,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": sl_config_to_dict(config),
        "best_val_detailed_metrics": best_val_detailed,
        "test_detailed_metrics": test_detailed,
        "full_trainval": full_trainval,
    }
