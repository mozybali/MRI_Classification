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
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
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
from ..common.evaluation import build_detailed_eval_report
from ..dl.dataset import (
    SINIF_ISIMLERI,
    _indices_for_groups,
    _missing_class_names,
    _split_group_keys,
    _split_group_keys_kfold,
    _summarize_grouping,
    _augmentasyon_kopyasi_mi,
    _validate_dataset_separation,
)
from ..dl.utils import (
    plot_classification_summary,
    plot_confusion_matrix,
    plot_multiclass_roc_pr_curves,
    plot_prediction_confidence,
)
from .dataset import build_feature_matrix
from .features import feature_group_slices
from .xgb_classifier import build_xgb_classifier, save_xgb_model

SUPPORTED_SELECTION_METRICS = {"loss", "accuracy", "precision", "recall", "f1"}


# ==================== Custom XGBoost eval metric callable'lari ====================
#
# XGBoost custom_metric'i varsayilan olarak minimize edildigi icin maximize
# istenen metriklerden 1.0 - metric donduren "loss-like" callable'lar uretiriz.
# Module-level isimli fonksiyonlar; functools.partial XGBoost 3.2'de
# `__name__` aramasinda hata veriyor. Multiprocessing/HPO icin de pickle-safe.


def _xgb_predicted_labels(y_pred: np.ndarray) -> tuple[np.ndarray, list[int]]:
    arr = np.asarray(y_pred)
    if arr.ndim != 2:
        raise ValueError(
            f"Custom metric multiclass softprob bekliyor; aldi shape={arr.shape}"
        )
    return arr.argmax(axis=1), list(range(arr.shape[1]))


def _xgb_macro_f1_loss(y_true, y_pred):
    preds, labels = _xgb_predicted_labels(y_pred)
    return 1.0 - f1_score(
        y_true, preds, labels=labels, average="macro", zero_division=0,
    )


def _xgb_macro_precision_loss(y_true, y_pred):
    preds, labels = _xgb_predicted_labels(y_pred)
    return 1.0 - precision_score(
        y_true, preds, labels=labels, average="macro", zero_division=0,
    )


def _xgb_macro_recall_loss(y_true, y_pred):
    preds, labels = _xgb_predicted_labels(y_pred)
    return 1.0 - recall_score(
        y_true, preds, labels=labels, average="macro", zero_division=0,
    )


def _xgb_accuracy_loss(y_true, y_pred):
    preds, _labels = _xgb_predicted_labels(y_pred)
    return 1.0 - accuracy_score(y_true, preds)


_SELECTION_METRIC_CALLABLES: dict[str, Any] = {
    "f1": _xgb_macro_f1_loss,
    "precision": _xgb_macro_precision_loss,
    "recall": _xgb_macro_recall_loss,
    "accuracy": _xgb_accuracy_loss,
}


def _xgb_eval_metric_for_selection(selection_metric: str) -> tuple[Any, str]:
    """selection_metric -> (eval_metric, xgb_early_stopping_metric_label).

    ``loss`` icin tek metrik ("mlogloss") doner; diger metrikler icin
    [mlogloss, custom_loss] listesi doner. XGBoost early stopping listenin
    son metrigine gore karar verdigi icin custom metric daima son sirada.
    """
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Gecersiz selection metric: {selection_metric!r}. "
            f"Desteklenen: {sorted(SUPPORTED_SELECTION_METRICS)}"
        )
    if selection_metric == "loss":
        return "mlogloss", "mlogloss"
    callable_metric = _SELECTION_METRIC_CALLABLES[selection_metric]
    label = f"1 - macro_{selection_metric}" if selection_metric != "accuracy" else "1 - accuracy"
    return ["mlogloss", callable_metric], label


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
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    n_jobs: int | None = None


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
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"--xgb-device 'auto', 'cpu' veya 'cuda' olmali (alindi: {config.device!r})."
        )
    if config.n_jobs is not None and config.n_jobs <= 0:
        raise ValueError("--xgb-n-jobs pozitif tamsayi olmali.")
    if not full_trainval and not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali.")
    if config.test_ratio < 0.0 or config.test_ratio >= 1.0:
        raise ValueError("--test-ratio 0 ile 1 arasinda olmali.")

    trainval_dir, test_dir = resolve_sl_data_dirs(config)
    if not trainval_dir.exists():
        raise FileNotFoundError(f"TrainVal veri dizini bulunamadi: {trainval_dir}")
    if test_dir is not None and not test_dir.exists():
        raise FileNotFoundError(f"Test veri dizini bulunamadi: {test_dir}")
    has_external_test = bool(test_dir is not None and test_dir.resolve() != trainval_dir.resolve())
    if not full_trainval and not has_external_test and config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio 1'den kucuk olmali.")
    if not full_trainval and require_test_dir and not has_external_test and config.test_ratio <= 0.0:
        raise ValueError("Harici test dizini yoksa --test-ratio pozitif olmali.")
    if full_trainval and require_test_dir:
        if not has_external_test:
            raise ValueError(
                "Full-trainval final egitim icin harici test dizini gerekli."
            )


# ==================== Split ====================


def _augmented_flags_from_paths(paths: list[str]) -> list[bool]:
    """Ozellik matrisindeki path bilgisinden augment/turev kopyalari belirle."""
    return [_augmentasyon_kopyasi_mi(Path(path).name) for path in paths]


def _ensure_split_has_all_classes(y_values: np.ndarray, split_name: str) -> None:
    """Model raporlarinin anlamli kalmasi icin split sinif kapsamini dogrula."""
    missing = _missing_class_names(y_values.astype(int).tolist(), len(SINIF_ISIMLERI))
    if missing:
        raise RuntimeError(f"{split_name} split'inde eksik siniflar: {', '.join(missing)}")


def _split_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    paths: list[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_test: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    """Ozellik matrisini DL akisiyle uyumlu sekilde train/val/test olarak bol."""
    num_classes = len(SINIF_ISIMLERI)
    augmented_flags = _augmented_flags_from_paths(paths)
    group_stats = _summarize_grouping(groups, augmented_flags)

    use_group_split = group_stats["grouping_reliable"]
    split_warnings: list[str] = []
    if not use_group_split and group_stats["augmented_samples"] > 0:
        split_warnings.append(
            "TrainVal dosya adlarindan tekrarli kaynak grup cikarilamadi; "
            "stratified split kullanildi ve augment turevleri icin leak-free garanti verilemiyor."
        )
        warnings.warn(
            "Augmentasyon kopyalari tespit edildi ancak grup-bazli split yapilamadi. "
            "Stratified split kullanildi; augmentasyon kopyalari farkli split'lere "
            "dusebilir ve veri sizintisina yol acabilir.",
            stacklevel=2,
        )

    internal_test_ratio = test_ratio if include_test else 0.0
    train_group_keys, val_group_keys, test_group_keys, strategy = _split_group_keys(
        labels=y.astype(int).tolist(),
        groups=groups,
        val_ratio=val_ratio,
        test_ratio=internal_test_ratio,
        seed=seed,
        use_group_split=use_group_split,
    )

    train_idxs = _indices_for_groups(groups, augmented_flags, train_group_keys, original_only=False)
    val_idxs = _indices_for_groups(groups, augmented_flags, val_group_keys, original_only=True)
    test_idxs = _indices_for_groups(groups, augmented_flags, test_group_keys, original_only=True)

    if not train_idxs:
        raise RuntimeError("Train split olusturulamadi.")
    if not val_idxs:
        raise RuntimeError("Validation split olusturulamadi; original validation ornegi bulunamadi.")
    if include_test and not test_idxs:
        raise RuntimeError("Test split olusturulamadi; original test ornegi bulunamadi.")

    X_train, y_train = X[train_idxs], y[train_idxs]
    X_val, y_val = X[val_idxs], y[val_idxs]
    X_test = X[test_idxs] if include_test else None
    y_test = y[test_idxs] if include_test else None

    _ensure_split_has_all_classes(y_train, "Train")
    _ensure_split_has_all_classes(y_val, "Validation")
    if y_test is not None:
        _ensure_split_has_all_classes(y_test, "Test")

    info = {
        "num_classes": num_classes,
        "train_size": len(train_idxs),
        "val_size": len(val_idxs),
        "test_size": len(test_idxs) if include_test else 0,
        "train_groups": len({groups[i] for i in train_idxs}),
        "val_groups": len({groups[i] for i in val_idxs}),
        "test_groups": len({groups[i] for i in test_idxs}) if include_test else 0,
        "split_strategy": strategy,
        "split_warnings": split_warnings,
        "trainval_grouping": group_stats,
        "uses_external_test_dir": False,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, info


def _kfold_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    paths: list[str],
    *,
    n_folds: int,
    test_ratio: float,
    seed: int,
    include_test: bool,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    """Ozellik matrisini K fold'a (+ ortak test) bol.

    Returns:
        fold_splits: her fold icin (X_train, y_train, X_val, y_val) tuple'i
        X_test, y_test: include_test ise dahili test seti, aksi halde (None, None)
        info: split stratejisi ve grup ozetini iceren sozluk
    """
    num_classes = len(SINIF_ISIMLERI)
    augmented_flags = _augmented_flags_from_paths(paths)
    group_stats = _summarize_grouping(groups, augmented_flags)

    use_group_split = group_stats["grouping_reliable"]
    split_warnings: list[str] = []
    if not use_group_split and group_stats["augmented_samples"] > 0:
        split_warnings.append(
            "TrainVal dosya adlarindan tekrarli kaynak grup cikarilamadi; "
            "stratified K-fold kullanildi ve augment turevleri icin leak-free garanti verilemiyor."
        )
        warnings.warn(
            "Augmentasyon kopyalari tespit edildi ancak grup-bazli K-fold yapilamadi.",
            stacklevel=2,
        )

    internal_test_ratio = test_ratio if include_test else 0.0
    fold_assignments, internal_test_groups, strategy = _split_group_keys_kfold(
        labels=y.astype(int).tolist(),
        groups=groups,
        n_folds=n_folds,
        test_ratio=internal_test_ratio,
        seed=seed,
        use_group_split=use_group_split,
    )

    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_size = 0
    if include_test:
        test_idxs = _indices_for_groups(
            groups, augmented_flags, internal_test_groups, original_only=True
        )
        if not test_idxs:
            raise RuntimeError(
                "Test split olusturulamadi; original test ornegi bulunamadi."
            )
        X_test = X[test_idxs]
        y_test = y[test_idxs]
        _ensure_split_has_all_classes(y_test, "Test")
        test_size = len(test_idxs)

    fold_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fold_size_info: list[dict[str, int]] = []
    for fold_index, (train_groups_set, val_groups_set) in enumerate(fold_assignments):
        train_idxs = _indices_for_groups(
            groups, augmented_flags, train_groups_set, original_only=False
        )
        val_idxs = _indices_for_groups(
            groups, augmented_flags, val_groups_set, original_only=True
        )
        if not train_idxs:
            raise RuntimeError(f"Fold {fold_index} train split bos.")
        if not val_idxs:
            raise RuntimeError(
                f"Fold {fold_index} validation split olusturulamadi; original ornegi bulunamadi."
            )
        X_train_fold = X[train_idxs]
        y_train_fold = y[train_idxs]
        X_val_fold = X[val_idxs]
        y_val_fold = y[val_idxs]
        _ensure_split_has_all_classes(y_train_fold, f"Fold {fold_index} Train")
        _ensure_split_has_all_classes(y_val_fold, f"Fold {fold_index} Validation")
        fold_splits.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        fold_size_info.append(
            {
                "fold_index": fold_index,
                "train_size": int(len(train_idxs)),
                "val_size": int(len(val_idxs)),
                "train_groups": len({groups[i] for i in train_idxs}),
                "val_groups": len({groups[i] for i in val_idxs}),
            }
        )

    info = {
        "num_classes": num_classes,
        "n_folds": n_folds,
        "test_size": test_size,
        "test_groups": len(internal_test_groups) if include_test else 0,
        "split_strategy": strategy,
        "split_warnings": split_warnings,
        "trainval_grouping": group_stats,
        "uses_external_test_dir": False,
        "folds": fold_size_info,
    }
    return fold_splits, X_test, y_test, info


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
    metric_names_seen: set[str] = set()
    for label, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            metric_names_seen.add(metric_name)
            ax.plot(range(1, len(values) + 1), values, label=f"{label} {metric_name}", linewidth=2)

    if best_iteration is not None:
        # XGBoost best_iteration 0-bazli; eksen 1-bazli oldugu icin +1.
        best_round_1based = int(best_iteration) + 1
        ax.axvline(
            best_round_1based, color="gray", linestyle=":", alpha=0.7,
            label=f"Best iter={best_round_1based}",
        )

    ax.set_xlabel("Boosting Round")
    if metric_names_seen == {"mlogloss"}:
        ax.set_ylabel("Log Loss")
    else:
        ax.set_ylabel("Loss / Metric")
    ax.set_title("XGBoost Egitim Egrisi")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[OK] XGBoost egitim egrisi kaydedildi: {save_path}")


def _plot_xgb_feature_importance(
    model: Any,
    save_path: Path,
    *,
    image_size: int,
    max_features: int = 30,
) -> Path | None:
    """XGBoost feature importance grafigini kaydet."""
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
    if importances.size == 0:
        warnings.warn("XGBoost feature_importances_ bos; feature importance grafigi atlandi.")
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)

    top_n = min(max_features, importances.size)
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_values = importances[top_indices]
    top_labels = [f"f{idx}" for idx in top_indices]

    group_values: dict[str, float] = {}
    covered_until = 0
    for group_name, group_slice in feature_group_slices(image_size).items():
        start = min(group_slice.start, importances.size)
        stop = min(group_slice.stop, importances.size)
        if stop > start:
            group_values[group_name] = float(importances[start:stop].sum())
            covered_until = max(covered_until, stop)
    if covered_until < importances.size:
        group_values["Other"] = float(importances[covered_until:].sum())

    fig, (ax_top, ax_group) = plt.subplots(1, 2, figsize=(18, 8))

    y_pos = np.arange(top_n)
    ax_top.barh(y_pos, top_values, color="#4C78A8")
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_labels)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Importance")
    ax_top.set_title(f"Top {top_n} XGBoost Features")
    ax_top.grid(True, axis="x", alpha=0.3)

    sorted_groups = sorted(group_values.items(), key=lambda item: item[1], reverse=True)
    group_names = [name for name, _ in sorted_groups]
    group_scores = [score for _, score in sorted_groups]
    group_pos = np.arange(len(group_names))
    ax_group.barh(group_pos, group_scores, color="#F58518")
    ax_group.set_yticks(group_pos)
    ax_group.set_yticklabels(group_names)
    ax_group.invert_yaxis()
    ax_group.set_xlabel("Total Importance")
    ax_group.set_title("Feature Group Importance")
    ax_group.grid(True, axis="x", alpha=0.3)

    fig.suptitle("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] XGBoost feature importance kaydedildi: {save_path}")
    return save_path


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
    preloaded_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """XGBoost ile tek bir egitim deneyimi calistir.

    ``preloaded_split`` parametresi K-fold CV orkestratoru icin kullanilir;
    verildiginde ozellik cikartimi ve dahili split adimlari atlanir, splitler
    dogrudan dictionary'den okunur. Beklenen anahtarlar:
    X_train, y_train, X_val, y_val, X_test (opsiyonel), y_test (opsiyonel),
    split_info, test_grouping (opsiyonel), trainval_dir, test_dir.
    """

    validate_sl_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=full_trainval,
    )

    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Gecersiz selection_metric: {selection_metric!r}. "
            f"Desteklenen: {sorted(SUPPORTED_SELECTION_METRICS)}"
        )

    np.random.seed(config.seed)

    if preloaded_split is not None:
        if full_trainval:
            raise ValueError("preloaded_split ile full_trainval birlikte kullanilamaz.")
        trainval_dir = Path(preloaded_split.get("trainval_dir", "."))
        test_dir = preloaded_split.get("test_dir")
        if isinstance(test_dir, str):
            test_dir = Path(test_dir)
        using_external_test = bool(
            evaluate_test_set
            and test_dir is not None
            and Path(test_dir).resolve() != trainval_dir.resolve()
        )
        X_train = preloaded_split["X_train"]
        y_train = preloaded_split["y_train"]
        X_val = preloaded_split.get("X_val")
        y_val = preloaded_split.get("y_val")
        X_test = preloaded_split.get("X_test")
        y_test = preloaded_split.get("y_test")
        split_info = dict(preloaded_split.get("split_info", {}))
        test_grouping = preloaded_split.get("test_grouping")

        if verbose:
            print("\n[INFO] XGBoost egitimi basliyor (CV fold; preloaded split)")
            print(f"  Veri dizini     : {trainval_dir}")
            print(f"  Test dizini     : {test_dir}")
    else:
        trainval_dir, test_dir = resolve_sl_data_dirs(config)
        if not evaluate_test_set:
            test_dir = None
        using_external_test = bool(
            evaluate_test_set
            and test_dir is not None
            and test_dir.resolve() != trainval_dir.resolve()
        )

        if verbose:
            print("\n[INFO] XGBoost egitimi basliyor")
            print(f"  Veri dizini     : {trainval_dir}")
            print(f"  Test dizini     : {test_dir}")

        tv_cache = Path(config.feature_cache) / f"trainval_img{config.image_size}.npz" if config.feature_cache else None
        X_tv, y_tv, groups_tv, paths_tv = build_feature_matrix(
            trainval_dir, image_size=config.image_size, cache_path=tv_cache,
        )

        X_test = None
        y_test = None
        test_grouping = None
        if using_external_test and test_dir is not None:
            te_cache = Path(config.feature_cache) / f"test_img{config.image_size}.npz" if config.feature_cache else None
            X_test_all, y_test_all, groups_te, paths_te = build_feature_matrix(
                test_dir, image_size=config.image_size, cache_path=te_cache,
            )
            _validate_dataset_separation(groups_tv, groups_te, trainval_dir, test_dir)
            aug_flags_te = _augmented_flags_from_paths(paths_te)
            test_grouping = _summarize_grouping(groups_te, aug_flags_te)
            external_test_idxs = _indices_for_groups(
                groups_te,
                aug_flags_te,
                set(groups_te),
                original_only=True,
            )
            if not external_test_idxs:
                raise RuntimeError(
                    "Harici test dizininde original goruntu bulunamadi; test split olusturulamiyor."
                )
            X_test = X_test_all[external_test_idxs]
            y_test = y_test_all[external_test_idxs]
            _ensure_split_has_all_classes(y_test, "Test")

        X_train = None
        y_train = None
        X_val = None
        y_val = None
        split_info = {}

    num_classes = len(SINIF_ISIMLERI)

    # Split
    if preloaded_split is not None:
        # Splitler hazir; dahili split atlanir.
        pass
    elif full_trainval:
        X_train, y_train = X_tv, y_tv
        X_val, y_val = None, None
        _ensure_split_has_all_classes(y_train, "Train")
        aug_flags_tv = _augmented_flags_from_paths(paths_tv)
        tv_grouping = _summarize_grouping(groups_tv, aug_flags_tv)
        split_info = {
            "num_classes": num_classes,
            "train_size": len(y_tv),
            "val_size": 0,
            "test_size": len(y_test) if y_test is not None else 0,
            "train_groups": tv_grouping["unique_groups"],
            "val_groups": 0,
            "test_groups": test_grouping["unique_groups"] if test_grouping is not None else 0,
            "split_strategy": "full_trainval_external_test",
            "split_warnings": [],
            "trainval_grouping": tv_grouping,
            "uses_external_test_dir": using_external_test,
        }
    else:
        include_internal_test = evaluate_test_set and not using_external_test
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test_internal,
            y_test_internal,
            split_info,
        ) = _split_feature_matrix(
            X_tv,
            y_tv,
            groups_tv,
            paths_tv,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
            include_test=include_internal_test,
        )
        if include_internal_test:
            X_test = X_test_internal
            y_test = y_test_internal
        if using_external_test:
            split_info["uses_external_test_dir"] = True

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
    eval_metric, xgb_early_stopping_metric = _xgb_eval_metric_for_selection(selection_metric)
    model = build_xgb_classifier(
        num_classes,
        xgb_params,
        eval_metric=eval_metric,
        device=config.device,
        n_jobs=config.n_jobs,
    )

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

    history: dict[str, list[float]] = {}
    for set_name, set_metrics in evals_result.items():
        for metric_name, values in set_metrics.items():
            history[f"{set_name}_{metric_name}"] = [float(v) for v in values]

    # Val metrikleri — predict/predict_proba erken durdurma sonrasi best_iteration
    # kullanir; train_loss da ayni iteration'a karsilik gelsin diye predict_proba
    # uzerinden log_loss ile hesaplanir.
    best_val_metrics: dict[str, float] | None = None
    best_val_detailed: dict[str, Any] | None = None
    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, labels=list(range(num_classes)), zero_division=0, average="macro",
        )
        val_loss = float(log_loss(y_val, val_probs, labels=list(range(num_classes))))
        best_val_metrics = {
            "loss": val_loss,
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
    train_loss = float(log_loss(y_train, train_probs, labels=list(range(num_classes))))
    best_train_metrics = {
        "loss": train_loss,
        "accuracy": train_acc,
        "precision": float(t_prec),
        "recall": float(t_rec),
        "f1": float(t_f1),
    }

    # Artifact kaydi
    artifact_stem = artifact_tag or "xgboost"
    output_dirs = None
    checkpoint_path = None
    feature_importance_path: Path | None = None
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
    if output_dirs is not None:
        feature_importance_path = _plot_xgb_feature_importance(
            model,
            output_dirs["visuals"] / f"feature_importance_{artifact_stem}.png",
            image_size=config.image_size,
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

        selection_mode = "fixed_epoch_full_trainval" if full_trainval else _selection_mode_for_metric(selection_metric)

        report: dict[str, Any] = {
            "model": "xgboost",
            "timestamp": timestamp,
            "n_estimators_trained": (
                int(best_iteration) + 1 if best_iteration is not None else config.n_estimators
            ),
            "best_iteration": best_iteration,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "xgb_early_stopping_metric": xgb_early_stopping_metric,
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
                "feature_importance": str(feature_importance_path) if feature_importance_path else None,
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
        "xgb_early_stopping_metric": xgb_early_stopping_metric,
        "test_metrics": test_metrics,
        "history": history,
        "data_info": {
            **split_info,
            "test_size": test_size,
            "test_grouping": test_grouping,
        },
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
        "feature_importance_path": feature_importance_path,
        "output_root": output_dirs["root"] if output_dirs else None,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": sl_config_to_dict(config),
        "best_val_detailed_metrics": best_val_detailed,
        "test_detailed_metrics": test_detailed,
        "full_trainval": full_trainval,
    }


_SL_CV_METRIC_KEYS = ("loss", "accuracy", "precision", "recall", "f1")


def _aggregate_sl_metric_dicts(
    metrics_list: list[dict[str, float] | None],
) -> dict[str, dict[str, Any]] | None:
    """Per-fold skaler metrik sozluklerinden mean/std/values uret."""
    cleaned = [m for m in metrics_list if m is not None]
    if not cleaned:
        return None
    aggregate: dict[str, dict[str, Any]] = {}
    for key in _SL_CV_METRIC_KEYS:
        values = [float(m[key]) for m in cleaned if key in m and m[key] is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in values],
        }
    return aggregate


def _aggregate_sl_cv_metrics(
    fold_results: list[dict[str, Any]],
    selection_metric: str,
) -> dict[str, Any]:
    val_metrics = [r.get("best_val_metrics") for r in fold_results]
    test_metrics = [r.get("test_metrics") for r in fold_results]
    selection_values = [
        float(r["best_selection_value"])
        for r in fold_results
        if r.get("best_selection_value") is not None
    ]
    selection_summary = None
    if selection_values:
        arr = np.asarray(selection_values, dtype=np.float64)
        selection_summary = {
            "metric": selection_metric,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "values": [float(v) for v in selection_values],
        }
    return {
        "val": _aggregate_sl_metric_dicts(val_metrics),
        "test": _aggregate_sl_metric_dicts(test_metrics),
        "selection": selection_summary,
        "completed_folds": len(fold_results),
    }


def run_sl_cv_training(
    config: SLTrainingConfig,
    *,
    n_folds: int,
    output_root: Path | None = None,
    artifact_tag: str | None = None,
    save_artifacts: bool = True,
    evaluate_test_set: bool = True,
    verbose: bool = True,
    selection_metric: str = "f1",
    on_fold_end: Any = None,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """K-fold cross-validation training (SL/XGBoost).

    Ozellik matrisi tek seferlik cikarilir; trainval seti grup-bazli K parcaya
    bolunup her fold icin bagimsiz XGBoost egitilir. Test seti (varsa) tum
    fold'lar arasinda paylasilir. Per-fold raporlar ve aggregate ``cv_summary``
    yazilir.
    """
    if n_folds < 2:
        raise ValueError(f"--folds en az 2 olmali (verilen: {n_folds}).")

    validate_sl_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=False,
    )

    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Gecersiz selection_metric: {selection_metric!r}. "
            f"Desteklenen: {sorted(SUPPORTED_SELECTION_METRICS)}"
        )
    _, xgb_early_stopping_metric = _xgb_eval_metric_for_selection(selection_metric)

    np.random.seed(config.seed)
    trainval_dir, test_dir = resolve_sl_data_dirs(config)
    if not evaluate_test_set:
        test_dir = None
    using_external_test = bool(
        evaluate_test_set
        and test_dir is not None
        and test_dir.resolve() != trainval_dir.resolve()
    )

    base_tag = artifact_tag or "xgboost"
    cv_root_path = Path(output_root) if output_root is not None else CIKTI_KLASORU
    cv_dirs: dict[str, Path] | None = None
    if save_artifacts:
        cv_dirs = _build_output_dirs(cv_root_path)

    if verbose:
        print(f"\n[INFO] SL/XGBoost {n_folds}-fold cross validation basliyor")
        print(f"  Veri dizini     : {trainval_dir}")
        print(f"  Test dizini     : {test_dir}")
        print(f"  Cikti kok dizin : {cv_root_path}")

    # Ozellik cikartimi (tek seferlik, cv boyunca paylasilir)
    tv_cache = (
        Path(config.feature_cache) / f"trainval_img{config.image_size}.npz"
        if config.feature_cache
        else None
    )
    X_tv, y_tv, groups_tv, paths_tv = build_feature_matrix(
        trainval_dir, image_size=config.image_size, cache_path=tv_cache,
    )

    X_test_shared: np.ndarray | None = None
    y_test_shared: np.ndarray | None = None
    test_grouping: dict[str, Any] | None = None
    if using_external_test and test_dir is not None:
        te_cache = (
            Path(config.feature_cache) / f"test_img{config.image_size}.npz"
            if config.feature_cache
            else None
        )
        X_test_all, y_test_all, groups_te, paths_te = build_feature_matrix(
            test_dir, image_size=config.image_size, cache_path=te_cache,
        )
        _validate_dataset_separation(groups_tv, groups_te, trainval_dir, test_dir)
        aug_flags_te = _augmented_flags_from_paths(paths_te)
        test_grouping = _summarize_grouping(groups_te, aug_flags_te)
        external_test_idxs = _indices_for_groups(
            groups_te, aug_flags_te, set(groups_te), original_only=True,
        )
        if not external_test_idxs:
            raise RuntimeError(
                "Harici test dizininde original goruntu bulunamadi; test split olusturulamiyor."
            )
        X_test_shared = X_test_all[external_test_idxs]
        y_test_shared = y_test_all[external_test_idxs]
        _ensure_split_has_all_classes(y_test_shared, "Test")

    include_internal_test = evaluate_test_set and not using_external_test
    fold_splits, X_test_internal, y_test_internal, kfold_info = _kfold_feature_matrix(
        X_tv,
        y_tv,
        groups_tv,
        paths_tv,
        n_folds=n_folds,
        test_ratio=config.test_ratio,
        seed=config.seed,
        include_test=include_internal_test,
    )
    if include_internal_test:
        X_test_shared = X_test_internal
        y_test_shared = y_test_internal

    fold_results: list[dict[str, Any]] = []
    for fold_index, ((X_tr, y_tr, X_va, y_va), fold_size) in enumerate(
        zip(fold_splits, kfold_info["folds"])
    ):
        fold_tag = f"{base_tag}_fold{fold_index:02d}"
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# SL FOLD {fold_index + 1}/{n_folds}")
            print(f"{'#' * 70}\n")

        fold_output_root: Path | None = None
        if save_artifacts:
            fold_output_root = cv_root_path / "folds" / f"fold_{fold_index:02d}"
            fold_output_root.mkdir(parents=True, exist_ok=True)

        # test_groups: harici test varsa harici grouping'ten, yoksa K-fold'dan
        # ayrilan internal test grup sayisindan al. Eski surumde her iki halde
        # de test_grouping'e bakildigi icin internal test'te 0 raporlaniyordu.
        if using_external_test and test_grouping is not None:
            test_groups_count = test_grouping["unique_groups"]
        else:
            test_groups_count = kfold_info.get("test_groups", 0)

        fold_split_info = {
            "num_classes": kfold_info["num_classes"],
            "train_size": fold_size["train_size"],
            "val_size": fold_size["val_size"],
            "test_size": kfold_info["test_size"],
            "train_groups": fold_size["train_groups"],
            "val_groups": fold_size["val_groups"],
            "test_groups": test_groups_count,
            "split_strategy": kfold_info["split_strategy"],
            "split_warnings": list(kfold_info["split_warnings"]),
            "trainval_grouping": kfold_info["trainval_grouping"],
            "uses_external_test_dir": using_external_test,
            "fold_index": fold_index,
            "n_folds": n_folds,
        }

        fold_extra = {"fold_index": fold_index, "n_folds": n_folds}
        if extra_report:
            fold_extra.update(extra_report)

        preloaded = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_va,
            "y_val": y_va,
            "X_test": X_test_shared,
            "y_test": y_test_shared,
            "split_info": fold_split_info,
            "test_grouping": test_grouping,
            "trainval_dir": trainval_dir,
            "test_dir": test_dir,
        }

        fold_result = run_sl_training(
            config,
            output_root=fold_output_root,
            artifact_tag=fold_tag,
            save_artifacts=save_artifacts,
            evaluate_test_set=evaluate_test_set,
            full_trainval=False,
            verbose=verbose,
            selection_metric=selection_metric,
            extra_report=fold_extra,
            preloaded_split=preloaded,
        )
        fold_summary = {"fold_index": fold_index, **fold_result}
        fold_results.append(fold_summary)
        if on_fold_end is not None:
            on_fold_end(fold_index, fold_summary)

    aggregate = _aggregate_sl_cv_metrics(fold_results, selection_metric)

    cv_summary_path: Path | None = None
    if save_artifacts and cv_dirs is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv_summary_path = cv_dirs["reports"] / f"cv_summary_{base_tag}_{timestamp}.json"
        cv_summary = {
            "model": "xgboost",
            "timestamp": timestamp,
            "n_folds": n_folds,
            "selection_metric": selection_metric,
            "selection_mode": _selection_mode_for_metric(selection_metric),
            "xgb_early_stopping_metric": xgb_early_stopping_metric,
            "aggregate": aggregate,
            "folds": [
                {
                    "fold_index": r["fold_index"],
                    "best_iteration": r.get("best_iteration"),
                    "best_val_metrics": r.get("best_val_metrics"),
                    "best_selection_value": r.get("best_selection_value"),
                    "test_metrics": r.get("test_metrics"),
                    "report_path": str(r.get("report_path")) if r.get("report_path") else None,
                    "checkpoint_path": (
                        str(r.get("checkpoint_path")) if r.get("checkpoint_path") else None
                    ),
                }
                for r in fold_results
            ],
            "config": sl_config_to_dict(config),
            "trainval_dir": str(trainval_dir),
            "test_dir": str(test_dir) if test_dir else None,
            "extra": extra_report or {},
        }
        with open(cv_summary_path, "w", encoding="utf-8") as fh:
            json.dump(cv_summary, fh, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"\n[OK] SL CV ozeti kaydedildi: {cv_summary_path}")

    if verbose and aggregate.get("val"):
        val_summary = aggregate["val"]
        print(f"\n{'=' * 70}")
        print(f"SL CV TAMAMLANDI ({n_folds} fold)")
        print(f"{'=' * 70}")
        for metric_name in ("loss", "accuracy", "f1"):
            if metric_name in val_summary:
                stat = val_summary[metric_name]
                print(f"  Val {metric_name:<9}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
        if aggregate.get("test"):
            test_summary = aggregate["test"]
            for metric_name in ("accuracy", "f1"):
                if metric_name in test_summary:
                    stat = test_summary[metric_name]
                    print(f"  Test {metric_name:<8}: {stat['mean']:.4f} +/- {stat['std']:.4f}")

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "cv_summary_path": cv_summary_path,
        "selection_metric": selection_metric,
        "selection_mode": _selection_mode_for_metric(selection_metric),
        "xgb_early_stopping_metric": xgb_early_stopping_metric,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": sl_config_to_dict(config),
        "output_root": cv_root_path if save_artifacts else None,
    }
