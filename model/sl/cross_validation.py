#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cross_validation.py  (SL / XGBoost)
-----------------------------------
Sig ogrenme (XGBoost) training_runner icin K-fold cross-validation
orkestratoru ve fold ozellik matrisi yardimcisi.

`run_sl_cv_training` ozellik matrisini bir kez cikartip trainval setini
grup-bazli K parcaya bolerek her fold icin `run_sl_training` cagrir; test
seti (varsa) tum fold'lar arasinda paylasilir. `_kfold_feature_matrix` bu
bolme adimini tek basina disariya verir.

Modul `model.sl.training_runner`'i bir modul olarak iceri alir (`_tr`) ve
gerekli sembolleri `_tr.X` uzerinden cozer. Bu, monkeypatch'lerin (testlerde
`sl_runner._kfold_feature_matrix`, `build_feature_matrix` vs.) buradan
yapilan cagrilara da yansimasini saglar ve dairesel import probleminden
kacinir.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import training_runner as _tr

_SL_CV_METRIC_KEYS = ("loss", "accuracy", "precision", "recall", "f1")


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
    num_classes = len(_tr.SINIF_ISIMLERI)
    augmented_flags = _tr._augmented_flags_from_paths(paths)
    group_stats = _tr._summarize_grouping(groups, augmented_flags)

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
    fold_assignments, internal_test_groups, strategy = _tr._split_group_keys_kfold(
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
        test_idxs = _tr._indices_for_groups(
            groups, augmented_flags, internal_test_groups, original_only=True
        )
        if not test_idxs:
            raise RuntimeError(
                "Test split olusturulamadi; original test ornegi bulunamadi."
            )
        X_test = X[test_idxs]
        y_test = y[test_idxs]
        _tr._ensure_split_has_all_classes(y_test, "Test")
        test_size = len(test_idxs)

    fold_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fold_size_info: list[dict[str, int]] = []
    for fold_index, (train_groups_set, val_groups_set) in enumerate(fold_assignments):
        train_idxs = _tr._indices_for_groups(
            groups, augmented_flags, train_groups_set, original_only=False
        )
        val_idxs = _tr._indices_for_groups(
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
        _tr._ensure_split_has_all_classes(y_train_fold, f"Fold {fold_index} Train")
        _tr._ensure_split_has_all_classes(y_val_fold, f"Fold {fold_index} Validation")
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
    config: "_tr.SLTrainingConfig",
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

    _tr.validate_sl_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=False,
    )

    if selection_metric not in _tr.SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Gecersiz selection_metric: {selection_metric!r}. "
            f"Desteklenen: {sorted(_tr.SUPPORTED_SELECTION_METRICS)}"
        )
    _, xgb_early_stopping_metric = _tr._xgb_eval_metric_for_selection(selection_metric)

    np.random.seed(config.seed)
    trainval_dir, test_dir = _tr.resolve_sl_data_dirs(config)
    if not evaluate_test_set:
        test_dir = None
    using_external_test = bool(
        evaluate_test_set
        and test_dir is not None
        and test_dir.resolve() != trainval_dir.resolve()
    )

    base_tag = artifact_tag or "xgboost"
    cv_root_path = Path(output_root) if output_root is not None else _tr.CIKTI_KLASORU
    cv_dirs: dict[str, Path] | None = None
    if save_artifacts:
        cv_dirs = _tr._build_output_dirs(cv_root_path)

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
    X_tv, y_tv, groups_tv, paths_tv = _tr.build_feature_matrix(
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
        X_test_all, y_test_all, groups_te, paths_te = _tr.build_feature_matrix(
            test_dir, image_size=config.image_size, cache_path=te_cache,
        )
        _tr._validate_dataset_separation(groups_tv, groups_te, trainval_dir, test_dir)
        aug_flags_te = _tr._augmented_flags_from_paths(paths_te)
        test_grouping = _tr._summarize_grouping(groups_te, aug_flags_te)
        external_test_idxs = _tr._indices_for_groups(
            groups_te, aug_flags_te, set(groups_te), original_only=True,
        )
        if not external_test_idxs:
            raise RuntimeError(
                "Harici test dizininde original goruntu bulunamadi; test split olusturulamiyor."
            )
        X_test_shared = X_test_all[external_test_idxs]
        y_test_shared = y_test_all[external_test_idxs]
        _tr._ensure_split_has_all_classes(y_test_shared, "Test")

    include_internal_test = evaluate_test_set and not using_external_test
    fold_splits, X_test_internal, y_test_internal, kfold_info = _tr._kfold_feature_matrix(
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
    for fold_index, ((X_tr_fold, y_tr_fold, X_va, y_va), fold_size) in enumerate(
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
            "X_train": X_tr_fold,
            "y_train": y_tr_fold,
            "X_val": X_va,
            "y_val": y_va,
            "X_test": X_test_shared,
            "y_test": y_test_shared,
            "split_info": fold_split_info,
            "test_grouping": test_grouping,
            "trainval_dir": trainval_dir,
            "test_dir": test_dir,
        }

        fold_result = _tr.run_sl_training(
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
            "selection_mode": _tr._selection_mode_for_metric(selection_metric),
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
            "config": _tr.sl_config_to_dict(config),
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
        "selection_mode": _tr._selection_mode_for_metric(selection_metric),
        "xgb_early_stopping_metric": xgb_early_stopping_metric,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": _tr.sl_config_to_dict(config),
        "output_root": cv_root_path if save_artifacts else None,
    }
