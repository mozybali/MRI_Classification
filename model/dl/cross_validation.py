#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cross_validation.py  (DL / ResNet)
----------------------------------
DL training_runner icin K-fold cross-validation orkestratoru.

`run_cv_training` trainval setini grup-bazli StratifiedKFold ile K parcaya
bolup her fold icin `run_training` cagrir; per-fold raporlari ile aggregate
``cv_summary`` JSON'unu yazar.

Modul `model.training_runner`'i bir modul olarak iceri alir (`_tr`) ve gerekli
sembolleri `_tr.X` uzerinden cozer. Bu, monkeypatch'lerin (testlerde
`training_runner.run_training` vs.) buradan yapilan cagrilara da yansimasini
saglar ve dairesel import probleminden kacinir.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import training_runner as _tr

_CV_METRIC_KEYS = ("loss", "accuracy", "precision", "recall", "f1")


def _aggregate_metric_dicts(
    metrics_list: list[dict[str, float] | None],
) -> dict[str, dict[str, Any]] | None:
    """Per-fold skaler metrik sozluklerinden mean/std/values uret."""
    cleaned = [m for m in metrics_list if m is not None]
    if not cleaned:
        return None
    aggregate: dict[str, dict[str, Any]] = {}
    for key in _CV_METRIC_KEYS:
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


def _aggregate_cv_metrics(
    fold_results: list[dict[str, Any]],
    selection_metric: str,
) -> dict[str, Any]:
    """Tum fold sonuclarindan val/test ortalama+std + selection ozeti uret."""
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
        "val": _aggregate_metric_dicts(val_metrics),
        "test": _aggregate_metric_dicts(test_metrics),
        "selection": selection_summary,
        "completed_folds": len(fold_results),
    }


def run_cv_training(
    config: "_tr.TrainingConfig",
    *,
    n_folds: int,
    output_root: Path | None = None,
    artifact_tag: str | None = None,
    save_artifacts: bool = True,
    evaluate_test_set: bool = True,
    verbose: bool = True,
    selection_metric: str = "loss",
    on_fold_end: Callable[[int, dict[str, Any]], None] | None = None,
    extra_report: dict[str, Any] | None = None,
    deterministic: bool = True,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """K-fold cross-validation training (DL/ResNet).

    Trainval setini grup-bazli StratifiedKFold ile K parcaya bolup her fold icin
    bagimsiz egitim calistirir; test seti (varsa) tum fold'lar arasinda paylasilir
    ve her fold'un en iyi modeliyle ayrica degerlendirilir. Per-fold raporlar
    ``output_root/folds/fold_NN/`` altina, aggregate ise ``cv_summary_<tag>.json``
    olarak yazilir.
    """
    if n_folds < 2:
        raise ValueError(f"--folds en az 2 olmali (verilen: {n_folds}).")

    _tr.validate_training_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=False,
    )

    trainval_dir, test_dir = _tr.resolve_data_dirs(config)
    if not evaluate_test_set:
        test_dir = None

    use_dataset_stats = not config.pretrained
    base_tag = artifact_tag or config.model

    cv_root_path = Path(output_root) if output_root is not None else _tr.CIKTI_KLASORU
    cv_dirs: dict[str, Path] | None = None
    if save_artifacts:
        cv_dirs = _tr._build_output_dirs(cv_root_path)

    fold_results: list[dict[str, Any]] = []
    if verbose:
        print(f"\n[INFO] {n_folds}-fold cross validation basliyor")
        print(f"  Veri dizini     : {trainval_dir}")
        print(f"  Test dizini     : {test_dir}")
        print(f"  Cikti kok dizin : {cv_root_path}")

    for fold_index, train_loader, val_loader, test_loader, fold_info in _tr.iter_kfold_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=test_dir,
        n_folds=n_folds,
        batch_size=config.batch_size,
        image_size=config.image_size,
        test_ratio=config.test_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        include_test=evaluate_test_set,
        hflip_p=config.hflip_p,
        rotation_degrees=config.rotation_degrees,
        color_jitter=config.color_jitter,
        use_dataset_stats=use_dataset_stats,
    ):
        fold_tag = f"{base_tag}_fold{fold_index:02d}"
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# FOLD {fold_index + 1}/{n_folds}")
            print(f"{'#' * 70}\n")

        fold_output_root: Path | None = None
        if save_artifacts:
            fold_output_root = cv_root_path / "folds" / f"fold_{fold_index:02d}"
            fold_output_root.mkdir(parents=True, exist_ok=True)

        fold_extra = {"fold_index": fold_index, "n_folds": n_folds}
        if extra_report:
            fold_extra.update(extra_report)

        fold_result = _tr.run_training(
            config,
            output_root=fold_output_root,
            artifact_tag=fold_tag,
            save_artifacts=save_artifacts,
            evaluate_test_set=evaluate_test_set,
            full_trainval=False,
            verbose=verbose,
            selection_metric=selection_metric,
            extra_report=fold_extra,
            preloaded_loaders=(train_loader, val_loader, test_loader, fold_info),
            deterministic=deterministic,
            use_amp=use_amp,
        )
        fold_result_summary = {
            "fold_index": fold_index,
            **fold_result,
        }
        fold_results.append(fold_result_summary)
        if on_fold_end is not None:
            on_fold_end(fold_index, fold_result_summary)

        # Fold sonu: bir sonraki fold yeni model + optimizer + DataLoader'lari
        # allocate etmeden once PyTorch caching allocator'in tutu bloklarini
        # geri ver. Ayni loop iterasyonundaki loader referanslari da bu noktada
        # serbest birakilir; aksi halde fragmente VRAM bir sonraki fold'a sarkar.
        del train_loader, val_loader, test_loader, fold_info, fold_result
        _tr.release_cuda_memory()

    aggregate = _aggregate_cv_metrics(fold_results, selection_metric)

    cv_summary_path: Path | None = None
    if save_artifacts and cv_dirs is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv_summary_path = cv_dirs["reports"] / f"cv_summary_{base_tag}_{timestamp}.json"
        cv_summary = {
            "model": config.model,
            "timestamp": timestamp,
            "n_folds": n_folds,
            "selection_metric": selection_metric,
            "aggregate": aggregate,
            "folds": [
                {
                    "fold_index": r["fold_index"],
                    "best_epoch": r.get("best_epoch"),
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
            "config": _tr.config_to_dict(config),
            "trainval_dir": str(trainval_dir),
            "test_dir": str(test_dir) if test_dir is not None else None,
            "extra": extra_report or {},
        }
        with open(cv_summary_path, "w", encoding="utf-8") as file:
            json.dump(cv_summary, file, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"\n[OK] CV ozeti kaydedildi: {cv_summary_path}")

    if verbose and aggregate.get("val"):
        val_summary = aggregate["val"]
        print(f"\n{'=' * 70}")
        print(f"CV TAMAMLANDI ({n_folds} fold)")
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
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": _tr.config_to_dict(config),
        "output_root": cv_root_path if save_artifacts else None,
    }
