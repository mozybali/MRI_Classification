#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_dl.py
---------
ResNet / DL modeli icin Optuna TPE tabanli hiperparametre arama bilesenleri.

Bu modul yalnizca DL akisina ozgu mantigi icerir (search space, sampler,
trial objective, final egitim). Ortak CLI/dispatch akisi icin bk. ``hpo.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.training_runner import (
        TrainingConfig,
        run_cv_training,
        run_training,
        validate_training_config,
    )
    from model.dl.utils import release_cuda_memory
else:
    from .training_runner import (
        TrainingConfig,
        run_cv_training,
        run_training,
        validate_training_config,
    )
    from .dl.utils import release_cuda_memory

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


def _search_space_summary_dl(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size_choices": sorted(set(args.batch_size_choices)),
        "image_size_choices": sorted(set(args.image_size_choices)),
        "lr_range": [args.lr_min, args.lr_max],
        "weight_decay_range": [args.weight_decay_min, args.weight_decay_max],
        "scheduler_factor_range": [args.scheduler_factor_min, args.scheduler_factor_max],
        "scheduler_patience_range": [
            args.scheduler_patience_min,
            args.scheduler_patience_max,
        ],
        "loss_choices": list(dict.fromkeys(args.loss_choices)),
        "focal_gamma_range": [args.focal_gamma_min, args.focal_gamma_max],
        "dropout_range": [args.dropout_min, args.dropout_max],
        "label_smoothing_range": [args.label_smoothing_min, args.label_smoothing_max],
        "hflip_p_choices": sorted(set(args.hflip_p_choices)),
        "rotation_degrees_range": [args.rotation_degrees_min, args.rotation_degrees_max],
        "color_jitter_range": [args.color_jitter_min, args.color_jitter_max],
        "search_pretrained": bool(args.search_pretrained and args.model == "resnet"),
    }


def _build_config_from_args(
    args: argparse.Namespace,
    *,
    batch_size: int = 32,
    image_size: int = 224,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    loss: str = "ce",
    focal_gamma: float = 2.0,
    pretrained: bool = False,
    epochs: int | None = None,
    dropout: float = 0.5,
    label_smoothing: float = 0.0,
    hflip_p: float = 0.0,
    rotation_degrees: float = 10.0,
    color_jitter: float = 0.1,
) -> TrainingConfig:
    """HPO argumanlari ve trial/final parametrelerinden TrainingConfig olustur."""
    return TrainingConfig(
        model=args.model,
        epochs=epochs if epochs is not None else args.epochs,
        batch_size=batch_size,
        lr=lr,
        patience=args.patience,
        image_size=image_size,
        trainval_dir=args.trainval_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        loss=loss,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=pretrained,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        focal_gamma=focal_gamma,
        dropout=dropout,
        label_smoothing=label_smoothing,
        hflip_p=hflip_p,
        rotation_degrees=rotation_degrees,
        color_jitter=color_jitter,
    )


def _validate_dl_args(args: argparse.Namespace) -> None:
    if not args.batch_size_choices:
        raise ValueError("--batch-size-choices bos olamaz.")
    if args.lr_min <= 0 or args.lr_max <= 0 or args.lr_min >= args.lr_max:
        raise ValueError("lr araligi pozitif olmali ve min < max olmali.")
    if (
        args.weight_decay_min < 0
        or args.weight_decay_max < 0
        or args.weight_decay_min >= args.weight_decay_max
    ):
        raise ValueError("weight decay araligi gecersiz.")
    if (
        args.scheduler_factor_min <= 0
        or args.scheduler_factor_max >= 1
        or args.scheduler_factor_min >= args.scheduler_factor_max
    ):
        raise ValueError("scheduler factor araligi 0 ile 1 arasinda olmali ve min < max olmali.")
    if args.scheduler_patience_min < 1 or args.scheduler_patience_min > args.scheduler_patience_max:
        raise ValueError("scheduler patience araligi gecersiz.")
    if args.focal_gamma_min < 0 or args.focal_gamma_min > args.focal_gamma_max:
        raise ValueError("focal gamma araligi gecersiz.")
    if (
        args.dropout_min < 0
        or args.dropout_max >= 1.0
        or args.dropout_min > args.dropout_max
    ):
        raise ValueError("dropout araligi 0 ile 1 arasinda olmali ve min <= max olmali.")
    if (
        args.label_smoothing_min < 0
        or args.label_smoothing_max >= 1.0
        or args.label_smoothing_min > args.label_smoothing_max
    ):
        raise ValueError("label smoothing araligi 0 ile 1 arasinda olmali ve min <= max olmali.")
    if not args.hflip_p_choices:
        raise ValueError("--hflip-p-choices bos olamaz.")
    if any(p < 0 or p > 1 for p in args.hflip_p_choices):
        raise ValueError("hflip olasiliklari 0 ile 1 arasinda olmali.")
    if (
        args.rotation_degrees_min < 0
        or args.rotation_degrees_min > args.rotation_degrees_max
    ):
        raise ValueError("rotation degrees araligi gecersiz.")
    if (
        args.color_jitter_min < 0
        or args.color_jitter_min > args.color_jitter_max
    ):
        raise ValueError("color jitter araligi gecersiz.")
    if not args.loss_choices:
        raise ValueError("--loss-choices bos olamaz.")

    base_config = _build_config_from_args(
        args,
        batch_size=min(args.batch_size_choices),
        image_size=min(args.image_size_choices),
        lr=args.lr_min,
        weight_decay=args.weight_decay_min,
        scheduler_factor=args.scheduler_factor_min,
        scheduler_patience=args.scheduler_patience_min,
        loss=args.loss_choices[0],
        focal_gamma=args.focal_gamma_min,
    )

    validate_training_config(
        base_config,
        require_test_dir=False,
        full_trainval=False,
    )
    if not args.skip_final_train:
        validate_training_config(
            base_config,
            require_test_dir=True,
            full_trainval=True,
        )


def _sample_params(trial, args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "batch_size": trial.suggest_categorical(
            "batch_size",
            sorted(set(args.batch_size_choices)),
        ),
        "image_size": trial.suggest_categorical(
            "image_size",
            sorted(set(args.image_size_choices)),
        ),
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            args.weight_decay_min,
            args.weight_decay_max,
            log=True,
        ),
        "scheduler_factor": trial.suggest_float(
            "scheduler_factor",
            args.scheduler_factor_min,
            args.scheduler_factor_max,
        ),
        "scheduler_patience": trial.suggest_int(
            "scheduler_patience",
            args.scheduler_patience_min,
            args.scheduler_patience_max,
        ),
        "loss": trial.suggest_categorical(
            "loss",
            list(dict.fromkeys(args.loss_choices)),
        ),
    }
    if params["loss"] == "focal":
        params["focal_gamma"] = trial.suggest_float(
            "focal_gamma",
            args.focal_gamma_min,
            args.focal_gamma_max,
        )
        params["label_smoothing"] = 0.0
    else:
        params["focal_gamma"] = 2.0
        params["label_smoothing"] = trial.suggest_float(
            "label_smoothing",
            args.label_smoothing_min,
            args.label_smoothing_max,
        )

    params["dropout"] = trial.suggest_float(
        "dropout",
        args.dropout_min,
        args.dropout_max,
    )
    params["hflip_p"] = trial.suggest_categorical(
        "hflip_p",
        sorted(set(args.hflip_p_choices)),
    )
    params["rotation_degrees"] = trial.suggest_int(
        "rotation_degrees",
        args.rotation_degrees_min,
        args.rotation_degrees_max,
    )
    params["color_jitter"] = trial.suggest_float(
        "color_jitter",
        args.color_jitter_min,
        args.color_jitter_max,
    )

    if args.model == "resnet" and args.search_pretrained:
        params["pretrained"] = trial.suggest_categorical("pretrained", [False, True])
    else:
        params["pretrained"] = False
    return params


def _on_epoch_end(trial, metric_name: str, epoch: int, val_metrics: dict[str, float]) -> None:
    trial.report(float(val_metrics[metric_name]), step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned(
            f"Trial {trial.number} prune edildi (epoch={epoch}, {metric_name}={val_metrics[metric_name]:.4f})"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _trial_dir(study_dir: Path, trial_number: int) -> Path:
    return study_dir / "trials" / f"trial_{trial_number:03d}"


def _objective_factory(args: argparse.Namespace, study_dir: Path):
    use_cv = args.hpo_folds > 1

    def objective(trial) -> float:
        params = _sample_params(trial, args)
        trial_dir = _trial_dir(study_dir, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = _build_config_from_args(
            args,
            batch_size=params["batch_size"],
            image_size=params["image_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            scheduler_factor=params["scheduler_factor"],
            scheduler_patience=params["scheduler_patience"],
            loss=params["loss"],
            focal_gamma=params["focal_gamma"],
            pretrained=params["pretrained"],
            dropout=params["dropout"],
            label_smoothing=params["label_smoothing"],
            hflip_p=params["hflip_p"],
            rotation_degrees=params["rotation_degrees"],
            color_jitter=params["color_jitter"],
        )

        cv_results: dict[str, Any] | None = None
        results: dict[str, Any] | None = None
        try:
            try:
                if use_cv:
                    cv_results = run_cv_training(
                        config,
                        n_folds=args.hpo_folds,
                        save_artifacts=False,
                        evaluate_test_set=False,
                        verbose=args.verbose_trials,
                        selection_metric=args.metric,
                        deterministic=False,
                    )
                else:
                    results = run_training(
                        config,
                        save_artifacts=False,
                        evaluate_test_set=False,
                        verbose=args.verbose_trials,
                        selection_metric=args.metric,
                        on_epoch_end=lambda epoch, _train, val: _on_epoch_end(
                            trial,
                            args.metric,
                            epoch,
                            val,
                        ),
                        deterministic=False,
                    )
            except Exception as exc:
                if optuna is not None and isinstance(exc, optuna.TrialPruned):
                    _write_json(
                        trial_dir / "trial_summary.json",
                        {
                            "trial_number": trial.number,
                            "state": "PRUNED",
                            "params": params,
                            "metric": args.metric,
                            "message": str(exc),
                        },
                    )
                    raise

                _write_json(
                    trial_dir / "trial_summary.json",
                    {
                        "trial_number": trial.number,
                        "state": "FAILED",
                        "params": params,
                        "metric": args.metric,
                        "error": str(exc),
                    },
                )
                raise

            if use_cv:
                aggregate = cv_results["aggregate"]
                val_summary = aggregate.get("val") or {}
                metric_summary = val_summary.get(args.metric)
                if metric_summary is None:
                    raise RuntimeError(
                        f"CV val metrikleri eksik: '{args.metric}' bulunamadi."
                    )
                objective_value = float(metric_summary["mean"])
                trial.set_user_attr("trial_dir", str(trial_dir))
                trial.set_user_attr("hpo_folds", args.hpo_folds)
                trial.set_user_attr(f"val_{args.metric}_mean", metric_summary["mean"])
                trial.set_user_attr(f"val_{args.metric}_std", metric_summary["std"])
                f1_summary = val_summary.get("f1")
                if f1_summary is not None:
                    trial.set_user_attr("best_val_f1_mean", f1_summary["mean"])
                    trial.set_user_attr("best_val_f1_std", f1_summary["std"])

                best_epochs = [
                    int(r["best_epoch"]) for r in cv_results["fold_results"]
                    if r.get("best_epoch") is not None
                ]
                if best_epochs:
                    rounded = int(round(sum(best_epochs) / len(best_epochs)))
                    trial.set_user_attr("best_epoch", rounded)
                    trial.set_user_attr("best_epochs_per_fold", best_epochs)

                _write_json(
                    trial_dir / "trial_summary.json",
                    {
                        "trial_number": trial.number,
                        "state": "COMPLETE",
                        "metric": args.metric,
                        "objective_value": objective_value,
                        "params": params,
                        "hpo_folds": args.hpo_folds,
                        "cv_aggregate": aggregate,
                        "config": cv_results["config"],
                    },
                )
                return objective_value

            best_val_metrics = results["best_val_metrics"]
            objective_value = float(best_val_metrics[args.metric])
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("best_epoch", results["best_epoch"])
            trial.set_user_attr("best_val_loss", results["best_val_loss"])
            trial.set_user_attr("lowest_val_loss", results["lowest_val_loss"])
            trial.set_user_attr("best_val_f1", best_val_metrics["f1"])

            _write_json(
                trial_dir / "trial_summary.json",
                {
                    "trial_number": trial.number,
                    "state": "COMPLETE",
                    "metric": args.metric,
                    "objective_value": objective_value,
                    "params": params,
                    "best_epoch": results["best_epoch"],
                    "best_val_loss": results["best_val_loss"],
                    "lowest_val_loss": results["lowest_val_loss"],
                    "best_val_metrics": best_val_metrics,
                    "config": results["config"],
                },
            )
            return objective_value
        finally:
            # Trial sonu (basari/hata/prune farketmez) buyuk referanslari dusur
            # ve PyTorch caching allocator'in tutu VRAM bloklarini surucuye geri
            # ver. HPO sirasinda batch_size/image_size trial bazinda degistigi
            # icin fragmentasyonun bir sonraki trial'a sarkmamasi kritik.
            cv_results = None
            results = None
            release_cuda_memory()

    return objective


def _run_final_training(
    args: argparse.Namespace,
    study_dir: Path,
    best_params: dict[str, Any],
    study_name: str,
    best_trial_number: int,
    best_epoch: int | None,
) -> dict[str, Any]:
    final_config = _build_config_from_args(
        args,
        epochs=int(best_epoch) if best_epoch is not None else args.epochs,
        batch_size=int(best_params["batch_size"]),
        lr=float(best_params["lr"]),
        image_size=int(best_params["image_size"]),
        loss=str(best_params["loss"]),
        pretrained=bool(best_params.get("pretrained", False)),
        weight_decay=float(best_params["weight_decay"]),
        scheduler_factor=float(best_params["scheduler_factor"]),
        scheduler_patience=int(best_params["scheduler_patience"]),
        focal_gamma=float(best_params.get("focal_gamma", 2.0)),
        dropout=float(best_params.get("dropout", 0.5)),
        label_smoothing=float(best_params.get("label_smoothing", 0.0)),
        hflip_p=float(best_params.get("hflip_p", 0.0)),
        rotation_degrees=float(best_params.get("rotation_degrees", 10.0)),
        color_jitter=float(best_params.get("color_jitter", 0.1)),
    )
    final_dir = study_dir / "best_run"
    return run_training(
        final_config,
        output_root=final_dir,
        artifact_tag=f"{args.model}_tuned",
        save_artifacts=True,
        evaluate_test_set=True,
        full_trainval=True,
        verbose=True,
        selection_metric=args.metric,
        extra_report={
            "study_name": study_name,
            "best_trial_number": best_trial_number,
            "best_trial_epoch_count": int(best_epoch) if best_epoch is not None else args.epochs,
            "optimized_metric": args.metric,
            "search_type": "bayesian_tpe",
        },
    )
