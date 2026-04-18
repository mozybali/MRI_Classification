#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo.py
------
Bayesian hyperparameter search for MRI classification models via Optuna TPE.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.ayarlar import HPO_KLASORU, RASTGELE_TOHUM, VARSAYILAN_EARLY_STOPPING_SABIR
    from model.training_runner import (
        SUPPORTED_SELECTION_METRICS,
        TrainingConfig,
        _selection_mode_for_metric,
        run_training,
        validate_training_config,
    )
    from model.sl.training_runner import (
        SLTrainingConfig,
        run_sl_training,
        validate_sl_config,
    )
else:
    from .ayarlar import HPO_KLASORU, RASTGELE_TOHUM, VARSAYILAN_EARLY_STOPPING_SABIR
    from .training_runner import (
        SUPPORTED_SELECTION_METRICS,
        TrainingConfig,
        _selection_mode_for_metric,
        run_training,
        validate_training_config,
    )
    from .sl.training_runner import (
        SLTrainingConfig,
        run_sl_training,
        validate_sl_config,
    )

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma icin Optuna TPE tabanli Bayes search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python -m model.hpo --model resnet --trials 20 --epochs 12
  python -m model.hpo --model resnet --trials 30 --metric loss --skip-final-train
  python -m model.hpo --model resnet --search-pretrained --batch-size-choices 16 32
  python -m model.hpo --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
        """,
    )
    parser.add_argument("--model", choices=["resnet", "xgboost"], default="resnet")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Hedef toplam trial sayisi",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Opsiyonel sure siniri (saniye)",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(SUPPORTED_SELECTION_METRICS),
        default="f1",
        help="Optimize edilecek validation metrik",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study adi",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Opsiyonel Optuna storage URL (ornegin sqlite:///study.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Arama ciktilarinin kaydedilecegi klasor",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=VARSAYILAN_EARLY_STOPPING_SABIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RASTGELE_TOHUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trainval-dir", type=str, default=None, help="Train+Val icin veri dizini")
    parser.add_argument("--test-dir", type=str, default=None, help="Test icin veri dizini")
    parser.add_argument(
        "--batch-size-choices",
        type=int,
        nargs="+",
        default=[16, 32, 48],
        help="Denenecek batch size adaylari",
    )
    parser.add_argument(
        "--image-size-choices",
        type=int,
        nargs="+",
        default=[160, 192, 224],
        help="Denenecek goruntu boyutlari",
    )
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=5e-4)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-2)
    parser.add_argument("--scheduler-factor-min", type=float, default=0.2)
    parser.add_argument("--scheduler-factor-max", type=float, default=0.7)
    parser.add_argument("--scheduler-patience-min", type=int, default=2)
    parser.add_argument("--scheduler-patience-max", type=int, default=6)
    parser.add_argument(
        "--loss-choices",
        choices=["ce", "focal"],
        nargs="+",
        default=["ce", "focal"],
        help="Aramada kullanilacak loss adaylari",
    )
    parser.add_argument("--focal-gamma-min", type=float, default=1.0)
    parser.add_argument("--focal-gamma-max", type=float, default=4.0)
    parser.add_argument("--dropout-min", type=float, default=0.1)
    parser.add_argument("--dropout-max", type=float, default=0.6)
    parser.add_argument("--label-smoothing-min", type=float, default=0.0)
    parser.add_argument("--label-smoothing-max", type=float, default=0.15)
    parser.add_argument(
        "--hflip-p-choices",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5],
        help="Denenecek RandomHorizontalFlip olasiliklari",
    )
    parser.add_argument("--rotation-degrees-min", type=int, default=0)
    parser.add_argument("--rotation-degrees-max", type=int, default=20)
    parser.add_argument("--color-jitter-min", type=float, default=0.0)
    parser.add_argument("--color-jitter-max", type=float, default=0.2)
    parser.add_argument(
        "--search-pretrained",
        action="store_true",
        help="ResNet icin pretrained secenegini de arama uzayina ekle",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="TPE sampler icin baslangic random trial sayisi",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=5,
        help="Median pruner baslamadan once tamamlanacak trial sayisi",
    )
    parser.add_argument(
        "--pruner-warmup-epochs",
        type=int,
        default=3,
        help="Pruning oncesi minimum epoch sayisi",
    )
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Arama sonunda en iyi parametrelerle final egitimi yapma",
    )
    parser.add_argument(
        "--verbose-trials",
        action="store_true",
        help="Her trial icin epoch loglarini goster",
    )
    parser.add_argument(
        "--feature-cache",
        type=str,
        default=None,
        help="XGBoost ozellik cache dizini (disk .npz)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _remaining_trials_to_run(requested_total_trials: int, existing_trial_count: int) -> int:
    return max(0, requested_total_trials - existing_trial_count)


def _default_study_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.model}_bayes_search_{timestamp}"


def _resolve_study_dir(args: argparse.Namespace, study_name: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return HPO_KLASORU / study_name


def _search_space_summary(args: argparse.Namespace) -> dict[str, Any]:
    if args.model == "xgboost":
        return _search_space_summary_xgb(args)
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


def _search_space_summary_xgb(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image_size_choices": sorted(set(args.image_size_choices)),
        "n_estimators_range": [100, 1000],
        "max_depth_range": [3, 10],
        "learning_rate_range": [0.01, 0.3],
        "subsample_range": [0.5, 1.0],
        "colsample_bytree_range": [0.5, 1.0],
        "reg_lambda_range": [1e-3, 10.0],
        "min_child_weight_range": [1, 10],
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
    hflip_p: float = 0.5,
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


def validate_search_args(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ModuleNotFoundError(
            "Bayes search icin 'optuna' gerekli. "
            "Kurulum: once requirements.txt ve uygun PyTorch requirements dosyasini "
            "yukleyin, sonra .\\.venv\\Scripts\\python.exe -m pip install -e .[dev] --no-deps calistirin."
        )
    if args.trials < 1:
        raise ValueError("--trials en az 1 olmali.")
    if args.timeout is not None and args.timeout < 1:
        raise ValueError("--timeout pozitif olmali.")
    if not args.batch_size_choices:
        raise ValueError("--batch-size-choices bos olamaz.")
    if not args.image_size_choices:
        raise ValueError("--image-size-choices bos olamaz.")
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
    if args.n_startup_trials < 1:
        raise ValueError("--n-startup-trials en az 1 olmali.")
    if args.pruner_startup_trials < 0:
        raise ValueError("--pruner-startup-trials negatif olamaz.")
    if args.pruner_warmup_epochs < 0:
        raise ValueError("--pruner-warmup-epochs negatif olamaz.")

    if args.model == "xgboost":
        base_sl_config = SLTrainingConfig(
            image_size=min(args.image_size_choices),
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        validate_sl_config(base_sl_config, require_test_dir=False, full_trainval=False)
        if not args.skip_final_train:
            validate_sl_config(base_sl_config, require_test_dir=True, full_trainval=True)
        return

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


def _trial_dir(study_dir: Path, trial_number: int) -> Path:
    return study_dir / "trials" / f"trial_{trial_number:03d}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _objective_factory(args: argparse.Namespace, study_dir: Path):
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

        try:
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

    return objective


def _on_epoch_end(trial, metric_name: str, epoch: int, val_metrics: dict[str, float]) -> None:
    trial.report(float(val_metrics[metric_name]), step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned(
            f"Trial {trial.number} prune edildi (epoch={epoch}, {metric_name}={val_metrics[metric_name]:.4f})"
        )


# ==================== XGBoost HPO ====================

def _sample_xgb_params(trial, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "image_size": trial.suggest_categorical(
            "image_size",
            sorted(set(args.image_size_choices)),
        ),
    }


def _xgb_objective_factory(args: argparse.Namespace, study_dir: Path):
    # NOT: XGBoost trial'larinda epoch bazli pruning desteklenmemektedir.
    # MedianPruner tanimli olsa da XGBoost objective'i trial.report() cagirmaz.
    # Dolayisiyla kotu parametreli trial'lar tam egitime tabi tutulur.
    def objective(trial) -> float:
        params = _sample_xgb_params(trial, args)
        trial_dir = _trial_dir(study_dir, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = SLTrainingConfig(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            min_child_weight=params["min_child_weight"],
            image_size=params["image_size"],
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            feature_cache=getattr(args, "feature_cache", None),
        )

        try:
            results = run_sl_training(
                config,
                save_artifacts=False,
                evaluate_test_set=False,
                verbose=args.verbose_trials,
                selection_metric=args.metric,
            )
        except Exception as exc:
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

        best_val_metrics = results["best_val_metrics"]
        if best_val_metrics is None:
            raise ValueError("XGBoost trial val metrikleri bos.")
        objective_value = float(best_val_metrics[args.metric])
        best_iteration = results.get("best_iteration")
        best_iteration_int = int(best_iteration) if best_iteration is not None else None
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("best_val_f1", best_val_metrics["f1"])
        if best_iteration_int is not None:
            trial.set_user_attr("best_iteration", best_iteration_int)

        _write_json(
            trial_dir / "trial_summary.json",
            {
                "trial_number": trial.number,
                "state": "COMPLETE",
                "metric": args.metric,
                "objective_value": objective_value,
                "params": params,
                "best_iteration": best_iteration_int,
                "best_val_metrics": best_val_metrics,
                "config": results["config"],
            },
        )
        return objective_value

    return objective


def _resolve_final_xgb_n_estimators(
    searched_n_estimators: int,
    best_iteration: Any | None,
) -> tuple[int, int | None]:
    searched_n_estimators = max(1, int(searched_n_estimators))
    if best_iteration is None:
        return searched_n_estimators, None

    try:
        best_iteration_int = int(best_iteration)
    except (TypeError, ValueError):
        return searched_n_estimators, None

    if best_iteration_int < 0:
        return searched_n_estimators, None

    final_n_estimators = min(searched_n_estimators, best_iteration_int + 1)
    return max(1, final_n_estimators), best_iteration_int


def _run_final_xgb_training(
    args: argparse.Namespace,
    study_dir: Path,
    best_params: dict[str, Any],
    study_name: str,
    best_trial_number: int,
    best_iteration: Any | None,
) -> dict[str, Any]:
    searched_n_estimators = int(best_params["n_estimators"])
    final_n_estimators, best_iteration_int = _resolve_final_xgb_n_estimators(
        searched_n_estimators,
        best_iteration,
    )
    config = SLTrainingConfig(
        n_estimators=final_n_estimators,
        max_depth=int(best_params["max_depth"]),
        learning_rate=float(best_params["learning_rate"]),
        subsample=float(best_params["subsample"]),
        colsample_bytree=float(best_params["colsample_bytree"]),
        reg_lambda=float(best_params["reg_lambda"]),
        min_child_weight=int(best_params["min_child_weight"]),
        image_size=int(best_params["image_size"]),
        trainval_dir=args.trainval_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        feature_cache=getattr(args, "feature_cache", None),
    )
    final_dir = study_dir / "best_run"
    return run_sl_training(
        config,
        output_root=final_dir,
        artifact_tag="xgboost_tuned",
        save_artifacts=True,
        evaluate_test_set=True,
        full_trainval=True,
        verbose=True,
        selection_metric=args.metric,
        extra_report={
            "study_name": study_name,
            "best_trial_number": best_trial_number,
            "optimized_metric": args.metric,
            "search_type": "bayesian_tpe",
            "best_trial_n_estimators": searched_n_estimators,
            "best_trial_best_iteration": best_iteration_int,
            "final_n_estimators": final_n_estimators,
            "final_n_estimators_source": (
                "best_iteration_plus_one"
                if best_iteration_int is not None
                else "best_trial_n_estimators"
            ),
        },
    )


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
        hflip_p=float(best_params.get("hflip_p", 0.5)),
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


def _figure_from_plot_result(plot_result: Any) -> Any | None:
    if hasattr(plot_result, "figure"):
        return plot_result.figure
    if hasattr(plot_result, "flat"):
        for item in plot_result.flat:
            fig = _figure_from_plot_result(item)
            if fig is not None:
                return fig
    if isinstance(plot_result, (list, tuple)):
        for item in plot_result:
            fig = _figure_from_plot_result(item)
            if fig is not None:
                return fig
    return None


def _save_single_hpo_plot(study: Any, plot_func: Any, save_path: Path, label: str) -> str | None:
    import logging
    import matplotlib.pyplot as plt

    optuna_logger = logging.getLogger("optuna")
    previous_level = optuna_logger.level
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optuna_logger.setLevel(logging.ERROR)
            plot_result = plot_func(study)
            fig = _figure_from_plot_result(plot_result)
            if fig is None:
                raise RuntimeError("Matplotlib figure bulunamadi.")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return str(save_path)
    except Exception as exc:
        plt.close("all")
        print(f"[UYARI] HPO {label} grafigi kaydedilemedi: {exc}")
        return None
    finally:
        optuna_logger.setLevel(previous_level)


def _save_hpo_visualizations(study: Any, study_dir: Path) -> dict[str, str]:
    """Optuna study icin analiz grafiklerini kaydet."""
    try:
        from optuna.visualization import matplotlib as optuna_mpl
    except Exception as exc:
        print(f"[UYARI] Optuna matplotlib gorselleri yuklenemedi: {exc}")
        return {}

    visuals_dir = study_dir / "gorseller"
    plotters = {
        "optimization_history": optuna_mpl.plot_optimization_history,
        "param_importances": optuna_mpl.plot_param_importances,
        "parallel_coordinate": optuna_mpl.plot_parallel_coordinate,
        "slice": optuna_mpl.plot_slice,
    }

    artifacts: dict[str, str] = {}
    for key, plot_func in plotters.items():
        saved_path = _save_single_hpo_plot(
            study,
            plot_func,
            visuals_dir / f"hpo_{key}.png",
            key,
        )
        if saved_path is not None:
            artifacts[key] = saved_path
    return artifacts


def _save_study_artifacts(
    study,
    args: argparse.Namespace,
    study_dir: Path,
    study_name: str,
    final_run: dict[str, Any] | None,
    existing_trial_count: int,
    trials_executed_this_run: int,
) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_dir / "trial_history.csv", index=False)
    hpo_visualizations = _save_hpo_visualizations(study, study_dir)

    state_counts = Counter(str(trial.state) for trial in study.trials)
    best_trial = study.best_trial
    summary = {
        "study_name": study_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "search_type": "bayesian_tpe",
        "metric": args.metric,
        "direction": _selection_mode_for_metric(args.metric),
        "trials_requested": args.trials,
        "timeout_seconds": args.timeout,
        "existing_trials_before_run": existing_trial_count,
        "trials_executed_this_run": trials_executed_this_run,
        "study_output_dir": str(study_dir),
        "storage": args.storage,
        "state_counts": dict(state_counts),
        "search_space": _search_space_summary(args),
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "final_run_dir": str(final_run["output_root"]) if final_run is not None else None,
        "final_report_path": str(final_run["report_path"]) if final_run is not None else None,
        "visualizations": hpo_visualizations,
    }
    _write_json(study_dir / "study_summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_search_args(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 1

    study_name = args.study_name or _default_study_name(args)
    study_dir = _resolve_study_dir(args, study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        warn_independent_sampling=False, #Mevcut durumda rastgele degerlerin denenmesini istedigimiz icin bu uyari kapatildi.
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction=_selection_mode_for_metric(args.metric),
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    existing_trial_count = len(study.trials)
    trials_to_run = _remaining_trials_to_run(args.trials, existing_trial_count)

    print(f"[INFO] Bayes search basliyor: study={study_name}, metric={args.metric}, trials={args.trials}")
    print(f"[INFO] Ciktilar: {study_dir}")
    if args.storage:
        print(
            f"[INFO] Storage study trial durumu: mevcut={existing_trial_count}, "
            f"hedef_toplam={args.trials}, bu_calismada={trials_to_run}"
        )

    if trials_to_run > 0:
        if args.model == "xgboost":
            objective_fn = _xgb_objective_factory(args, study_dir)
        else:
            objective_fn = _objective_factory(args, study_dir)
        study.optimize(
            objective_fn,
            n_trials=trials_to_run,
            timeout=args.timeout,
            catch=(RuntimeError, ValueError, FileNotFoundError),
            gc_after_trial=True,
        )
    else:
        print("[INFO] Mevcut study zaten istenen toplam trial sayisina ulasmis; yeni trial calistirilmadi.")

    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed_trials:
        print("[HATA] Hic tamamlanan trial yok. Arama sonlandirildi.")
        return 1

    final_run = None
    if not args.skip_final_train:
        print(
            "[INFO] En iyi trial bulundu: "
            f"#{study.best_trial.number} ({args.metric}={study.best_value:.4f}). "
            "Tum trainval uzerinde final egitim baslatiliyor."
        )
        if args.model == "xgboost":
            final_run = _run_final_xgb_training(
                args=args,
                study_dir=study_dir,
                best_params=study.best_trial.params,
                study_name=study_name,
                best_trial_number=study.best_trial.number,
                best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            )
        else:
            final_run = _run_final_training(
                args=args,
                study_dir=study_dir,
                best_params=study.best_trial.params,
                study_name=study_name,
                best_trial_number=study.best_trial.number,
                best_epoch=study.best_trial.user_attrs.get("best_epoch"),
            )

    _save_study_artifacts(
        study=study,
        args=args,
        study_dir=study_dir,
        study_name=study_name,
        final_run=final_run,
        existing_trial_count=existing_trial_count,
        trials_executed_this_run=trials_to_run,
    )

    print(
        f"[OK] Bayes search tamamlandi. En iyi trial=#{study.best_trial.number}, "
        f"{args.metric}={study.best_value:.4f}"
    )
    print(f"[OK] Ozet: {study_dir / 'study_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
