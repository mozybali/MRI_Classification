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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.ayarlar import HPO_KLASORU, RASTGELE_TOHUM
    from model.training_runner import TrainingConfig, run_training, validate_training_config
else:
    from .ayarlar import HPO_KLASORU, RASTGELE_TOHUM
    from .training_runner import TrainingConfig, run_training, validate_training_config

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


SUPPORTED_METRICS = ("loss", "accuracy", "precision", "recall", "f1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma icin Optuna TPE tabanli Bayes search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python -m model.hpo --model resnet --trials 20 --epochs 12
  python -m model.hpo --model unet --trials 30 --metric loss --skip-final-train
  python -m model.hpo --model resnet --search-pretrained --batch-size-choices 16 32
  python -m model.hpo --model resnet --use-processed-trainval --trainval-dir goruntu_isleme/cikti
        """,
    )
    parser.add_argument("--model", choices=["resnet", "unet"], default="resnet")
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
        choices=SUPPORTED_METRICS,
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
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RASTGELE_TOHUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trainval-dir", type=str, default=None, help="Train+Val icin veri dizini")
    parser.add_argument("--test-dir", type=str, default=None, help="Test icin veri dizini")
    parser.add_argument(
        "--use-processed-trainval",
        action="store_true",
        help="Train+validation icin islenmis goruntu dizinini varsayilan kaynak yapar.",
    )
    parser.add_argument(
        "--use-processed-test",
        action="store_true",
        help="Test icin islenmis goruntu dizinini varsayilan kaynak yapar.",
    )
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _metric_direction(metric: str) -> str:
    return "minimize" if metric == "loss" else "maximize"


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
        "search_pretrained": bool(args.search_pretrained and args.model == "resnet"),
    }


def validate_search_args(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ModuleNotFoundError(
            "Bayes search icin 'optuna' gerekli. "
            "Kurulum: .\\.venv\\Scripts\\python.exe -m pip install -e .[dev]"
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
    if not args.loss_choices:
        raise ValueError("--loss-choices bos olamaz.")
    if args.n_startup_trials < 1:
        raise ValueError("--n-startup-trials en az 1 olmali.")
    if args.pruner_startup_trials < 0:
        raise ValueError("--pruner-startup-trials negatif olamaz.")
    if args.pruner_warmup_epochs < 0:
        raise ValueError("--pruner-warmup-epochs negatif olamaz.")

    validate_training_config(
        TrainingConfig(
            model=args.model,
            epochs=args.epochs,
            patience=args.patience,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            use_processed_trainval=args.use_processed_trainval,
            use_processed_test=args.use_processed_test,
            batch_size=min(args.batch_size_choices),
            image_size=min(args.image_size_choices),
            lr=args.lr_min,
            weight_decay=args.weight_decay_min,
            scheduler_factor=args.scheduler_factor_min,
            scheduler_patience=args.scheduler_patience_min,
            loss=args.loss_choices[0],
            focal_gamma=args.focal_gamma_min,
        ),
        require_test_dir=not args.skip_final_train,
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
    else:
        params["focal_gamma"] = 2.0

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

        config = TrainingConfig(
            model=args.model,
            epochs=args.epochs,
            batch_size=params["batch_size"],
            lr=params["lr"],
            patience=args.patience,
            image_size=params["image_size"],
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            loss=params["loss"],
            seed=args.seed,
            num_workers=args.num_workers,
            use_processed_trainval=args.use_processed_trainval,
            use_processed_test=args.use_processed_test,
            pretrained=params["pretrained"],
            weight_decay=params["weight_decay"],
            scheduler_factor=params["scheduler_factor"],
            scheduler_patience=params["scheduler_patience"],
            focal_gamma=params["focal_gamma"],
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


def _run_final_training(
    args: argparse.Namespace,
    study_dir: Path,
    best_params: dict[str, Any],
    study_name: str,
    best_trial_number: int,
) -> dict[str, Any]:
    final_config = TrainingConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=int(best_params["batch_size"]),
        lr=float(best_params["lr"]),
        patience=args.patience,
        image_size=int(best_params["image_size"]),
        trainval_dir=args.trainval_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        loss=str(best_params["loss"]),
        seed=args.seed,
        num_workers=args.num_workers,
        use_processed_trainval=args.use_processed_trainval,
        use_processed_test=args.use_processed_test,
        pretrained=bool(best_params.get("pretrained", False)),
        weight_decay=float(best_params["weight_decay"]),
        scheduler_factor=float(best_params["scheduler_factor"]),
        scheduler_patience=int(best_params["scheduler_patience"]),
        focal_gamma=float(best_params.get("focal_gamma", 2.0)),
    )
    final_dir = study_dir / "best_run"
    return run_training(
        final_config,
        output_root=final_dir,
        artifact_tag=f"{args.model}_tuned",
        save_artifacts=True,
        evaluate_test_set=True,
        verbose=True,
        selection_metric=args.metric,
        extra_report={
            "study_name": study_name,
            "best_trial_number": best_trial_number,
            "optimized_metric": args.metric,
            "search_type": "bayesian_tpe",
        },
    )


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

    state_counts = Counter(str(trial.state) for trial in study.trials)
    best_trial = study.best_trial
    summary = {
        "study_name": study_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "search_type": "bayesian_tpe",
        "metric": args.metric,
        "direction": _metric_direction(args.metric),
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
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction=_metric_direction(args.metric),
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
        study.optimize(
            _objective_factory(args, study_dir),
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
            "Final egitim baslatiliyor."
        )
        final_run = _run_final_training(
            args=args,
            study_dir=study_dir,
            best_params=study.best_trial.params,
            study_name=study_name,
            best_trial_number=study.best_trial.number,
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
