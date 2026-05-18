#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_xgb.py
----------
XGBoost modeli icin Optuna TPE tabanli hiperparametre arama bilesenleri.

Bu modul yalnizca SL/XGBoost akisina ozgu mantigi icerir (search space,
sampler, trial objective, final egitim). Ortak CLI/dispatch akisi icin
bk. ``hpo.py``.
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

    from model.sl.training_runner import (
        SLTrainingConfig,
        run_sl_cv_training,
        run_sl_training,
        validate_sl_config,
    )
else:
    from .sl.training_runner import (
        SLTrainingConfig,
        run_sl_cv_training,
        run_sl_training,
        validate_sl_config,
    )


def _search_space_summary_xgb(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image_size_choices": sorted(set(args.image_size_choices)),
        "n_estimators_range": [100, 1000],
        "max_depth_range": [3, 10],
        "learning_rate_range": [0.01, 0.3],
        "subsample_range": [0.4, 1.0],
        "colsample_bytree_range": [0.4, 1.0],
        "reg_lambda_range": [1e-3, 10.0],
        "reg_alpha_range": [1e-4, 10.0],
        "gamma_range": [1e-4, 5.0],
        "min_child_weight_range": [1, 20],
        "max_delta_step_range": [0, 20],
    }


def _resolve_feature_cache(args: argparse.Namespace) -> str | None:
    if getattr(args, "no_feature_cache", False):
        return None
    cache = getattr(args, "feature_cache", None)
    return cache if cache else None


def _validate_xgb_args(args: argparse.Namespace) -> None:
    if (
        getattr(args, "xgb_n_jobs", None) is not None
        and int(args.xgb_n_jobs) < 1
    ):
        raise ValueError("--xgb-n-jobs pozitif tamsayi olmali.")

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


def _sample_xgb_params(trial, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 20),
        "image_size": trial.suggest_categorical(
            "image_size",
            sorted(set(args.image_size_choices)),
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _trial_dir(study_dir: Path, trial_number: int) -> Path:
    return study_dir / "trials" / f"trial_{trial_number:03d}"


def _xgb_objective_factory(args: argparse.Namespace, study_dir: Path):
    # NOT: XGBoost trial'larinda epoch bazli pruning desteklenmemektedir.
    # XGBoost mod'unda study NopPruner ile olusturulur (bk. hpo.main()) ve
    # trial.report() cagrilmaz; her trial tam egitime tabi tutulur.
    use_cv = args.hpo_folds > 1

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
            reg_alpha=params["reg_alpha"],
            gamma=params["gamma"],
            min_child_weight=params["min_child_weight"],
            max_delta_step=params["max_delta_step"],
            image_size=params["image_size"],
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            feature_cache=_resolve_feature_cache(args),
            device=getattr(args, "xgb_device", "auto"),
            n_jobs=getattr(args, "xgb_n_jobs", None),
            class_balance=getattr(args, "xgb_class_balance", "none"),
        )

        try:
            if use_cv:
                cv_results = run_sl_cv_training(
                    config,
                    n_folds=args.hpo_folds,
                    save_artifacts=False,
                    evaluate_test_set=False,
                    verbose=args.verbose_trials,
                    selection_metric=args.metric,
                )
            else:
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

        if use_cv:
            aggregate = cv_results["aggregate"]
            val_summary = aggregate.get("val") or {}
            metric_summary = val_summary.get(args.metric)
            if metric_summary is None:
                raise RuntimeError(
                    f"XGBoost CV val metrikleri eksik: '{args.metric}' bulunamadi."
                )
            objective_value = float(metric_summary["mean"])
            best_iterations = [
                int(r["best_iteration"]) for r in cv_results["fold_results"]
                if r.get("best_iteration") is not None
            ]
            best_iteration_avg = (
                int(round(sum(best_iterations) / len(best_iterations)))
                if best_iterations
                else None
            )
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("hpo_folds", args.hpo_folds)
            trial.set_user_attr(f"val_{args.metric}_mean", metric_summary["mean"])
            trial.set_user_attr(f"val_{args.metric}_std", metric_summary["std"])
            f1_summary = val_summary.get("f1")
            if f1_summary is not None:
                trial.set_user_attr("best_val_f1_mean", f1_summary["mean"])
            if best_iteration_avg is not None:
                trial.set_user_attr("best_iteration", best_iteration_avg)
                trial.set_user_attr("best_iterations_per_fold", best_iterations)
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
                    "best_iteration": best_iteration_avg,
                    "config": cv_results["config"],
                },
            )
            return objective_value

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
        reg_alpha=float(best_params.get("reg_alpha", 0.0)),
        gamma=float(best_params.get("gamma", 0.0)),
        min_child_weight=int(best_params["min_child_weight"]),
        max_delta_step=int(best_params.get("max_delta_step", 0)),
        image_size=int(best_params["image_size"]),
        trainval_dir=args.trainval_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        feature_cache=_resolve_feature_cache(args),
        device=getattr(args, "xgb_device", "auto"),
        n_jobs=getattr(args, "xgb_n_jobs", None),
        class_balance=getattr(args, "xgb_class_balance", "none"),
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
