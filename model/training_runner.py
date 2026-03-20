#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py
------------------
Shared training utilities for single-run training and Bayesian search.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from .ayarlar import (
    CIKTI_KLASORU,
    GORSELLER_KLASORU,
    ISLENMIS_TEST_VERI_DIZINI,
    ISLENMIS_TRAINVAL_VERI_DIZINI,
    ISLENMIS_VERI_KLASORU,
    MODELS_KLASORU,
    RASTGELE_TOHUM,
    RAPORLAR_KLASORU,
    TEST_VERI_DIZINI,
    TRAINVAL_VERI_DIZINI,
    VARSAYILAN_VERI_DIZINI,
)
from .dl.dataset import SINIF_ISIMLERI, create_dataloaders
from .dl.engine import EarlyStopping, evaluate, train_one_epoch
from .dl.losses import FocalLoss, compute_class_weights
from .dl.models.resnet_classifier import ResNetClassifier
from .dl.models.unet_classifier import UNetClassifier
from .dl.utils import get_device, plot_confusion_matrix, plot_training_curves, set_seed

SUPPORTED_SELECTION_METRICS = {"loss", "accuracy", "precision", "recall", "f1"}


@dataclass(slots=True)
class TrainingConfig:
    model: str = "resnet"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    patience: int = 30
    image_size: int = 224
    trainval_dir: Path | str | None = None
    test_dir: Path | str | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    loss: str = "ce"
    seed: int = RASTGELE_TOHUM
    num_workers: int = 0
    use_processed_trainval: bool = False
    use_processed_test: bool = False
    pretrained: bool = False
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    focal_gamma: float = 2.0


def _contains_class_dirs(data_dir: Path) -> bool:
    return any((data_dir / class_name).is_dir() for class_name in SINIF_ISIMLERI)


def _resolve_split_subdir(data_dir: Path, split_name: str) -> Path | None:
    """Split root verilirse ilgili alt dizini, aksi halde sinif dizinini dondur."""
    data_dir = Path(data_dir)
    split_dir = data_dir / split_name
    if split_dir.exists() and _contains_class_dirs(split_dir):
        return split_dir
    if data_dir.exists() and _contains_class_dirs(data_dir):
        return data_dir
    return None


def _resolve_processed_trainval_dir() -> Path:
    if ISLENMIS_TRAINVAL_VERI_DIZINI.exists() and _contains_class_dirs(ISLENMIS_TRAINVAL_VERI_DIZINI):
        return ISLENMIS_TRAINVAL_VERI_DIZINI
    if ISLENMIS_VERI_KLASORU.exists() and _contains_class_dirs(ISLENMIS_VERI_KLASORU):
        return ISLENMIS_VERI_KLASORU
    return ISLENMIS_TRAINVAL_VERI_DIZINI


def _resolve_processed_test_dir() -> Path | None:
    if ISLENMIS_TEST_VERI_DIZINI.exists() and _contains_class_dirs(ISLENMIS_TEST_VERI_DIZINI):
        return ISLENMIS_TEST_VERI_DIZINI
    return None


def _resolve_default_trainval_dir() -> Path:
    """Varsayilan train/val kaynagini ayarlardan sec."""
    if VARSAYILAN_VERI_DIZINI.exists():
        return VARSAYILAN_VERI_DIZINI
    return TRAINVAL_VERI_DIZINI


def _resolve_default_test_dir(trainval_dir: Path) -> Path | None:
    """Ayar dosyasindaki harici test dizini kullanilabiliyorsa dondur."""
    if not TEST_VERI_DIZINI.exists() or not _contains_class_dirs(TEST_VERI_DIZINI):
        return None

    if TEST_VERI_DIZINI.resolve() == trainval_dir.resolve():
        return None
    return TEST_VERI_DIZINI


def build_model(
    name: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Build a model instance by name."""
    if name == "resnet":
        model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif name == "unet":
        model = UNetClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")
    return model.to(device)


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize config into JSON-safe values."""
    data = asdict(config)
    for key in ("trainval_dir", "test_dir"):
        if data[key] is not None:
            data[key] = str(Path(data[key]))
    return data


def resolve_data_dirs(
    config: TrainingConfig,
    *,
    require_test_dir: bool = True,
) -> tuple[Path, Path | None]:
    """Resolve split source directory and optional external test directory."""
    explicit_trainval_root: Path | None = None
    if config.trainval_dir:
        explicit_trainval_root = Path(config.trainval_dir)
        trainval_dir = _resolve_split_subdir(explicit_trainval_root, "trainval") or explicit_trainval_root
    elif config.use_processed_trainval:
        trainval_dir = _resolve_processed_trainval_dir()
    else:
        trainval_dir = _resolve_default_trainval_dir()

    if config.test_dir:
        explicit_test_root = Path(config.test_dir)
        test_dir = _resolve_split_subdir(explicit_test_root, "test") or explicit_test_root
    elif config.use_processed_test:
        test_dir = _resolve_processed_test_dir() or ISLENMIS_TEST_VERI_DIZINI
    elif config.use_processed_trainval:
        if explicit_trainval_root is not None:
            test_dir = _resolve_split_subdir(explicit_trainval_root, "test")
        else:
            test_dir = _resolve_processed_test_dir()
    else:
        test_dir = _resolve_default_test_dir(trainval_dir)
    return trainval_dir, test_dir


def validate_training_config(
    config: TrainingConfig,
    *,
    require_test_dir: bool = True,
) -> None:
    """Validate training config before starting a run."""
    if config.epochs < 1:
        raise ValueError("--epochs en az 1 olmali.")
    if config.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmali.")
    if config.lr <= 0:
        raise ValueError("--lr pozitif olmali.")
    if config.patience < 1:
        raise ValueError("--patience en az 1 olmali.")
    if config.image_size < 32:
        raise ValueError("--image-size en az 32 olmali.")
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali.")
    if config.test_ratio < 0.0 or config.test_ratio >= 1.0:
        raise ValueError("--test-ratio 0 ile 1 arasinda olmali.")
    if config.loss not in {"ce", "focal"}:
        raise ValueError(f"Gecersiz loss secimi: {config.loss}")
    if config.num_workers < 0:
        raise ValueError("--num-workers negatif olamaz.")
    if config.weight_decay < 0:
        raise ValueError("--weight-decay negatif olamaz.")
    if not 0.0 < config.scheduler_factor < 1.0:
        raise ValueError("--scheduler-factor 0 ile 1 arasinda olmali.")
    if config.scheduler_patience < 1:
        raise ValueError("--scheduler-patience en az 1 olmali.")
    if config.focal_gamma < 0:
        raise ValueError("--focal-gamma negatif olamaz.")

    trainval_dir, test_dir = resolve_data_dirs(config, require_test_dir=require_test_dir)
    if test_dir is None and config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio 1'den kucuk olmali.")
    if require_test_dir and test_dir is None and config.test_ratio <= 0.0:
        raise ValueError("Harici test dizini yoksa --test-ratio pozitif olmali.")
    if not trainval_dir.exists():
        raise FileNotFoundError(f"TrainVal veri dizini bulunamadi: {trainval_dir}")
    if test_dir is not None and not test_dir.exists():
        raise FileNotFoundError(f"Test veri dizini bulunamadi: {test_dir}")


def _selection_mode_for_metric(metric: str) -> Literal["minimize", "maximize"]:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Gecersiz selection metric: {metric}")
    return "minimize" if metric == "loss" else "maximize"


def _is_improved(
    candidate_value: float,
    best_value: float | None,
    mode: Literal["minimize", "maximize"],
) -> bool:
    if best_value is None:
        return True
    if mode == "minimize":
        return candidate_value < best_value
    return candidate_value > best_value


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    if output_root.resolve() == CIKTI_KLASORU.resolve():
        models_dir = MODELS_KLASORU
        reports_dir = RAPORLAR_KLASORU
        visuals_dir = GORSELLER_KLASORU
    else:
        models_dir = output_root / "modeller"
        reports_dir = output_root / "raporlar"
        visuals_dir = output_root / "gorseller"
    for directory in (models_dir, reports_dir, visuals_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_root,
        "models": models_dir,
        "reports": reports_dir,
        "visuals": visuals_dir,
    }


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def _checkpoint_payload(
    config: TrainingConfig,
    epoch: int,
    best_val_loss: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
        "model_name": config.model,
        "pretrained": config.pretrained,
        "num_classes": num_classes,
        "image_size": config.image_size,
        "class_names": SINIF_ISIMLERI,
        "training_config": config_to_dict(config),
    }


def run_training(
    config: TrainingConfig,
    *,
    output_root: Path | None = None,
    artifact_tag: str | None = None,
    save_artifacts: bool = True,
    evaluate_test_set: bool = True,
    verbose: bool = True,
    selection_metric: str = "loss",
    on_epoch_end: Callable[[int, dict[str, float], dict[str, float]], None] | None = None,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single training experiment and optionally persist artifacts."""
    selection_mode = _selection_mode_for_metric(selection_metric)
    validate_training_config(config, require_test_dir=evaluate_test_set)

    set_seed(config.seed)
    device = get_device(verbose=verbose)
    trainval_dir, test_dir = resolve_data_dirs(config, require_test_dir=evaluate_test_set)

    if verbose:
        print("\n[INFO] Veri yukleniyor:")
        print(f"  Veri dizini     : {trainval_dir}")
        print(f"  Test dizini     : {test_dir}")
        print(f"  Val orani       : {config.val_ratio}")
        print(f"  Test orani      : {config.test_ratio}")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=test_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        include_test=evaluate_test_set,
    )

    if verbose:
        print(f"\n  Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
        print(f"  Grup: Train={info['train_groups']}, Val={info['val_groups']}")
        print(f"  Split stratejisi : {info['split_strategy']}")
        if info["split_warnings"]:
            print("  [UYARI] Split ile ilgili notlar:")
            for warning in info["split_warnings"]:
                print(f"    - {warning}")

    num_classes = info["num_classes"]
    model = build_model(config.model, num_classes, device, pretrained=config.pretrained)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n[INFO] Model: {config.model.upper()}")
        print(f"  Egitilebilir parametre: {param_count:,}")

    class_weights = compute_class_weights(info["train_labels"], num_classes).to(device)
    if config.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if verbose:
        if config.loss == "focal":
            print(f"  Loss: FOCAL (gamma={config.focal_gamma:.3f}, class weights aktif)")
        else:
            print("  Loss: CE (class weights aktif)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    early_stopping = EarlyStopping(patience=config.patience)

    artifact_stem = artifact_tag or config.model
    output_dirs = None
    if save_artifacts:
        output_dirs = _build_output_dirs(output_root or CIKTI_KLASORU)

    best_checkpoint_path = (
        output_dirs["models"] / f"best_{artifact_stem}.pt" if output_dirs else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    lowest_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_state_dict: dict[str, Any] | None = None
    best_selection_value: float | None = None
    selected_epoch_val_loss: float | None = None

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            "EGITIM BASLIYOR - "
            f"{config.epochs} epoch, batch={config.batch_size}, lr={config.lr}"
        )
        print(f"{'=' * 70}\n")

    epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        train_scalars = _scalar_metrics(train_metrics)
        val_scalars = _scalar_metrics(val_metrics)

        train_losses.append(train_scalars["loss"])
        val_losses.append(val_scalars["loss"])
        train_accs.append(train_scalars["accuracy"])
        val_accs.append(val_scalars["accuracy"])

        lr_current = optimizer.param_groups[0]["lr"]
        if verbose:
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train Loss: {train_scalars['loss']:.4f} Acc: {train_scalars['accuracy']:.4f} | "
                f"Val Loss: {val_scalars['loss']:.4f} Acc: {val_scalars['accuracy']:.4f} "
                f"F1: {val_scalars['f1']:.4f} | LR: {lr_current:.2e}"
            )

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_scalars, val_scalars)

        scheduler.step(val_scalars["loss"])

        if val_scalars["loss"] < lowest_val_loss:
            lowest_val_loss = val_scalars["loss"]

        current_selection_value = float(val_scalars[selection_metric])
        if _is_improved(current_selection_value, best_selection_value, selection_mode):
            best_selection_value = current_selection_value
            best_epoch = epoch
            best_val_metrics = dict(val_scalars)
            best_state_dict = copy.deepcopy(model.state_dict())
            selected_epoch_val_loss = val_scalars["loss"]
            if best_checkpoint_path is not None:
                torch.save(
                    _checkpoint_payload(
                        config=config,
                        epoch=epoch,
                        best_val_loss=val_scalars["loss"],
                        model=model,
                        optimizer=optimizer,
                        num_classes=num_classes,
                    ),
                    best_checkpoint_path,
                )
                if verbose:
                    print(
                        "  [OK] Best checkpoint kaydedildi "
                        f"({selection_metric}={current_selection_value:.4f})"
                    )

        if early_stopping(val_scalars["loss"]):
            if verbose:
                print(
                    f"\n[INFO] Early stopping: {config.patience} epoch boyunca "
                    "iyilesme olmadi."
                )
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_val_metrics = dict(val_scalars)
        best_epoch = epoch
        best_selection_value = float(val_scalars[selection_metric])
        selected_epoch_val_loss = val_scalars["loss"]

    model.load_state_dict(best_state_dict)

    test_metrics = None
    if evaluate_test_set:
        if verbose:
            print(f"\n{'=' * 70}")
            print("TEST DEGERLENDIRMESI")
            print(f"{'=' * 70}\n")

        test_eval = evaluate(model, test_loader, criterion, device)
        test_metrics = _scalar_metrics(test_eval)

        if verbose:
            print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall   : {test_metrics['recall']:.4f}")
            print(f"  F1 (macro): {test_metrics['f1']:.4f}")

        if output_dirs is not None:
            plot_confusion_matrix(
                test_eval["labels"],
                test_eval["preds"],
                SINIF_ISIMLERI,
                output_dirs["visuals"] / f"confusion_matrix_{artifact_stem}.png",
            )

    if output_dirs is not None:
        plot_training_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            output_dirs["visuals"] / f"training_curves_{artifact_stem}.png",
        )

    report_path = None
    if output_dirs is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dirs["reports"] / f"rapor_{artifact_stem}_{timestamp}.json"
        report = {
            "model": config.model,
            "timestamp": timestamp,
            "epochs_trained": epoch,
            "best_epoch": best_epoch,
            "pretrained": config.pretrained,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_value": round(best_selection_value, 6),
            "lowest_val_loss": round(lowest_val_loss, 6),
            "selected_epoch_val_loss": round(selected_epoch_val_loss, 6),
            "best_val_loss": round(selected_epoch_val_loss, 6),
            "best_val_metrics": {
                "accuracy": round(best_val_metrics["accuracy"], 4),
                "precision": round(best_val_metrics["precision"], 4),
                "recall": round(best_val_metrics["recall"], 4),
                "f1_macro": round(best_val_metrics["f1"], 4),
            },
            "test_metrics": (
                {
                    "accuracy": round(test_metrics["accuracy"], 4),
                    "precision": round(test_metrics["precision"], 4),
                    "recall": round(test_metrics["recall"], 4),
                    "f1_macro": round(test_metrics["f1"], 4),
                }
                if test_metrics is not None
                else None
            ),
            "data_split": {
                "strategy": info["split_strategy"],
                "trainval_dir": str(trainval_dir),
                "test_dir": str(test_dir) if test_dir is not None else None,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "train_size": info["train_size"],
                "val_size": info["val_size"],
                "test_size": info["test_size"],
                "trainval_grouping": info["trainval_grouping"],
                "test_grouping": info["test_grouping"],
                "warnings": info["split_warnings"],
            },
            "config": config_to_dict(config),
            "history": {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_accuracy": train_accs,
                "val_accuracy": val_accs,
            },
        }
        if extra_report:
            report["extra"] = extra_report

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n[OK] Rapor kaydedildi: {report_path}")
            if best_checkpoint_path is not None:
                print(f"[OK] Model kaydedildi: {best_checkpoint_path}")
            print(f"\n{'=' * 70}")
            print("EGITIM TAMAMLANDI")
            print(f"{'=' * 70}\n")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": selected_epoch_val_loss,
        "lowest_val_loss": lowest_val_loss,
        "best_val_metrics": best_val_metrics,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selected_epoch_val_loss": selected_epoch_val_loss,
        "test_metrics": test_metrics,
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accs,
            "val_accuracy": val_accs,
        },
        "data_info": info,
        "checkpoint_path": best_checkpoint_path,
        "report_path": report_path,
        "output_root": output_dirs["root"] if output_dirs is not None else None,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": config_to_dict(config),
    }
