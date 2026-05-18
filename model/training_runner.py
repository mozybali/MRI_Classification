#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py
------------------
Shared training utilities for single-run training and Bayesian search.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

from .ayarlar import (
    CIKTI_KLASORU,
    GORSELLER_KLASORU,
    MODELS_KLASORU,
    RASTGELE_TOHUM,
    RAPORLAR_KLASORU,
    TEST_VERI_DIZINI,
    TRAINVAL_VERI_DIZINI,
    VARSAYILAN_EARLY_STOPPING_SABIR,
)
from .dl.dataset import (
    SINIF_ISIMLERI,
    create_dataloaders,
    create_full_train_test_loaders,
    iter_kfold_dataloaders,
)
from .dl.engine import EarlyStopping, evaluate, train_one_epoch
from .dl.losses import FocalLoss, compute_class_weights
from .dl.models.resnet_classifier import ResNetClassifier
from .dl.utils import (
    clone_state_dict_to_cpu,
    configure_torch_runtime,
    get_device,
    plot_classification_summary,
    plot_confusion_matrix,
    plot_multiclass_roc_pr_curves,
    plot_prediction_confidence,
    plot_training_curves,
    release_cuda_memory,
    set_seed,
)
from .common.evaluation import (
    build_detailed_eval_report as _build_detailed_eval_report,
    scalar_metrics as _scalar_metrics,
)

SUPPORTED_SELECTION_METRICS = {"loss", "accuracy", "precision", "recall", "f1"}


@dataclass(slots=True)
class TrainingConfig:
    model: str = "resnet"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    patience: int = VARSAYILAN_EARLY_STOPPING_SABIR
    image_size: int = 224
    trainval_dir: Path | str | None = None
    test_dir: Path | str | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    loss: str = "ce"
    seed: int = RASTGELE_TOHUM
    num_workers: int = 0
    pretrained: bool = False
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    focal_gamma: float = 2.0
    dropout: float = 0.5
    label_smoothing: float = 0.0
    hflip_p: float = 0.0
    rotation_degrees: float = 10.0
    color_jitter: float = 0.1


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


def _resolve_default_trainval_dir() -> Path:
    """Varsayilan train/val kaynagini ayarlardan sec."""
    return TRAINVAL_VERI_DIZINI


def _resolve_default_test_dir(trainval_dir: Path) -> Path | None:
    """Ayar dosyasindaki harici test dizini kullanilabiliyorsa dondur."""
    if not TEST_VERI_DIZINI.exists() or not _contains_class_dirs(TEST_VERI_DIZINI):
        return None

    if TEST_VERI_DIZINI.resolve() == trainval_dir.resolve():
        return None
    return TEST_VERI_DIZINI


SUPPORTED_MODELS = ("resnet",)


def build_model(
    name: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
    dropout: float = 0.5,
) -> torch.nn.Module:
    """Build a model instance by name."""
    if name == "resnet":
        model = ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Bilinmeyen model: {name}. Desteklenen modeller: {', '.join(SUPPORTED_MODELS)}"
        )
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
) -> tuple[Path, Path | None]:
    """Resolve split source directory and optional external test directory."""
    explicit_trainval_root: Path | None = None
    if config.trainval_dir:
        explicit_trainval_root = Path(config.trainval_dir)
        trainval_dir = _resolve_split_subdir(explicit_trainval_root, "trainval") or explicit_trainval_root
    else:
        trainval_dir = _resolve_default_trainval_dir()

    if config.test_dir:
        explicit_test_root = Path(config.test_dir)
        test_dir = _resolve_split_subdir(explicit_test_root, "test") or explicit_test_root
    elif (
        explicit_trainval_root is not None
        and trainval_dir != explicit_trainval_root
    ):
        # Kullanici split kokunu verdi (trainval/ alt dizinine inildi);
        # global default yerine ayni kokun altindaki test/ kardesini tercih et.
        sibling_test = explicit_trainval_root / "test"
        if sibling_test.exists() and _contains_class_dirs(sibling_test):
            test_dir = sibling_test
        else:
            test_dir = _resolve_default_test_dir(trainval_dir)
    else:
        test_dir = _resolve_default_test_dir(trainval_dir)
    return trainval_dir, test_dir


def validate_training_config(
    config: TrainingConfig,
    *,
    require_test_dir: bool = True,
    full_trainval: bool = False,
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
    if not full_trainval and not 0.0 < config.val_ratio < 1.0:
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
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("--dropout 0 ile 1 arasinda olmali.")
    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError("--label-smoothing 0 ile 1 arasinda olmali.")
    if not 0.0 <= config.hflip_p <= 1.0:
        raise ValueError("--hflip-p 0 ile 1 arasinda olmali.")
    if config.rotation_degrees < 0:
        raise ValueError("--rotation-degrees negatif olamaz.")
    if config.color_jitter < 0:
        raise ValueError("--color-jitter negatif olamaz.")

    trainval_dir, test_dir = resolve_data_dirs(config)
    if (
        not full_trainval
        and require_test_dir
        and test_dir is None
        and config.val_ratio + config.test_ratio >= 1.0
    ):
        raise ValueError("--val-ratio + --test-ratio 1'den kucuk olmali.")
    if require_test_dir and test_dir is None and config.test_ratio <= 0.0:
        raise ValueError("Harici test dizini yoksa --test-ratio pozitif olmali.")
    if not trainval_dir.exists():
        raise FileNotFoundError(f"TrainVal veri dizini bulunamadi: {trainval_dir}")
    if test_dir is not None and not test_dir.exists():
        raise FileNotFoundError(f"Test veri dizini bulunamadi: {test_dir}")
    if full_trainval:
        # Full-trainval val split uretmez; degerlendirilecek harici test sart.
        if test_dir is None:
            raise ValueError(
                "Full-trainval final egitim icin harici test dizini gerekli. "
                "--test-dir verin veya once preprocess ile trainval/test ayirin."
            )
        if test_dir.resolve() == trainval_dir.resolve():
            raise ValueError(
                "Full-trainval final egitim icin test dizini trainval'den farkli olmali."
            )


def _selection_mode_for_metric(metric: str) -> Literal["minimize", "maximize"]:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Gecersiz selection metric: {metric}")
    return "minimize" if metric == "loss" else "maximize"


def _is_improved(
    candidate_value: float,
    best_value: float | None,
    mode: Literal["minimize", "maximize"],
) -> bool:
    import math
    if not math.isfinite(candidate_value):
        return False
    if best_value is None or not math.isfinite(best_value):
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


def _checkpoint_payload(
    config: TrainingConfig,
    epoch: int,
    selection_metric: str,
    selection_value: float | None,
    selection_source: Literal["val", "train"],
    val_loss: float | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> dict[str, Any]:
    """Checkpoint sozlugu olustur.

    ``val_loss`` yalnizca validation seti varken doldurulur. ``selection_value``
    secim icin kullanilan metrigin (``selection_metric``) o epoch'taki degerini
    saklar; ``selection_source`` bu sinyalin val'dan mi yoksa full-trainval'da
    train'den mi geldigini belirtir.
    """
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "selection_source": selection_source,
        "model_name": config.model,
        "pretrained": config.pretrained,
        "num_classes": num_classes,
        "image_size": config.image_size,
        "class_names": SINIF_ISIMLERI,
        "training_config": config_to_dict(config),
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
    }


def run_training(
    config: TrainingConfig,
    *,
    output_root: Path | None = None,
    artifact_tag: str | None = None,
    save_artifacts: bool = True,
    evaluate_test_set: bool = True,
    full_trainval: bool = False,
    verbose: bool = True,
    selection_metric: str = "loss",
    on_epoch_end: Callable[[int, dict[str, float], dict[str, float]], None] | None = None,
    extra_report: dict[str, Any] | None = None,
    preloaded_loaders: tuple[Any, Any, Any, dict[str, Any]] | None = None,
    deterministic: bool = True,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Run a single training experiment and optionally persist artifacts.

    ``preloaded_loaders`` parametresi K-fold CV orkestratoru icin kullanilir;
    verildiginde (train_loader, val_loader, test_loader, info) tuple'i icteki
    DataLoader olusturma adimini atlatir. Bu sayede CV her fold icin run_training
    govdesini tek bir kod yolundan gecirebilir.
    """
    selection_mode = _selection_mode_for_metric(selection_metric)
    if full_trainval and not evaluate_test_set:
        # Full-trainval val split uretmez; degerlendirilecek harici test
        # kapatilirsa egitim sinyali kalmaz. Kombinasyonu basta reddet.
        raise ValueError(
            "full_trainval=True yalnizca harici test ile anlamli; "
            "evaluate_test_set=True olmali."
        )
    validate_training_config(
        config,
        require_test_dir=evaluate_test_set,
        full_trainval=full_trainval,
    )

    set_seed(config.seed)
    configure_torch_runtime(deterministic=deterministic, allow_tf32=True)
    device = get_device(verbose=verbose)
    resolved_use_amp = bool(use_amp) if use_amp is not None else (device.type == "cuda")
    trainval_dir, test_dir = resolve_data_dirs(config)
    if not evaluate_test_set:
        test_dir = None

    if verbose:
        print("\n[INFO] Veri yukleniyor:")
        print(f"  Veri dizini     : {trainval_dir}")
        print(f"  Test dizini     : {test_dir}")
        if full_trainval:
            print("  Mod             : full-trainval final egitim")
        elif preloaded_loaders is not None:
            print("  Mod             : K-fold CV (fold dataloader'lari icten enjekte edildi)")
        else:
            print(f"  Val orani       : {config.val_ratio}")
        print(f"  Test orani      : {config.test_ratio}")

    # Pretrained ResNet ImageNet stats ile uyumlu; from-scratch egitimde
    # train split'ten hesaplanan dataset-bazli mean/std hizli yakinsama saglar.
    use_dataset_stats = not config.pretrained

    if preloaded_loaders is not None:
        if full_trainval:
            raise ValueError("preloaded_loaders ile full_trainval birlikte kullanilamaz.")
        train_loader, val_loader, test_loader, info = preloaded_loaders
    elif full_trainval:
        if test_dir is None:
            raise ValueError(
                "Full-trainval final egitim icin harici test dizini gerekli."
            )
        train_loader, test_loader, info = create_full_train_test_loaders(
            trainval_dir=trainval_dir,
            test_dir=test_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            seed=config.seed,
            num_workers=config.num_workers,
            hflip_p=config.hflip_p,
            rotation_degrees=config.rotation_degrees,
            color_jitter=config.color_jitter,
            use_dataset_stats=use_dataset_stats,
        )
        val_loader = None
    else:
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
            hflip_p=config.hflip_p,
            rotation_degrees=config.rotation_degrees,
            color_jitter=config.color_jitter,
            use_dataset_stats=use_dataset_stats,
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
    model = build_model(
        config.model,
        num_classes,
        device,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n[INFO] Model: {config.model.upper()}")
        print(f"  Egitilebilir parametre: {param_count:,}")

    # Class weights orijinal egitim orneklerinden hesaplanir; aug-kopyalari
    # post-augmentasyon dagilimi yansittigi icin loss'u cifte duzeltebilir.
    if "train_original_labels" in info:
        class_weight_labels = info["train_original_labels"]
        if class_weight_labels is None or len(class_weight_labels) == 0:
            raise RuntimeError(
                "Original-only train etiketleri bos; class weight hesaplamak icin "
                "train split'inde en az bir original ornek bulunmali."
            )
    else:
        class_weight_labels = info["train_labels"]
    class_weights_cpu = compute_class_weights(class_weight_labels, num_classes)
    # MPS backend'inde CrossEntropyLoss(weight=...) float32 gerektiriyor;
    # bfloat16/float64 ya da device uyumsuzlugu NaN loss uretiyor.
    # Criterion her zaman CPU float32 weight ile kurulup sonra device'a tasiniyor.
    class_weights_cpu = class_weights_cpu.to(dtype=torch.float32)
    class_weights = class_weights_cpu.to(device)
    if verbose:
        import math as _math
        weights_ok = all(_math.isfinite(w) for w in class_weights_cpu.tolist())
        print(f"  Class weights  : {[round(w, 4) for w in class_weights_cpu.tolist()]} finite={weights_ok}")
    if config.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )

    if verbose:
        if config.loss == "focal":
            print(f"  Loss: FOCAL (gamma={config.focal_gamma:.3f}, class weights aktif)")
        else:
            print(
                "  Loss: CE (class weights aktif, "
                f"label_smoothing={config.label_smoothing:.3f})"
            )

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

    # GradScaler bf16'da gereksiz; fp16 secimi runtime'da yapildigi icin
    # CUDA fp16 yolu icin scaler'i burada olusturup epoch dongusu boyunca paylas.
    # Yeni torch.amp API'si onerilen; eski torch.cuda.amp surumunde fallback.
    scaler = None
    if resolved_use_amp and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    if verbose and resolved_use_amp and device.type == "cuda":
        precision_label = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        print(f"  AMP aktif: autocast dtype={precision_label}")

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
    train_precisions: list[float] = []
    val_precisions: list[float] = []
    train_recalls: list[float] = []
    val_recalls: list[float] = []
    train_f1s: list[float] = []
    val_f1s: list[float] = []
    lowest_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_state_dict: dict[str, Any] | None = None
    best_selection_value: float | None = None
    selected_epoch_val_loss: float | None = None
    selected_epoch_train_loss: float | None = None
    best_val_eval_cache: dict[str, Any] | None = None

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            "EGITIM BASLIYOR - "
            f"{config.epochs} epoch, batch={config.batch_size}, lr={config.lr}"
        )
        print(f"{'=' * 70}\n")

    epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=resolved_use_amp,
            scaler=scaler,
        )
        train_scalars = _scalar_metrics(train_metrics)

        train_losses.append(train_scalars["loss"])
        train_accs.append(train_scalars["accuracy"])
        train_precisions.append(train_scalars["precision"])
        train_recalls.append(train_scalars["recall"])
        train_f1s.append(train_scalars["f1"])

        lr_current = optimizer.param_groups[0]["lr"]
        if full_trainval:
            if verbose:
                print(
                    f"Epoch {epoch:3d}/{config.epochs} | "
                    f"Train Loss: {train_scalars['loss']:.4f} Acc: {train_scalars['accuracy']:.4f} "
                    f"F1: {train_scalars['f1']:.4f} | LR: {lr_current:.2e}"
                )
            scheduler.step(train_scalars["loss"])
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_scalars, {})
            best_train_metrics = dict(train_scalars)
            best_epoch = epoch
            selected_epoch_train_loss = train_scalars["loss"]
            continue

        val_metrics = evaluate(
            model, val_loader, criterion, device, use_amp=resolved_use_amp
        )
        val_scalars = _scalar_metrics(val_metrics)
        val_losses.append(val_scalars["loss"])
        val_accs.append(val_scalars["accuracy"])
        val_precisions.append(val_scalars["precision"])
        val_recalls.append(val_scalars["recall"])
        val_f1s.append(val_scalars["f1"])

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
            # CPU'da klonla: deepcopy GPU tensor'lari icin GPU'da yeni allocation
            # yapardi ve trial-ici VRAM tepe noktasini ~1x model agirligi kadar
            # sisirirdi. CPU klonu, model.load_state_dict cagrisinda otomatik
            # olarak cihaza yuklenir.
            best_state_dict = clone_state_dict_to_cpu(model.state_dict())
            selected_epoch_val_loss = val_scalars["loss"]
            best_val_eval_cache = val_metrics
            if best_checkpoint_path is not None:
                torch.save(
                    _checkpoint_payload(
                        config=config,
                        epoch=epoch,
                        selection_metric=selection_metric,
                        selection_value=current_selection_value,
                        selection_source="val",
                        val_loss=val_scalars["loss"],
                        model=model,
                        optimizer=optimizer,
                        num_classes=num_classes,
                        normalize_mean=info.get("normalize_mean"),
                        normalize_std=info.get("normalize_std"),
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

    if full_trainval:
        best_state_dict = clone_state_dict_to_cpu(model.state_dict())
        if best_train_metrics is not None:
            best_selection_value = float(best_train_metrics[selection_metric])
        if best_checkpoint_path is not None and selected_epoch_train_loss is not None:
            torch.save(
                _checkpoint_payload(
                    config=config,
                    epoch=best_epoch,
                    selection_metric=selection_metric,
                    selection_value=best_selection_value,
                    selection_source="train",
                    val_loss=None,
                    model=model,
                    optimizer=optimizer,
                    num_classes=num_classes,
                    normalize_mean=info.get("normalize_mean"),
                    normalize_std=info.get("normalize_std"),
                ),
                best_checkpoint_path,
            )
            if verbose:
                print("  [OK] Final checkpoint kaydedildi (full-trainval)")
    elif best_state_dict is None and val_losses:
        best_state_dict = clone_state_dict_to_cpu(model.state_dict())
        best_val_metrics = dict(val_scalars)
        best_epoch = epoch
        best_selection_value = float(val_scalars[selection_metric])
        selected_epoch_val_loss = val_scalars["loss"]
        if best_val_eval_cache is None:
            best_val_eval_cache = val_metrics
    elif best_state_dict is None:
        raise RuntimeError(
            "Egitim calismadi: config.epochs >= 1 olmali veya --full-trainval kullanilmali."
        )

    model.load_state_dict(best_state_dict)

    best_val_detailed_metrics = None
    if not full_trainval and best_val_eval_cache is not None:
        if {"labels", "preds"}.issubset(best_val_eval_cache):
            best_val_detailed_metrics = _build_detailed_eval_report(
                best_val_eval_cache["labels"],
                best_val_eval_cache["preds"],
                best_val_eval_cache.get("probs"),
                SINIF_ISIMLERI,
            )

    test_metrics = None
    test_detailed_metrics = None
    if evaluate_test_set:
        if verbose:
            print(f"\n{'=' * 70}")
            print("TEST DEGERLENDIRMESI")
            print(f"{'=' * 70}\n")

        test_eval = evaluate(
            model, test_loader, criterion, device, use_amp=resolved_use_amp
        )
        test_metrics = _scalar_metrics(test_eval)
        if {"labels", "preds"}.issubset(test_eval):
            test_detailed_metrics = _build_detailed_eval_report(
                test_eval["labels"],
                test_eval["preds"],
                test_eval.get("probs"),
                SINIF_ISIMLERI,
            )

        if verbose:
            print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall   : {test_metrics['recall']:.4f}")
            print(f"  F1 (macro): {test_metrics['f1']:.4f}")
            if test_detailed_metrics is not None and test_detailed_metrics["macro_auc_ovr"] is not None:
                print(f"  ROC-AUC (macro OVR): {test_detailed_metrics['macro_auc_ovr']:.4f}")
            if (
                test_detailed_metrics is not None
                and test_detailed_metrics["macro_average_precision"] is not None
            ):
                print(
                    "  Avg Precision (macro): "
                    f"{test_detailed_metrics['macro_average_precision']:.4f}"
                )

        if output_dirs is not None and {"labels", "preds"}.issubset(test_eval):
            test_confidences = np.asarray(
                test_eval.get("confidences", np.array([], dtype=np.float32)),
                dtype=np.float32,
            )
            if not test_confidences.size and np.asarray(test_eval.get("probs")).size:
                test_confidences = np.asarray(test_eval["probs"], dtype=np.float32).max(axis=1)

            plot_confusion_matrix(
                test_eval["labels"],
                test_eval["preds"],
                SINIF_ISIMLERI,
                output_dirs["visuals"] / f"confusion_matrix_{artifact_stem}.png",
            )
            plot_confusion_matrix(
                test_eval["labels"],
                test_eval["preds"],
                SINIF_ISIMLERI,
                output_dirs["visuals"] / f"confusion_matrix_normalized_{artifact_stem}.png",
                normalize=True,
            )
            plot_classification_summary(
                test_eval["labels"],
                test_eval["preds"],
                SINIF_ISIMLERI,
                output_dirs["visuals"] / f"classification_summary_{artifact_stem}.png",
            )
            plot_prediction_confidence(
                test_confidences,
                test_eval["labels"],
                test_eval["preds"],
                output_dirs["visuals"] / f"prediction_confidence_{artifact_stem}.png",
            )
            if np.asarray(test_eval.get("probs")).size:
                plot_multiclass_roc_pr_curves(
                    test_eval["labels"],
                    test_eval["probs"],
                    SINIF_ISIMLERI,
                    output_dirs["visuals"] / f"roc_pr_curves_{artifact_stem}.png",
                )

    if output_dirs is not None:
        plot_training_curves(
            train_losses,
            val_losses if not full_trainval else None,
            train_accs,
            val_accs if not full_trainval else None,
            output_dirs["visuals"] / f"training_curves_{artifact_stem}.png",
            train_precisions=train_precisions,
            val_precisions=val_precisions if not full_trainval else None,
            train_recalls=train_recalls,
            val_recalls=val_recalls if not full_trainval else None,
            train_f1s=train_f1s,
            val_f1s=val_f1s if not full_trainval else None,
            best_epoch=best_epoch,
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
            "selection_mode": "fixed_epoch_full_trainval" if full_trainval else selection_mode,
            "best_selection_value": round(best_selection_value, 6) if best_selection_value is not None else None,
            "lowest_val_loss": round(lowest_val_loss, 6) if lowest_val_loss != float("inf") else None,
            "selected_epoch_val_loss": (
                round(selected_epoch_val_loss, 6) if selected_epoch_val_loss is not None else None
            ),
            "selected_epoch_train_loss": (
                round(selected_epoch_train_loss, 6) if selected_epoch_train_loss is not None else None
            ),
            "best_val_metrics": (
                {
                    "accuracy": round(best_val_metrics["accuracy"], 4),
                    "precision": round(best_val_metrics["precision"], 4),
                    "recall": round(best_val_metrics["recall"], 4),
                    "f1_macro": round(best_val_metrics["f1"], 4),
                }
                if best_val_metrics is not None
                else None
            ),
            "best_train_metrics": (
                {
                    "accuracy": round(best_train_metrics["accuracy"], 4),
                    "precision": round(best_train_metrics["precision"], 4),
                    "recall": round(best_train_metrics["recall"], 4),
                    "f1_macro": round(best_train_metrics["f1"], 4),
                }
                if best_train_metrics is not None
                else None
            ),
            "best_val_detailed_metrics": best_val_detailed_metrics,
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
            "test_detailed_metrics": test_detailed_metrics,
            "data_split": {
                "strategy": info["split_strategy"],
                "trainval_dir": str(trainval_dir),
                "test_dir": str(test_dir) if test_dir is not None else None,
                "val_ratio": config.val_ratio if not full_trainval else None,
                "test_ratio": config.test_ratio,
                "train_size": info["train_size"],
                "val_size": info["val_size"],
                "test_size": info["test_size"],
                "trainval_grouping": info["trainval_grouping"],
                "test_grouping": info["test_grouping"],
                "warnings": info["split_warnings"],
                "full_trainval_run": full_trainval,
            },
            "config": config_to_dict(config),
            "history": {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_accuracy": train_accs,
                "val_accuracy": val_accs,
                "train_precision": train_precisions,
                "val_precision": val_precisions,
                "train_recall": train_recalls,
                "val_recall": val_recalls,
                "train_f1": train_f1s,
                "val_f1": val_f1s,
            },
            "artifacts": {
                "training_dashboard": str(
                    output_dirs["visuals"] / f"training_curves_{artifact_stem}.png"
                ),
                "confusion_matrix": (
                    str(output_dirs["visuals"] / f"confusion_matrix_{artifact_stem}.png")
                    if test_metrics is not None
                    else None
                ),
                "confusion_matrix_normalized": (
                    str(
                        output_dirs["visuals"]
                        / f"confusion_matrix_normalized_{artifact_stem}.png"
                    )
                    if test_metrics is not None
                    else None
                ),
                "classification_summary": (
                    str(output_dirs["visuals"] / f"classification_summary_{artifact_stem}.png")
                    if test_metrics is not None
                    else None
                ),
                "prediction_confidence": (
                    str(output_dirs["visuals"] / f"prediction_confidence_{artifact_stem}.png")
                    if test_metrics is not None
                    else None
                ),
                "roc_pr_curves": (
                    str(output_dirs["visuals"] / f"roc_pr_curves_{artifact_stem}.png")
                    if test_metrics is not None
                    else None
                ),
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
        "lowest_val_loss": lowest_val_loss if best_val_metrics is not None else None,
        "best_val_metrics": best_val_metrics,
        "best_train_metrics": best_train_metrics,
        "best_selection_value": best_selection_value,
        "selection_metric": selection_metric,
        "selection_mode": "fixed_epoch_full_trainval" if full_trainval else selection_mode,
        "selected_epoch_val_loss": selected_epoch_val_loss,
        "selected_epoch_train_loss": selected_epoch_train_loss,
        "test_metrics": test_metrics,
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accs,
            "val_accuracy": val_accs,
            "train_precision": train_precisions,
            "val_precision": val_precisions,
            "train_recall": train_recalls,
            "val_recall": val_recalls,
            "train_f1": train_f1s,
            "val_f1": val_f1s,
        },
        "data_info": info,
        "checkpoint_path": best_checkpoint_path,
        "report_path": report_path,
        "output_root": output_dirs["root"] if output_dirs is not None else None,
        "trainval_dir": trainval_dir,
        "test_dir": test_dir,
        "config": config_to_dict(config),
        "best_val_detailed_metrics": best_val_detailed_metrics,
        "test_detailed_metrics": test_detailed_metrics,
        "full_trainval": full_trainval,
    }


# ==================== Cross-validation ====================
#
# K-fold CV orkestratoru ayrı bir modulde tutulur (model/dl/cross_validation.py).
# Geriye donuk uyumluluk icin (hpo_dl, train.py ve testler `training_runner`
# uzerinden import ediyor) burada yeniden disa actirilir.

from .dl.cross_validation import (  # noqa: E402
    _CV_METRIC_KEYS,
    _aggregate_cv_metrics,
    _aggregate_metric_dicts,
    run_cv_training,
)
