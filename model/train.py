#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
MRI siniflandirma derin ogrenme egitim scripti.

Kullanim:
    python model/train.py --model resnet --epochs 50 --batch-size 32
    python model/train.py --model unet --epochs 50 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.training_runner import TrainingConfig, build_model, run_training
else:
    from .training_runner import TrainingConfig, build_model, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma - derin ogrenme egitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python model/train.py --model resnet --epochs 50 --batch-size 32
  python model/train.py --model unet --epochs 50 --batch-size 16
  python model/train.py --model resnet --loss focal --lr 3e-4 --focal-gamma 2.5
  python model/train.py --model resnet --weight-decay 1e-3 --scheduler-factor 0.3
  python model/train.py --model resnet --pretrained
  python model/train.py --model resnet --trainval-dir Veri_Seti/OriginalDataset
  python model/train.py --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
  python model/train.py --model unet --val-ratio 0.2
        """,
    )
    parser.add_argument(
        "--model",
        choices=["resnet", "unet"],
        default="resnet",
        help="Model tipi (varsayilan: resnet)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Ogrenme hizi")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping sabir degeri")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--trainval-dir",
        type=str,
        default=None,
        help="Train/validation/test split kaynagi veya train+val veri dizini (ham veya islenmis)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Opsiyonel harici test veri dizini (ham veya islenmis)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation orani (varsayilan: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Harici test dizini yoksa internal test orani (varsayilan: 0.15)",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="Kayip fonksiyonu: ce=CrossEntropy, focal=FocalLoss",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Sadece ResNet icin: ImageNet pretrained agirliklarini kullan",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay degeri",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau icin LR carpani",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau sabir degeri",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Loss=focal iken gamma parametresi",
    )
    parser.add_argument(
        "--full-trainval",
        action="store_true",
        help=(
            "Final model icin validation ayirmadan tum trainval uzerinde egit; "
            "test icin harici test dizini gerekir."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainingConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        image_size=args.image_size,
        trainval_dir=args.trainval_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        loss=args.loss,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        focal_gamma=args.focal_gamma,
    )

    try:
        run_training(
            config,
            artifact_tag=args.model,
            full_trainval=args.full_trainval,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
