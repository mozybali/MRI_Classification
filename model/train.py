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

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.ayarlar import (
        MODELS_KLASORU,
        RAPORLAR_KLASORU,
        GORSELLER_KLASORU,
        TRAINVAL_VERI_DIZINI,
        TEST_VERI_DIZINI,
        RASTGELE_TOHUM,
    )
    from model.dl.dataset import create_dataloaders, SINIF_ISIMLERI
    from model.dl.engine import train_one_epoch, evaluate, EarlyStopping
    from model.dl.losses import FocalLoss, compute_class_weights
    from model.dl.models.resnet_classifier import ResNetClassifier
    from model.dl.models.unet_classifier import UNetClassifier
    from model.dl.utils import (
        set_seed,
        get_device,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
    )
else:
    from .ayarlar import (
        MODELS_KLASORU,
        RAPORLAR_KLASORU,
        GORSELLER_KLASORU,
        TRAINVAL_VERI_DIZINI,
        TEST_VERI_DIZINI,
        RASTGELE_TOHUM,
    )
    from .dl.dataset import create_dataloaders, SINIF_ISIMLERI
    from .dl.engine import train_one_epoch, evaluate, EarlyStopping
    from .dl.losses import FocalLoss, compute_class_weights
    from .dl.models.resnet_classifier import ResNetClassifier
    from .dl.models.unet_classifier import UNetClassifier
    from .dl.utils import (
        set_seed,
        get_device,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
    )


def build_model(
    name: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Model adina gore model nesnesi olustur."""
    if name == "resnet":
        model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif name == "unet":
        model = UNetClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma - derin ogrenme egitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python model/train.py --model resnet --epochs 50 --batch-size 32
  python model/train.py --model unet --epochs 50 --batch-size 16
  python model/train.py --model resnet --loss focal --lr 3e-4
  python model/train.py --model resnet --pretrained
  python model/train.py --model resnet --trainval-dir Veri_Seti/AugmentedAlzheimerDataset --test-dir Veri_Seti/OriginalDataset
  python model/train.py --model unet --val-ratio 0.2
        """,
    )
    parser.add_argument("--model", choices=["resnet", "unet"], default="resnet",
                        help="Model tipi (varsayilan: resnet)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Ogrenme hizi")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping sabir degeri")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trainval-dir", type=str, default=None,
                        help="Train+Val icin augmented veri dizini")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Test icin original veri dizini")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation orani (varsayilan: 0.15)")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce",
                        help="Kayip fonksiyonu: ce=CrossEntropy, focal=FocalLoss")
    parser.add_argument("--seed", type=int, default=RASTGELE_TOHUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Sadece ResNet icin: ImageNet pretrained agirliklarini kullan",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        print("[HATA] --epochs en az 1 olmali.")
        return 1

    set_seed(args.seed)
    device = get_device()

    trainval_dir = Path(args.trainval_dir) if args.trainval_dir else TRAINVAL_VERI_DIZINI
    test_dir = Path(args.test_dir) if args.test_dir else TEST_VERI_DIZINI
    if not trainval_dir.exists():
        print(f"[HATA] TrainVal veri dizini bulunamadi: {trainval_dir}")
        return 1
    if not test_dir.exists():
        print(f"[HATA] Test veri dizini bulunamadi: {test_dir}")
        return 1

    for d in [MODELS_KLASORU, RAPORLAR_KLASORU, GORSELLER_KLASORU]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Veri yukleniyor:")
    print(f"  TrainVal dizini : {trainval_dir}")
    print(f"  Test dizini     : {test_dir}")
    print(f"  Val orani       : {args.val_ratio}")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"\n  Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
    print(
        f"  Grup: Train={info['train_groups']}, "
        f"Val={info['val_groups']}"
    )
    if info["split_warnings"]:
        print("  [UYARI] Leak-free split sinif kapsami tam degil:")
        for warning in info["split_warnings"]:
            print(f"    - {warning}")

    num_classes = info["num_classes"]
    model = build_model(args.model, num_classes, device, pretrained=args.pretrained)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model: {args.model.upper()}")
    print(f"  Egitilebilir parametre: {param_count:,}")

    class_weights = compute_class_weights(info["train_labels"], num_classes).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Loss: {args.loss.upper()} (class weights aktif)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    early_stopping = EarlyStopping(patience=args.patience)

    best_val_loss = float("inf")
    best_checkpoint_path = MODELS_KLASORU / f"best_{args.model}.pt"
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n{'='*70}")
    print(f"EGITIM BASLIYOR - {args.epochs} epoch, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*70}\n")

    epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} | LR: {lr_current:.2e}"
        )

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_name": args.model,
                    "pretrained": args.pretrained,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "class_names": SINIF_ISIMLERI,
                },
                best_checkpoint_path,
            )
            print(f"  [OK] Best checkpoint kaydedildi (val_loss: {best_val_loss:.4f})")

        if early_stopping(val_metrics["loss"]):
            print(f"\n[INFO] Early stopping: {args.patience} epoch boyunca iyilesme olmadi.")
            break

    print(f"\n{'='*70}")
    print("TEST DEGERLENDIRMESI")
    print(f"{'='*70}\n")

    checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1']:.4f}")

    plot_confusion_matrix(
        test_metrics["labels"],
        test_metrics["preds"],
        SINIF_ISIMLERI,
        GORSELLER_KLASORU / f"confusion_matrix_{args.model}.png",
    )

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        GORSELLER_KLASORU / f"training_curves_{args.model}.png",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "model": args.model,
        "timestamp": timestamp,
        "epochs_trained": epoch,
        "pretrained": args.pretrained,
        "best_val_loss": round(best_val_loss, 6),
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
            "f1_macro": round(test_metrics["f1"], 4),
        },
        "data_split": {
            "strategy": "augmented_trainval_original_test",
            "trainval_dir": str(trainval_dir),
            "test_dir": str(test_dir),
            "val_ratio": args.val_ratio,
            "train_size": info["train_size"],
            "val_size": info["val_size"],
            "test_size": info["test_size"],
        },
        "args": vars(args),
    }
    report_path = RAPORLAR_KLASORU / f"rapor_{args.model}_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Rapor kaydedildi: {report_path}")
    print(f"[OK] Model kaydedildi: {best_checkpoint_path}")
    print(f"\n{'='*70}")
    print("EGITIM TAMAMLANDI")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
