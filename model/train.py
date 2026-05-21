#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
MRI siniflandirma derin ogrenme egitim scripti.

Kullanim:
    python model/train.py --model resnet --epochs 50 --batch-size 32
"""

from __future__ import annotations

import sys

# macOS OpenMP çakışmasını önlemek için xgboost'u torch'tan (dl modülleri) önce yüklüyoruz.
# Ancak bu işlemi sadece xgboost modeli seçildiyse yapıyoruz, aksi takdirde resnet mps üzerinde segfault alıyor.
# Yalnızca macOS: Windows'ta xgboost'u torch'tan önce yüklemek torch'un c10.dll'ini
# bozuyor (OSError WinError 1114). Windows'ta torch zaten dl.dataset üzerinden
# xgboost'tan önce yüklendiği için bu workaround'a gerek yok.
if sys.platform == "darwin" and any("xgboost" in arg for arg in sys.argv):
    try:
        import xgboost
    except ImportError:
        pass

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.ayarlar import VARSAYILAN_EARLY_STOPPING_SABIR
else:
    from .ayarlar import VARSAYILAN_EARLY_STOPPING_SABIR


def build_model(*args, **kwargs):
    if __package__ in {None, ""}:
        from model.training_runner import build_model as _impl
    else:
        from .training_runner import build_model as _impl
    return _impl(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma - derin ogrenme / sig ogrenme egitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python model/train.py --model resnet --epochs 50 --batch-size 32
  python model/train.py --model resnet --loss focal --lr 3e-4 --focal-gamma 2.5
  python model/train.py --model resnet --weight-decay 1e-3 --scheduler-factor 0.3
  python model/train.py --model resnet --pretrained
  python model/train.py --model resnet --trainval-dir Veri_Seti/OriginalDataset
  python model/train.py --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
  python model/train.py --model resnet --val-ratio 0.2
  python model/train.py --model xgboost --xgb-max-depth 6 --xgb-n-estimators 300
  python model/train.py --model resnet --folds 5
  python model/train.py --model xgboost --folds 5
        """,
    )
    parser.add_argument(
        "--model",
        choices=["resnet", "xgboost"],
        default="resnet",
        help="Model tipi (varsayilan: resnet)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Ogrenme hizi")
    parser.add_argument(
        "--patience",
        type=int,
        default=VARSAYILAN_EARLY_STOPPING_SABIR,
        help="Early stopping sabir degeri",
    )
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
        "--dropout",
        type=float,
        default=0.5,
        help="ResNet classifier head dropout orani",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="CE loss icin label smoothing (focal iken yok sayilir)",
    )
    parser.add_argument(
        "--hflip-p",
        type=float,
        default=0.0,
        help=(
            "Egitim augmentasyonunda RandomHorizontalFlip olasiligi. "
            "Beyin MR'larinda anatomik lateralite (orn. hipokampal asimetri) "
            "tani icin bilgi tasidigindan varsayilan 0.0'dir."
        ),
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=10.0,
        help="Egitim augmentasyonunda RandomRotation sinir derecesi",
    )
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.1,
        help="Egitim augmentasyonunda ColorJitter brightness/contrast siddeti",
    )
    parser.add_argument(
        "--full-trainval",
        action="store_true",
        help=(
            "Final model icin validation ayirmadan tum trainval uzerinde egit; "
            "test icin harici test dizini gerekir."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help=(
            "K-fold cross-validation icin fold sayisi (>=2). 1 ise tek hold-out "
            "egitimi yapilir (varsayilan davranis). DL ve XGBoost icin desteklenir."
        ),
    )

    # XGBoost parametreleri
    xgb_group = parser.add_argument_group("XGBoost")
    xgb_group.add_argument("--xgb-n-estimators", type=int, default=300, help="Boosting round sayisi")
    xgb_group.add_argument("--xgb-max-depth", type=int, default=6, help="Maksimum agac derinligi")
    xgb_group.add_argument("--xgb-learning-rate", type=float, default=0.1, help="XGBoost ogrenme hizi")
    xgb_group.add_argument("--xgb-subsample", type=float, default=0.8, help="Satir ornekleme orani")
    xgb_group.add_argument("--xgb-colsample-bytree", type=float, default=0.8, help="Sutun ornekleme orani")
    xgb_group.add_argument("--xgb-reg-lambda", type=float, default=1.0, help="L2 regularizasyon")
    xgb_group.add_argument("--xgb-reg-alpha", type=float, default=0.0, help="L1 regularizasyon")
    xgb_group.add_argument(
        "--xgb-gamma",
        type=float,
        default=0.0,
        help="Bir yapragin daha fazla bolunmesi icin gereken minimum loss azalmasi (min_split_loss).",
    )
    xgb_group.add_argument("--xgb-min-child-weight", type=int, default=1, help="Min child weight")
    xgb_group.add_argument(
        "--xgb-max-delta-step",
        type=int,
        default=0,
        help="XGBoost max_delta_step degeri. Dengesiz veri setleri icin 1-10 arasi onerilir."
    )
    xgb_group.add_argument("--feature-cache", type=str, default=None, help="Ozellik cache dizini (.npz)")
    xgb_group.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "XGBoost device modu (sadece --model xgboost icin). 'auto' hem "
            "torch CUDA hem XGBoost CUDA build'i mevcutsa GPU, aksi halde "
            "CPU secer. 'cuda' acik istek; CPU-only XGBoost build'inde "
            "uyari verilir ve fit asamasinda XGBoost hata atar."
        ),
    )
    xgb_group.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost icin worker thread sayisi. None ise os.cpu_count().",
    )
    xgb_group.add_argument(
        "--xgb-class-balance",
        choices=["none", "balanced"],
        default="none",
        help=(
            "Sinif dengesizligi telafisi. 'none' (varsayilan): mevcut davranis, "
            "agirlik uygulanmaz. 'balanced': sklearn compute_sample_weight ile "
            "her ornege class-frequency'ye ters orantili agirlik atanir; hem "
            "XGBoost loss'una hem eval_set/erken durdurma metriklerine gecirilir."
        ),
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.folds < 1:
        print("[HATA] --folds en az 1 olmali.")
        return 1
    if args.folds > 1 and args.full_trainval:
        print("[HATA] --folds > 1 ile --full-trainval birlikte kullanilamaz.")
        return 1

    if args.model == "xgboost":
        import warnings
        dl_arg_names = (
            "epochs", "batch_size", "lr", "patience", "loss", "pretrained",
            "weight_decay", "scheduler_factor", "scheduler_patience",
            "focal_gamma", "num_workers", "dropout", "label_smoothing",
            "hflip_p", "rotation_degrees", "color_jitter",
        )
        # Not: focal_gamma (DL) ile xgb_gamma (XGBoost min_split_loss) farkli
        # hiperparametrelerdir; ayni isim cakismasi olmasin diye XGBoost
        # gamma'si --xgb-gamma flag'i ile expose edilmistir.
        used_dl_args = [
            name for name in dl_arg_names
            if getattr(args, name) != parser.get_default(name)
        ]
        if used_dl_args:
            warnings.warn(
                f"XGBoost modunda DL'ye ozgu arguman(lar) yok sayildi: {', '.join(used_dl_args)}",
                stacklevel=1,
            )

        if __package__ in {None, ""}:
            from model.sl.training_runner import (
                SLTrainingConfig,
                run_sl_cv_training,
                run_sl_training,
            )
        else:
            from .sl.training_runner import (
                SLTrainingConfig,
                run_sl_cv_training,
                run_sl_training,
            )

        config = SLTrainingConfig(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            reg_alpha=args.xgb_reg_alpha,
            gamma=args.xgb_gamma,
            min_child_weight=args.xgb_min_child_weight,
            max_delta_step=args.xgb_max_delta_step,
            image_size=args.image_size,
            trainval_dir=args.trainval_dir,
            test_dir=args.test_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            feature_cache=args.feature_cache,
            device=args.xgb_device,
            n_jobs=args.xgb_n_jobs,
            class_balance=args.xgb_class_balance,
        )
        try:
            if args.folds > 1:
                run_sl_cv_training(
                    config,
                    n_folds=args.folds,
                    artifact_tag="xgboost",
                )
            else:
                run_sl_training(
                    config,
                    artifact_tag="xgboost",
                    full_trainval=args.full_trainval,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[HATA] {exc}")
            return 1
        return 0

    # DL (ResNet) akisi
    import warnings
    sl_arg_names = (
        "xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate",
        "xgb_subsample", "xgb_colsample_bytree", "xgb_reg_lambda",
        "xgb_reg_alpha", "xgb_gamma",
        "xgb_min_child_weight", "feature_cache", "xgb_device", "xgb_n_jobs",
        "xgb_class_balance",
    )
    used_sl_args = [
        name for name in sl_arg_names
        if getattr(args, name) != parser.get_default(name)
    ]
    if used_sl_args:
        warnings.warn(
            f"ResNet modunda XGBoost'a ozgu arguman(lar) yok sayildi: {', '.join(used_sl_args)}",
            stacklevel=1,
        )

    if __package__ in {None, ""}:
        from model.training_runner import (
            TrainingConfig,
            run_cv_training,
            run_training,
        )
    else:
        from .training_runner import (
            TrainingConfig,
            run_cv_training,
            run_training,
        )

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
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        hflip_p=args.hflip_p,
        rotation_degrees=args.rotation_degrees,
        color_jitter=args.color_jitter,
    )

    try:
        if args.folds > 1:
            run_cv_training(
                config,
                n_folds=args.folds,
                artifact_tag=args.model,
            )
        else:
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
