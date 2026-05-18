#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_trainer.py
---------------
YOLOv8 siniflandirma egitimi.

mri-train (ResNet) ve mri-xgb-train'den tamamen bagimsiz; mevcut
pipeline'a dokunmaz. Egitilen model mevcut 'modeller/' dizinine
kaydedilir ve yani sira bir sidecar .yolo.meta.json dosyasi yazilir.

Kullanim:
    mri-yolo-train
    mri-yolo-train --backbone yolov8s-cls.pt --epochs 100 --image-size 224
    mri-yolo-train --output-name demans_yolo --trainval-dir /custom/trainval --test-dir /custom/test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.ayarlar import (
        MODELS_KLASORU,
        RAPORLAR_KLASORU,
        RASTGELE_TOHUM,
        TEST_VERI_DIZINI,
        TRAINVAL_VERI_DIZINI,
    )
    from model.dl.dataset import SINIF_ISIMLERI
else:
    from .ayarlar import (
        MODELS_KLASORU,
        RAPORLAR_KLASORU,
        RASTGELE_TOHUM,
        TEST_VERI_DIZINI,
        TRAINVAL_VERI_DIZINI,
    )
    from .dl.dataset import SINIF_ISIMLERI


# Desteklenen YOLOv8 siniflandirma backbone'lari (kucukten buyuge)
YOLO_BACKBONES = ("yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt")



def run_yolo_training(
    *,
    backbone: str = "yolov8n-cls.pt",
    epochs: int = 50,
    batch_size: int = 32,
    image_size: int = 224,
    trainval_dir: Path | None = None,
    test_dir: Path | None = None,
    output_name: str = "yolo_cls",
    seed: int = RASTGELE_TOHUM,
    workers: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    YOLOv8 siniflandirma egitimini calistir ve model + meta dosyalarini kaydet.

    Returns:
        {model_path, meta_path, metrics, report_path}
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics paketi bulunamadi. Kurmak icin: pip install ultralytics"
        ) from exc

    _trainval = Path(trainval_dir) if trainval_dir else TRAINVAL_VERI_DIZINI
    _test = Path(test_dir) if test_dir else TEST_VERI_DIZINI

    if not _trainval.exists():
        raise FileNotFoundError(f"Trainval dizini bulunamadi: {_trainval}")
    if not _test.exists():
        raise FileNotFoundError(f"Test dizini bulunamadi: {_test}")

    MODELS_KLASORU.mkdir(parents=True, exist_ok=True)
    RAPORLAR_KLASORU.mkdir(parents=True, exist_ok=True)

    # Trainval ve test dizinleri ayni kokten gelmeyebilir;
    # YAML sadece trainval'in parent'ini referans alir, test dizini
    # ayri bir konumdaysa mutlak yol kullanilmali.
    # En guvenlisi: tmp dizinine her ikisini de sembolik linkle bagla.
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        # Sembolik linkler olustur (train/ ve val/ isimleriyle)
        train_link = tmp_dir / "train"
        val_link = tmp_dir / "val"
        train_link.symlink_to(_trainval.resolve())
        val_link.symlink_to(_test.resolve())

        # Ultralytics classification: data = train/ ve val/ iceren ust dizin
        run_dir = tmp_dir / "runs"
        run_dir.mkdir()

        if device:
            ul_device = device
        else:
            import torch
            if torch.backends.mps.is_available():
                ul_device = "mps"
            elif torch.cuda.is_available():
                ul_device = "cuda"
            else:
                ul_device = "cpu"

        if verbose:
            print(f"\n[INFO] YOLO egitimi basliyor")
            print(f"  Backbone  : {backbone}")
            print(f"  Epochs    : {epochs}")
            print(f"  Batch     : {batch_size}")
            print(f"  Image     : {image_size}x{image_size}")
            print(f"  Trainval  : {_trainval}")
            print(f"  Test      : {_test}")
            print(f"  Device    : {ul_device}")

        ul_model = YOLO(backbone)
        results = ul_model.train(
            data=str(tmp_dir),  # tmp_dir icinde train/ ve val/ linkleri var
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=ul_device,
            seed=seed,
            workers=workers,
            project=str(run_dir),
            name="train",
            exist_ok=True,
            verbose=verbose,
        )

        # Egitilmis en iyi agirligi al
        weights_dir = run_dir / "train" / "weights"
        best_weights = weights_dir / "best.pt"
        if not best_weights.exists():
            # Bazi versiyonlarda last.pt ile biter
            last_weights = weights_dir / "last.pt"
            if last_weights.exists():
                best_weights = last_weights
            else:
                raise RuntimeError(
                    f"Egitim tamamlandi ancak agirlik dosyasi bulunamadi: {weights_dir}"
                )

        # Hedef yol
        dest_model = MODELS_KLASORU / f"best_{output_name}.pt"
        shutil.copy2(best_weights, dest_model)

        # Sidecar meta dosyasi
        meta = {
            "model_type": "yolo",
            "backbone": backbone,
            "image_size": image_size,
            "class_names": SINIF_ISIMLERI,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "trainval_dir": str(_trainval),
            "test_dir": str(_test),
            "trained_at": datetime.now().isoformat(),
        }
        # Metrics varsa ekle
        if results is not None:
            try:
                results_dict = results.results_dict
                meta["metrics"] = {k: float(v) for k, v in results_dict.items()}
            except (AttributeError, TypeError):
                pass

        dest_meta = MODELS_KLASORU / f"best_{output_name}.yolo.meta.json"
        with open(dest_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        # Egitim raporu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RAPORLAR_KLASORU / f"rapor_yolo_{output_name}_{timestamp}.json"
        report = {
            "model": "yolo",
            "backbone": backbone,
            "timestamp": timestamp,
            "epochs_trained": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "trainval_dir": str(_trainval),
            "test_dir": str(_test),
            "model_path": str(dest_model),
            "meta_path": str(dest_meta),
            "metrics": meta.get("metrics"),
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n[OK] Model kaydedildi  : {dest_model}")
        print(f"[OK] Meta kaydedildi   : {dest_meta}")
        print(f"[OK] Rapor kaydedildi  : {report_path}")

    return {
        "model_path": dest_model,
        "meta_path": dest_meta,
        "report_path": report_path,
        "metrics": meta.get("metrics"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="YOLOv8 MRI siniflandirma egitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  mri-yolo-train
  mri-yolo-train --backbone yolov8s-cls.pt --epochs 100
  mri-yolo-train --output-name demans_v2 --image-size 224
        """,
    )
    parser.add_argument(
        "--backbone",
        default="yolov8n-cls.pt",
        choices=YOLO_BACKBONES,
        help="YOLOv8 backbone (varsayilan: yolov8n-cls.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epoch sayisi (varsayilan: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch boyutu (varsayilan: 32)")
    parser.add_argument("--image-size", type=int, default=224, help="Goruntu boyutu (varsayilan: 224)")
    parser.add_argument("--trainval-dir", type=str, default=None, help="Trainval dizini (varsayilan: ayarlar.py)")
    parser.add_argument("--test-dir", type=str, default=None, help="Test dizini (varsayilan: ayarlar.py)")
    parser.add_argument("--output-name", type=str, default="yolo_cls", help="Cikti model adi (varsayilan: yolo_cls)")
    parser.add_argument("--seed", type=int, default=RASTGELE_TOHUM, help=f"Rastgele tohum (varsayilan: {RASTGELE_TOHUM})")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker sayisi (varsayilan: 0)")
    parser.add_argument("--device", type=str, default=None, help="Cihaz: cpu, cuda, mps (varsayilan: otomatik)")
    args = parser.parse_args(argv)

    try:
        run_yolo_training(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            trainval_dir=Path(args.trainval_dir) if args.trainval_dir else None,
            test_dir=Path(args.test_dir) if args.test_dir else None,
            output_name=args.output_name,
            seed=args.seed,
            workers=args.workers,
            device=args.device,
            verbose=True,
        )
    except (RuntimeError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
