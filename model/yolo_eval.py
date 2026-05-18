#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_eval.py
------------
YOLOv8 siniflandirma modeli icin test/val seti degerlendirmesi.

Test setindeki her sinif klasorunu gezer, tahmin yapar, ResNet raporlariyla
ayni format ve gorsel tarzinda metrik + confusion matrix uretir.

Kullanim:
    mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt
    mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt --split val
    mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt --data-dir goruntu_isleme/cikti/test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.ayarlar import (
        GORSELLER_KLASORU,
        RAPORLAR_KLASORU,
        TEST_VERI_DIZINI,
        TRAINVAL_VERI_DIZINI,
    )
    from model.dl.dataset import GORUNTU_UZANTILARI, SINIF_ISIMLERI
    from model.dl.utils import plot_classification_summary, plot_confusion_matrix
    from model.common.evaluation import build_detailed_eval_report
    from model.yolo_inference import load_yolo_model, predict_image_yolo
else:
    from .ayarlar import (
        GORSELLER_KLASORU,
        RAPORLAR_KLASORU,
        TEST_VERI_DIZINI,
        TRAINVAL_VERI_DIZINI,
    )
    from .dl.dataset import GORUNTU_UZANTILARI, SINIF_ISIMLERI
    from .dl.utils import plot_classification_summary, plot_confusion_matrix
    from .common.evaluation import build_detailed_eval_report
    from .yolo_inference import load_yolo_model, predict_image_yolo


def _collect_images(data_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    """
    Sinif klasorlerinden goruntu yol + etiket cifti topla.
    Veri dizini yapisi: data_dir/sinif_adi/goruntu.jpg
    """
    samples: list[tuple[Path, int]] = []
    for idx, name in enumerate(class_names):
        cls_dir = data_dir / name
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if p.is_file() and p.suffix.lower() in GORUNTU_UZANTILARI:
                samples.append((p, idx))
    return samples


def run_eval(
    model_path: Path,
    data_dir: Path,
    *,
    verbose: bool = True,
    save_artifacts: bool = True,
) -> dict:
    """
    YOLO modelini data_dir uzerinde degerlendir.

    Returns:
        {accuracy, per_class, macro_auc_ovr, macro_average_precision,
         confusion_matrix_path, report_path}
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri dizini bulunamadi: {data_dir}")

    model, image_size, class_names = load_yolo_model(model_path)

    samples = _collect_images(data_dir, class_names)
    if not samples:
        raise RuntimeError(
            f"Goruntu bulunamadi: {data_dir}\n"
            "Dizin yapisi bekleniyor: data_dir/sinif_adi/goruntu.jpg"
        )

    if verbose:
        print(f"\n[INFO] YOLO Degerlendirme")
        print(f"  Model     : {model_path.name}")
        print(f"  Veri      : {data_dir}")
        print(f"  Goruntu   : {len(samples)}")
        print(f"  Siniflar  : {class_names}")
        print()

    labels: list[int] = []
    preds: list[int] = []
    probs_list: list[list[float]] = []
    failures: list[tuple[Path, str]] = []

    for i, (img_path, true_idx) in enumerate(samples, 1):
        try:
            result = predict_image_yolo(model, img_path, image_size, class_names)
        except Exception as exc:
            failures.append((img_path, str(exc)))
            continue

        labels.append(true_idx)
        preds.append(result["tahmin_sinif"])
        probs_list.append([result["olasiliklar"][name] for name in class_names])

        if verbose and i % 200 == 0:
            done = len(labels)
            acc_so_far = sum(l == p for l, p in zip(labels, preds)) / done
            print(f"  [{done}/{len(samples)}] gecici accuracy: {acc_so_far:.2%}")

    labels_arr = np.array(labels, dtype=int)
    preds_arr = np.array(preds, dtype=int)
    probs_arr = np.array(probs_list, dtype=np.float32)

    accuracy = float((labels_arr == preds_arr).mean())
    detailed = build_detailed_eval_report(labels_arr, preds_arr, probs_arr, class_names)

    if verbose:
        print(f"\n{'=' * 60}")
        print("DEGERLENDIRME SONUCLARI")
        print(f"{'=' * 60}")
        print(f"  Goruntu sayisi  : {len(labels)}")
        print(f"  Accuracy        : {accuracy:.4f}  ({accuracy:.2%})")
        if detailed.get("macro_auc_ovr") is not None:
            print(f"  ROC-AUC (macro) : {detailed['macro_auc_ovr']:.4f}")
        if detailed.get("macro_average_precision") is not None:
            print(f"  Avg Precision   : {detailed['macro_average_precision']:.4f}")
        print(f"\n  Sinif bazli:")
        for cls_name, m in detailed["per_class"].items():
            print(
                f"    {cls_name:22s}  "
                f"P:{m['precision']:.3f}  R:{m['recall']:.3f}  "
                f"F1:{m['f1']:.3f}  n={m['support']}"
            )
        if failures:
            print(f"\n  [UYARI] {len(failures)} goruntu islenemedi.")
        print(f"{'=' * 60}\n")

    cm_path = None
    cm_norm_path = None
    cls_summary_path = None
    report_path = None

    if save_artifacts:
        GORSELLER_KLASORU.mkdir(parents=True, exist_ok=True)
        RAPORLAR_KLASORU.mkdir(parents=True, exist_ok=True)

        stem = model_path.stem  # ornek: best_yolo_cls

        cm_path = GORSELLER_KLASORU / f"confusion_matrix_{stem}.png"
        plot_confusion_matrix(labels_arr, preds_arr, class_names, cm_path)

        cm_norm_path = GORSELLER_KLASORU / f"confusion_matrix_normalized_{stem}.png"
        plot_confusion_matrix(labels_arr, preds_arr, class_names, cm_norm_path, normalize=True)

        cls_summary_path = GORSELLER_KLASORU / f"classification_summary_{stem}.png"
        plot_classification_summary(labels_arr, preds_arr, class_names, cls_summary_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RAPORLAR_KLASORU / f"eval_yolo_{stem}_{timestamp}.json"
        report = {
            "model": stem,
            "timestamp": timestamp,
            "data_dir": str(data_dir),
            "total_images": len(labels),
            "failed_images": len(failures),
            "accuracy": round(accuracy, 4),
            "macro_auc_ovr": detailed.get("macro_auc_ovr"),
            "macro_average_precision": detailed.get("macro_average_precision"),
            "per_class": detailed["per_class"],
            "confidence": detailed.get("confidence"),
            "artifacts": {
                "confusion_matrix": str(cm_path),
                "confusion_matrix_normalized": str(cm_norm_path),
                "classification_summary": str(cls_summary_path),
            },
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        if verbose:
            print(f"[OK] Confusion matrix     : {cm_path}")
            print(f"[OK] Norm. confusion matrix: {cm_norm_path}")
            print(f"[OK] Classification summary: {cls_summary_path}")
            print(f"[OK] Rapor                : {report_path}")

    return {
        "accuracy": accuracy,
        "detailed": detailed,
        "confusion_matrix_path": cm_path,
        "report_path": report_path,
        "total": len(labels),
        "failed": len(failures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="YOLOv8 MRI modeli degerlendirme (confusion matrix + metrikler)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt
  mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt --split val
  mri-yolo-eval --model-path model/ciktilar/modeller/best_yolo_cls.pt --data-dir /custom/test
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="YOLO model dosya yolu (.pt)",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split",
        choices=["test", "val", "trainval"],
        default="test",
        help="Hangi split kullanilsin (varsayilan: test)",
    )
    split_group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Ozel veri dizini (--split ile birlikte kullanilmaz)",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.split == "test":
        data_dir = TEST_VERI_DIZINI
    else:
        data_dir = TRAINVAL_VERI_DIZINI

    try:
        run_eval(model_path, data_dir, verbose=True, save_artifacts=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
