#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_inference.py
-----------------
YOLOv8 siniflandirma inference fonksiyonlari.

ResNet (model.inference) ve XGBoost ile ayni cikti formatini uretir:
    {dosya, tahmin_sinif, tahmin_adi, guven_skoru, olasiliklar}

YOLO egitilirken sinif siralamasini alfabetik belirler; bu modul sinif
adi uzerinden eslestirme yaparak kanonik SINIF_ISIMLERI siralamasini
her zaman korur.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.dl.dataset import SINIF_ISIMLERI
    from model.dl.models.yolo_classifier import YoloClassifier
else:
    from .dl.dataset import SINIF_ISIMLERI
    from .dl.models.yolo_classifier import YoloClassifier


def _meta_path(model_path: Path) -> Path:
    """Model yolundan sidecar meta dosya yolunu hesapla."""
    return model_path.parent / (model_path.stem + ".yolo.meta.json")


def load_yolo_model(model_path: Path) -> tuple[YoloClassifier, int, list[str]]:
    """
    YOLO modelini ve sidecar meta dosyasini yukle.

    Returns:
        (YoloClassifier, image_size, class_names)
    """
    meta = _meta_path(model_path)
    if not meta.exists():
        raise FileNotFoundError(
            f"YOLO sidecar meta dosyasi bulunamadi: {meta}\n"
            "mri-yolo-train ile egitilen model bekleniyor."
        )

    with open(meta, encoding="utf-8") as fh:
        raw = json.load(fh)

    image_size = int(raw.get("image_size", 224))
    class_names = raw.get("class_names", SINIF_ISIMLERI)

    from model.dl.utils import get_device
    device = str(get_device(verbose=False))
    model = YoloClassifier(model_path, device=device)
    return model, image_size, class_names


def predict_image_yolo(
    model: YoloClassifier,
    image_path: Path,
    image_size: int,
    class_names: list[str],
    *,
    apply_mri_preprocessing: bool = False,
) -> dict:
    """
    Tek bir goruntu icin YOLOv8 siniflandirma tahmin yap.

    Cikti formati ResNet / XGBoost ile ayni:
        {dosya, tahmin_sinif, tahmin_adi, guven_skoru, olasiliklar}
    """
    tmp_path: Path | None = None

    try:
        if apply_mri_preprocessing:
            from goruntu_isleme.goruntu_isleyici import GorselIsleyici
            from PIL import Image as PilImage

            isleyici = GorselIsleyici()
            processed = isleyici.goruntu_isle(str(image_path))
            if processed is None:
                raise RuntimeError(
                    f"MRI on isleme basarisiz (kalite kontrol veya yukleme hatasi): {image_path}"
                )
            img = PilImage.fromarray(processed).convert("RGB")
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            img.save(tmp)
            tmp_path = Path(tmp)
            inference_path = tmp_path
        else:
            inference_path = image_path

        # {sinif_adi: olasilik} — YOLO ic siralamasindan bagimsiz
        name_to_prob: dict[str, float] = model.predict_probs(inference_path, image_size)

    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    # Kanonik class_names siralamasina gore cikti uret
    olasiliklar = {name: name_to_prob.get(name, 0.0) for name in class_names}
    pred_name = max(olasiliklar, key=olasiliklar.get)  # type: ignore[arg-type]
    pred_idx = class_names.index(pred_name)

    return {
        "dosya": str(image_path),
        "tahmin_sinif": pred_idx,
        "tahmin_adi": pred_name,
        "guven_skoru": olasiliklar[pred_name],
        "olasiliklar": olasiliklar,
    }
