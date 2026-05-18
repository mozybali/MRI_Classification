#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_classifier.py
------------------
Ultralytics YOLOv8 siniflandirici wrapper.

Ultralytics YOLO nesnesi etrafinda ince bir katman saglar: model yukler,
sinif isimlerini (YOLO'nun kendi siralama duzeninden bagimsiz) olasilik
sozlugune donusturur.
"""

from pathlib import Path


class YoloClassifier:
    """Ultralytics YOLOv8-cls wrapper — siniflandirma gorevleri icin."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        self._device = device

    def predict_probs(self, image_path: Path, image_size: int = 224) -> dict[str, float]:
        """
        Goruntu uzerinde siniflandirma yap.

        Returns:
            {sinif_adi: olasilik} sozlugu — YOLO'nun ic indeks siralamasindan
            bagimsiz olarak sinif adi anahtarlidir.
        """
        results = self._model.predict(
            str(image_path), verbose=False, device=self._device, imgsz=image_size
        )
        r = results[0]
        probs = r.probs.data.cpu().numpy()
        # r.names: {0: 'ClassName', 1: 'ClassName', ...} YOLO'nun egitim sirasi
        return {r.names[i]: float(probs[i]) for i in range(len(probs))}
