"""
model_loader.py
---------------
Uygulama başlarken ML modellerini bir kez yükleyen singleton servisi.
Her Django request'inde model dosyasını yeniden açmaz — bellekte tutar.
"""

import threading
from pathlib import Path
from typing import Any

from django.conf import settings


class _ModelRegistry:
    """Thread-safe singleton model deposu."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}          # key → yüklenmiş model nesnesi
        self._meta:  dict[str, dict] = {}          # key → {image_size, class_names, …}

    def _make_key(self, model_path: Path) -> str:
        return str(model_path.resolve())

    # ── ResNet ──────────────────────────────────────────────────────────────
    def get_resnet(self, model_path: Path):
        """ResNet checkpoint'ini yükle (ilk çağrıda) ya da önbellekten döndür."""
        key = self._make_key(model_path)
        if key in self._cache:
            return self._cache[key], self._meta[key]

        with self._lock:
            if key in self._cache:                # double-check after acquiring lock
                return self._cache[key], self._meta[key]

            try:
                import torch
                from model.dl.utils import get_device, load_checkpoint
                from model.training_runner import build_model
                from model.dl.dataset import IMAGENET_MEAN, IMAGENET_STD, SINIF_ISIMLERI

                device = get_device()
                checkpoint = load_checkpoint(model_path, map_location=device)
                model_name = checkpoint.get("model_name")
                if not model_name:
                    raise ValueError("Checkpoint 'model_name' eksik.")

                num_classes = checkpoint.get("num_classes", 4)
                model = build_model(model_name, num_classes, device, pretrained=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                meta = {
                    "image_size":  checkpoint.get("image_size", 224),
                    "class_names": checkpoint.get("class_names", SINIF_ISIMLERI),
                    "mean": tuple(checkpoint.get("normalize_mean") or IMAGENET_MEAN),
                    "std":  tuple(checkpoint.get("normalize_std")  or IMAGENET_STD),
                    "device": device,
                }
                self._cache[key] = model
                self._meta[key]  = meta
            except Exception as exc:
                raise RuntimeError(f"ResNet model yüklenemedi ({model_path.name}): {exc}") from exc

        return self._cache[key], self._meta[key]

    # ── YOLO ────────────────────────────────────────────────────────────────
    def get_yolo(self, model_path: Path):
        """YOLOv8 modelini yükle (ilk çağrıda) ya da önbellekten döndür."""
        key = self._make_key(model_path)
        if key in self._cache:
            return self._cache[key], self._meta[key]

        with self._lock:
            if key in self._cache:
                return self._cache[key], self._meta[key]

            try:
                from model.yolo_inference import load_yolo_model

                yolo_model, image_size, class_names = load_yolo_model(model_path)
                meta = {
                    "image_size": image_size,
                    "class_names": class_names,
                }
                self._cache[key] = yolo_model
                self._meta[key] = meta
            except Exception as exc:
                raise RuntimeError(f"YOLO model yüklenemedi ({model_path.name}): {exc}") from exc

        return self._cache[key], self._meta[key]

    # ── XGBoost ─────────────────────────────────────────────────────────────
    def get_xgboost(self, model_path: Path):
        """XGBoost modelini yükle (ilk çağrıda) ya da önbellekten döndür."""
        key = self._make_key(model_path)
        if key in self._cache:
            return self._cache[key], self._meta[key]

        with self._lock:
            if key in self._cache:
                return self._cache[key], self._meta[key]

            try:
                from model.sl.xgb_classifier import load_xgb_model_with_meta
                from model.dl.dataset import SINIF_ISIMLERI

                xgb_model, raw_meta = load_xgb_model_with_meta(model_path)
                meta = {
                    "image_size":  raw_meta.get("image_size", 224),
                    "class_names": raw_meta.get("class_names", SINIF_ISIMLERI),
                }
                self._cache[key] = xgb_model
                self._meta[key]  = meta
            except Exception as exc:
                raise RuntimeError(f"XGBoost model yüklenemedi ({model_path.name}): {exc}") from exc

        return self._cache[key], self._meta[key]

    def list_available_models(self) -> list[dict]:
        """MODEL_DIR içindeki .pt ve .json dosyalarını listele."""
        model_dir: Path = settings.MODEL_DIR
        if not model_dir.exists():
            return []
        models = []
        for path in sorted(model_dir.iterdir()):
            if path.suffix.lower() == ".pt":
                yolo_meta = model_dir / (path.stem + ".yolo.meta.json")
                if yolo_meta.exists():
                    models.append({"name": path.name, "type": "yolo", "path": str(path)})
                else:
                    models.append({"name": path.name, "type": "resnet", "path": str(path)})
            elif path.suffix.lower() == ".json" and not path.name.endswith(".meta.json"):
                models.append({"name": path.name, "type": "xgboost", "path": str(path)})
        return models

    def invalidate(self, model_path: Path):
        """Önbellekten belirli bir modeli temizle."""
        key = self._make_key(model_path)
        with self._lock:
            self._cache.pop(key, None)
            self._meta.pop(key, None)


# Modül seviyesi singleton — tüm view'lardan import edilir
registry = _ModelRegistry()
