#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgb_classifier.py
------------------
XGBClassifier olusturma, kaydetme ve yukleme yardimcilari.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xgboost import XGBClassifier


def build_xgb_classifier(
    num_classes: int,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Siniflandirma icin XGBClassifier olustur.

    Parameters
    ----------
    num_classes : int
        Sinif sayisi.
    params : dict | None
        Opsiyonel XGBoost parametreleri.

    Returns
    -------
    XGBClassifier
    """
    defaults: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": 42,
    }
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def save_xgb_model(
    model: XGBClassifier,
    path: Path | str,
    *,
    image_size: int | None = None,
    class_names: list[str] | None = None,
    seed: int | None = None,
) -> Path:
    """XGBClassifier modelini JSON formatinda kaydet.

    Opsiyonel metadata (image_size, class_names, feature_dim, seed) side-car
    JSON dosyasina yazilir.  Inference sirasinda load_xgb_model_with_meta
    ile okunabilir.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    if image_size is not None or class_names is not None:
        meta: dict[str, Any] = {}
        if image_size is not None:
            meta["image_size"] = image_size
        if class_names is not None:
            meta["class_names"] = class_names
        if seed is not None:
            meta["seed"] = seed
        # feature_dim from booster
        try:
            meta["feature_dim"] = model.n_features_in_
        except AttributeError:
            pass
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    return path


def load_xgb_model(path: Path | str) -> XGBClassifier:
    """JSON formatindan XGBClassifier yukle."""
    path = Path(path)
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def load_xgb_model_with_meta(path: Path | str) -> tuple[XGBClassifier, dict[str, Any]]:
    """XGBClassifier ve side-car metadata yukle.

    Returns
    -------
    model : XGBClassifier
    meta : dict  — image_size, class_names, feature_dim, seed (varsa)
    """
    import json

    path = Path(path)
    model = load_xgb_model(path)
    meta: dict[str, Any] = {}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    return model, meta
