#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgb_classifier.py
------------------
XGBClassifier olusturma, kaydetme ve yukleme yardimcilari.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

from xgboost import XGBClassifier

XgbDeviceMode = Literal["auto", "cpu", "cuda"]


def _xgboost_has_cuda_build() -> bool:
    """Yuklu XGBoost paketinin CUDA build'iyle gelip gelmedigini dogrula.

    ``xgboost.build_info()`` derleme zamani bayraklarini (USE_CUDA dahil)
    dondurur. CPU-only build'lerde ``USE_CUDA=False`` gelir; bu fonksiyon
    XGBoost tarafindaki gercek CUDA destegini kontrol eder, PyTorch CUDA
    durumuna degil. ``build_info`` cok eski XGBoost surumlerinde olmayabilir;
    bu durumda muhafazakarca ``False`` doneriz (CPU'ya dus).
    """
    try:
        import xgboost  # type: ignore

        build_info_fn = getattr(xgboost, "build_info", None)
        if build_info_fn is None:
            return False
        info = build_info_fn()
    except Exception:
        return False
    return bool(info.get("USE_CUDA", False))


def _resolve_xgb_device(device: XgbDeviceMode) -> str:
    """``--xgb-device`` icin gercek XGBoost device string'i sec.

    ``auto``: hem XGBoost CUDA build'i hem torch CUDA mevcutsa ``cuda``,
    aksi halde ``cpu``. ``torch.cuda.is_available()`` tek basina yetersiz:
    CUDA-li PyTorch + CPU-only XGBoost ikilisinde fit asamasinda XGBoost
    patlardi.
    ``cuda``: kullanicinin acik isteyini koruruz; ancak yuklu XGBoost CUDA
    build'i degilse erken bir uyari basariz (XGBoost zaten anlamli runtime
    hatasi atar, sadece sebebi onceden netlestiriyoruz).
    ``cpu``: her zaman CPU.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    if device == "auto":
        if not _xgboost_has_cuda_build():
            return "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    raise ValueError(
        f"Gecersiz xgb device modu: {device!r}. Beklenen: 'auto', 'cpu', 'cuda'."
    )


def build_xgb_classifier(
    num_classes: int,
    params: dict[str, Any] | None = None,
    *,
    eval_metric: Any = None,
    custom_metric: Any = None,
    device: XgbDeviceMode = "auto",
    n_jobs: int | None = None,
) -> XGBClassifier:
    """Siniflandirma icin XGBClassifier olustur.

    Parameters
    ----------
    num_classes : int
        Sinif sayisi. Sanity-check icin kullanilir; XGBoost'a dogrudan
        aktarilmaz (num_class fit sirasinda y'den cikarilir).
    params : dict | None
        Opsiyonel XGBoost parametreleri.
    eval_metric : str | callable | list | None
        Override icin eval_metric. None ise "mlogloss" kullanilir. Liste
        verilirse XGBoost early stopping listenin son metrigine gore karar
        verir.
    device : {"auto", "cpu", "cuda"}
        XGBoost device modu. ``auto`` (default) torch.cuda mevcutsa GPU,
        aksi halde CPU secer. ``cuda`` zorla GPU; CUDA-li XGBoost build'i
        yoksa XGBoost runtime'da hata verir.
    n_jobs : int | None
        XGBoost icin worker thread sayisi. None ise os.cpu_count().

    Returns
    -------
    XGBClassifier
    """
    if num_classes < 2:
        raise ValueError(
            f"num_classes en az 2 olmali, {num_classes} verildi."
        )
    resolved_device = _resolve_xgb_device(device)
    resolved_n_jobs = int(n_jobs) if n_jobs is not None else (os.cpu_count() or 1)

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
        "tree_method": "hist",
        "device": resolved_device,
        "n_jobs": resolved_n_jobs,
        "verbosity": 0,
        "random_state": 42,
    }
    if params:
        # Caller params'in kendi tree_method/device/n_jobs/random_state'ini
        # vermisse onlara saygi gosteriyoruz; aksi halde defaults kalir.
        defaults.update(params)
    if eval_metric is not None:
        defaults["eval_metric"] = eval_metric
    if custom_metric is not None:
        defaults["custom_metric"] = custom_metric

    if defaults.get("device") == "cuda" and not _xgboost_has_cuda_build():
        # Kullanici acikca cuda istedi, ama yuklu XGBoost CPU-only build.
        # Sessizce CPU'ya dusmek yerine ne yapilmasi gerektigini soyleyen
        # bir uyari basiyoruz; fit/predict asamasinda da XGBoost kendi
        # hatasini atar.
        warnings.warn(
            "XGBoost device='cuda' istendi ancak yuklu XGBoost CPU-only "
            "build (xgboost.build_info()['USE_CUDA']=False). CUDA-uyumlu "
            "build kurun veya --xgb-device cpu/auto kullanin; aksi halde "
            "fit/predict asamasinda XGBoost hata verecek.",
            stacklevel=2,
        )

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
    if path.suffix == "":
        path = path.with_suffix(".json")
    elif path.suffix.lower() != ".json":
        raise ValueError(
            f"XGBoost model yolu .json uzantili olmali, alindi: {path.suffix}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    if image_size is not None or class_names is not None or seed is not None:
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
