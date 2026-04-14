#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py
----------
Klasor agacindan ozellik matrisi (X, y, groups, paths) olusturma.

DL pipeline'indaki SINIF_ISIMLERI ve kaynak_id_belirle fonksiyonlarini
reuse ederek leak-free grup bilgisini korur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..dl.dataset import (
    GORUNTU_UZANTILARI,
    SINIF_ETIKETI,
    SINIF_ISIMLERI,
    kaynak_id_belirle,
)
from .features import extract_features


def build_feature_matrix(
    data_dir: Path | str,
    image_size: int = 224,
    cache_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Klasor agacindan ozellik matrisi olustur.

    Parameters
    ----------
    data_dir : Path
        Sinif alt klasorleri iceren veri dizini.
    image_size : int
        Goruntulerin yeniden olceklenecegi boyut (kare).
    cache_path : Path | None
        Opsiyonel .npz disk cache yolu. Varsa yukler, yoksa hesaplar ve kaydeder.

    Returns
    -------
    X : np.ndarray
        (n_samples, n_features) ozellik matrisi.
    y : np.ndarray
        (n_samples,) etiket dizisi.
    groups : list[str]
        Leak-free split icin kaynak grup anahtarlari.
    paths : list[str]
        Goruntu dosya yollarinin string listesi.
    """
    data_dir = Path(data_dir)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            # Metadata dogrulama: cache dosyasi farkli image_size ile
            # olusturulmussa stale cache kullanilmasini onle
            cached_image_size = int(data["image_size"]) if "image_size" in data else None
            if cached_image_size is not None and cached_image_size != image_size:
                raise ValueError(
                    f"Cache dosyasi farkli image_size ile olusturulmus: "
                    f"cache={cached_image_size}, istenen={image_size}. "
                    f"Cache dosyasini silin veya farkli cache yolu kullanin: {cache_path}"
                )
            return (
                data["X"],
                data["y"],
                data["groups"].tolist(),
                data["paths"].tolist(),
            )

    image_paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []

    for class_name, label in SINIF_ETIKETI.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in GORUNTU_UZANTILARI:
                image_paths.append(img_file)
                labels.append(label)
                groups.append(f"{class_name}::{kaynak_id_belirle(img_file.name)}")

    if not image_paths:
        raise FileNotFoundError(f"Veri bulunamadi: {data_dir}")

    features_list: list[np.ndarray] = []
    for img_path in tqdm(image_paths, desc="Ozellik cikarma", unit="img"):
        img = Image.open(img_path).convert("L")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        features_list.append(extract_features(arr))

    X = np.stack(features_list, axis=0)
    y = np.array(labels, dtype=np.int64)
    paths_str = [str(p) for p in image_paths]

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            groups=np.array(groups, dtype=object),
            paths=np.array(paths_str, dtype=object),
            image_size=np.array(image_size),
        )

    return X, y, groups, paths_str
