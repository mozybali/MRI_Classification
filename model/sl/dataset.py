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

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..dl.dataset import (
    GORUNTU_UZANTILARI,
    SINIF_ETIKETI,
    _validate_expected_classes,
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
    _validate_expected_classes(data_dir, "Trainval")
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            # Context manager ile NpzFile'i kapatiyoruz; paralel HPO'da yazici
            # `os.replace` cagirdiginda Windows dahil tum platformlarda dosya
            # handle'i sicakta kalmasin diye tum okumalari with bloku icinde
            # yapiyoruz.
            with np.load(cache_path, allow_pickle=True) as data:
                # Metadata dogrulama: cache dosyasi farkli image_size ile
                # veya veri diziniyle olusturulmussa stale cache kullanilmasini onle
                if "image_size" not in data:
                    raise ValueError(
                        "Cache dosyasi image_size metadata'si icermiyor. "
                        f"Stale cache riskini onlemek icin dosyayi silin veya farkli cache yolu kullanin: {cache_path}"
                    )
                cached_image_size = int(data["image_size"])
                if cached_image_size != image_size:
                    raise ValueError(
                        f"Cache dosyasi farkli image_size ile olusturulmus: "
                        f"cache={cached_image_size}, istenen={image_size}. "
                        f"Cache dosyasini silin veya farkli cache yolu kullanin: {cache_path}"
                    )
                cached_data_dir = str(data["data_dir"].item()) if "data_dir" in data else None
                current_data_dir = str(data_dir.resolve())
                if cached_data_dir is None:
                    raise ValueError(
                        "Cache dosyasi veri dizini metadata'si icermiyor. "
                        f"Stale cache riskini onlemek icin dosyayi silin veya farkli cache yolu kullanin: {cache_path}"
                    )
                if cached_data_dir != current_data_dir:
                    raise ValueError(
                        "Cache dosyasi farkli veri dizini ile olusturulmus: "
                        f"cache={cached_data_dir}, istenen={current_data_dir}. "
                        f"Cache dosyasini silin veya farkli cache yolu kullanin: {cache_path}"
                    )
                X_cached = np.array(data["X"])
                y_cached = np.array(data["y"])
                groups_cached = data["groups"].tolist()
                paths_cached = data["paths"].tolist()
            return X_cached, y_cached, groups_cached, paths_cached

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
        with Image.open(img_path) as raw:
            img = raw.convert("L").resize((image_size, image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        features_list.append(extract_features(arr))

    X = np.stack(features_list, axis=0)
    y = np.array(labels, dtype=np.int64)
    paths_str = [str(p) for p in image_paths]

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomik yazim: paralel HPO trial'lari (n_jobs > 1) ayni cache_path'e
        # ayni anda yazabilir. np.savez_compressed dogrudan target dosyaya
        # yazarsa baska bir worker yarim/bozuk .npz okuyabilir. Once ayni
        # dizinde benzersiz bir gecici dosyaya yazip os.replace ile atomik
        # olarak yerlestiriyoruz; son yazan kazanir ama her okuma daima
        # tutarli bir .npz gorur. File-object kullaniyoruz cunku savez
        # path argumanina otomatik ".npz" uzantisi ekliyor.
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=cache_path.name + ".",
            suffix=".tmp",
            dir=str(cache_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_fh:
                np.savez_compressed(
                    tmp_fh,
                    X=X,
                    y=y,
                    groups=np.array(groups, dtype=object),
                    paths=np.array(paths_str, dtype=object),
                    image_size=np.array(image_size),
                    data_dir=np.array(str(data_dir.resolve())),
                )
                tmp_fh.flush()
                os.fsync(tmp_fh.fileno())
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    return X, y, groups, paths_str
