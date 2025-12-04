"""
io_utils.py
-----------
Veri okuma, görüntü yükleme ve temel yardımcı fonksiyonlar.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    RANDOM_SEED,
    OUTPUT_DIR,
    LABEL_NAME_MAP,
    N_PIXELS_PER_IMAGE_SAMPLE,
    CONVERT_TO_GRAYSCALE,
)


def cikti_klasorunu_olustur():
    """Çıktı klasörünün var olduğundan emin ol."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def rastgele_tohum_ayarla(seed: int = RANDOM_SEED):
    """Tüm rastgelelik kaynakları için sabit tohum ayarla."""
    random.seed(seed)
    np.random.seed(seed)


def etiket_tablosu_yukle(csv_yolu: str) -> pd.DataFrame:
    """
    labels.csv dosyasını oku ve temel kontrolleri yap.

    Beklenen kolonlar: id, filepath, label
    """
    df = pd.read_csv(csv_yolu)
    beklenen_kolonlar = {"id", "filepath", "label"}
    if not beklenen_kolonlar.issubset(df.columns):
        raise ValueError(f"CSV şu kolonları içermeli: {beklenen_kolonlar}, mevcut: {df.columns}")

    # Dosya yollarının gerçekten var olup olmadığına hızlı bir bakış
    eksik = []
    for fp in df["filepath"]:
        if not Path(fp).exists():
            eksik.append(fp)
    if len(eksik) > 0:
        print(f"[UYARI] {len(eksik)} adet dosya bulunamadı. İlk birkaç örnek:")
        print("\n".join(map(str, eksik[:10])))

    # label'ı kategorik stringe çevir
    if LABEL_NAME_MAP is not None:
        df["label_name"] = df["label"].map(LABEL_NAME_MAP).astype(str)
    else:
        df["label_name"] = df["label"].astype(str)

    return df


def goruntu_yukle_yoksa_gri(path: str) -> np.ndarray:
    """
    JPEG/PNG görüntüyü oku ve numpy dizisi olarak döndür.

    - Eğer CONVERT_TO_GRAYSCALE True ise: (H, W) gri tonlamalı
    - Aksi halde: (H, W, C) RGB
    """
    img = Image.open(path)
    if CONVERT_TO_GRAYSCALE:
        img = img.convert("L")  # 8-bit gri tonlama
    else:
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    return arr


def normalize_goruntu(goruntu: np.ndarray) -> np.ndarray:
    """
    Görselleştirme için 2B görüntüyü normalize et:
    - %1 ve %99 percentilleri arasında kırp
    - 0-1 aralığına ölçekle
    """
    flat = goruntu.flatten()
    vmin, vmax = np.percentile(flat, [1, 99])
    goruntu_kirp = np.clip(goruntu, vmin, vmax)
    if vmax - vmin < 1e-6:
        return np.zeros_like(goruntu_kirp)
    return (goruntu_kirp - vmin) / (vmax - vmin)


def rastgele_piksel_ornekle(path: str,
                             n_piksel: int = N_PIXELS_PER_IMAGE_SAMPLE) -> np.ndarray:
    """
    Global yoğunluk histogramı için tek bir görüntüden rastgele piksel örnekle.
    """
    goruntu = goruntu_yukle_yoksa_gri(path)
    flat = goruntu.flatten()
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        return np.array([])

    n = min(n_piksel, flat.size)
    idx = np.random.choice(flat.size, size=n, replace=False)
    return flat[idx]
