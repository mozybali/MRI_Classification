"""
stats_utils.py
--------------
Görüntü bazlı özet istatistiklerin hesaplanması.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import goruntu_yukle_yoksa_gri


def tek_goruntu_istatistikleri(path: str):
    """
    Tek bir JPEG/PNG görüntü için:

    - yükseklik (height), genişlik (width)
    - kanal sayısı (channel_count)
    - yoğunluk istatistikleri: mean, std, min, max, p1, p99
    """
    goruntu = goruntu_yukle_yoksa_gri(path)

    if goruntu.ndim == 2:  # gri
        yukseklik, genislik = goruntu.shape
        kanal_sayisi = 1
        flat = goruntu.flatten()
    else:  # RGB
        yukseklik, genislik, kanal_sayisi = goruntu.shape
        # İstatistikleri tek kanal üzerinden yapmak istersen,
        # burada griye dönüştürülebilir; şimdilik tüm kanalları birlikte kullanıyoruz.
        flat = goruntu.reshape(-1, kanal_sayisi).mean(axis=1)

    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        ort = std = min_val = max_val = p1 = p99 = 0.0
    else:
        ort = float(np.mean(flat))
        std = float(np.std(flat))
        min_val = float(np.min(flat))
        max_val = float(np.max(flat))
        p1 = float(np.percentile(flat, 1))
        p99 = float(np.percentile(flat, 99))

    return {
        "genislik": genislik,
        "yukseklik": yukseklik,
        "kanal_sayisi": kanal_sayisi,
        "int_ort": ort,
        "int_std": std,
        "int_min": min_val,
        "int_max": max_val,
        "int_p1": p1,
        "int_p99": p99,
        "en_boy_orani": genislik / yukseklik if yukseklik > 0 else 0.0,
    }


def tum_gorseller_icin_istatistik_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm dataset için görüntü bazlı özet istatistikleri hesaplar.
    """
    kayitlar = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Görüntü istatistikleri hesaplanıyor"):
        stats = tek_goruntu_istatistikleri(row["filepath"])
        stats["id"] = row["id"]
        kayitlar.append(stats)

    stats_df = pd.DataFrame(kayitlar)
    birlesik = df.merge(stats_df, on="id", how="left")
    return birlesik


def gomuleme_icin_ozellik_matrisi(df: pd.DataFrame):
    """
    PCA / t-SNE için kullanılacak öznitelik matrisi.

    Kullanılan öznitelikler:
    - genislik, yukseklik, en_boy_orani
    - int_ort, int_std, int_min, int_max, int_p1, int_p99
    """
    ozellik_kolonlari = [
        "genislik", "yukseklik", "en_boy_orani",
        "int_ort", "int_std", "int_min", "int_max", "int_p1", "int_p99",
    ]
    X = df[ozellik_kolonlari].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    return X, ozellik_kolonlari
