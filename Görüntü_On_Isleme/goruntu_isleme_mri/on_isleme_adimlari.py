"""
on_isleme_adimlari.py
---------------------
Tek bir MRI görüntüsü üzerinde uygulanacak ön işleme adımlarını içerir:
- Arka plan tespiti
- Maske oluşturma ve kırpma
- Yoğunluk normalizasyonu
- Histogram eşitleme (isteğe bağlı)
- Yeniden boyutlandırma
"""

import numpy as np
import cv2
from skimage import exposure

from .ayarlar import (
    HEDEF_GENISLIK,
    HEDEF_YUKSEKLIK,
    KIRPMA_YUZDELERI,
    HISTOGRAM_ESITLEME_AKTIF,
    MASKE_KENAR_PAYI,
)
from .arka_plan_isleme import (
    arka_plan_tipi_belirle,
    ikili_maske_olustur,
    maske_sinir_kutusu_bul,
    maske_sinir_kutusunu_genislet,
)


def yogunluk_normalizasyonu(goruntu: np.ndarray, yuzdelikler=(1, 99)) -> np.ndarray:
    """
    MRI görüntüsünün yoğunluğunu normalize eder:
    - Belirtilen yüzdeliklere göre alt/üst kırpma (örn: %1 ve %99)
    - 0-255 aralığına yeniden ölçekleme

    Çıktı: uint8 [0, 255] şeklinde normalize edilmiş görüntü.
    """
    alt_yuzde, ust_yuzde = yuzdelikler
    flat = goruntu.flatten()
    alt_deger, ust_deger = np.percentile(flat, [alt_yuzde, ust_yuzde])

    goruntu_kirp = np.clip(goruntu, alt_deger, ust_deger)
    if ust_deger - alt_deger < 1e-6:
        norm = np.zeros_like(goruntu_kirp)
    else:
        norm = (goruntu_kirp - alt_deger) / (ust_deger - alt_deger)

    norm = (norm * 255.0).astype("uint8")
    return norm


def histogram_esitle(goruntu_uint8: np.ndarray) -> np.ndarray:
    """
    Adaptif histogram eşitleme (CLAHE benzeri) uygular.
    skimage.exposure.equalize_adapthist kullanır.

    Girdi: uint8 [0,255] görüntü
    Çıktı: uint8 [0,255] görüntü
    """
    # 0-1 aralığına getir
    goruntu_float = goruntu_uint8.astype("float32") / 255.0
    goruntu_eq = exposure.equalize_adapthist(goruntu_float, clip_limit=0.01)
    goruntu_eq_uint8 = (goruntu_eq * 255.0).clip(0, 255).astype("uint8")
    return goruntu_eq_uint8


def yeniden_boyutlandir(goruntu_uint8: np.ndarray, genislik: int = HEDEF_GENISLIK, yukseklik: int = HEDEF_YUKSEKLIK) -> np.ndarray:
    """
    Görüntüyü hedef boyuta yeniden boyutlandırır (bilineer interpolasyon).
    """
    hedef_boyut = (genislik, yukseklik)
    # cv2.resize boyut parametresi (width, height) sırasıyla alır
    yeniden = cv2.resize(goruntu_uint8, hedef_boyut, interpolation=cv2.INTER_LINEAR)
    return yeniden


def tek_goruntu_on_isle(goruntu_gri: np.ndarray):
    """
    Tek bir gri tonlamalı MRI görüntüsüne tüm ön işleme adımlarını uygular.
    Girdi: goruntu_gri - float32 [0,255] veya benzeri aralıkta 2B numpy dizisi

    Çıktı:
        on_islenmis_goruntu: uint8 [0,255], (HEDEF_YUKSEKLIK, HEDEF_GENISLIK)
        meta_bilgi: dict
    """
    orijinal_h, orijinal_w = goruntu_gri.shape

    # 1) Arka plan tipini belirle (bilgi amaçlı)
    arka_plan_tipi = arka_plan_tipi_belirle(goruntu_gri)

    # 2) Maske oluştur
    maske = ikili_maske_olustur(goruntu_gri)

    # 3) Maske sınır kutusunu bul ve kenar payı ile genişlet
    sinir_kutusu = maske_sinir_kutusu_bul(maske)
    sinir_kutusu_genis = maske_sinir_kutusunu_genislet(sinir_kutusu, goruntu_gri.shape, MASKE_KENAR_PAYI)

    if sinir_kutusu_genis is not None:
        y_min, y_max, x_min, x_max = sinir_kutusu_genis
        kirpilmis = goruntu_gri[y_min:y_max, x_min:x_max]
    else:
        # Maske boş ise görüntüyü kırpmadan kullan
        y_min, y_max, x_min, x_max = 0, orijinal_h, 0, orijinal_w
        kirpilmis = goruntu_gri

    # 4) Yoğunluk normalizasyonu
    norm = yogunluk_normalizasyonu(kirpilmis, yuzdelikler=KIRPMA_YUZDELERI)

    # 5) İsteğe bağlı histogram eşitleme
    if HISTOGRAM_ESITLEME_AKTIF:
        norm = histogram_esitle(norm)

    # 6) Hedef boyuta yeniden boyutlandırma
    yeniden = yeniden_boyutlandir(norm, HEDEF_GENISLIK, HEDEF_YUKSEKLIK)

    meta_bilgi = {
        "orijinal_genislik": int(orijinal_w),
        "orijinal_yukseklik": int(orijinal_h),
        "kirp_y_min": int(y_min),
        "kirp_y_max": int(y_max),
        "kirp_x_min": int(x_min),
        "kirp_x_max": int(x_max),
        "arka_plan_tipi": arka_plan_tipi,
    }

    return yeniden, meta_bilgi
