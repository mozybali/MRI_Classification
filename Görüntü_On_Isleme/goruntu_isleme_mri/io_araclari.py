"""
io_araclari.py
--------------
Dosya sisteminden görüntüleri bulma, okuma ve kaydetme ile ilgili yardımcı fonksiyonlar.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    GORUNTU_UZANTILARI,
    RASTGELE_TOHUM,
)


def rastgele_tohum_ayarla(tohum: int = RASTGELE_TOHUM):
    """Tüm rastgelelik kaynakları için sabit tohum ayarla."""
    random.seed(tohum)
    np.random.seed(tohum)


def klasor_olustur_yoksa(klasor_yolu: str):
    """Verilen klasör yolu yoksa oluştur."""
    Path(klasor_yolu).mkdir(parents=True, exist_ok=True)


def girdi_gorsellerini_listele(klasor_yolu: str = GIRDİ_KLASORU):
    """
    Girdi klasörü altında izin verilen uzantılara sahip tüm görüntü dosyalarını listele.
    Alt klasörler de taranır.
    """
    klasor = Path(klasor_yolu)
    dosya_listesi = []
    for kok, alt_klasorler, dosyalar in os.walk(klasor):
        for dosya in dosyalar:
            alt_uzanti = Path(dosya).suffix.lower()
            if alt_uzanti in GORUNTU_UZANTILARI:
                tam_yol = str(Path(kok) / dosya)
                dosya_listesi.append(tam_yol)
    dosya_listesi.sort()
    return dosya_listesi


def goruntu_oku_gri(path: str) -> np.ndarray:
    """
    Verilen dosya yolundaki görüntüyü oku ve gri tonlamaya çevir.
    Çıktı numpy dizisi (H, W) float32 [0, 255] aralığında olur.
    """
    img = Image.open(path).convert("L")  # 8-bit gri tonlama
    arr = np.array(img).astype(np.float32)
    return arr


def goruntu_kaydet(path: str, goruntu: np.ndarray):
    """
    Verilen numpy dizisini (H, W) veya (H, W, 3) JPEG/PNG olarak kaydet.
    Değerler [0, 255] aralığında olmalı.
    """
    arr = goruntu
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype("uint8"), mode="L")
    else:
        img = Image.fromarray(arr.astype("uint8"), mode="RGB")

    kayit_yolu = Path(path)
    kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(kayit_yolu))


def cikti_yolu_olustur(girdi_dosyasi: str, girdi_kok: str = GIRDİ_KLASORU, cikti_kok: str = CIKTI_KLASORU):
    """
    Girdi dosyasının CIKTI_KLASORU içindeki karşılık gelen yolunu üretir.
    Örnek:
        girdi: veri/girdi/alt/hasta_001.jpg
        çıktı: veri/cikti/alt/hasta_001.jpg
    """
    girdi_path = Path(girdi_dosyasi)
    girdi_kok = Path(girdi_kok)
    nispi_yol = girdi_path.relative_to(girdi_kok)
    cikti_path = Path(cikti_kok) / nispi_yol
    return str(cikti_path)
