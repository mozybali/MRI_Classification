"""
artirma.py
----------
Basit veri artırma (data augmentation) fonksiyonları.
Bu modül, ön işlenmiş görüntüler üzerine dönüşler, yansıtma ve küçük parlaklık/kontrast değişimleri uygular.
"""

import numpy as np
import random


def yatay_ayna(goruntu: np.ndarray) -> np.ndarray:
    """Görüntüyü yatay eksende (sol-sağ) ayna yansıması ile çevirir."""
    return np.fliplr(goruntu)


def dikey_ayna(goruntu: np.ndarray) -> np.ndarray:
    """Görüntüyü dikey eksende (yukarı-aşağı) ayna yansıması ile çevirir."""
    return np.flipud(goruntu)


def rastgele_dondur(goruntu: np.ndarray) -> np.ndarray:
    """
    Görüntüyü 0, 90, 180 veya 270 derece rastgele döndürür.
    """
    acilar = [0, 90, 180, 270]
    aci = random.choice(acilar)
    if aci == 0:
        return goruntu
    elif aci == 90:
        return np.rot90(goruntu, k=1)
    elif aci == 180:
        return np.rot90(goruntu, k=2)
    else:
        return np.rot90(goruntu, k=3)


def parlaklik_kontrast_oyna(goruntu: np.ndarray,
                            parlaklik_aralik=(-20, 20),
                            kontrast_aralik=(0.9, 1.1)) -> np.ndarray:
    """
    Parlaklık ve kontrast değerleriyle küçük rastgele değişiklikler yapar.
    Girdi ve çıktı uint8 [0,255] varsayılır.
    """
    b = random.uniform(*parlaklik_aralik)  # bias
    c = random.uniform(*kontrast_aralik)   # gain

    goruntu_float = goruntu.astype("float32")
    degismis = goruntu_float * c + b
    degismis = np.clip(degismis, 0, 255).astype("uint8")
    return degismis


def rastgele_artirma_uygula(goruntu: np.ndarray):
    """
    Tek bir görüntüye birkaç temel artırma işlemini rastgele uygular.
    Çıktı: artırılmış görüntü.
    """
    g = goruntu.copy()

    # Rastgele yatay veya dikey ayna
    if random.random() < 0.5:
        g = yatay_ayna(g)
    if random.random() < 0.5:
        g = dikey_ayna(g)

    # Rastgele döndür
    g = rastgele_dondur(g)

    # Rastgele parlaklık/kontrast
    g = parlaklik_kontrast_oyna(g)

    return g
