"""
arka_plan_isleme.py
-------------------
Siyah veya gri arka plana sahip MRI görüntülerinden arka planı ayırma ve
ilgi bölgesinin (ROI) sınır kutusunu bulma işlemleri.
"""

import numpy as np
from skimage.filters import threshold_otsu


def arka_plan_tonu_tahmin_et(goruntu: np.ndarray) -> float:
    """
    Görüntünün kenar piksellerine bakarak arka planın ortalama tonunu tahmin eder.
    Kenarlar genelde arka plan olduğu için kenar şeridinin ortalaması iyi bir tahmin verir.
    """
    h, w = goruntu.shape
    kenar_kalinligi = max(1, min(h, w) // 20)  # görüntünün %5'i kadar kenar

    ust = goruntu[:kenar_kalinligi, :]
    alt = goruntu[-kenar_kalinligi:, :]
    sol = goruntu[:, :kenar_kalinligi]
    sag = goruntu[:, -kenar_kalinligi:]

    kenar_pikseller = np.concatenate([ust.flatten(), alt.flatten(), sol.flatten(), sag.flatten()])
    arka_plan_tonu = float(np.mean(kenar_pikseller))
    return arka_plan_tonu


def arka_plan_tipi_belirle(goruntu: np.ndarray) -> str:
    """
    Kenar bölgelerin ortalama tonuna göre arka planın "siyah", "gri" veya "diger" olduğunu kabaca sınıflandırır.
    """
    ton = arka_plan_tonu_tahmin_et(goruntu)
    if ton < 30:
        return "siyah"
    elif ton < 120:
        return "gri"
    else:
        return "diger"


def ikili_maske_olustur(goruntu: np.ndarray) -> np.ndarray:
    """
    Otsu eşiği kullanarak ikili maske oluşturur.
    MRI'larda genellikle ilgi bölgesi (vücut dokusu) arka plandan daha parlak olduğu için
    maske = goruntu > eşik şeklinde alınır.

    Eğer bazı görüntülerde durum ters ise (ilgi bölgesi arka plandan daha koyu),
    bu fonksiyonu projede kolayca değiştirip maske = goruntu < eşik şeklinde kullanabilirsin.
    """
    # Otsu eşiğini hesapla
    esik = threshold_otsu(goruntu)
    maske = goruntu > esik

    return maske.astype("uint8")


def maske_sinir_kutusu_bul(maske: np.ndarray):
    """
    Verilen ikili maskede (0/1) 1'lerin bulunduğu en küçük sınır kutusunu (bounding box) bulur.
    Çıktı: (y_min, y_max, x_min, x_max)
    Eğer maske tamamen boşsa None döner.
    """
    koordinatlar = np.argwhere(maske > 0)
    if koordinatlar.size == 0:
        return None

    y_min, x_min = koordinatlar.min(axis=0)
    y_max, x_max = koordinatlar.max(axis=0) + 1  # Python slicing için +1

    return int(y_min), int(y_max), int(x_min), int(x_max)


def maske_sinir_kutusunu_genislet(sinir_kutusu, goruntu_sekli, kenar_payi: int):
    """
    Bulunan sınır kutusunu (y_min, y_max, x_min, x_max) belirtilen kenar payı kadar genişlet.
    Görüntü boyutlarını aşmayacak şekilde kırpar.
    """
    if sinir_kutusu is None:
        return None

    y_min, y_max, x_min, x_max = sinir_kutusu
    h, w = goruntu_sekli

    y_min_genis = max(0, y_min - kenar_payi)
    y_max_genis = min(h, y_max + kenar_payi)
    x_min_genis = max(0, x_min - kenar_payi)
    x_max_genis = min(w, x_max + kenar_payi)

    return int(y_min_genis), int(y_max_genis), int(x_min_genis), int(x_max_genis)
