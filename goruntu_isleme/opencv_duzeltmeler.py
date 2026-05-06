"""OpenCV tabanli yardimcilar.

Kenar artefakt temizleme ve egim/skew duzeltme akislarinin OpenCV
omurgasini bu modulde toplanir. Tum yardimcilar stateless'tir; ayarlar
cagiran tarafa birakilir, boylece on_isleme/kalite_io modullerindeki
mevcut monkeypatch davranisi degismez.
"""

from typing import Tuple

import cv2
import numpy as np

__all__ = [
    "uint8_goruntu",
    "bool_maske_uint8",
    "kenar_serit_kalinliklari",
    "kenar_serit_maskesi",
    "parlak_maske",
    "baglantili_bilesenler",
    "pca_compute_2d",
    "affine_dondur_padding",
]


def uint8_goruntu(goruntu: np.ndarray) -> np.ndarray:
    """Goruntuyu OpenCV islemleri icin guvenli uint8 araligina al."""
    arr = np.asarray(goruntu)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def bool_maske_uint8(mask: np.ndarray) -> np.ndarray:
    """Bool maskeyi OpenCV'nin bekledigi 0/255 uint8 formuna cevir."""
    return (np.asarray(mask).astype(bool) * 255).astype(np.uint8)


def kenar_serit_kalinliklari(sekil: Tuple[int, int], oran: float) -> Tuple[int, int]:
    """Verilen oran icin yukseklik/genislik serit kalinliklarini hesapla."""
    h, w = int(sekil[0]), int(sekil[1])
    oran_f = float(oran)
    oran_f = min(max(oran_f, 0.0), 0.49)
    kalinlik_y = max(1, int(round(h * oran_f)))
    kalinlik_x = max(1, int(round(w * oran_f)))
    return kalinlik_y, kalinlik_x


def kenar_serit_maskesi(
    sekil: Tuple[int, int], kalinlik_y: int, kalinlik_x: int
) -> np.ndarray:
    """Ust/alt/sol/sag seritleri True olan bool maske uret."""
    h, w = int(sekil[0]), int(sekil[1])
    serit = np.zeros((h, w), dtype=bool)
    ky = max(0, min(int(kalinlik_y), h))
    kx = max(0, min(int(kalinlik_x), w))
    if ky > 0:
        serit[:ky, :] = True
        serit[h - ky:, :] = True
    if kx > 0:
        serit[:, :kx] = True
        serit[:, w - kx:] = True
    return serit


def parlak_maske(arr_u8: np.ndarray, esik: int) -> np.ndarray:
    """uint8 goruntu icin parlak >= esik maskesini 0/255 olarak don.

    cv2.inRange semantigi: lower <= src <= upper, dolayisiyla `arr >= esik`
    iliskisini birebir esler.
    """
    if arr_u8.dtype != np.uint8:
        arr_u8 = uint8_goruntu(arr_u8)
    esik_int = int(esik)
    if esik_int < 0:
        esik_int = 0
    if esik_int > 255:
        return np.zeros(arr_u8.shape, dtype=np.uint8)
    return cv2.inRange(arr_u8, esik_int, 255)


def baglantili_bilesenler(mask_u8: np.ndarray, connectivity: int = 8):
    """cv2.connectedComponentsWithStats sarmalayicisi."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    return cv2.connectedComponentsWithStats(mask_u8, connectivity=connectivity)


def pca_compute_2d(coords_xy: np.ndarray):
    """2B koordinatlar uzerinde OpenCV PCA uygula.

    Args:
        coords_xy: (N, 2) seklinde (x, y) sirali koordinatlar.

    Returns:
        (mean, eigenvectors, eigenvalues) - eigenvalues azalan sirada,
        eigenvectors'in satirlari ana eksenleri verir.
    """
    data = np.ascontiguousarray(coords_xy, dtype=np.float64)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, mean=None)
    return mean.ravel(), eigenvectors, eigenvalues.ravel()


def affine_dondur_padding(
    arr_u8: np.ndarray, aci_deg: float, pad: int, dolgu: int
) -> np.ndarray:
    """Padding'li affine rotasyon uygula ve orijinal boyuta merkezden kirp."""
    if arr_u8.dtype != np.uint8:
        arr_u8 = uint8_goruntu(arr_u8)
    h, w = arr_u8.shape[:2]
    pad = max(0, int(pad))
    dolgu_int = int(np.clip(dolgu, 0, 255))
    padded = cv2.copyMakeBorder(
        arr_u8,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=dolgu_int,
    )
    ph, pw = padded.shape[:2]
    merkez = ((pw - 1) / 2.0, (ph - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(merkez, float(aci_deg), 1.0)
    rotated = cv2.warpAffine(
        padded,
        matrix,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=dolgu_int,
    )
    y0 = (ph - h) // 2
    x0 = (pw - w) // 2
    return rotated[y0:y0 + h, x0:x0 + w]
