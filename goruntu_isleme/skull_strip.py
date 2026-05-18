"""Foreground/Otsu maske yardimcilari.

Yogunluk normalizasyonu (`yogunluk_normalize`) ve pipeline sonu kalite
kontrolu (`_pipeline_sonu_kalite_kontrol`) gibi adimlarin ihtiyac
duydugu beyin aday maskesi uretimini saglar. Skull stripping akisi
veri setinin beyin-kirpilmis 2D dilimlerden olusmasi nedeniyle
pipeline'dan kaldirildi; bu modul yalnizca yardimcilari barindirir.
"""

from typing import Optional

import cv2
import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselSkullStripMixin:
    @staticmethod
    def _kenar_maskesini_temizle(mask: np.ndarray) -> np.ndarray:
        """Maske kenarlarindaki artefaktlari ayarlanabilir pay ile temizle."""
        temiz = mask.astype(bool, copy=True)
        pay = max(0, int(MASKE_KENAR_PAYI))
        if pay == 0:
            return temiz

        h, w = temiz.shape
        if pay * 2 >= min(h, w):
            return np.zeros_like(temiz, dtype=bool)

        temiz[:pay, :] = False
        temiz[-pay:, :] = False
        temiz[:, :pay] = False
        temiz[:, -pay:] = False
        return temiz

    @staticmethod
    def _morfolojik_yapi(kernel_boyutu: Optional[int] = None) -> np.ndarray:
        """Ayarlardaki kernel boyutunu kullanarak eliptik OpenCV yapi elemani uret."""
        boyut = int(kernel_boyutu or MORFOLOJIK_KERNEL_BOYUTU)
        boyut = max(1, boyut)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boyut, boyut))

    def _maskeyi_duzenle(
        self,
        mask: np.ndarray,
        *,
        closing_scale: int = 2,
        dilation_scale: int = 0,
    ) -> np.ndarray:
        """Maske kenarlarini ve morfolojik temizligini ayarlara gore uygula."""
        duzenli = self._kenar_maskesini_temizle(mask)
        if not MORFOLOJIK_OPERASYONLAR_AKTIF:
            return duzenli

        temel = self._morfolojik_yapi()
        close_kernel = self._morfolojik_yapi(MORFOLOJIK_KERNEL_BOYUTU * max(1, closing_scale))
        maske_u8 = self._bool_maske_uint8(duzenli)
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_OPEN,
            temel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_CLOSE,
            close_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if dilation_scale > 0:
            dilate_kernel = self._morfolojik_yapi(MORFOLOJIK_KERNEL_BOYUTU * dilation_scale)
            maske_u8 = cv2.dilate(
                maske_u8,
                dilate_kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return maske_u8 > 0

    def _otsu_maskesi(self, goruntu: np.ndarray) -> np.ndarray:
        """OpenCV Otsu thresholding ile beyin aday maskesini uret."""
        _, mask = cv2.threshold(
            self._uint8_goruntu(goruntu),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask > 0

    @staticmethod
    def _min_foreground_piksel_sayisi(toplam_piksel: int, oran: float = 0.001) -> int:
        """Kucuk test goruntulerini cezalandirmadan foreground alt siniri hesapla."""
        toplam = max(1, int(toplam_piksel))
        return min(max(16, int(round(toplam * oran))), max(1, toplam // 4))

    def _foreground_maskesi(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Per-image foreground/beyin aday maskesi uret.

        Maske yalnizca normalizasyon ve kalite kontrol gibi istatistik
        adimlarinda kullanilir; goruntuyu tek basina kirpmaz veya reddetmez.
        """
        if goruntu is None:
            return np.zeros((0, 0), dtype=bool)

        arr = self._uint8_goruntu(goruntu)
        if arr.ndim != 2 or arr.size == 0:
            return np.zeros(arr.shape[:2], dtype=bool)

        min_piksel = self._min_foreground_piksel_sayisi(arr.size)
        if int(arr.max()) <= int(arr.min()):
            return np.zeros(arr.shape, dtype=bool)

        try:
            maske = self._otsu_maskesi(arr)
        except cv2.error:
            maske = arr > 0

        if not maske.any():
            maske = arr > 0

        if min(arr.shape[:2]) >= GORUNTU_MIN_BOYUT and maske.any():
            maske = self._maskeyi_duzenle(maske, closing_scale=2)
            maske = self._kucuk_bilesenleri_temizle(maske, min_size=min_piksel)
            if maske.any():
                maske = self._kucuk_delikleri_doldur(
                    maske,
                    area_threshold=min_piksel,
                )

        maske = maske.astype(bool) & (arr > 0)
        if int(maske.sum()) >= min_piksel:
            return maske

        nonzero = arr > 0
        if int(nonzero.sum()) >= min_piksel:
            return nonzero

        return np.zeros(arr.shape, dtype=bool)

    def _kucuk_bilesenleri_temizle(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """OpenCV connected components ile min_size altindaki nesneleri sil."""
        if min_size <= 1:
            return mask.astype(bool)

        maske_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(maske_u8, connectivity=8)
        temiz = np.zeros_like(maske_u8, dtype=bool)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
                temiz[labels == label_idx] = True
        return temiz

    def _kucuk_delikleri_doldur(self, mask: np.ndarray, area_threshold: int) -> np.ndarray:
        """OpenCV connected components ile icteki kucuk delikleri doldur."""
        if area_threshold <= 0:
            return mask.astype(bool)

        dolu = mask.astype(bool, copy=True)
        ters = (~dolu).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ters, connectivity=8)
        h, w = dolu.shape

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area > area_threshold:
                continue

            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            genislik = stats[label_idx, cv2.CC_STAT_WIDTH]
            yukseklik = stats[label_idx, cv2.CC_STAT_HEIGHT]
            kenara_degiyor = x == 0 or y == 0 or x + genislik >= w or y + yukseklik >= h
            if not kenara_degiyor:
                dolu[labels == label_idx] = True

        return dolu
