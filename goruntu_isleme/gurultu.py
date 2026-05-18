"""Gurultu giderme filtreleri (median / gaussian / bilateral)."""

import cv2
import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselGurultuMixin:
    def _bilateral_filtre_uygula(self, goruntu: np.ndarray) -> np.ndarray:
        """Kenar korumali bilateral filtre uygula."""
        filtered = cv2.bilateralFilter(goruntu, d=5, sigmaColor=35, sigmaSpace=35)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def gurultu_gider(self, goruntu: np.ndarray, metod: str = 'auto') -> np.ndarray:
        """
        Görüntüden gürültüyü temizle.

        Args:
            goruntu: Girdi görüntüsü
            metod: 'auto' (FILTRE_METODU ayarini kullanir), 'off', 'median',
                   'gaussian' veya 'bilateral'.

        Returns:
            Gürültüsü azaltılmış görüntü
        """
        if metod == 'auto':
            metod = FILTRE_METODU

        if metod == 'off':
            # auto+off durumunda median 3x3'e dusmek kortikal dokuyu siler;
            # bu yuzden goruntu aynen donulur.
            return goruntu
        if metod == 'median':
            filtered = cv2.medianBlur(self._uint8_goruntu(goruntu), 3)
            return filtered.astype(np.uint8)
        if metod == 'gaussian':
            filtered = self._gaussian_blur_cv(
                self._uint8_goruntu(goruntu),
                sigma=GAUSSIAN_BLUR_SIGMA,
            )
            return filtered.astype(np.uint8)
        if metod == 'bilateral':
            return self._bilateral_filtre_uygula(goruntu)
        return goruntu
