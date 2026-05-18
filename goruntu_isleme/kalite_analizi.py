"""Goruntu kalite analizi ortak yardimcilari."""

import warnings
from typing import Optional, Tuple, TypedDict

import cv2
import numpy as np

try:
    from .ayarlar import *
    from . import opencv_duzeltmeler as _cv_yardimci
except ImportError:
    from ayarlar import *
    import opencv_duzeltmeler as _cv_yardimci


class PipelineSonucu(TypedDict):
    """Tekil goruntu pipeline sonucu icin tip kontrati.

    `goruntu_isle_sonuc` ve `_tek_goruntu_isle` arasindaki sozlesme bu
    sozlukle sabittir. Yeni alan eklemeden once tum okuyucular gunceller.
    """

    processed_image: Optional[np.ndarray]
    quality_rejected: bool
    quality_reason: str


class GorselKaliteAnaliziMixin:
    @staticmethod
    def _uint8_goruntu(goruntu: np.ndarray) -> np.ndarray:
        """OpenCV islemleri icin goruntuyu guvenli uint8 araligina al."""
        return _cv_yardimci.uint8_goruntu(goruntu)

    @staticmethod
    def _bool_maske_uint8(mask: np.ndarray) -> np.ndarray:
        """Bool maskeyi OpenCV'nin bekledigi 0/255 uint8 formuna cevir."""
        return _cv_yardimci.bool_maske_uint8(mask)

    @staticmethod
    def _gaussian_blur_cv(goruntu: np.ndarray, sigma: float) -> np.ndarray:
        """SciPy gaussian_filter yerine OpenCV GaussianBlur uygula."""
        return cv2.GaussianBlur(
            goruntu,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _oran_ayari_dogrula(
        ad: str,
        deger: object,
        varsayilan: float,
        alt: float = 0.0,
        ust: float = 1.0,
        *,
        clamp: bool = True,
    ) -> float:
        """Oran ayarlarini uyarili sekilde dogrula."""
        try:
            sayi = float(deger)
        except (TypeError, ValueError):
            warnings.warn(
                f"{ad} gecersiz; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)

        if not np.isfinite(sayi):
            warnings.warn(
                f"{ad} sonlu bir sayi olmali; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)

        if alt <= sayi <= ust:
            return sayi

        if clamp:
            kirpilmis = min(max(sayi, alt), ust)
            warnings.warn(
                f"{ad} [{alt}, {ust}] araliginda olmali; {kirpilmis} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(kirpilmis)

        warnings.warn(
            f"{ad} [{alt}, {ust}] araliginda olmali; {varsayilan} kullaniliyor.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float(varsayilan)

    @staticmethod
    def _temel_goruntu_on_kontrol(goruntu: np.ndarray) -> Tuple[bool, str]:
        """Kenar temizligi oncesi cok temel bozuk/bos goruntu kontrolu."""
        if goruntu is None or np.asarray(goruntu).size == 0:
            return False, "Boş görüntü"

        arr = np.asarray(goruntu)
        if arr.ndim != 2:
            return False, "Geçersiz görüntü boyutu"

        if not np.isfinite(arr.astype(np.float32, copy=False)).all():
            return False, "Geçersiz piksel değerleri"

        if float(np.nanmax(arr)) <= 0.0:
            return False, "Boş/siyah görüntü"

        return True, ""

    @staticmethod
    def _pipeline_sonucu(
        processed_image: Optional[np.ndarray] = None,
        *,
        quality_rejected: bool = False,
        quality_reason: str = "",
    ) -> PipelineSonucu:
        """Tekil goruntu pipeline'i icin standart sonuc sozlugu uret."""
        return PipelineSonucu(
            processed_image=processed_image,
            quality_rejected=bool(quality_rejected),
            quality_reason=str(quality_reason),
        )
