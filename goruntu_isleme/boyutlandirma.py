"""Goruntu yeniden boyutlandirma ve padding yardimcilari."""

import warnings

import cv2
import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselBoyutlandirmaMixin:
    def boyutlandir(self, goruntu: np.ndarray,
                    genislik: int = HEDEF_GENISLIK,
                    yukseklik: int = HEDEF_YUKSEKLIK) -> np.ndarray:
        """
        Görüntüyü hedef boyuta yeniden boyutlandır.

        Makine öğrenmesi modellerinde tüm görüntülerin aynı boyutta olması gerekir.
        Bu fonksiyon MRI görüntülerini standart boyuta (örn: 256x256) getirir.

        BOYUTLANDIRMA_MODU ayarina gore iki strateji desteklenir:
        - "stretch": Goruntu dogrudan hedef boyuta gerilir (eski davranis).
                     En-boy orani korunmaz, ancak basit ve hizlidir.
        - "pad"    : En-boy orani korunarak goruntu hedef cerceveye sigdirilir
                     ve bos kenarlar PADDING_DEGERI ile doldurulur. MRI 2D
                     dilimlerinde anatomik distorsiyonu engeller.

        İnterpolasyon: LINEAR (bilinear interpolation)

        Args:
            goruntu: Kaynak görüntü (numpy array)
            genislik: Hedef genişlik (pixel)
            yukseklik: Hedef yükseklik (pixel)

        Returns:
            (yukseklik, genislik) seklinde yeniden boyutlandirilmis goruntu.
            Girdi dtype'i korunur.
        """
        mod = BOYUTLANDIRMA_MODU
        if mod == "stretch":
            return self._boyutlandir_dogrudan(goruntu, genislik, yukseklik)
        if mod != "pad":
            print(f"[UYARI] Bilinmeyen BOYUTLANDIRMA_MODU: {mod}, 'pad' kullanılıyor")
        return self._boyutlandir_pad(goruntu, genislik, yukseklik)

    def _boyutlandir_dogrudan(self, goruntu: np.ndarray,
                              genislik: int,
                              yukseklik: int) -> np.ndarray:
        """En-boy orani gozetmeden hedef boyuta dogrudan resize."""
        return cv2.resize(goruntu, (genislik, yukseklik), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _padding_fallback_degeri(dtype) -> int:
        """PADDING_DEGERI'ni uint8 araliginda guvenli oku."""
        try:
            deger = float(PADDING_DEGERI)
        except (TypeError, ValueError):
            warnings.warn(
                "PADDING_DEGERI gecersiz; 0 kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0

        if not np.isfinite(deger):
            warnings.warn(
                "PADDING_DEGERI sonlu bir sayi olmali; 0 kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0

        if np.issubdtype(np.dtype(dtype), np.integer):
            if deger < 0 or deger > 255:
                kirpilmis = int(np.clip(deger, 0, 255))
                warnings.warn(
                    f"PADDING_DEGERI [0, 255] araliginda olmali; {kirpilmis} kullaniliyor.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return kirpilmis
            return int(round(deger))
        return int(np.clip(round(deger), 0, 255))

    def _padding_degeri_hesapla(self, olceklenmis: np.ndarray) -> int:
        """
        Padding icin post-normalizasyon arka plan degeri tahmin et.

        Tahmin yalnizca kose yamalari goruntunun dusuk yogunluklu arka
        planini guvenilir temsil ediyorsa kullanilir; aksi halde
        PADDING_DEGERI fallback olarak kalir.
        """
        fallback = self._padding_fallback_degeri(olceklenmis.dtype)
        if not PADDING_OTOMATIK_ARKAPLAN:
            return fallback

        arr = self._uint8_goruntu(olceklenmis)
        if arr.ndim != 2 or arr.size == 0 or min(arr.shape[:2]) < 4:
            return fallback

        h, w = arr.shape[:2]
        patch = max(2, min(12, h // 8, w // 8))
        kose_degerleri = np.concatenate(
            [
                arr[:patch, :patch].ravel(),
                arr[:patch, w - patch:].ravel(),
                arr[h - patch:, :patch].ravel(),
                arr[h - patch:, w - patch:].ravel(),
            ]
        )
        if kose_degerleri.size == 0:
            return fallback

        p25 = float(np.percentile(arr, 25))
        p90 = float(np.percentile(arr, 90))
        kose_medyan = float(np.median(kose_degerleri))

        # Tek renkli veya koseleri anatomik doku gibi parlak gorunen
        # goruntulerde otomatik tahmin kullanmak non-anatomik padding
        # uretebilir; fallback daha guvenlidir.
        if p90 - p25 < 1.0:
            return fallback
        if kose_medyan <= p25 + 5.0:
            return int(np.clip(round(kose_medyan), 0, 255))
        return fallback

    def _boyutlandir_pad(self, goruntu: np.ndarray,
                         genislik: int,
                         yukseklik: int) -> np.ndarray:
        """En-boy oranini koruyarak hedef cerceveye sigdir ve kenarlari doldur."""
        h, w = goruntu.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Geçersiz görüntü boyutu")

        olcek = min(genislik / float(w), yukseklik / float(h))
        yeni_g = max(1, int(round(w * olcek)))
        yeni_y = max(1, int(round(h * olcek)))
        # Yuvarlama hedefi asabilir; kirpalim
        yeni_g = min(yeni_g, genislik)
        yeni_y = min(yeni_y, yukseklik)

        olceklenmis = self._boyutlandir_dogrudan(goruntu, yeni_g, yeni_y)

        padding_degeri = self._padding_degeri_hesapla(olceklenmis)
        ust = (yukseklik - yeni_y) // 2
        sol = (genislik - yeni_g) // 2
        alt = yukseklik - yeni_y - ust
        sag = genislik - yeni_g - sol
        return cv2.copyMakeBorder(
            olceklenmis,
            ust,
            alt,
            sol,
            sag,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_degeri,
        )
