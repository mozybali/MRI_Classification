"""Goruntu yukleme, kalite kontrol ve kaydetme islemleri."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselKaliteIOMixin:
    @staticmethod
    def _kenar_serit_kalinliklari(sekil: Tuple[int, int]) -> Tuple[int, int]:
        """KENAR_SERIT_ORANI'na gore yukseklik ve genislik kalinliklarini hesapla."""
        h, w = int(sekil[0]), int(sekil[1])
        oran = float(KENAR_SERIT_ORANI)
        oran = min(max(oran, 0.0), 0.49)
        kalinlik_y = max(1, int(round(h * oran)))
        kalinlik_x = max(1, int(round(w * oran)))
        return kalinlik_y, kalinlik_x

    @staticmethod
    def _kenar_serit_metrikleri(serit: np.ndarray, kenar_yon: str) -> Dict[str, float]:
        """Tek bir serit icin parlaklik ve baglantili bilesen metriklerini hesapla."""
        serit_u8 = np.asarray(serit)
        if serit_u8.dtype != np.uint8:
            serit_u8 = np.clip(serit_u8, 0, 255).astype(np.uint8)

        toplam = int(serit_u8.size)
        if toplam == 0:
            return {
                "boyut": 0,
                "ortalama": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "parlak_oran": 0.0,
                "cok_parlak_oran": 0.0,
                "buyuk_bilesen_orani": 0.0,
                "suspicious": False,
            }

        parlak_esigi = int(KENAR_PARLAKLIK_ESIGI)
        cok_parlak_esigi = int(KENAR_COK_PARLAKLIK_ESIGI)

        ortalama = float(serit_u8.mean())
        p95 = float(np.percentile(serit_u8, 95))
        p99 = float(np.percentile(serit_u8, 99))
        parlak_maske = serit_u8 >= parlak_esigi
        cok_parlak_maske = serit_u8 >= cok_parlak_esigi
        parlak_oran = float(parlak_maske.sum()) / float(toplam)
        cok_parlak_oran = float(cok_parlak_maske.sum()) / float(toplam)

        buyuk_bilesen_orani = 0.0
        if parlak_maske.any():
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                parlak_maske.astype(np.uint8), connectivity=8
            )
            if num_labels > 1:
                en_buyuk_alan = int(stats[1:, cv2.CC_STAT_AREA].max())
                buyuk_bilesen_orani = float(en_buyuk_alan) / float(toplam)

        suspicious = (
            parlak_oran >= float(KENAR_PARLAK_PIXEL_ORANI_ESIGI)
            or buyuk_bilesen_orani >= float(KENAR_BILESEN_ORANI_ESIGI)
            or p99 >= float(KENAR_COK_PARLAKLIK_ESIGI)
        )

        return {
            "boyut": toplam,
            "ortalama": ortalama,
            "p95": p95,
            "p99": p99,
            "parlak_oran": parlak_oran,
            "cok_parlak_oran": cok_parlak_oran,
            "buyuk_bilesen_orani": buyuk_bilesen_orani,
            "suspicious": bool(suspicious),
        }

    def kenar_artefakt_analiz(self, goruntu: np.ndarray) -> Dict[str, object]:
        """
        Kenar (top/bottom/left/right) seritlerinde parlaklik artefakti tespit et.

        Bu yardimci konservatif bir tespit yapar; merkezi beyin dokusuna
        bakmaz. KENAR_SERIT_ORANI ile belirlenen kalinliktaki kenar
        seritleri uzerinden dort yon icin metrik toplar.

        Args:
            goruntu: Gri tonlamali 2D numpy.ndarray (uint8 onerilir).

        Returns:
            Yon bazli metrik sozlugu ve genel `artefakt_var` boolean'i.
            Anahtarlar: 'top', 'bottom', 'left', 'right', 'artefakt_var',
            'suspicious_yonler'.
        """
        if goruntu is None or goruntu.ndim != 2:
            return {
                "top": {},
                "bottom": {},
                "left": {},
                "right": {},
                "artefakt_var": False,
                "suspicious_yonler": [],
            }

        h, w = goruntu.shape[:2]
        kalinlik_y, kalinlik_x = self._kenar_serit_kalinliklari((h, w))

        seritler = {
            "top": goruntu[:kalinlik_y, :],
            "bottom": goruntu[h - kalinlik_y:, :],
            "left": goruntu[:, :kalinlik_x],
            "right": goruntu[:, w - kalinlik_x:],
        }

        sonuc: Dict[str, object] = {}
        suspicious_yonler = []
        for yon, serit in seritler.items():
            metrik = self._kenar_serit_metrikleri(serit, yon)
            sonuc[yon] = metrik
            if metrik.get("suspicious", False):
                suspicious_yonler.append(yon)

        sonuc["suspicious_yonler"] = suspicious_yonler
        sonuc["artefakt_var"] = bool(suspicious_yonler)
        return sonuc

    def goruntu_kalite_kontrol(self, goruntu: np.ndarray) -> Tuple[bool, str]:
        """
        Görüntü kalitesini kontrol et.
        
        Bu fonksiyon, bozuk, düşük kaliteli veya hatalı görüntüleri tespit eder.
        Model eğitiminde kullanılmaması gereken görüntüleri filtreler.
        
        Kontroller:
        1. Çok karanlık görüntü (ortalama yoğunluk < MIN_MEAN_INTENSITY)
        2. Çok aydınlık görüntü (ortalama yoğunluk > MAX_MEAN_INTENSITY)
        3. Düşük kontrast (std < MIN_STD_INTENSITY) - düz/tek renkli görüntü
        4. Çok fazla siyah piksel (> MAX_BLACK_RATIO) - boş/hatalı tarama
        
        Args:
            goruntu: Kontrol edilecek görüntü
            
        Returns:
            Tuple[bool, str]: (kalite_ok, hata_mesaji)
        """
        if not KALITE_KONTROL_AKTIF:
            return True, ""
        
        if goruntu is None or goruntu.size == 0:
            return False, "Boş görüntü"
        
        # Temel istatistikler
        mean_intensity = np.mean(goruntu)
        std_intensity = np.std(goruntu)
        
        # 1. Çok karanlık kontrol
        if mean_intensity < MIN_MEAN_INTENSITY:
            return False, f"Çok karanlık (mean={mean_intensity:.1f})"
        
        # 2. Çok aydınlık kontrol
        if mean_intensity > MAX_MEAN_INTENSITY:
            return False, f"Çok aydınlık (mean={mean_intensity:.1f})"
        
        # 3. Düşük kontrast kontrol (düz görüntü)
        if std_intensity < MIN_STD_INTENSITY:
            return False, f"Düşük kontrast (std={std_intensity:.1f})"
        
        # 4. Siyah piksel oranı kontrol (boş görüntü)
        black_pixels = np.sum(goruntu < 10)
        black_ratio = black_pixels / goruntu.size
        if black_ratio > MAX_BLACK_RATIO:
            return False, f"Çok fazla siyah piksel ({black_ratio*100:.1f}%)"
        
        # Tüm kontroller başarılı
        return True, ""
    
    def goruntu_yukle(self, dosya_yolu: str) -> Optional[np.ndarray]:
        """
        Görüntü dosyasını yükle ve gri tonlamaya çevir.
        
        Args:
            dosya_yolu: Görüntü dosyasının yolu
            
        Returns:
            np.ndarray veya None
        """
        try:
            buffer = np.fromfile(str(dosya_yolu), dtype=np.uint8)
            goruntu = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if goruntu is None:
                raise ValueError("OpenCV görüntüyü decode edemedi")
            return goruntu
        except Exception as e:
            print(f"[HATA] Görüntü yüklenemedi {dosya_yolu}: {e}")
            return None

    def goruntu_kaydet(self, goruntu: np.ndarray, cikti_yolu: str) -> bool:
        """İşlenmiş görüntüyü kaydet; başarı durumunu döndür."""
        try:
            arr = np.asarray(goruntu)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            uzanti = Path(cikti_yolu).suffix.lower() or ".png"
            basarili, encoded = cv2.imencode(uzanti, arr)
            if not basarili or encoded is None:
                raise ValueError("OpenCV görüntüyü encode edemedi")
            encoded.tofile(str(cikti_yolu))
            hedef = Path(cikti_yolu)
            if not hedef.is_file() or hedef.stat().st_size <= 0:
                raise IOError("Kayit sonrasi dosya bulunamadi veya bos")
            return True
        except Exception as e:
            print(f"[HATA] Görüntü kaydedilemedi {cikti_yolu}: {e}")
            return False
