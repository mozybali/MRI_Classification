"""Goruntu yukleme, kalite kontrol ve kaydetme islemleri."""

from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselKaliteIOMixin:
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
            goruntu = Image.open(dosya_yolu)
            if goruntu.mode != 'L':
                goruntu = goruntu.convert('L')
            return np.array(goruntu)
        except Exception as e:
            print(f"[HATA] Görüntü yüklenemedi {dosya_yolu}: {e}")
            return None

    def goruntu_kaydet(self, goruntu: np.ndarray, cikti_yolu: str):
        """İşlenmiş görüntüyü kaydet."""
        try:
            pil_img = Image.fromarray(goruntu)
            pil_img.save(cikti_yolu)
        except Exception as e:
            print(f"[HATA] Görüntü kaydedilemedi {cikti_yolu}: {e}")
