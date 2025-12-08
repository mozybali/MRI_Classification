#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
veri_olustur_ve_scale_et.py
---------------------------
CSV oluşturma ve Min-Max scaling'i bir arada yapan master script.

Adımlar:
  1) Tüm görüntüler için CSV oluştur
  2) İstatistikleri hesapla
  3) Min-Max scaling uygula
  4) Sonuçları kaydet

Çalıştırma:
    python scripts/veri_olustur_ve_scale_et.py
"""

import os
import sys
from pathlib import Path

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.ayarlar import CIKTI_KLASORU
from goruntu_isleme_mri.csv_olusturucu import (
    tum_gorseller_icin_csv_olustur,
    istatistikleri_kaydet,
)
from goruntu_isleme_mri.veri_normalizasyon import (
    csv_dosyasina_minmax_scaling_uygula,
    csv_dosyasina_robust_scaling_uygula,
)


def main():
    """Ana fonksiyon: CSV oluştur ve scaling uygula."""
    
    print("=" * 70)
    print("VERİ OLUŞTURMA VE NORMALIZASYON İŞLEMLERİ")
    print("=" * 70)
    
    # Adım 1: Temel CSV oluştur
    print("\n[ADIM 1] Tüm görüntüler için CSV dosyası oluşturuluyor...")
    csv_yolu = tum_gorseller_icin_csv_olustur(
        cikti_klasoru=CIKTI_KLASORU,
        csv_dosya_adi="goruntu_ozellikleri.csv"
    )
    
    if not os.path.exists(csv_yolu):
        print("[HATA] CSV dosyası oluşturulamadı!")
        return
    
    # Adım 2: İstatistikleri hesapla
    print("\n[ADIM 2] İstatistikler hesaplanıyor...")
    istat_yolu = istatistikleri_kaydet(
        csv_dosya_yolu=csv_yolu,
        istatistik_csv_adi="istatistikler.csv"
    )
    
    # Adım 3: Min-Max Scaling uygula
    print("\n[ADIM 3] Min-Max Scaling uygulanıyor (0-1 aralığı)...")
    minmax_csv_yolu, minmax_scaler = csv_dosyasina_minmax_scaling_uygula(
        csv_dosya_yolu=csv_yolu,
        cikti_dosya_adi="goruntu_ozellikleri_minmax_scaled.csv"
    )
    
    # Adım 4: Robust Scaling uygula (isteğe bağlı)
    print("\n[ADIM 4] Robust Scaling uygulanıyor (istatistiksel aykırı değerlere dayanıklı)...")
    robust_csv_yolu, robust_scaler = csv_dosyasina_robust_scaling_uygula(
        csv_dosya_yolu=csv_yolu,
        cikti_dosya_adi="goruntu_ozellikleri_robust_scaled.csv"
    )
    
    # Özet
    print("\n" + "=" * 70)
    print("İŞLEMLER TAMAMLANDI")
    print("=" * 70)
    print(f"\n[ÇIKTI DOSYALARI]")
    print(f"  1. Orijinal CSV: {csv_yolu}")
    print(f"  2. İstatistikler: {istat_yolu}")
    print(f"  3. Min-Max Scaled CSV: {minmax_csv_yolu}")
    print(f"  4. Robust Scaled CSV: {robust_csv_yolu}")
    
    print(f"\n[AÇIKLAMA]")
    print(f"  • Min-Max Scaling: Tüm değerleri [0, 1] aralığına normalize eder")
    print(f"    Formül: X_scaled = (X - min) / (max - min)")
    print(f"    Avantaj: Değer aralığı sabit ve anlaşılır")
    print(f"    Dezavantaj: Aykırı değerlere duyarlı")
    print(f"")
    print(f"  • Robust Scaling: Medyan ve IQR (çeyreklik aralığı) kullanır")
    print(f"    Formül: X_scaled = (X - median) / IQR")
    print(f"    Avantaj: Aykırı değerlere dayanıklı")
    print(f"    Dezavantaj: Değer aralığı değişken olabilir")
    
    print(f"\n[KULLANIM ÖNERİLERİ]")
    print(f"  ✓ Min-Max Scaled: Neural Networks, SVM, KNN için ideal")
    print(f"  ✓ Robust Scaled: Aykırı değerlerin olduğu durumlarda tercih edin")


if __name__ == "__main__":
    main()
