#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
minmax_scaling_ornegi.py
------------------------
Min-Max Scaling'in basit bir örneği.

Çalıştırma:
    python scripts/minmax_scaling_ornegi.py
"""

import sys
from pathlib import Path
import pandas as pd

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.ayarlar import CIKTI_KLASORU
from goruntu_isleme_mri.csv_olusturucu import (
    tum_gorseller_icin_csv_olustur,
    csv_ye_minmax_scaling_uygula,
)


def main():
    """Min-Max scaling örneği."""
    
    print("=" * 70)
    print("MIN-MAX SCALING ÖRNEĞİ")
    print("=" * 70)
    
    # Adım 1: CSV oluştur
    print("\n[ADIM 1] CSV dosyası oluşturuluyor...")
    csv_yolu = tum_gorseller_icin_csv_olustur(
        cikti_klasoru=CIKTI_KLASORU,
        csv_dosya_adi="goruntu_ozellikleri.csv"
    )
    
    if not Path(csv_yolu).exists():
        print("[HATA] CSV dosyası oluşturulamadı!")
        return
    
    print(f"✓ CSV başarıyla oluşturuldu: {csv_yolu}")
    
    # Adım 2: Orijinal CSV'yi göster
    print("\n[ADIM 2] Orijinal CSV'den örnek satırlar:")
    df_orijinal = pd.read_csv(csv_yolu)
    print(f"Şekil: {df_orijinal.shape}")
    print("\nSayısal sütunların istatistikleri:")
    print(df_orijinal.describe())
    
    # Adım 3: Min-Max Scaling uygula
    print("\n[ADIM 3] Min-Max Scaling uygulanıyor...")
    scaled_csv_yolu, scaler_stats = csv_ye_minmax_scaling_uygula(
        csv_dosya_yolu=csv_yolu,
        cikti_dosya_adi="goruntu_ozellikleri_scaled.csv"
    )
    
    print(f"✓ Min-Max Scaled CSV kaydedildi: {scaled_csv_yolu}")
    
    # Adım 4: Scaled CSV'yi göster
    print("\n[ADIM 4] Ölçeklenmiş CSV'den örnek satırlar:")
    df_scaled = pd.read_csv(scaled_csv_yolu)
    print(f"Şekil: {df_scaled.shape}")
    print("\nSayısal sütunların istatistikleri (Scaled):")
    print(df_scaled.describe())
    
    # Adım 5: Karşılaştırma
    print("\n[ADIM 5] Karşılaştırma:")
    karsilastir_sutunlar = [
        col for col in df_orijinal.select_dtypes(include=['float64', 'int64']).columns
        if col not in ['genislik', 'yukseklik', 'piksel_sayisi', 'boyut_bayt', 'etiket']
    ]
    
    if karsilastir_sutunlar:
        print(f"\nÖrnek sütun: '{karsilastir_sutunlar[0]}'")
        print(f"{'Orijinal (Min-Max)':30} → {'Scaled (0-1)':30}")
        print("-" * 65)
        
        for idx in range(min(5, len(df_orijinal))):
            orig = df_orijinal[karsilastir_sutunlar[0]].iloc[idx]
            scl = df_scaled[karsilastir_sutunlar[0]].iloc[idx]
            print(f"{orig:30.4f} → {scl:30.4f}")
    
    print("\n" + "=" * 70)
    print("[AÇIKLAMA]")
    print("=" * 70)
    print("""
Min-Max Scaling (0-1 Normalizasyonu):
  • Tüm değerleri [0, 1] aralığına dönüştürür
  • Formül: X_scaled = (X - min) / (max - min)
  • Avantaj: 
    - Değer aralığı sabit ve anlaşılır
    - Neural Networks, SVM, KNN gibi algoritmalar için ideal
  • Dezavantaj:
    - Aykırı değerlere (outliers) duyarlı
    - Eğer yeni veri min/max'dan dışarıda olursa [0, 1] dışına çıkabilir

Kullanım Senaryoları:
  ✓ Görüntü işleme özellikleri (yoğunluk, kontrast, entropi)
  ✓ Makine öğrenmesi veri ön işleme
  ✓ Derin öğrenme (Neural Networks)
  ✓ Benzerlik tabanlı algoritmalar (KNN, K-Means)
""")
    
    print(f"[DOSYALARIN YERİ]")
    print(f"  Orijinal CSV: {csv_yolu}")
    print(f"  Scaled CSV: {scaled_csv_yolu}")


if __name__ == "__main__":
    main()
