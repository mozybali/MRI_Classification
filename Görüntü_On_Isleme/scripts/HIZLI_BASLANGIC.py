#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HIZLI_BASLANGIC.py
------------------
Min-Max Scaling'i 3 adımda kullanmak için en hızlı örnek.

Çalıştırma:
    python scripts/HIZLI_BASLANGIC.py

Bu script:
1. CSV oluşturur
2. Min-Max scaling uygular
3. Sonuçları gösterir
"""

import sys
from pathlib import Path

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 70)
    print("MIN-MAX SCALING - HIZLI BAŞLANGIÇ")
    print("=" * 70)
    
    # Import'lar
    from goruntu_isleme_mri.ayarlar import CIKTI_KLASORU
    from goruntu_isleme_mri.csv_olusturucu import (
        tum_gorseller_icin_csv_olustur,
        csv_ye_minmax_scaling_uygula
    )
    import pandas as pd
    import os
    
    print("\n[ADIM 1/3] CSV dosyası oluşturuluyor...")
    
    # Çıktı klasörü kontrol et
    if not os.path.exists(CIKTI_KLASORU):
        print(f"[UYARI] Çıktı klasörü bulunamadı: {CIKTI_KLASORU}")
        print("[IPUCU] Önce toplu_on_isleme.py çalıştırın")
        return
    
    # CSV oluştur
    csv_yolu = tum_gorseller_icin_csv_olustur(
        cikti_klasoru=CIKTI_KLASORU,
        csv_dosya_adi="goruntu_ozellikleri.csv"
    )
    
    if not os.path.exists(csv_yolu):
        print("[HATA] CSV dosyası oluşturulamadı!")
        return
    
    print(f"✓ CSV oluşturuldu: {csv_yolu}")
    
    # CSV'yi yükle ve göster
    print("\n[ADIM 2/3] CSV'ye Min-Max scaling uygulanıyor...")
    df_orijinal = pd.read_csv(csv_yolu)
    print(f"  Veri şekli: {df_orijinal.shape} (satır, sütun)")
    print(f"  Sütunlar: {list(df_orijinal.columns)}")
    
    # Scaling uygula
    scaled_csv_yolu, scaler_stats = csv_ye_minmax_scaling_uygula(
        csv_dosya_yolu=csv_yolu,
        cikti_dosya_adi="goruntu_ozellikleri_scaled.csv"
    )
    
    print(f"✓ Scaled CSV oluşturuldu: {scaled_csv_yolu}")
    
    # Sonuçları göster
    print("\n[ADIM 3/3] Sonuçlar:")
    print("-" * 70)
    
    df_scaled = pd.read_csv(scaled_csv_yolu)
    
    # Örnek karşılaştırma
    numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in 
                   ['genislik', 'yukseklik', 'piksel_sayisi', 'boyut_bayt', 'etiket']]
    
    if feature_cols:
        col = feature_cols[0]
        print(f"\nÖrnek: '{col}' sütunu")
        print(f"{'Orijinal':20} | {'Scaled':20}")
        print("-" * 45)
        
        for i in range(min(5, len(df_orijinal))):
            orig = df_orijinal[col].iloc[i]
            scl = df_scaled[col].iloc[i]
            print(f"{orig:20.4f} | {scl:20.4f}")
    
    print("\n[İSTATİSTİKLER]")
    print(f"Toplam sütun: {len(scaler_stats)}")
    print(f"Ölçeklenen sütunlar:")
    for col, stats in list(scaler_stats.items())[:3]:  # İlk 3'ü göster
        print(f"  • {col}")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ TAMAMLANDI!")
    print("=" * 70)
    print(f"\nÇıktı dosyaları:")
    print(f"  1. Orijinal CSV: {csv_yolu}")
    print(f"  2. Scaled CSV: {scaled_csv_yolu}")
    print(f"\nŞimdi bu dosyaları makine öğrenmesi modellerinize girebilirsiniz!")
    print(f"\nDetaylı rehber için: MINMAX_SCALING_REHBERI.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[HATA] {str(e)}")
        import traceback
        traceback.print_exc()
