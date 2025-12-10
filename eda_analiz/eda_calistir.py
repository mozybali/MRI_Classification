#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eda_calistir.py
---------------
EDA analizi çalıştırma scripti.
"""

import sys
from pathlib import Path

# Modül yolunu ekle
sys.path.insert(0, str(Path(__file__).parent))

from eda_araclar import EDAAnaLiz


def main():
    """Ana program."""
    print("\nMRI Veri Seti EDA Analizi Başlatılıyor...\n")
    
    # Kullanıcıdan girdi al
    veri_klasoru = input("Veri seti klasörü (Enter=varsayılan: ../../Veri_Seti): ").strip()
    if not veri_klasoru:
        veri_klasoru = "../../Veri_Seti"
    
    cikti_klasoru = input("Çıktı klasörü (Enter=varsayılan: eda_ciktilar): ").strip()
    if not cikti_klasoru:
        cikti_klasoru = "eda_ciktilar"
    
    # Analiz yap
    try:
        analizci = EDAAnaLiz(
            veri_klasoru=veri_klasoru,
            cikti_klasoru=cikti_klasoru
        )
        
        df = analizci.tam_analiz_yap()
        
        # İsteğe bağlı: DataFrame'i kaydet
        csv_yolu = Path(cikti_klasoru) / "veri_seti_istatistikler.csv"
        df.to_csv(csv_yolu, index=False, encoding='utf-8')
        print(f"\n✓ Veri seti CSV kaydedildi: {csv_yolu}")
        
    except Exception as e:
        print(f"\n[HATA] Analiz sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
