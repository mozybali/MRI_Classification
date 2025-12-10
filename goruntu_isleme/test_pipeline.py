#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pipeline.py
----------------
Güncellenmiş görüntü işleme pipeline'ını test eden script.
Tek bir görüntü üzerinde tüm adımları gösterir.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Modül yolunu ekle
sys.path.insert(0, str(Path(__file__).parent))

from ayarlar import *
from goruntu_isleyici import GorselIsleyici


def pipeline_test(goruntu_yolu: str):
    """
    Pipeline'ın tüm adımlarını test et ve görselleştir.
    
    Args:
        goruntu_yolu: Test edilecek görüntünün yolu
    """
    print("\n" + "="*70)
    print("GÖRÜNTÜ İŞLEME PIPELINE TEST")
    print("="*70)
    print(f"\nTest görüntüsü: {goruntu_yolu}")
    
    isleyici = GorselIsleyici()
    
    # Ham görüntüyü yükle
    goruntu_ham = isleyici.goruntu_yukle(goruntu_yolu)
    if goruntu_ham is None:
        print("\n❌ HATA: Görüntü yüklenemedi!")
        return
    
    print(f"\n✓ Görüntü yüklendi: {goruntu_ham.shape}")
    
    # Adım adım işleme
    asamalar = {}
    
    print("\n" + "-"*70)
    print("PİPELİNE AŞAMALARI")
    print("-"*70)
    
    # 1. Gürültü giderme
    print("\n1. Gürültü giderme (median filter)...")
    g1 = isleyici.gurultu_gider(goruntu_ham.copy(), metod='median')
    asamalar["1. Orijinal"] = goruntu_ham
    asamalar["2. Gürültü Giderme"] = g1
    
    # 2. Bias field correction
    print("2. Bias field correction...")
    g2 = isleyici.bias_field_correction(g1.copy())
    asamalar["3. Bias Correction"] = g2
    
    # 3. Skull stripping
    print("3. Skull stripping (kafatası çıkarma)...")
    g3 = isleyici.skull_strip(g2.copy())
    asamalar["4. Skull Stripping"] = g3
    
    # 4. Center of mass alignment
    print("4. Center of mass alignment (hizalama)...")
    g4 = isleyici.center_of_mass_alignment(g3.copy())
    asamalar["5. Alignment"] = g4
    
    # 5. Yoğunluk normalizasyonu
    print("5. Yoğunluk normalizasyonu...")
    g5 = isleyici.yogunluk_normalize(g4.copy())
    asamalar["6. Yoğunluk Norm."] = g5
    
    # 6. Histogram eşitleme (adaptive)
    print("6. Adaptive histogram eşitleme (CLAHE)...")
    g6 = isleyici.histogram_esitle(g5.copy(), adaptive=True)
    asamalar["7. CLAHE"] = g6
    
    # 7. Boyutlandırma
    print("7. Boyutlandırma (256x256)...")
    g7 = isleyici.boyutlandir(g6.copy())
    asamalar["8. Boyutlandırma"] = g7
    
    # 8. Z-score normalizasyonu
    print("8. Z-score normalizasyonu...")
    g8 = isleyici.z_score_normalize(g7.copy())
    asamalar["9. Z-score"] = g8
    
    print("\n✓ Tüm aşamalar tamamlandı!")
    
    # Görselleştirme
    print("\n" + "-"*70)
    print("GÖRSELLEŞTİRME")
    print("-"*70)
    print("\nGrafik oluşturuluyor...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (baslik, goruntu) in enumerate(asamalar.items()):
        if idx < 9:
            axes[idx].imshow(goruntu, cmap='gray')
            axes[idx].set_title(baslik, fontsize=10, weight='bold')
            axes[idx].axis('off')
            
            # İstatistikler
            ort = np.mean(goruntu)
            std = np.std(goruntu)
            axes[idx].text(0.05, 0.95, f'μ={ort:.1f}, σ={std:.1f}',
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=8)
    
    plt.tight_layout()
    
    # Kaydet
    cikti_klasoru = CIKTI_KLASORU / "test"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    cikti_yolu = cikti_klasoru / "pipeline_test.png"
    plt.savefig(cikti_yolu, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: {cikti_yolu}")
    
    # Augmentation testi
    print("\n" + "-"*70)
    print("VERİ ARTIRMA (AUGMENTATION) TEST")
    print("-"*70)
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    axes2 = axes2.flatten()
    
    axes2[0].imshow(g8, cmap='gray')
    axes2[0].set_title('İşlenmiş Orijinal', weight='bold')
    axes2[0].axis('off')
    
    aug_metodlar = [
        ('Elastic Deform', lambda x: isleyici.elastic_deformation(x)),
        ('Random Crop', lambda x: isleyici.random_crop_resize(x)),
        ('Gaussian Noise', lambda x: isleyici.gaussian_noise(x)),
        ('Intensity Shift', lambda x: isleyici.intensity_shift(x)),
        ('Yatay Ayna', lambda x: isleyici.yatay_ayna(x)),
        ('Dikey Ayna', lambda x: isleyici.dikey_ayna(x)),
        ('Veri Artır (Mix)', lambda x: isleyici.veri_artir(x))
    ]
    
    for idx, (metod_adi, metod_func) in enumerate(aug_metodlar, 1):
        print(f"  {idx}. {metod_adi}...")
        aug_goruntu = metod_func(g8.copy())
        axes2[idx].imshow(aug_goruntu, cmap='gray')
        axes2[idx].set_title(metod_adi, weight='bold')
        axes2[idx].axis('off')
    
    plt.tight_layout()
    
    cikti_yolu2 = cikti_klasoru / "augmentation_test.png"
    plt.savefig(cikti_yolu2, dpi=150, bbox_inches='tight')
    print(f"\n✓ Augmentation grafiği kaydedildi: {cikti_yolu2}")
    
    print("\n" + "="*70)
    print("TEST TAMAMLANDI!")
    print("="*70)
    print(f"\nÇıktılar:")
    print(f"  • Pipeline: {cikti_yolu}")
    print(f"  • Augmentation: {cikti_yolu2}")
    print(f"\nGörüntüleri görüntülemek için:")
    print(f"  open {cikti_klasoru}")
    print()


if __name__ == "__main__":
    # Kullanıcıdan görüntü yolu al
    if len(sys.argv) > 1:
        test_goruntu = sys.argv[1]
    else:
        # Varsayılan: Veri setinden ilk görüntüyü al
        print("\nVeri setinden test görüntüsü aranıyor...")
        
        for sinif in SINIF_KLASORLERI:
            sinif_klasoru = VERI_SETI_KLASORU / sinif
            if sinif_klasoru.exists():
                dosyalar = list(sinif_klasoru.glob("*.jpg")) + list(sinif_klasoru.glob("*.png"))
                if dosyalar:
                    test_goruntu = str(dosyalar[0])
                    break
        else:
            print("\n❌ HATA: Veri setinde görüntü bulunamadı!")
            print("Kullanım: python test_pipeline.py [goruntu_yolu]")
            sys.exit(1)
    
    # Test çalıştır
    pipeline_test(test_goruntu)
