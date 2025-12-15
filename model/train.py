#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
MRI sınıflandırma modeli eğitim scripti.
Kullanıcı dostu, interaktif model eğitim arayüzü.

Kullanım:
    python3 train.py                    # İnteraktif mod
    python3 train.py --model xgboost    # Hızlı başlatma
    python3 train.py --auto             # Tüm işlemleri otomatik yap
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Modül yolunu ekle
sys.path.insert(0, str(Path(__file__).parent))

from ayarlar import *
from model_egitici import ModelEgitici


def banner():
    """Hoş geldin banner'ı göster."""
    print("\n" + "="*70)
    print(" "*15 + "MRI SINIFLANDIRMA MODEL EĞİTİMİ")
    print("="*70)
    print("\nDemans Seviyesi Sınıflandırması")
    print("  • NonDemented (0)")
    print("  • VeryMildDemented (1)")
    print("  • MildDemented (2)")
    print("  • ModerateDemented (3)")
    print()


def kontrol_veri_seti():
    """CSV dosyasının varlığını kontrol et."""
    if not VERI_CSV.exists():
        print(f"\n❌ HATA: Veri dosyası bulunamadı!")
        print(f"Aranan: {VERI_CSV}")
        print(f"\n⚠️  Önce görüntü işleme adımlarını tamamlayın:")
        print(f"   1. cd ../goruntu_isleme")
        print(f"   2. python3 ana_islem.py")
        print(f"   3. Menüden '6' seçerek tüm işlemleri yapın\n")
        return False
    
    print(f"✓ Veri dosyası bulundu: {VERI_CSV}")
    return True


def model_sec():
    """
    Kullanıcıdan model tipini al.
    
    3 farklı model seçeneği sunar:
    1. XGBoost - Yüksek performans, gradient boosting (önerilen)
    2. LightGBM - Hızlı eğitim, büyük veri setleri için
    3. Linear SVM - Basit, hızlı ama düşük doğruluk
    
    Returns:
        str: Model tipi ('xgboost', 'lightgbm', 'svm')
    """
    print("\n" + "-"*70)
    print("MODEL SEÇİMİ")
    print("-"*70)
    print("\n1. XGBoost (Önerilen)")
    print("   • Yüksek doğruluk")
    print("   • Gradient boosting tabanlı")
    print("   • Orta hız")
    print()
    print("2. LightGBM")
    print("   • Çok hızlı eğitim")
    print("   • Büyük veri setleri için ideal")
    print("   • XGBoost'a yakın performans")
    print()
    print("3. Linear SVM")
    print("   • Çok hızlı")
    print("   • Basit model")
    print("   • Düşük doğruluk")
    print()
    
    while True:
        secim = input("Seçiminiz (1-3, varsayılan=1): ").strip()
        
        if secim == "" or secim == "1":
            return "xgboost"
        elif secim == "2":
            return "lightgbm"
        elif secim == "3":
            return "svm"
        else:
            print("❌ Geçersiz seçim! 1, 2 veya 3 girin.")


def smote_sec():
    """
    SMOTE kullanımını sor.
    
    SMOTE (Synthetic Minority Over-sampling Technique):
    - Az olan sınıflar için yapay örnekler üretir
    - Sınıf dengesizliğini giderir
    - Model performansını artırır
    
    Veri setimizde:
    - NonDemented: ~9600 (çok)
    - ModerateDemented: ~6464 (az) <- SMOTE bu sınıfı dengeler
    
    Returns:
        bool: SMOTE kullanılsın mı?
    """
    print("\n" + "-"*70)
    print("VERİ DENGELEME (SMOTE)")
    print("-"*70)
    print("\nSınıf dengesizliği var:")
    print("  • NonDemented: ~9600 örnek")
    print("  • MildDemented: ~8960 örnek")
    print("  • VeryMildDemented: ~8960 örnek")
    print("  • ModerateDemented: ~6464 örnek (en az)")
    print()
    print("SMOTE (Synthetic Minority Over-sampling):")
    print("  ✓ Azınlık sınıflar için sentetik örnekler üretir")
    print("  ✓ Model dengesizliğini azaltır")
    print("  ✗ Eğitim süresini artırır")
    print()
    
    secim = input("SMOTE kullanılsın mı? (E/h, varsayılan=E): ").strip().lower()
    return secim != "h" and secim != "n" and secim != "no"


def feature_selection_sec():
    """Feature selection kullanımını sor."""
    print("\n" + "-"*70)
    print("ÖZELLİK SEÇİMİ (Feature Selection)")
    print("-"*70)
    print("\nEn önemli özellikleri seçerek:")
    print("  ✓ Model basitleşir")
    print("  ✓ Overfitting azalır")
    print("  ✓ Eğitim hızlanır")
    print("  ✗ Biraz doğruluk kaybı olabilir")
    print()
    
    secim = input("Feature selection kullanılsın mı? (e/H, varsayılan=H): ").strip().lower()
    return secim == "e" or secim == "yes"


def grid_search_sec():
    """Grid search kullanımını sor."""
    print("\n" + "-"*70)
    print("HİPERPARAMETRE OPTİMİZASYONU (Grid Search)")
    print("-"*70)
    print("\nOtomatik parametre ayarlama:")
    print("  ✓ En iyi parametreleri bulur")
    print("  ✓ Model performansını artırır")
    print("  ✗ ÇOK uzun sürer (saatler)")
    print()
    print("⚠️  Önerilmez (ilk eğitimde varsayılan parametreler yeterli)")
    print()
    
    secim = input("Grid search kullanılsın mı? (e/H, varsayılan=H): ").strip().lower()
    return secim == "e" or secim == "yes"


def egitim_yap(model_tipi, smote_aktif, feature_selection_aktif, grid_search_aktif):
    """Model eğitimini başlat."""
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("="*70)
    print(f"\nAyarlar:")
    print(f"  • Model: {model_tipi.upper()}")
    print(f"  • SMOTE: {'Evet' if smote_aktif else 'Hayır'}")
    print(f"  • Feature Selection: {'Evet' if feature_selection_aktif else 'Hayır'}")
    print(f"  • Grid Search: {'Evet' if grid_search_aktif else 'Hayır'}")
    print()
    
    input("Devam etmek için ENTER'a basın (Çıkmak için Ctrl+C)...")
    
    try:
        # Model eğitici oluştur
        egitici = ModelEgitici(
            model_tipi=model_tipi,
            smote_aktif=smote_aktif,
            feature_selection_aktif=feature_selection_aktif
        )
        
        # Veri yükle
        X_train, X_val, X_test, y_train, y_val, y_test = egitici.veri_yukle()
        
        # Feature selection
        if feature_selection_aktif:
            X_train = egitici.feature_selection(X_train, y_train, k=15)
            # Validation ve test setlerine de uygula
            if egitici.selected_features:
                X_val = X_val[egitici.selected_features]
                X_test = X_test[egitici.selected_features]
        
        # Not: Eğer feature selection yapılmadıysa bile,
        # feature isimlerinin korunduğundan emin olalım
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=egitici.feature_names)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=egitici.selected_features or egitici.feature_names)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=egitici.selected_features or egitici.feature_names)
        
        # Model oluştur
        egitici.model_olustur()
        
        # Grid search veya normal eğitim
        if grid_search_aktif:
            print("\n⚠️  Grid search başlıyor... Bu uzun sürebilir!")
            egitici.grid_search(X_train, y_train)
        else:
            egitici.egit(X_train, y_train, X_val, y_val)
        
        # Değerlendirme - metrikleri kaydet
        egitici.metrikler = egitici.degerlendir(X_test, y_test, set_adi="Test")
        
        # Çapraz doğrulama
        egitici.cross_validate(X_train, y_train)
        
        # Model kaydet
        model_yolu = egitici.model_kaydet()
        
        # Rapor oluştur
        egitici.rapor_olustur()
        
        # Grafikler
        egitici.grafik_ciz(X_test, y_test)
        
        print("\n" + "="*70)
        print("✓ EĞİTİM TAMAMLANDI!")
        print("="*70)
        print(f"\nModel kaydedildi: {model_yolu}")
        print(f"Raporlar: {RAPORLAR_KLASORU}")
        print(f"Grafikler: {GORSELLER_KLASORU}")
        print()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Eğitim kullanıcı tarafından iptal edildi.")
        return False
    except Exception as e:
        print(f"\n\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        return False


def otomatik_mod(model_tipi="xgboost"):
    """Otomatik mod - tüm işlemleri varsayılan ayarlarla yap."""
    print("\n🚀 OTOMATİK MOD")
    print("Varsayılan ayarlarla eğitim başlatılıyor...")
    return egitim_yap(
        model_tipi=model_tipi,
        smote_aktif=True,
        feature_selection_aktif=False,
        grid_search_aktif=False
    )


def interaktif_mod():
    """İnteraktif mod - kullanıcıya sor."""
    banner()
    
    # Veri kontrolü
    if not kontrol_veri_seti():
        return False
    
    # Kullanıcı seçimleri
    model_tipi = model_sec()
    smote_aktif = smote_sec()
    feature_selection_aktif = feature_selection_sec()
    grid_search_aktif = grid_search_sec()
    
    # Eğitim
    return egitim_yap(model_tipi, smote_aktif, feature_selection_aktif, grid_search_aktif)


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="MRI sınıflandırma modeli eğitimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python3 train.py                      # İnteraktif mod
  python3 train.py --auto               # Otomatik eğitim (XGBoost)
  python3 train.py --model lightgbm     # LightGBM ile hızlı başlat
  python3 train.py --auto --model svm   # SVM ile otomatik eğitim
        """
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Otomatik mod (varsayılan ayarlarla eğit)"
    )
    
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "svm"],
        default="xgboost",
        help="Model tipi (varsayılan: xgboost)"
    )
    
    args = parser.parse_args()
    
    # Mod seçimi
    if args.auto:
        basarili = otomatik_mod(args.model)
    else:
        basarili = interaktif_mod()
    
    return 0 if basarili else 1


if __name__ == "__main__":
    sys.exit(main())
