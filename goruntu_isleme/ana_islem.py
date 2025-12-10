#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ana_islem.py
------------
MRI görüntü işleme ana menü ve işlem yöneticisi.
Tüm işlemleri tek bir yerden yönetir.
"""

import sys
from pathlib import Path

# Modül yolunu ekle
sys.path.insert(0, str(Path(__file__).parent))

from ayarlar import *
from goruntu_isleyici import GorselIsleyici
from ozellik_cikarici import OzellikCikarici, veri_boluntule


def ana_menu():
    """
    Ana menüyü göster.
    
    Kullanıcıya 6 farklı işlem seçeneği sunar:
    1. Görüntü ön işleme (normalizasyon, histogram eşitleme, vb.)
    2. Özellik çıkarma ve CSV oluşturma
    3. CSV'ye ölçeklendirme uygulama
    4. Veri seti bölme (eğitim/doğrulama/test)
    5. İstatistik raporu gösterme
    6. Tüm işlemleri otomatik yapma
    """
    print("\n" + "="*60)
    print("MRI GÖRÜNTÜ İŞLEME SİSTEMİ")
    print("="*60)
    print("\n1. Görüntüleri ön işle")
    print("2. Özellik çıkar ve CSV oluştur")
    print("3. CSV'ye ölçeklendirme uygula")
    print("4. Veri setini böl (eğitim/doğrulama/test)")
    print("5. İstatistik raporu göster")
    print("6. TÜM İŞLEMLERİ OTOMATIK YAP")
    print("0. Çıkış")
    print("\n" + "="*60)


def goruntu_on_isleme():
    """
    Görüntüleri ön işle.
    
    Bu fonksiyon, ham MRI görüntülerini alır ve şu işlemleri uygular:
    - Yoğunluk normalizasyonu (kontrast iyileştirme)
    - Histogram eşitleme (CLAHE)
    - Yeniden boyutlandırma (standart boyuta getirme)
    - Veri artırma (augmentation)
    
    |şlem sonunda, standartlaştırılmış görüntüler çıktı klasörüne kaydedilir.
    """
    print("\n[1] GÖRÜNTÜ ÖN İŞLEME")
    print("-" * 60)
    
    isleyici = GorselIsleyici()
    
    # Kullanıcıdan girdi al
    giris = input(f"\nGirdi klasörü (varsayılan: {VERI_SETI_KLASORU}): ").strip()
    if giris:
        giris_klasoru = Path(giris)
    else:
        giris_klasoru = VERI_SETI_KLASORU
    
    cikis = input(f"Çıktı klasörü (varsayılan: {CIKTI_KLASORU}): ").strip()
    if cikis:
        cikti_klasoru = Path(cikis)
    else:
        cikti_klasoru = CIKTI_KLASORU
    
    # İşlemi başlat
    istatistikler = isleyici.tum_gorselleri_isle(cikti_klasoru)
    
    if istatistikler:
        print("\n✓ Görüntü işleme tamamlandı!")
    else:
        print("\n✗ Görüntü işleme başarısız!")


def ozellik_cikar():
    """
    Özellik çıkar ve CSV oluştur.
    
    Bu fonksiyon, işlenmiş görüntülerden makine öğrenmesi için
    sayısal özellikler çıkarır ve bir CSV dosyasına kaydeder.
    
    Çıkarılan özellikler:
    - Boyut bilgileri (genişlik, yükseklik)
    - Yoğunluk istatistikleri (ortalama, std, min, max, percentile'ler)
    - Doku özellikleri (entropi, kontrast, homojenlik)
    
    Bu CSV, model eğitiminde kullanılır.
    """
    print("\n[2] ÖZELLİK ÇIKARMA VE CSV OLUŞTURMA")
    print("-" * 60)
    
    cikarici = OzellikCikarici()
    
    # Kullanıcıdan girdi al
    giris = input(f"\nİşlenmiş görüntüler klasörü (varsayılan: {CIKTI_KLASORU}): ").strip()
    if giris:
        giris_klasoru = Path(giris)
    else:
        giris_klasoru = CIKTI_KLASORU
    
    # Özellik çıkar
    df = cikarici.csv_olustur(giris_klasoru)
    
    if not df.empty:
        print("\n✓ Özellik çıkarma tamamlandı!")
    else:
        print("\n✗ Özellik çıkarma başarısız!")


def scaling_uygula():
    """
    CSV'ye ölçeklendirme uygula.
    
    Makine öğrenmesi modelleri, farklı ölçeklerdeki özelliklerle
    iyi çalışamaz. Bu fonksiyon, tüm özellikleri aynı ölçeğe getirir.
    
    Ölçeklendirme metodları:
    - MinMax: Tüm değerleri [0, 1] arasına sıkıştırır
    - Robust: Aykırı değerlere karşı daha dayanıklıdır (medyan + IQR)
    - Standard: Z-score normalizasyonu (mean=0, std=1)
    - MaxAbs: Değerleri [-1, 1] aralığına ölçeklendirir
    """
    print("\n[3] ÖLÇEKLENDİRME UYGULAMA")
    print("-" * 60)
    
    cikarici = OzellikCikarici()
    
    print(f"\nMevcut metod: {SCALING_METODU}")
    print("\nMevcut ölçeklendirme metodları:")
    print("  1. minmax   - [0, 1] aralığına ölçeklendirir")
    print("  2. robust   - Medyan ve IQR (aykırı değerlere dayanıklı)")
    print("  3. standard - Z-score normalizasyonu (mean=0, std=1)")
    print("  4. maxabs   - [-1, 1] aralığına ölçeklendirir")
    
    metod = input("\nMetod seçin (minmax/robust/standard/maxabs, Enter=varsayılan): ").strip().lower()
    
    if metod not in ['minmax', 'robust', 'standard', 'maxabs', '']:
        print("[HATA] Geçersiz metod!")
        return
    
    if not metod:
        metod = SCALING_METODU
    
    df = cikarici.scaling_uygula(metod=metod)
    
    if not df.empty:
        print("\n✓ Ölçeklendirme tamamlandı!")
    else:
        print("\n✗ Ölçeklendirme başarısız!")


def veri_bol():
    """
    Veri setini böl.
    
    Veri setini üç parçaya böler:
    1. Eğitim seti (%70): Model parametrelerini öğrenmek için
    2. Doğrulama seti (%15): Hiperparametre ayarlama için
    3. Test seti (%15): Son performans değerlendirmesi için
    
    Sınıf dengesi korunur (stratified split).
    Her sınıftan aynı oranda veri her sete dağıtılır.
    """
    print("\n[4] VERİ SETİ BÖLME")
    print("-" * 60)
    print(f"\nOranlar: Eğitim={EGITIM_ORANI}, Doğrulama={DOGRULAMA_ORANI}, Test={TEST_ORANI}")
    
    veri_boluntule()


def istatistik_goster():
    """İstatistik raporu göster."""
    print("\n[5] İSTATİSTİK RAPORU")
    print("-" * 60)
    
    cikarici = OzellikCikarici()
    cikarici.istatistik_raporu()


def tum_islemleri_yap():
    """
    Tüm işlemleri otomatik yap.
    
    Bu fonksiyon, tüm işleme adımlarını sırasıyla otomatik olarak yapar:
    1. Ham görüntüleri işle (normalizasyon, boyutlandırma, artırma)
    2. İşlenmiş görüntülerden özellik çıkar
    3. Özelliklere ölçeklendirme uygula (MinMax/Robust)
    4. Veri setini eğitim/doğrulama/test olarak böl
    5. İstatistik raporu oluştur
    
    Kullanıcı müdahalesi gerektirmez, baştan sona otomatik çalışır.
    Yeni başlayanlar veya hızlı işleme için ideal.
    """"
    print("\n[6] TÜM İŞLEMLER OTOMATİK")
    print("-" * 60)
    print("\nŞu işlemler sırayla yapılacak:")
    print("  1. Görüntü ön işleme")
    print("  2. Özellik çıkarma")
    print("  3. Ölçeklendirme")
    print("  4. Veri bölme")
    print("  5. İstatistik raporu")
    
    onay = input("\nDevam etmek istiyor musunuz? (e/h): ").strip().lower()
    if onay != 'e':
        print("İşlem iptal edildi.")
        return
    
    # 1. Görüntü işleme
    print("\n\n" + "="*60)
    print("ADIM 1/5: GÖRÜNTÜ ÖN İŞLEME")
    print("="*60)
    isleyici = GorselIsleyici()
    isleyici.tum_gorselleri_isle(CIKTI_KLASORU)
    
    # 2. Özellik çıkarma
    print("\n\n" + "="*60)
    print("ADIM 2/5: ÖZELLİK ÇIKARMA")
    print("="*60)
    cikarici = OzellikCikarici()
    df = cikarici.csv_olustur(CIKTI_KLASORU)
    
    if df.empty:
        print("\n✗ Özellik çıkarma başarısız! İşlem durduruluyor.")
        return
    
    # 3. Ölçeklendirme
    print("\n\n" + "="*60)
    print("ADIM 3/5: ÖLÇEKLENDİRME")
    print("="*60)
    cikarici.scaling_uygula()
    
    # 4. Veri bölme
    print("\n\n" + "="*60)
    print("ADIM 4/5: VERİ BÖLME")
    print("="*60)
    veri_boluntule()
    
    # 5. İstatistik raporu
    print("\n\n" + "="*60)
    print("ADIM 5/5: İSTATİSTİK RAPORU")
    print("="*60)
    cikarici.istatistik_raporu()
    
    print("\n\n" + "="*60)
    print("✓ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print("="*60)


def main():
    """Ana program."""
    while True:
        try:
            ana_menu()
            secim = input("\nSeçiminiz: ").strip()
            
            if secim == '0':
                print("\nÇıkılıyor...")
                break
            elif secim == '1':
                goruntu_on_isleme()
            elif secim == '2':
                ozellik_cikar()
            elif secim == '3':
                scaling_uygula()
            elif secim == '4':
                veri_bol()
            elif secim == '5':
                istatistik_goster()
            elif secim == '6':
                tum_islemleri_yap()
            else:
                print("\n[HATA] Geçersiz seçim! Lütfen 0-6 arası bir sayı girin.")
            
            input("\nDevam etmek için Enter'a basın...")
            
        except KeyboardInterrupt:
            print("\n\nProgram kullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            print(f"\n[HATA] Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
            input("\nDevam etmek için Enter'a basın...")


if __name__ == "__main__":
    main()
