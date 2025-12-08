#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
goruntu_isleme_kontrol_paneli.py
---------------------------------
Görüntü işleme pipeline'ının tüm aşamalarını kontrol eden interaktif uygulama.

Menü:
  1. Tek görüntü işle ve görüntüle
  2. Toplu görüntü işleme
  3. CSV oluştur
  4. Min-Max scaling uygula
  5. İstatistikleri göster
  6. Veri seti kontrol et

Çalıştırma:
    python goruntu_isleme_kontrol_paneli.py
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Parent dizini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from goruntu_isleme_mri.ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    VERI_ARTIRMA_AKTIF,
    SINIF_KLASORLERI,
)
from goruntu_isleme_mri.io_araclari import (
    klasor_olustur_yoksa,
    girdi_gorsellerini_listele,
    goruntu_gri_olarak_oku,
    goruntu_dosyaya_kaydet,
)
from goruntu_isleme_mri.on_isleme_adimlari import tek_goruntu_on_isle
from goruntu_isleme_mri.csv_olusturucu import (
    tum_gorseller_icin_csv_olustur,
    istatistikleri_kaydet,
    csv_ye_minmax_scaling_uygula,
)
from goruntu_isleme_mri.veri_artirma import rastgele_artirma_uygula


class GoruntulemeKontrolPaneli:
    """Görüntü işleme pipeline'ını kontrol eden ana sınıf."""
    
    def __init__(self):
        self.girdi_klasoru = GIRDİ_KLASORU
        self.cikti_klasoru = CIKTI_KLASORU
        self.baslik = "GÖRÜNTÜ İŞLEME KONTROL PANELİ"
        
    def ekrani_temizle(self):
        """Terminal ekranını temizle."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def ana_menu(self) -> int:
        """Ana menüyü göster ve seçim al."""
        self.ekrani_temizle()
        print("=" * 70)
        print(self.baslik.center(70))
        print("=" * 70)
        
        print("\n[MENÜ]")
        print("  1. Tek görüntü işle ve görüntüle")
        print("  2. Toplu görüntü işleme")
        print("  3. CSV oluştur")
        print("  4. Min-Max scaling uygula")
        print("  5. İstatistikleri göster")
        print("  6. Veri seti kontrol et")
        print("  7. Çıkış")
        print("-" * 70)
        
        try:
            secim = int(input("\nSeçim yapınız (1-7): "))
            return secim
        except ValueError:
            input("\nHatalı giriş! Enter'e basınız...")
            return 0
    
    def tek_goruntu_isle(self):
        """Tek bir görüntüyü işle ve sonuçları göster."""
        self.ekrani_temizle()
        print("=" * 70)
        print("TEK GÖRÜNTÜ İŞLEME".center(70))
        print("=" * 70)
        
        # Görüntü dosyalarını listele
        girdi_listesi = girdi_gorsellerini_listele(self.girdi_klasoru)
        
        if not girdi_listesi:
            print("\n[UYARI] Girdi klasöründe görüntü bulunamadı!")
            print(f"Klasör: {self.girdi_klasoru}")
            input("\nDevam etmek için Enter'e basınız...")
            return
        
        # Görüntüleri listele
        print(f"\nToplam {len(girdi_listesi)} görüntü bulundu:\n")
        for i, dosya_bilgisi in enumerate(girdi_listesi[:10], 1):
            print(f"  {i}. {Path(dosya_bilgisi['path']).name} ({dosya_bilgisi['sinif']})")
        
        if len(girdi_listesi) > 10:
            print(f"  ... ve {len(girdi_listesi) - 10} tane daha")
        
        # Seçim al
        try:
            secim = int(input(f"\nGörüntü numarası seçiniz (1-{len(girdi_listesi)}): "))
            if 1 <= secim <= len(girdi_listesi):
                dosya_bilgisi = girdi_listesi[secim - 1]
                girdi_yolu = dosya_bilgisi["path"]
                
                # Görüntüyü işle
                print(f"\nİşleniyor: {Path(girdi_yolu).name}...")
                goruntu_gri = goruntu_gri_olarak_oku(girdi_yolu)
                on_islenmis, meta = tek_goruntu_on_isle(goruntu_gri)
                
                # Sonuçları göster
                print("\n[SONUÇLAR]")
                print(f"  Girdi boyutu: {goruntu_gri.shape}")
                print(f"  Çıktı boyutu: {on_islenmis.shape}")
                print(f"  Yoğunluk aralığı: [{on_islenmis.min()}, {on_islenmis.max()}]")
                
                if meta:
                    print("\n[META BİLGİLER]")
                    for anahtar, deger in list(meta.items())[:5]:
                        print(f"  {anahtar}: {deger}")
                
                # Görüntü kaydet
                cikti_yolu = os.path.join(
                    self.cikti_klasoru,
                    dosya_bilgisi['sinif'],
                    Path(girdi_yolu).name
                )
                klasor_olustur_yoksa(os.path.dirname(cikti_yolu))
                goruntu_dosyaya_kaydet(cikti_yolu, on_islenmis)
                print(f"\n✓ İşlenmiş görüntü kaydedildi: {cikti_yolu}")
                
            else:
                print("\nGeçersiz seçim!")
        except ValueError:
            print("\nHatalı giriş!")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def toplu_goruntu_isle(self):
        """Toplu görüntü işleme yapılan."""
        self.ekrani_temizle()
        print("=" * 70)
        print("TOPLU GÖRÜNTÜ İŞLEME".center(70))
        print("=" * 70)
        
        girdi_listesi = girdi_gorsellerini_listele(self.girdi_klasoru)
        
        if not girdi_listesi:
            print("\n[UYARI] Girdi klasöründe görüntü bulunamadı!")
            input("\nDevam etmek için Enter'e basınız...")
            return
        
        print(f"\nToplam {len(girdi_listesi)} görüntü işlenecek.")
        
        try:
            devam = input("Devam etmek istiyor musunuz? (E/H): ").upper()
            if devam != 'E':
                return
        except:
            return
        
        print("\n[İŞLEM BAŞLANIYOR...]")
        islem_sayisi = 0
        hata_sayisi = 0
        
        for i, dosya_bilgisi in enumerate(girdi_listesi, 1):
            girdi_yolu = dosya_bilgisi["path"]
            sinif_adi = dosya_bilgisi["sinif"]
            
            try:
                # Görüntüyü işle
                goruntu_gri = goruntu_gri_olarak_oku(girdi_yolu)
                on_islenmis, meta = tek_goruntu_on_isle(goruntu_gri)
                
                # Kaydet
                cikti_yolu = os.path.join(
                    self.cikti_klasoru,
                    sinif_adi,
                    Path(girdi_yolu).name
                )
                klasor_olustur_yoksa(os.path.dirname(cikti_yolu))
                goruntu_dosyaya_kaydet(cikti_yolu, on_islenmis)
                
                islem_sayisi += 1
                
                # Veri artırma (isteğe bağlı)
                if VERI_ARTIRMA_AKTIF:
                    for j in range(2):  # 2 ekstra kopya
                        aug_goruntu = rastgele_artirma_uygula(on_islenmis)
                        aug_yolu = cikti_yolu.replace(
                            Path(girdi_yolu).stem,
                            f"{Path(girdi_yolu).stem}_aug{j+1}"
                        )
                        goruntu_dosyaya_kaydet(aug_yolu, aug_goruntu)
                
                if i % 10 == 0:
                    print(f"  [{i}/{len(girdi_listesi)}] işlendi...")
                    
            except Exception as e:
                print(f"  [HATA] {Path(girdi_yolu).name}: {str(e)}")
                hata_sayisi += 1
        
        print(f"\n[TAMAMLANDI]")
        print(f"  Başarılı: {islem_sayisi}")
        print(f"  Hatalar: {hata_sayisi}")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def csv_olustur_menu(self):
        """CSV oluşturma menüsü."""
        self.ekrani_temizle()
        print("=" * 70)
        print("CSV OLUŞTURMA".center(70))
        print("=" * 70)
        
        if not os.path.exists(self.cikti_klasoru):
            print("\n[HATA] Çıktı klasörü bulunamadı!")
            print(f"Lütfen önce toplu görüntü işleme yapınız.")
            input("\nDevam etmek için Enter'e basınız...")
            return
        
        print("\n[İŞLEM]")
        print("  CSV dosyası oluşturuluyor...")
        
        try:
            csv_yolu = tum_gorseller_icin_csv_olustur(
                cikti_klasoru=self.cikti_klasoru,
                csv_dosya_adi="goruntu_ozellikleri.csv"
            )
            
            print(f"\n✓ CSV başarıyla oluşturuldu!")
            print(f"  Dosya: {csv_yolu}")
            
            # CSV istatistiklerini göster
            if os.path.exists(csv_yolu):
                df = pd.read_csv(csv_yolu)
                print(f"  Satır sayısı: {len(df)}")
                print(f"  Sütun sayısı: {len(df.columns)}")
                
        except Exception as e:
            print(f"\n[HATA] CSV oluşturulamadı: {str(e)}")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def scaling_uygula_menu(self):
        """Min-Max scaling menüsü."""
        self.ekrani_temizle()
        print("=" * 70)
        print("MIN-MAX SCALING".center(70))
        print("=" * 70)
        
        csv_yolu = os.path.join(self.cikti_klasoru, "goruntu_ozellikleri.csv")
        
        if not os.path.exists(csv_yolu):
            print(f"\n[HATA] CSV dosyası bulunamadı!")
            print(f"Lütfen önce CSV oluşturunuz.")
            input("\nDevam etmek için Enter'e basınız...")
            return
        
        print("\n[İŞLEM]")
        print("  Min-Max scaling uygulanıyor...")
        
        try:
            scaled_csv, stats = csv_ye_minmax_scaling_uygula(
                csv_dosya_yolu=csv_yolu,
                cikti_dosya_adi="goruntu_ozellikleri_scaled.csv"
            )
            
            print(f"\n✓ Scaling başarıyla uygulandı!")
            print(f"  Dosya: {scaled_csv}")
            print(f"  Ölçeklenen sütun sayısı: {len(stats)}")
            
            # İlk 3 sütunun istatistiklerini göster
            print("\n  [ÖLÇEKLEME İSTATİSTİKLERİ]")
            for i, (col, stat) in enumerate(list(stats.items())[:3]):
                print(f"    {col}")
                print(f"      Min: {stat['min']:.4f}, Max: {stat['max']:.4f}")
                
        except Exception as e:
            print(f"\n[HATA] Scaling uygulanamadı: {str(e)}")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def istatistikleri_goster(self):
        """Veri seti istatistiklerini göster."""
        self.ekrani_temizle()
        print("=" * 70)
        print("VERİ SETİ İSTATİSTİKLERİ".center(70))
        print("=" * 70)
        
        csv_yolu = os.path.join(self.cikti_klasoru, "goruntu_ozellikleri.csv")
        
        if not os.path.exists(csv_yolu):
            print(f"\n[HATA] CSV dosyası bulunamadı!")
            input("\nDevam etmek için Enter'e basınız...")
            return
        
        try:
            df = pd.read_csv(csv_yolu)
            
            print(f"\n[GENEL BİLGİLER]")
            print(f"  Toplam satır: {len(df)}")
            print(f"  Toplam sütun: {len(df.columns)}")
            
            # Sınıf dağılımı
            if 'sinif' in df.columns:
                print(f"\n[SINIF DAĞILIMI]")
                sinif_dagit = df['sinif'].value_counts()
                for sinif, sayi in sinif_dagit.items():
                    yuzde = (sayi / len(df)) * 100
                    print(f"  {sinif}: {sayi} ({yuzde:.1f}%)")
            
            # Sayısal sütunların istatistikleri
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n[SAYISAL ÖZELLİKLER]")
                for col in list(numeric_cols)[:5]:
                    print(f"  {col}")
                    print(f"    Ort: {df[col].mean():.4f}, Std: {df[col].std():.4f}")
                    print(f"    Min: {df[col].min():.4f}, Max: {df[col].max():.4f}")
            
        except Exception as e:
            print(f"\n[HATA] İstatistikler gösterilemedi: {str(e)}")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def veri_seti_kontrol_et(self):
        """Veri setinin durumunu kontrol et."""
        self.ekrani_temizle()
        print("=" * 70)
        print("VERİ SETİ KONTROL".center(70))
        print("=" * 70)
        
        print("\n[KONTROL EDILIYOR]")
        
        # Girdi klasörü
        print(f"\n1. GİRDİ KLASÖRÜ")
        if os.path.exists(self.girdi_klasoru):
            girdi_listesi = girdi_gorsellerini_listele(self.girdi_klasoru)
            print(f"   ✓ Klasör var: {self.girdi_klasoru}")
            print(f"   ✓ Görüntü sayısı: {len(girdi_listesi)}")
            
            if girdi_listesi:
                sinif_sayilari = {}
                for dosya in girdi_listesi:
                    sinif = dosya['sinif']
                    sinif_sayilari[sinif] = sinif_sayilari.get(sinif, 0) + 1
                
                for sinif, sayi in sinif_sayilari.items():
                    print(f"     - {sinif}: {sayi}")
        else:
            print(f"   ✗ Klasör bulunamadı: {self.girdi_klasoru}")
        
        # Çıktı klasörü
        print(f"\n2. ÇIKTI KLASÖRÜ")
        if os.path.exists(self.cikti_klasoru):
            print(f"   ✓ Klasör var: {self.cikti_klasoru}")
            
            # İşlenmiş görüntü sayısı
            toplam_goruntu = 0
            for sinif_adi in SINIF_KLASORLERI:
                sinif_klasoru = os.path.join(self.cikti_klasoru, sinif_adi)
                if os.path.exists(sinif_klasoru):
                    goruntu_sayisi = len([f for f in os.listdir(sinif_klasoru) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if goruntu_sayisi > 0:
                        print(f"     - {sinif_adi}: {goruntu_sayisi}")
                        toplam_goruntu += goruntu_sayisi
            
            print(f"   ✓ Toplam işlenmiş görüntü: {toplam_goruntu}")
        else:
            print(f"   ✗ Klasör bulunamadı: {self.cikti_klasoru}")
        
        # CSV dosyaları
        print(f"\n3. CSV DOSYALARI")
        csv_yolu = os.path.join(self.cikti_klasoru, "goruntu_ozellikleri.csv")
        scaled_csv_yolu = os.path.join(self.cikti_klasoru, "goruntu_ozellikleri_scaled.csv")
        
        if os.path.exists(csv_yolu):
            df = pd.read_csv(csv_yolu)
            print(f"   ✓ goruntu_ozellikleri.csv ({len(df)} satır)")
        else:
            print(f"   ✗ goruntu_ozellikleri.csv bulunamadı")
        
        if os.path.exists(scaled_csv_yolu):
            df_scaled = pd.read_csv(scaled_csv_yolu)
            print(f"   ✓ goruntu_ozellikleri_scaled.csv ({len(df_scaled)} satır)")
        else:
            print(f"   ✗ goruntu_ozellikleri_scaled.csv bulunamadı")
        
        input("\nDevam etmek için Enter'e basınız...")
    
    def calistir(self):
        """Ana loop'u çalıştır."""
        while True:
            secim = self.ana_menu()
            
            if secim == 1:
                self.tek_goruntu_isle()
            elif secim == 2:
                self.toplu_goruntu_isle()
            elif secim == 3:
                self.csv_olustur_menu()
            elif secim == 4:
                self.scaling_uygula_menu()
            elif secim == 5:
                self.istatistikleri_goster()
            elif secim == 6:
                self.veri_seti_kontrol_et()
            elif secim == 7:
                self.ekrani_temizle()
                print("Hoşça kalınız!")
                break
            else:
                input("\nGeçersiz seçim! Enter'e basınız...")


def main():
    """Ana fonksiyon."""
    # Başlangıç kontrolleri
    klasor_olustur_yoksa(GIRDİ_KLASORU)
    klasor_olustur_yoksa(CIKTI_KLASORU)
    
    # Kontrol panelini başlat
    panel = GoruntulemeKontrolPaneli()
    panel.calistir()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram durduruldu.")
    except Exception as e:
        print(f"\n[HATA] {str(e)}")
        import traceback
        traceback.print_exc()
