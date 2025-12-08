"""
csv_olusturucu.py
-----------------
Ön işlenmiş MRI görüntülerini CSV formatına dönüştürme modülü.

Fonksiyonlar:
  - goruntu_ozelliklerini_cikart(): Tek görüntü için öznitelikleri hesapla
  - tum_gorseller_icin_csv_olustur(): Tüm dataset için CSV oluştur
  - istatistikleri_kaydet(): Dataset istatistiklerini CSV'ye kaydet
  - on_isleme_log_kaydet(): Ön işleme işlemlerinin log dosyasını CSV'ye kaydet
  - csv_ye_scaling_uygula(): CSV'ye Min-Max veya Robust scaling uygula
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .ayarlar import (
    CIKTI_KLASORU,
    SINIF_KLASORLERI,
    SINIF_ETIKETI,
)
from .io_araclari import (
    girdi_gorsellerini_listele,
    klasor_olustur_yoksa,
)


def goruntu_ozelliklerini_cikart(goruntu_yolu: str) -> Dict:
    """
    Tek bir görüntü dosyasından öznitelikleri çıkar.
    
    Öznitelikler:
    - dosya_adı: Görüntü dosyasının adı
    - boyut_bayt: Dosya boyutu (byte)
    - genislik, yukseklik: Görüntü boyutları
    - en_boy_orani: Genişlik/Yükseklik oranı
    - piksel_sayisi: Toplam piksel sayısı
    - ort_yogunluk: Ortalama piksel yoğunluğu
    - std_yogunluk: Standart sapma
    - min_yogunluk, max_yogunluk: Min-Max yoğunluk
    - p1_yogunluk, p99_yogunluk: 1. ve 99. percentile
    - medyan_yogunluk: Medyan yoğunluk
    - entropi: Shannon entropisi (bilgi miktarı)
    
    Parametreler:
    - goruntu_yolu: Görüntü dosyasının tam yolu
    
    Döndürülen: Dict with features or None if error
    """
    try:
        # Dosya bilgileri
        dosya_adı = os.path.basename(goruntu_yolu)
        boyut_bayt = os.path.getsize(goruntu_yolu)
        
        # Görüntü yükle
        goruntu = Image.open(goruntu_yolu)
        if goruntu.mode != 'L':
            goruntu = goruntu.convert('L')
        
        # Görüntü boyutları
        genislik, yukseklik = goruntu.size
        en_boy_orani = genislik / yukseklik if yukseklik > 0 else 0.0
        piksel_sayisi = genislik * yukseklik
        
        # Piksel verilerini numpy arrayına dönüştür
        piksel_array = np.array(goruntu, dtype=np.float32)
        
        # Yoğunluk istatistikleri
        ort_yogunluk = float(np.mean(piksel_array))
        std_yogunluk = float(np.std(piksel_array))
        min_yogunluk = float(np.min(piksel_array))
        max_yogunluk = float(np.max(piksel_array))
        p1_yogunluk = float(np.percentile(piksel_array, 1))
        p99_yogunluk = float(np.percentile(piksel_array, 99))
        medyan_yogunluk = float(np.median(piksel_array))
        
        # Shannon Entropisi (histogram tabanlı)
        hist, _ = np.histogram(piksel_array.astype(np.uint8), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Sıfır olanları çıkar
        entropi = float(-np.sum(hist * np.log2(hist)))
        
        # Kontrast (Laplacian filtresi)
        from scipy import ndimage
        laplacian = ndimage.laplace(piksel_array)
        kontrast = float(np.std(laplacian))
        
        return {
            "dosya_adı": dosya_adı,
            "boyut_bayt": int(boyut_bayt),
            "genislik": int(genislik),
            "yukseklik": int(yukseklik),
            "en_boy_orani": round(en_boy_orani, 4),
            "piksel_sayisi": int(piksel_sayisi),
            "ort_yogunluk": round(ort_yogunluk, 2),
            "std_yogunluk": round(std_yogunluk, 2),
            "min_yogunluk": round(min_yogunluk, 2),
            "max_yogunluk": round(max_yogunluk, 2),
            "p1_yogunluk": round(p1_yogunluk, 2),
            "p99_yogunluk": round(p99_yogunluk, 2),
            "medyan_yogunluk": round(medyan_yogunluk, 2),
            "entropi": round(entropi, 4),
            "kontrast": round(kontrast, 4),
        }
    
    except Exception as e:
        print(f"[HATA] {goruntu_yolu} işlenirken hata: {str(e)}")
        return None


def tum_gorseller_icin_csv_olustur(cikti_klasoru: str = CIKTI_KLASORU, 
                                   csv_dosya_adi: str = "goruntu_ozellikleri.csv") -> str:
    """
    Çıktı klasöründeki tüm ön işlenmiş görüntüler için CSV dosyası oluştur.
    
    Parametreler:
    - cikti_klasoru: Ön işlenmiş görüntülerin bulunduğu klasör
    - csv_dosya_adi: Oluşturulacak CSV dosyasının adı
    
    Döndürülen: CSV dosyasının tam yolu
    """
    csv_yolu = os.path.join(cikti_klasoru, csv_dosya_adi)
    
    print(f"[BILGI] {cikti_klasoru} klasöründe CSV oluşturuluyor...")
    
    # Tüm görüntüleri bul
    goruntu_dosyalari = []
    for sinif_adi in SINIF_KLASORLERI:
        sinif_klasoru = os.path.join(cikti_klasoru, sinif_adi)
        if os.path.exists(sinif_klasoru):
            etiket = SINIF_ETIKETI.get(sinif_adi, -1)
            for dosya in os.listdir(sinif_klasoru):
                if dosya.lower().endswith(('.png', '.jpg', '.jpeg')):
                    dosya_yolu = os.path.join(sinif_klasoru, dosya)
                    goruntu_dosyalari.append({
                        "dosya_yolu": dosya_yolu,
                        "sinif": sinif_adi,
                        "etiket": etiket,
                    })
    
    if not goruntu_dosyalari:
        print(f"[UYARI] {cikti_klasoru} klasöründe görüntü bulunamadı!")
        return csv_yolu
    
    print(f"[BILGI] Toplam {len(goruntu_dosyalari)} görüntü bulundu.")
    
    # Tüm görüntüler için öznitelikleri hesapla
    satirlar = []
    for item in tqdm(goruntu_dosyalari, desc="Görüntüler işleniyor"):
        ozellikler = goruntu_ozelliklerini_cikart(item["dosya_yolu"])
        
        if ozellikler is not None:
            # Sınıf bilgisini ekle
            ozellikler["sinif"] = item["sinif"]
            ozellikler["etiket"] = item["etiket"]
            ozellikler["dosya_yolu"] = item["dosya_yolu"]
            satirlar.append(ozellikler)
    
    if not satirlar:
        print("[HATA] Hiçbir görüntü işlenemedi!")
        return csv_yolu
    
    # CSV dosyasına yaz
    alanlar = sorted(satirlar[0].keys())
    
    with open(csv_yolu, 'w', newline='', encoding='utf-8') as f:
        yazici = csv.DictWriter(f, fieldnames=alanlar)
        yazici.writeheader()
        yazici.writerows(satirlar)
    
    print(f"[TAMAMLANDI] CSV dosyası kaydedildi: {csv_yolu}")
    print(f"[BILGI] Toplam {len(satirlar)} satır yazıldı")
    
    return csv_yolu


def istatistikleri_kaydet(csv_dosya_yolu: str, 
                          istatistik_csv_adi: str = "istatistikler.csv") -> str:
    """
    CSV dosyasından istatistikleri hesapla ve ayrı bir CSV'ye kaydet.
    
    Sınıf bazında:
    - Örnek sayısı
    - Ortalama/Std boyut
    - Ortalama/Std yoğunluk
    - Ortalama/Std entropi
    - vb.
    
    Parametreler:
    - csv_dosya_yolu: İşlenecek CSV dosyasının yolu
    - istatistik_csv_adi: Istatistik CSV dosyasının adı
    
    Döndürülen: Istatistik CSV dosyasının tam yolu
    """
    import pandas as pd
    
    # CSV yükle
    try:
        df = pd.read_csv(csv_dosya_yolu)
    except FileNotFoundError:
        print(f"[HATA] {csv_dosya_yolu} dosyası bulunamadı!")
        return ""
    
    print(f"[BILGI] CSV dosyası yüklendi: {len(df)} satır")
    
    # Sınıf bazında istatistikler
    istatistikler = []
    
    for sinif_adi in df['sinif'].unique():
        sinif_verisi = df[df['sinif'] == sinif_adi]
        
        istat = {
            "sinif": sinif_adi,
            "ornek_sayisi": len(sinif_verisi),
            "etiket": int(sinif_verisi['etiket'].iloc[0]),
            
            # Boyut istatistikleri
            "ort_genislik": round(sinif_verisi['genislik'].mean(), 1),
            "std_genislik": round(sinif_verisi['genislik'].std(), 1),
            "ort_yukseklik": round(sinif_verisi['yukseklik'].mean(), 1),
            "std_yukseklik": round(sinif_verisi['yukseklik'].std(), 1),
            
            # Yoğunluk istatistikleri
            "ort_ort_yogunluk": round(sinif_verisi['ort_yogunluk'].mean(), 2),
            "std_ort_yogunluk": round(sinif_verisi['ort_yogunluk'].std(), 2),
            "ort_min_yogunluk": round(sinif_verisi['min_yogunluk'].mean(), 2),
            "ort_max_yogunluk": round(sinif_verisi['max_yogunluk'].mean(), 2),
            
            # Entropi istatistikleri
            "ort_entropi": round(sinif_verisi['entropi'].mean(), 4),
            "std_entropi": round(sinif_verisi['entropi'].std(), 4),
            "min_entropi": round(sinif_verisi['entropi'].min(), 4),
            "max_entropi": round(sinif_verisi['entropi'].max(), 4),
            
            # Kontrast istatistikleri
            "ort_kontrast": round(sinif_verisi['kontrast'].mean(), 4),
            "std_kontrast": round(sinif_verisi['kontrast'].std(), 4),
        }
        
        istatistikler.append(istat)
    
    # CSV'ye kaydet
    istatistik_yolu = os.path.join(os.path.dirname(csv_dosya_yolu), istatistik_csv_adi)
    
    if istatistikler:
        alanlar = sorted(istatistikler[0].keys())
        
        with open(istatistik_yolu, 'w', newline='', encoding='utf-8') as f:
            yazici = csv.DictWriter(f, fieldnames=alanlar)
            yazici.writeheader()
            yazici.writerows(istatistikler)
        
        print(f"[TAMAMLANDI] Istatistikler kaydedildi: {istatistik_yolu}")
    
    # Konsola da yazdır
    print("\n" + "="*80)
    print("SINIF BAZLI ISTATISTIKLER")
    print("="*80)
    for istat in istatistikler:
        print(f"\n{istat['sinif']} (Etiket: {istat['etiket']})")
        print(f"  Örnek Sayısı: {istat['ornek_sayisi']}")
        print(f"  Ort. Genişlik: {istat['ort_genislik']:.1f} ± {istat['std_genislik']:.1f}")
        print(f"  Ort. Yükseklik: {istat['ort_yukseklik']:.1f} ± {istat['std_yukseklik']:.1f}")
        print(f"  Ort. Yoğunluk: {istat['ort_ort_yogunluk']:.2f} ± {istat['std_ort_yogunluk']:.2f}")
        print(f"  Ort. Entropi: {istat['ort_entropi']:.4f} ± {istat['std_entropi']:.4f}")
        print(f"  Ort. Kontrast: {istat['ort_kontrast']:.4f} ± {istat['std_kontrast']:.4f}")
    
    return istatistik_yolu


def on_isleme_log_kaydet(log_kayitlari: List[Dict], 
                         cikti_klasoru: str = CIKTI_KLASORU,
                         log_dosya_adi: str = "on_isleme_log.csv") -> str:
    """
    Ön işleme işlemlerinin log dosyasını CSV formatında kaydet.
    
    Bu fonksiyon, her görüntü için yapılan ön işlemlerin detaylı log'unu
    CSV dosyası olarak kaydeder.
    
    Parametreler:
    - log_kayitlari: Her işlenen görüntü için log kaydı içeren sözlüklerin listesi
                     Her kayıt: {"girdi_yolu": ..., "sinif": ..., "etiket": ..., 
                                "cikti_yolu": ..., "aug_kopya": ..., ...}
    - cikti_klasoru: Log dosyasının kaydedileceği klasör
    - log_dosya_adi: Log dosyasının adı
    
    Döndürülen: Log CSV dosyasının tam yolu
    """
    if not log_kayitlari:
        print("[UYARI] Boş log kaydı listesi!")
        return ""
    
    # Çıktı klasörünü oluştur (yoksa)
    klasor_olustur_yoksa(cikti_klasoru)
    
    # Log dosyası yolunu oluştur
    log_dosyasi_yolu = Path(cikti_klasoru) / log_dosya_adi
    
    # Tüm olası alanları topla
    alanlar = sorted({anahtar for kayit in log_kayitlari for anahtar in kayit.keys()})
    
    # CSV dosyasına yaz
    with open(log_dosyasi_yolu, "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=alanlar)
        yazici.writeheader()
        for kayit in log_kayitlari:
            yazici.writerow(kayit)
    
    print(f"[TAMAMLANDI] Ön işleme log dosyası kaydedildi: {log_dosyasi_yolu}")
    print(f"[BILGI] Toplam {len(log_kayitlari)} işlem kaydedildi")
    
    return str(log_dosyasi_yolu)


def csv_ye_minmax_scaling_uygula(
    csv_dosya_yolu: str,
    ozellik_sutunlari: Optional[List[str]] = None,
    hariç_tutulacak_sutunlar: Optional[List[str]] = None,
    cikti_dosya_adi: str = "goruntu_ozellikleri_scaled.csv"
) -> Tuple[str, Dict]:
    """
    CSV dosyasındaki sayısal özelliklere Min-Max scaling (0-1 normalizasyonu) uygula.
    
    Min-Max Scaling Formula: X_scaled = (X - min) / (max - min)
    
    Parametreler:
    - csv_dosya_yolu: İnput CSV dosyasının yolu
    - ozellik_sutunlari: Ölçeklenecek sütunlar (None ise tüm sayısal sütunlar)
    - hariç_tutulacak_sutunlar: Ölçeklenmeyecek sütunlar (örn: sinif, etiket, dosya_adı)
    - cikti_dosya_adi: Ölçeklenmiş CSV dosyasının adı
    
    Döndürülen: (Ölçeklenmiş CSV dosyasının yolu, scaler istatistikleri dict)
    """
    # CSV dosyasını yükle
    try:
        df = pd.read_csv(csv_dosya_yolu)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_dosya_yolu}")
    
    print(f"[BILGI] CSV dosyası yüklendi: {csv_dosya_yolu} ({len(df)} satır)")
    
    # Varsayılan hariç tutulacak sütunlar
    if hariç_tutulacak_sutunlar is None:
        hariç_tutulacak_sutunlar = [
            'dosya_adı', 'dosya_yolu', 'sinif', 'etiket',
            'genislik', 'yukseklik', 'piksel_sayisi', 'boyut_bayt'
        ]
    
    # Ölçeklenecek sütunları belirle
    if ozellik_sutunlari is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ozellik_sutunlari = [
            col for col in numeric_cols 
            if col not in hariç_tutulacak_sutunlar
        ]
    
    print(f"[BILGI] Ölçeklenecek sütunlar ({len(ozellik_sutunlari)}): {ozellik_sutunlari}")
    
    # Min-Max scaling hesaplamaları
    scaler_stats = {}
    df_scaled = df.copy()
    
    for col in ozellik_sutunlari:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            
            if range_val < 1e-8:
                # Sabit değerler için özel durum
                df_scaled[col] = 0.0
            else:
                df_scaled[col] = (df[col] - min_val) / range_val
            
            scaler_stats[col] = {
                'min': float(min_val),
                'max': float(max_val),
                'range': float(range_val)
            }
    
    # Ölçeklenmiş CSV'yi kaydet
    output_dir = os.path.dirname(csv_dosya_yolu)
    cikti_yolu = os.path.join(output_dir, cikti_dosya_adi)
    
    df_scaled.to_csv(cikti_yolu, index=False, encoding='utf-8')
    
    print(f"[TAMAMLANDI] Min-Max Scaled CSV kaydedildi: {cikti_yolu}")
    
    # İstatistikleri göster
    print("\n[ÖLÇEKLEME İSTATİSTİKLERİ]")
    for col, stats in scaler_stats.items():
        print(f"  {col:20} → Min: {stats['min']:10.4f}, Max: {stats['max']:10.4f}")
    
    return cikti_yolu, scaler_stats


if __name__ == "__main__":
    # Test
    csv_yolu = tum_gorseller_icin_csv_olustur()
    istatistik_yolu = istatistikleri_kaydet(csv_yolu)
