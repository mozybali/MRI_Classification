"""
ozellik_cikarici.py
-------------------
İşlenmiş görüntülerden özellik çıkarma ve CSV oluşturma modülü.
"""

import os
import pickle
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from multiprocessing import Pool, cpu_count
from functools import partial

from ayarlar import *

KATEGORIK_SUTUNLAR = [
    'dosya_adi', 'sinif', 'etiket', 'tam_yol',
    'kaynak_id', 'kaynak_grup', 'augmentasyon_mu'
]
MODELE_DAHIL_EDILMEYEN_SAYISAL_SUTUNLAR = [
    'boyut_bayt', 'genislik', 'yukseklik', 'en_boy_orani', 'piksel_sayisi'
]


def _ozellik_cikar_wrapper(goruntu_yolu: str, sinif_adi: str) -> Optional[Dict]:
    """⚡ Paralel özellik çıkarma için wrapper fonksiyon."""
    try:
        cikarici = OzellikCikarici()
        ozellikler = cikarici.tek_goruntu_ozellikleri(str(goruntu_yolu))

        if ozellikler:
            ozellikler["sinif"] = sinif_adi
            ozellikler["etiket"] = SINIF_ETIKETI[sinif_adi]
            ozellikler["tam_yol"] = str(goruntu_yolu)
            ozellikler["kaynak_id"] = cikarici.kaynak_id_belirle(Path(goruntu_yolu).name)
            ozellikler["kaynak_grup"] = f"{sinif_adi}::{ozellikler['kaynak_id']}"
            ozellikler["augmentasyon_mu"] = Path(goruntu_yolu).stem != ozellikler["kaynak_id"]
            return ozellikler
    except Exception as e:
        print(f"[HATA] Ozellik cikarma basarisiz {goruntu_yolu}: {type(e).__name__}: {e}")
    return None


class OzellikCikarici:
    """Görüntülerden özellik çıkarma ve CSV oluşturma sınıfı."""
    
    def __init__(self):
        """Özellik çıkarıcıyı başlat."""
        self.n_jobs = max(1, cpu_count() - 1)  # Bir çekirdek sisteme bırak

    @staticmethod
    def kaynak_id_belirle(dosya_adi: str) -> str:
        """Augment edilmiş dosyalardan kaynak görüntü kimliğini çıkar."""
        stem = Path(str(dosya_adi)).stem
        stem = re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"\s*\(\d+\)$", "", stem)
        return stem

    @classmethod
    def kaynak_kolonlarini_hazirla(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Split ve ölçekleme için kaynak görüntü kolonlarını güvenli şekilde ekle."""
        df = df.copy()

        if 'kaynak_id' not in df.columns:
            if 'dosya_adi' in df.columns:
                df['kaynak_id'] = df['dosya_adi'].fillna("").map(cls.kaynak_id_belirle)
            elif 'tam_yol' in df.columns:
                df['kaynak_id'] = df['tam_yol'].fillna("").map(lambda yol: cls.kaynak_id_belirle(Path(str(yol)).name))
            else:
                df['kaynak_id'] = [f"satir_{i}" for i in range(len(df))]

        if 'kaynak_grup' not in df.columns:
            if 'sinif' in df.columns:
                df['kaynak_grup'] = df['sinif'].astype(str) + "::" + df['kaynak_id'].astype(str)
            else:
                df['kaynak_grup'] = df['kaynak_id'].astype(str)

        if 'augmentasyon_mu' not in df.columns:
            if 'dosya_adi' in df.columns:
                kok_adlari = df['dosya_adi'].fillna("").map(lambda ad: Path(str(ad)).stem)
                df['augmentasyon_mu'] = kok_adlari != df['kaynak_id'].astype(str)
            else:
                df['augmentasyon_mu'] = False

        return df

    @staticmethod
    def _sayisal_sutunlari_bul(df: pd.DataFrame) -> List[str]:
        """Ölçeklenecek ve modele girecek sayısal sütunları bul."""
        haric_sutunlar = set(KATEGORIK_SUTUNLAR + MODELE_DAHIL_EDILMEYEN_SAYISAL_SUTUNLAR)
        return [col for col in df.columns if col not in haric_sutunlar]

    @staticmethod
    def _stratify_serisi_uygun_mu(seri: pd.Series) -> bool:
        """Stratify için her sınıfta yeterli örnek var mı kontrol et."""
        if seri.nunique() < 2:
            return False
        return bool((seri.value_counts() >= 2).all())

    @staticmethod
    def _scaler_olustur(metod: str):
        """İstenen ölçekleyiciyi oluştur."""
        if metod == "minmax":
            return MinMaxScaler(), "MinMaxScaler: Degerleri [0, 1] araligina olceklendirir"
        if metod == "robust":
            return RobustScaler(), "RobustScaler: Medyan ve IQR kullanir (aykiri degerlere dayanikli)"
        if metod == "standard":
            return StandardScaler(), "StandardScaler: Z-score normalizasyonu (mean=0, std=1)"
        if metod == "maxabs":
            return MaxAbsScaler(), "MaxAbsScaler: Degerleri [-1, 1] araligina olceklendirir"
        raise ValueError(f"Bilinmeyen scaling metodu: {metod}")

    def _df_olceklendir(
        self,
        df: pd.DataFrame,
        scaler,
        fit: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """DataFrame üzerindeki sayısal sütunlara scaler uygula."""
        df_scaled = df.copy()
        sayisal_sutunlar = self._sayisal_sutunlari_bul(df_scaled)

        if not sayisal_sutunlar:
            return df_scaled, sayisal_sutunlar

        if fit:
            df_scaled[sayisal_sutunlar] = scaler.fit_transform(df_scaled[sayisal_sutunlar])
        else:
            df_scaled[sayisal_sutunlar] = scaler.transform(df_scaled[sayisal_sutunlar])

        return df_scaled, sayisal_sutunlar

    @staticmethod
    def _scaler_kaydet(scaler, sayisal_sutunlar: List[str], metod: str, cikti_klasoru: Path):
        """Inference tarafında kullanılmak üzere scaler'ı kaydet."""
        scaler_yolu = cikti_klasoru / SCALER_DOSYA_ADI
        with open(scaler_yolu, 'wb') as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "columns": sayisal_sutunlar,
                    "method": metod,
                },
                f
            )
    
    def tek_goruntu_ozellikleri(self, goruntu_yolu: str) -> Optional[Dict]:
        """
        Tek bir görüntüden özellikler çıkar.
        
        Bu fonksiyon, bir MRI görüntüsünden makine öğrenmesi için kullanılacak
        sayısal özellikleri hesaplar. Bu özellikler görüntünün içeriğini temsil eder.
        
        Çıkarılan özellikler:
        - Dosya bilgileri: ad, boyut (byte)
        - Görüntü boyutları: genişlik, yükseklik, en-boy oranı, piksel sayısı
        - Yoğunluk istatistikleri: ortalama, std sapma, min, max, medyan, percentile'ler
        - Doku özellikleri: entropi (bilgi miktarı), kontrast, homojenlik, enerji
        
        Args:
            goruntu_yolu: Görüntü dosyasının tam yolu
            
        Returns:
            Özellikler sözlüğü veya hata durumunda None
        """
        try:
            # 1. DOSYA BİLGİLERİNİ AL
            dosya_adi = os.path.basename(goruntu_yolu)  # Sadece dosya adı
            boyut_bayt = os.path.getsize(goruntu_yolu)  # Dosya boyutu (byte)
            
            # 2. GÖRÜNTÜYÜ YÜKLE VE GRİ TONLAMAYA ÇEVİR
            goruntu = Image.open(goruntu_yolu)
            if goruntu.mode != 'L':  # Eğer renkli ise
                goruntu = goruntu.convert('L')  # Gri tonlamaya çevir
            
            # 3. BOYUT BİLGİLERİNİ HESAPLA
            genislik, yukseklik = goruntu.size
            en_boy_orani = genislik / yukseklik if yukseklik > 0 else 0.0
            
            # 4. PİKSEL VERİLERİNİ NUMPY ARRAY'E ÇEVİR
            piksel_array = np.array(goruntu, dtype=np.float32)
            
            # 5. YOĞUNLUK İSTATİSTİKLERİNİ HESAPLA
            # Temel istatistikler
            ort_yogunluk = float(np.mean(piksel_array))     # Ortalama piksel değeri
            std_yogunluk = float(np.std(piksel_array))      # Standart sapma (dağılım)
            min_yogunluk = float(np.min(piksel_array))      # En düşük değer
            max_yogunluk = float(np.max(piksel_array))      # En yüksek değer
            
            # Yüzdelik değerleri (percentiles) - dağılımı daha iyi anlamak için
            p1_yogunluk = float(np.percentile(piksel_array, 1))    # %1'lik dilim
            p25_yogunluk = float(np.percentile(piksel_array, 25))  # 1. çeyrek
            p50_yogunluk = float(np.percentile(piksel_array, 50))  # Medyan (ortanca)
            p75_yogunluk = float(np.percentile(piksel_array, 75))  # 3. çeyrek
            p99_yogunluk = float(np.percentile(piksel_array, 99))  # %99'luk dilim
            
            # 6. SHANNON ENTROPİSİNİ HESAPLA
            # Entropi, görüntüdeki bilgi miktarını / karmaşıklığı ölçer
            # Yüksek entropi = fazla detay, düşük entropi = düz/homojen görüntü
            hist, _ = np.histogram(piksel_array.astype(np.uint8), bins=256, range=(0, 256))
            hist = hist / hist.sum()  # Histogramı normalize et (olasılıklara çevir)
            hist = hist[hist > 0]     # Sıfır olmayan değerleri al
            entropi = float(-np.sum(hist * np.log2(hist)))  # Shannon entropi formülü
            
            # 7. KONTRAST HESAPLA (Laplacian varyansı)
            # Kontrast, görüntüdeki keskinliği / kenar yoğunluğunu ölçer
            laplacian = ndimage.laplace(piksel_array)  # Laplacian filtresi uygula
            kontrast = float(np.var(laplacian))        # Varyansı al
            
            # 8. EK DOKU ÖZELLİKLERİNİ HESAPLA
            # Homojenlik: Görüntü ne kadar düzgün? (düşük std = yüksek homojenlik)
            homojenlik = 1.0 / (1.0 + std_yogunluk)
            
            # Enerji: Histogramın ikinci momenti (üniformluğu ölçer)
            enerji = float(np.sum(hist ** 2))
            
            # 9. GELİŞMİŞ İSTATİSTİKSEL ÖZELLİKLER
            # Skewness (Çarpıklık): Dağılımın simetrisini ölçer
            # Pozitif = sağa çarpık, negatif = sola çarpık, 0 = simetrik
            from scipy.stats import skew, kurtosis
            carpiklik = float(skew(piksel_array.flatten()))
            
            # Kurtosis (Basıklık): Dağılımın kuyruk kalınlığını ölçer
            # Yüksek = sivri tepe ve kalın kuyruklar, düşük = düz dağılım
            basiklik = float(kurtosis(piksel_array.flatten()))
            
            # 10. GRADYAN ÖZELLİKLERİ (Kenar Yoğunluğu)
            # Gradyan, görüntüdeki değişim hızını ölçer (kenarları yakalar)
            grad_y, grad_x = np.gradient(piksel_array.astype(np.float32))
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            ortalama_gradyan = float(np.mean(gradient_magnitude))
            max_gradyan = float(np.max(gradient_magnitude))
            
            # 11. OTSU EŞİĞİ ANALİZİ
            # Otsu yöntemi optimal eşik değerini otomatik bulur
            # Bu değer, beyin-arka plan ayrımı için ipucu verir
            try:
                from skimage.filters import threshold_otsu
                otsu_esik = float(threshold_otsu(piksel_array))
            except ImportError:
                # scikit-image yoksa basit hesaplama
                otsu_esik = float(np.mean(piksel_array))
            
            # 12. TÜM ÖZELLİKLERİ SÖZLÜKTE TOPLA VE DÖNDÜR
            return {
                "dosya_adi": dosya_adi,
                "boyut_bayt": int(boyut_bayt),
                "genislik": int(genislik),
                "yukseklik": int(yukseklik),
                "en_boy_orani": round(en_boy_orani, 4),
                "piksel_sayisi": int(genislik * yukseklik),
                "ort_yogunluk": round(ort_yogunluk, 2),
                "std_yogunluk": round(std_yogunluk, 2),
                "min_yogunluk": round(min_yogunluk, 2),
                "max_yogunluk": round(max_yogunluk, 2),
                "p1_yogunluk": round(p1_yogunluk, 2),
                "p25_yogunluk": round(p25_yogunluk, 2),
                "medyan_yogunluk": round(p50_yogunluk, 2),
                "p75_yogunluk": round(p75_yogunluk, 2),
                "p99_yogunluk": round(p99_yogunluk, 2),
                "entropi": round(entropi, 4),
                "kontrast": round(kontrast, 2),
                "homojenlik": round(homojenlik, 4),
                "enerji": round(enerji, 4),
                # Yeni gelişmiş özellikler
                "carpiklik": round(carpiklik, 4),
                "basiklik": round(basiklik, 4),
                "ortalama_gradyan": round(ortalama_gradyan, 2),
                "max_gradyan": round(max_gradyan, 2),
                "otsu_esik": round(otsu_esik, 2),
            }
            
        except Exception as e:
            print(f"[HATA] Özellik çıkarılamadı {goruntu_yolu}: {e}")
            return None
    
    def csv_olustur(self, giris_klasoru: Path = CIKTI_KLASORU, 
                    cikti_csv: Optional[Path] = None) -> pd.DataFrame:
        """
        Klasördeki tüm görüntülerden özellik çıkar ve CSV oluştur.
        
        Bu fonksiyon, makine öğrenmesi için kullanılacak veri setini hazırlar.
        İşlenmiş görüntü klasörünü tarar, her görüntü için 20+ özellik hesaplar
        ve sonuçları tek bir CSV dosyasına birleştirir.
        
        Çalışma akışı:
        1. Tüm sınıf klasörlerini tara (NonDemented, MildDemented, vb.)
        2. Her görüntü için tek_goruntu_ozellikleri() çağır
        3. Sınıf adı ve etiketi ekle
        4. Tüm sonuçları DataFrame'de birleştir
        5. CSV'ye kaydet
        
        CSV formatı:
        | dosya_adi | sinif | etiket | genislik | yukseklik | ... | entropi |
        |-----------|-------|--------|----------|-----------|-----|----------|
        | img1.jpg  | NonDemented | 0 | 256 | 256 | ... | 5.23 |
        
        Bu CSV, model/train.py tarafından okunur ve eğitimde kullanılır.
        
        Args:
            giris_klasoru: İşlenmiş görüntülerin bulunduğu klasör
            cikti_csv: CSV dosyasının kaydedileceği yol (None ise varsayılan)
            
        Returns:
            Pandas DataFrame (tüm özellikler ve etiketler)
        """
        # Varsayılan CSV yolunu belirle
        if cikti_csv is None:
            cikti_csv = CIKTI_KLASORU / CSV_DOSYA_ADI
        
        tum_ozellikler = []  # Tüm görüntülerin özelliklerini saklayacak liste
        
        print(f"\n⚡ Özellikler çıkarılıyor (paralel: {self.n_jobs} çekirdek)...\n")
        
        # Her sınıf için döngü
        for sinif_adi in SINIF_KLASORLERI:
            sinif_klasoru = giris_klasoru / sinif_adi
            
            # Klasör yoksa uyar ve devam et
            if not sinif_klasoru.exists():
                print(f"[UYARI] Klasör bulunamadı: {sinif_klasoru}")
                continue
            
            # Klasördeki tüm görüntüleri bul (GORUNTU_UZANTILARI ile uyumlu)
            gorseller = []
            for uzanti in GORUNTU_UZANTILARI:
                gorseller.extend(sinif_klasoru.glob(f"*{uzanti}"))
            gorseller.sort(key=lambda p: p.name)
            
            # ⚡ Paralel özellik çıkarma
            with Pool(processes=self.n_jobs) as pool:
                partial_func = partial(_ozellik_cikar_wrapper, sinif_adi=sinif_adi)
                sonuclar = list(tqdm(
                    pool.imap(partial_func, gorseller),
                    total=len(gorseller),
                    desc=f"{sinif_adi} işleniyor (paralel)"
                ))
            
            # None olmayan sonuçları ekle
            tum_ozellikler.extend([s for s in sonuclar if s is not None])
        
        if not tum_ozellikler:
            print("[HATA] Hiç özellik çıkarılamadı!")
            return pd.DataFrame()
        
        # DataFrame oluştur
        df = self.kaynak_kolonlarini_hazirla(pd.DataFrame(tum_ozellikler))
        
        # CSV'ye kaydet
        df.to_csv(cikti_csv, index=False, encoding='utf-8')
        print(f"\n[BASARILI] CSV kaydedildi: {cikti_csv}")
        print(f"  Toplam {len(df)} goruntu")
        print(f"\nSınıf dağılımı:")
        print(df['sinif'].value_counts().to_string())
        
        return df
    
    def nan_temizle(self, csv_dosyasi: Optional[Path] = None, 
                    metod: str = 'drop') -> pd.DataFrame:
        """
        CSV'deki NaN (eksik) değerleri temizle.
        
        Args:
            csv_dosyasi: CSV dosyası (None ise varsayılan)
            metod: Temizleme metodu
                - 'drop': NaN içeren satırları çıkar (önerilen)
                - 'mean': NaN'ları sütun ortalamasıyla doldur
                - 'median': NaN'ları sütun medyanıyla doldur
                - 'zero': NaN'ları 0 ile doldur
                
        Returns:
            Temizlenmiş DataFrame
        """
        if csv_dosyasi is None:
            csv_dosyasi = CIKTI_KLASORU / CSV_DOSYA_ADI
        
        try:
            df = pd.read_csv(csv_dosyasi)
        except Exception as e:
            print(f"[HATA] CSV okunamadı: {e}")
            return pd.DataFrame()
        
        total_nan = df.isnull().sum().sum()
        if total_nan == 0:
            print(f"[BILGI] CSV'de NaN deger YOK - temizlemeye gerek yok")
            return df
        
        print(f"\n[BILGI] {total_nan} NaN deger bulundu")
        
        df = self.kaynak_kolonlarini_hazirla(df)
        sayisal_sutunlar = self._sayisal_sutunlari_bul(df)
        
        if metod == 'drop':
            df_temiz = df.dropna()
            print(f"[ISLEM] NaN iceren {len(df) - len(df_temiz)} satir cikarildi")
            print(f"   Kalan satir: {len(df_temiz)}")
        elif metod == 'mean':
            df_temiz = df.copy()
            for col in sayisal_sutunlar:
                if df_temiz[col].isnull().any():
                    ort = df_temiz[col].mean()
                    df_temiz[col].fillna(ort, inplace=True)
                    print(f"   * {col}: NaN -> {ort:.2f} (ortalama)")
        elif metod == 'median':
            df_temiz = df.copy()
            for col in sayisal_sutunlar:
                if df_temiz[col].isnull().any():
                    med = df_temiz[col].median()
                    df_temiz[col].fillna(med, inplace=True)
                    print(f"   * {col}: NaN -> {med:.2f} (medyan)")
        elif metod == 'zero':
            df_temiz = df.copy()
            df_temiz[sayisal_sutunlar] = df_temiz[sayisal_sutunlar].fillna(0)
            print(f"[ISLEM] Tum NaN'lar 0 ile dolduruldu")
        else:
            print(f"[HATA] Bilinmeyen metod: {metod}")
            print(f"   Gecerli metodlar: drop, mean, median, zero")
            return df
        
        # Kaydet
        df_temiz.to_csv(csv_dosyasi, index=False, encoding='utf-8')
        print(f"\n[BASARILI] Temizlenmis CSV kaydedildi: {csv_dosyasi}")
        
        return df_temiz
    
    def scaling_uygula(self, giris_csv: Optional[Path] = None,
                      cikti_csv: Optional[Path] = None,
                      metod: str = SCALING_METODU) -> pd.DataFrame:
        """
        CSV dosyasındaki özelliklere ölçeklendirme (scaling) uygula.
        
        Neden ölçeklendirme gerekli?
        - Makine öğrenmesi modelleri, farklı ölçeklerdeki özelliklerle iyi çalışamaz
        - Örn: genislik=256, entropi=5.2 -> model genislik'e aşırı ağırlık verir
        - Tüm özellikleri aynı ölçeğe getirerek adil bir öğrenme sağlanır
        
        Desteklenen ölçeklendirme metodları:
        
        1. minmax: Min-Max normalizasyonu
           - Tüm değerleri [0, 1] aralığına sıkıştırır
           - Formül: (x - min) / (max - min)
           - Avantaj: Basit, hızlı
           - Dezavantaj: Aykırı değerlere duyarlı
        
        2. robust: Robust Scaler (Önerilen ⭐)
           - Medyan ve IQR (interquartile range) kullanır
           - Formül: (x - median) / IQR
           - Avantaj: Aykırı değerlere karşı dayanıklı
           - MRI görüntülerinde gürültü olabilir, bu yüzden robust tercih edilir
        
        3. standard: Standart (Z-score) normalizasyonu
           - Ortalama ve standart sapma kullanır
           - Formül: (x - mean) / std
           - Avantaj: Normal dağılım sağlar (mean=0, std=1)
           - Kullanım: SVM ve lineer modeller için uygun
        
        4. maxabs: Max Absolute Scaler
           - Maksimum mutlak değere böler
           - Formül: x / max(|x|)
           - Değerler [-1, 1] aralığında olur
        
        Args:
            giris_csv: Okunacak CSV dosyası (None ise varsayılan)
            cikti_csv: Kaydedilecek CSV dosyası (None ise orijinal üzerine yazar)
            metod: Ölçeklendirme metodu ('minmax', 'robust', 'standard', 'maxabs')
            
        Returns:
            Ölçeklendirilmiş DataFrame
        """
        if giris_csv is None:
            giris_csv = CIKTI_KLASORU / CSV_DOSYA_ADI
        
        if cikti_csv is None:
            cikti_csv = CIKTI_KLASORU / CSV_SCALED_DOSYA_ADI
        
        # CSV'yi oku
        try:
            df = pd.read_csv(giris_csv)
        except Exception as e:
            print(f"[HATA] CSV okunamadı: {e}")
            return pd.DataFrame()
        
        # NaN değer kontrolü
        total_nan = df.isnull().sum().sum()
        if total_nan > 0:
            print(f"\n[UYARI] CSV'de {total_nan} adet NaN deger bulundu!")
            nan_cols = df.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            print(f"   NaN iceren sutunlar:")
            for col, count in nan_cols.items():
                print(f"   * {col}: {count} NaN ({count/len(df)*100:.2f}%)")
            
            print(f"\n   [SECENEKLER]")
            print(f"   1. NaN degerleri koruyarak devam et (scaler NaN'lari atlar)")
            print(f"   2. NaN iceren satirlari cikar (onerilen)")
            print(f"   3. NaN degerleri sutun ortalamasiyla doldur")
            print(f"\n   Simdilik devam ediliyor... (NaN'lar korunacak)")
        
        # Ölçeklendirilecek sütunları belirle (sayısal olanlar)
        df = self.kaynak_kolonlarini_hazirla(df)
        sayisal_sutunlar = self._sayisal_sutunlari_bul(df)
        
        # Sabit sütunları tespit et (std = 0 olanlar)
        sabit_sutunlar = []
        for col in sayisal_sutunlar:
            if df[col].std() == 0:
                sabit_sutunlar.append(col)
        
        if sabit_sutunlar:
            print(f"\n[UYARI] {len(sabit_sutunlar)} sabit sutun bulundu (tum degerler ayni):")
            for col in sabit_sutunlar[:5]:
                print(f"   * {col} = {df[col].iloc[0]}")
            if len(sabit_sutunlar) > 5:
                print(f"   ... ve {len(sabit_sutunlar) - 5} tane daha")
            print(f"   Bu sutunlar model egitiminde kullanissiz olabilir.")
        
        # Scaling seçimi
        try:
            scaler, bilgi = self._scaler_olustur(metod)
            print(f"\n[BILGI] {bilgi}")
        except ValueError:
            print(f"[HATA] Bilinmeyen scaling metodu: {metod}")
            print("       Gecerli metodlar: minmax, robust, standard, maxabs")
            return df
        
        # Ölçeklendirme uygula
        print(f"\n[ISLEM] Olceklendirme uygulanıyor...")
        try:
            df_scaled, sayisal_sutunlar = self._df_olceklendir(df, scaler, fit=True)
        except Exception as e:
            print(f"\n[HATA] Olceklendirme basarisiz: {e}")
            return df
        
        # Ölçeklendirme sonrası NaN kontrolü
        nan_after_scaling = df_scaled[sayisal_sutunlar].isnull().sum().sum()
        if nan_after_scaling > 0:
            print(f"\n[UYARI] Olceklendirme sonrasi {nan_after_scaling} NaN degeri korundu")
            print(f"   (sklearn scaler'lari NaN degerleri oldugu gibi birakir)")
        
        # Kaydet
        try:
            df_scaled.to_csv(cikti_csv, index=False, encoding='utf-8')
            print(f"\n[BASARILI] Olceklendirilmis CSV kaydedildi!")
            print(f"   Dosya: {cikti_csv}")
            print(f"   Metod: {metod}")
            print(f"   Islenen ozellik sayisi: {len(sayisal_sutunlar)}")
            print(f"   Toplam satir sayisi: {len(df_scaled)}")
        except Exception as e:
            print(f"\n[HATA] CSV kaydedilemedi: {e}")
            return df
        
        # Scaling istatistikleri göster
        print(f"\n[ISTATISTIK] Olceklendirme sonrasi deger araliklari:")
        
        # Değişken sütunları filtrele (sabit olmayanlar)
        degisken_sutunlar = [col for col in sayisal_sutunlar if col not in sabit_sutunlar]
        
        if degisken_sutunlar:
            for col in degisken_sutunlar[:5]:  # İlk 5 değişken özelliği göster
                min_val = df_scaled[col].min()
                max_val = df_scaled[col].max()
                ort_val = df_scaled[col].mean()
                print(f"   * {col}: [{min_val:.4f}, {max_val:.4f}] (ort: {ort_val:.4f})")
            if len(degisken_sutunlar) > 5:
                print(f"   ... ve {len(degisken_sutunlar) - 5} degisken ozellik daha")
        else:
            print(f"   [UYARI] Hic degisken ozellik yok (tum sutunlar sabit)")
        
        if sabit_sutunlar:
            print(f"\n[IPUCU] Sabit sutunlar ({len(sabit_sutunlar)} adet) model egitiminde")
            print(f"   cikarilebilir cunku bilgi icermiyorlar.")
        
        return df_scaled
    
    def istatistik_raporu(self, csv_dosyasi: Optional[Path] = None):
        """
        CSV dosyasından istatistik raporu oluştur.
        
        Args:
            csv_dosyasi: CSV dosyası yolu
        """
        if csv_dosyasi is None:
            csv_dosyasi = CIKTI_KLASORU / CSV_DOSYA_ADI
        
        try:
            df = pd.read_csv(csv_dosyasi)
        except Exception as e:
            print(f"[HATA] CSV okunamadı: {e}")
            return
        
        print("\n" + "="*60)
        print("VERİ SETİ İSTATİSTİK RAPORU")
        print("="*60)
        
        print(f"\nToplam görüntü sayısı: {len(df)}")
        print(f"\nSınıf dağılımı:")
        print(df['sinif'].value_counts().to_string())
        
        print(f"\n\nTemel istatistikler:")
        print(df.describe().to_string())
        
        # Eksik degerler
        eksik = df.isnull().sum()
        if eksik.sum() > 0:
            print(f"\n\nEksik degerler:")
            print(eksik[eksik > 0].to_string())
        else:
            print(f"\n\n[BASARILI] Eksik deger yok")
        
        print("\n" + "="*60)


def veri_boluntule(csv_dosyasi: Optional[Path] = None,
                   cikti_klasoru: Optional[Path] = None):
    """
    Veri setini eğitim, doğrulama ve test setlerine böl.

    Raises:
        ValueError: Oran toplamı 1.0 değilse veya yeterli örnek yoksa
    """
    from sklearn.model_selection import train_test_split

    # Oran doğrulaması
    oran_toplam = EGITIM_ORANI + DOGRULAMA_ORANI + TEST_ORANI
    if abs(oran_toplam - 1.0) > 1e-6:
        raise ValueError(
            f"Bolme oranlari toplami 1.0 olmali, ancak {oran_toplam:.4f} "
            f"(egitim={EGITIM_ORANI}, dogrulama={DOGRULAMA_ORANI}, test={TEST_ORANI})"
        )

    if csv_dosyasi is None:
        csv_dosyasi = CIKTI_KLASORU / CSV_DOSYA_ADI

    if cikti_klasoru is None:
        cikti_klasoru = CIKTI_KLASORU

    # CSV'yi oku
    try:
        df = pd.read_csv(csv_dosyasi)
    except Exception as e:
        print(f"[HATA] CSV okunamadı: {e}")
        return None

    cikarici = OzellikCikarici()
    df = cikarici.kaynak_kolonlarini_hazirla(df)

    group_df = df[['kaynak_grup', 'etiket']].drop_duplicates().reset_index(drop=True)

    # Split yapısına göre minimum örnek doğrulaması
    toplam_grup = len(group_df)
    temp_oran = 1 - EGITIM_ORANI
    val_oran = DOGRULAMA_ORANI / (DOGRULAMA_ORANI + TEST_ORANI)

    temp_grup_sayisi = math.ceil(toplam_grup * temp_oran)
    train_grup_sayisi = toplam_grup - temp_grup_sayisi
    test_grup_sayisi = math.ceil(temp_grup_sayisi * (1 - val_oran))
    val_grup_sayisi = temp_grup_sayisi - test_grup_sayisi

    if train_grup_sayisi < 1 or val_grup_sayisi < 1 or test_grup_sayisi < 1:
        min_split_groups = 4  # varsayılan oranlarda güvenli alt sınır
        raise ValueError(
            f"Veri setinde yeterli benzersiz kaynak grup yok "
            f"(bulunan: {toplam_grup}, gereken minimum: {min_split_groups}). "
            f"Daha fazla veri ekleyin veya bolme oranlarini ayarlayin."
        )

    stratify_groups = group_df['etiket'] if cikarici._stratify_serisi_uygun_mu(group_df['etiket']) else None
    if stratify_groups is not None:
        sinif_sayisi = int(group_df['etiket'].nunique())
        if train_grup_sayisi < sinif_sayisi or temp_grup_sayisi < sinif_sayisi:
            stratify_groups = None

    # Aynı kaynak görüntünün augmentasyonları farklı split'lere düşmesin.
    train_groups, temp_groups = train_test_split(
        group_df,
        test_size=(1 - EGITIM_ORANI),
        stratify=stratify_groups,
        random_state=RASTGELE_TOHUM
    )

    temp_stratify = temp_groups['etiket'] if cikarici._stratify_serisi_uygun_mu(temp_groups['etiket']) else None
    if temp_stratify is not None:
        temp_sinif_sayisi = int(temp_groups['etiket'].nunique())
        if val_grup_sayisi < temp_sinif_sayisi or test_grup_sayisi < temp_sinif_sayisi:
            temp_stratify = None
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(1 - val_oran),
        stratify=temp_stratify,
        random_state=RASTGELE_TOHUM
    )

    train_df = df[df['kaynak_grup'].isin(train_groups['kaynak_grup'])].copy()
    val_df = df[df['kaynak_grup'].isin(val_groups['kaynak_grup'])].copy()
    test_df = df[df['kaynak_grup'].isin(test_groups['kaynak_grup'])].copy()

    # Kaydet
    train_df.to_csv(cikti_klasoru / EGITIM_CSV_DOSYA_ADI, index=False)
    val_df.to_csv(cikti_klasoru / DOGRULAMA_CSV_DOSYA_ADI, index=False)
    test_df.to_csv(cikti_klasoru / TEST_CSV_DOSYA_ADI, index=False)

    print("\n[BASARILI] Veri seti bolundu:")
    print(f"  Eğitim: {len(train_df)} ({EGITIM_ORANI*100:.0f}%)")
    print(f"  Doğrulama: {len(val_df)} ({DOGRULAMA_ORANI*100:.0f}%)")
    print(f"  Test: {len(test_df)} ({TEST_ORANI*100:.0f}%)")

    return train_df, val_df, test_df


def veri_setini_bol_ve_olceklendir(
    csv_dosyasi: Optional[Path] = None,
    cikti_klasoru: Optional[Path] = None,
    metod: str = SCALING_METODU
):
    """
    Veri setini önce grup-bazlı böl, sonra scaler'ı sadece eğitim verisinde fit et.
    """
    if cikti_klasoru is None:
        cikti_klasoru = CIKTI_KLASORU

    cikarici = OzellikCikarici()

    try:
        boluntuleme_sonucu = veri_boluntule(csv_dosyasi=csv_dosyasi, cikti_klasoru=cikti_klasoru)
    except ValueError as e:
        print(f"[HATA] Veri boluntuleme basarisiz: {e}")
        return None

    if boluntuleme_sonucu is None:
        print("[HATA] Veri boluntuleme basarisiz oldu, olceklendirme yapilamiyor.")
        return None

    train_df, val_df, test_df = boluntuleme_sonucu

    scaler, _ = cikarici._scaler_olustur(metod)
    train_scaled, sayisal_sutunlar = cikarici._df_olceklendir(train_df, scaler, fit=True)
    val_scaled, _ = cikarici._df_olceklendir(val_df, scaler, fit=False)
    test_scaled, _ = cikarici._df_olceklendir(test_df, scaler, fit=False)

    train_scaled.to_csv(cikti_klasoru / EGITIM_SCALED_CSV_DOSYA_ADI, index=False, encoding='utf-8')
    val_scaled.to_csv(cikti_klasoru / DOGRULAMA_SCALED_CSV_DOSYA_ADI, index=False, encoding='utf-8')
    test_scaled.to_csv(cikti_klasoru / TEST_SCALED_CSV_DOSYA_ADI, index=False, encoding='utf-8')

    tum_scaled = pd.concat([train_scaled, val_scaled, test_scaled], ignore_index=True)
    tum_scaled.to_csv(cikti_klasoru / CSV_SCALED_DOSYA_ADI, index=False, encoding='utf-8')
    cikarici._scaler_kaydet(scaler, sayisal_sutunlar, metod, cikti_klasoru)

    print("\n[BASARILI] Leakage-free split + scaling tamamlandi:")
    print(f"  Egitim scaled: {cikti_klasoru / EGITIM_SCALED_CSV_DOSYA_ADI}")
    print(f"  Dogrulama scaled: {cikti_klasoru / DOGRULAMA_SCALED_CSV_DOSYA_ADI}")
    print(f"  Test scaled: {cikti_klasoru / TEST_SCALED_CSV_DOSYA_ADI}")
    print(f"  Scaler: {cikti_klasoru / SCALER_DOSYA_ADI}")

    return train_scaled, val_scaled, test_scaled
