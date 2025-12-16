"""
eda_araclar.py
--------------
MRI görüntü veri seti için keşifsel veri analizi (EDA) araçları.
İstatistik hesaplama ve görselleştirme fonksiyonları.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Union
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERI_KLASORU = PROJECT_ROOT / "Veri_Seti"
DEFAULT_CIKTI_KLASORU = Path(__file__).resolve().parent / "eda_ciktilar"


def _istatistik_hesapla_wrapper(satir_dict: Dict) -> Optional[Dict]:
    """⚡ Paralel istatistik hesaplama için wrapper fonksiyon."""
    try:
        with Image.open(satir_dict["filepath"]) as goruntu:
            if goruntu.mode != 'L':
                goruntu = goruntu.convert('L')

            arr = np.array(goruntu)
            genislik, yukseklik = goruntu.size

            istat = {
                "id": satir_dict["id"],
                "genislik": genislik,
                "yukseklik": yukseklik,
                "en_boy_orani": genislik / yukseklik if yukseklik > 0 else 0,
                "int_ort": float(np.mean(arr)),
                "int_std": float(np.std(arr)),
                "int_min": float(np.min(arr)),
                "int_max": float(np.max(arr)),
                "int_p1": float(np.percentile(arr, 1)),
                "int_p25": float(np.percentile(arr, 25)),
                "int_p50": float(np.percentile(arr, 50)),
                "int_p75": float(np.percentile(arr, 75)),
                "int_p99": float(np.percentile(arr, 99))
            }
            return istat
    except Exception:
        return None


class EDAAnaLiz:
    """MRI görüntü veri seti için EDA sınıfı."""
    
    def __init__(self, veri_klasoru: Union[str, Path] = DEFAULT_VERI_KLASORU,
                 cikti_klasoru: Union[str, Path] = DEFAULT_CIKTI_KLASORU,
                 rastgele_tohum: int = 42):
        """
        EDA analizörünü başlat.
        
        EDA (Exploratory Data Analysis - Keşifsel Veri Analizi), veri setini
        anlamak ve görselleştirmek için yapılan ilk adımdır. Bu sınıf,
        MRI görüntü veri setini kapsamlı şekilde analiz eder.
        
        Args:
            veri_klasoru: MRI görüntülerinin bulunduğu klasör
            cikti_klasoru: Grafiklerin kaydedileceği klasör
            rastgele_tohum: Tekrarlanabilirlik için rastgeleliği sabitleme tohumu
        """
        self.veri_klasoru = Path(veri_klasoru).expanduser().resolve()
        self.cikti_klasoru = Path(cikti_klasoru).expanduser().resolve()
        self.cikti_klasoru.mkdir(parents=True, exist_ok=True)
        self.tohum = rastgele_tohum
        
        # Sınıf tanımları (demans seviyeleri)
        
        self.sinif_klasorleri = [
            "NonDemented",
            "VeryMildDemented",
            "MildDemented",
            "ModerateDemented"
        ]
        
        self.sinif_etiketi = {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3
        }
        
        np.random.seed(self.tohum)
        self.n_jobs = max(1, cpu_count() - 1)  # ⚡ Paralel işleme için
    
    def _veri_klasorunu_dogrula(self):
        """Veri klasörü var mı ve beklenen alt klasörlerden en az biri mevcut mu kontrol et."""
        if not self.veri_klasoru.exists():
            raise FileNotFoundError(f"Veri klasörü bulunamadı: {self.veri_klasoru}")
        
        alt_klasor_var = any((self.veri_klasoru / klasor).exists() for klasor in self.sinif_klasorleri)
        if not alt_klasor_var:
            raise FileNotFoundError(
                f"Veri klasörü beklenen sınıf klasörlerini içermiyor: {self.veri_klasoru}"
            )

    def veri_yukle(self) -> pd.DataFrame:
        """
        Veri setinden tüm görüntü yollarını ve etiketlerini yükle.
        
        Bu fonksiyon, veri seti klasöründeki tüm sınıf klasörlerini tarar ve
        her görüntü için bir kayıt oluşturur. Bu kayıtlar daha sonra
        analizlerde kullanılır.
        
        Returns:
            DataFrame: id, filepath, label, label_name kolonları içeren tablo
        """
        self._veri_klasorunu_dogrula()
        kayitlar = []  # Tüm görüntü kayıtlarını sakla
        idx = 0        # Benzersiz ID sayacı
        
        for sinif_adi in self.sinif_klasorleri:
            sinif_klasoru = self.veri_klasoru / sinif_adi
            
            if not sinif_klasoru.exists():
                print(f"[UYARI] Klasör bulunamadı: {sinif_klasoru}")
                continue
            
            for dosya in sinif_klasoru.glob("*"):
                if dosya.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    kayitlar.append({
                        "id": idx,
                        "filepath": str(dosya),
                        "label": self.sinif_etiketi[sinif_adi],
                        "label_name": sinif_adi
                    })
                    idx += 1
        
        df = pd.DataFrame(kayitlar)
        if df.empty:
            raise ValueError(f"Veri klasöründe desteklenen uzantılarda görüntü bulunamadı: {self.veri_klasoru}")
        return df
    
    def goruntu_istatistikleri_hesapla(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Her görüntü için temel istatistikleri hesapla.
        
        Bu fonksiyon, her görüntü için boyut, yoğunluk ve doku özelliklerini
        hesaplayarak DataFrame'e ekler. Bu istatistikler, veri setinin
        genel yapısını anlamamıza yardımcı olur.
        
        Args:
            df: Görüntü yollarını içeren DataFrame
            
        Returns:
            İstatistiklerle genişletilmiş DataFrame
        """
        if df.empty:
            raise ValueError("İstatistik hesaplamak için en az bir satır gerekli.")

        print(f"⚡ İstatistikler hesaplanıyor (paralel: {self.n_jobs} çekirdek)...")
        
        # DataFrame'i dict listesine çevir (multiprocessing için)
        satir_listesi = df.to_dict('records')
        
        # ⚡ Paralel istatistik hesaplama
        with Pool(processes=self.n_jobs) as pool:
            istatistikler = list(tqdm(
                pool.imap(_istatistik_hesapla_wrapper, satir_listesi),
                total=len(satir_listesi),
                desc="İstatistikler hesaplanıyor (paralel)"
            ))
        
        # None olmayan sonuçları al
        istatistikler = [i for i in istatistikler if i is not None]
        if not istatistikler:
            raise ValueError("Hiçbir görüntüden istatistik hesaplanamadı. Dosyalar okunabilir mi kontrol edin.")
        if len(istatistikler) != len(df):
            print(f"[UYARI] {len(df) - len(istatistikler)} görüntüden istatistik alınamadı; dosyalar atlandı.")
        
        istat_df = pd.DataFrame(istatistikler)
        return df.merge(istat_df, on="id", how="left")
    
    def grafik_kaydet(self, fig, dosya_adi: str):
        """
        Matplotlib grafiğini dosyaya kaydet ve belleği temizle.
        
        Args:
            fig: Matplotlib figure nesnesi
            dosya_adi: Kaydedilecek dosya adı (.png uzantısı ile)
        """
        yol = self.cikti_klasoru / dosya_adi
        fig.savefig(yol, dpi=200, bbox_inches='tight')  # Yüksek çözünürlük, kırpılmadan kaydet
        plt.close(fig)  # Belleği temizle (memory leak önlemek için önemli!)
        print(f"✓ Kaydedildi: {yol}")
    
    def sinif_dagilimi_ciz(self, df: pd.DataFrame):
        """Sınıf dağılımı grafiği çiz.
        
        Her sınıfta kaç görüntü olduğunu gösteren çubuk grafik.
        Dengesiz veri setlerini tespit etmek için önemlidir.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="label_name", ax=ax)
        ax.set_xlabel("Sınıf")
        ax.set_ylabel("Görüntü Sayısı")
        ax.set_title("Sınıf Dağılımı")
        plt.xticks(rotation=45)
        self.grafik_kaydet(fig, "1_sinif_dagilimi.png")
    
    def boyut_analizi_ciz(self, df: pd.DataFrame):
        """Görüntü boyut analizi grafiği çiz.
        
        Görüntülerin genişlik, yükseklik ve en-boy oranı dağılımlarını gösterir.
        Boyut tutarlılığını ve standartlaştırma ihtiyacını anlamak için kullanılır.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Genişlik histogramı
        sns.histplot(df["genislik"], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Genişlik Dağılımı")
        axes[0, 0].set_xlabel("Genişlik (piksel)")
        
        # Yükseklik histogramı
        sns.histplot(df["yukseklik"], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Yükseklik Dağılımı")
        axes[0, 1].set_xlabel("Yükseklik (piksel)")
        
        # En-boy oranı
        sns.histplot(df["en_boy_orani"], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("En-Boy Oranı Dağılımı")
        axes[1, 0].set_xlabel("En-Boy Oranı")
        
        # Genişlik vs Yükseklik scatter
        for sinif in df["label_name"].unique():
            alt_df = df[df["label_name"] == sinif]
            axes[1, 1].scatter(alt_df["genislik"], alt_df["yukseklik"], 
                             label=sinif, alpha=0.6, s=20)
        axes[1, 1].set_xlabel("Genişlik")
        axes[1, 1].set_ylabel("Yükseklik")
        axes[1, 1].set_title("Genişlik vs Yükseklik")
        axes[1, 1].legend()
        
        plt.tight_layout()
        self.grafik_kaydet(fig, "2_boyut_analizi.png")
    
    def yogunluk_analizi_ciz(self, df: pd.DataFrame):
        """Yoğunluk (intensity) analizi grafiği çiz.
        
        Her sınıf için piksel yoğunluk istatistiklerini karşılaştırır.
        Sınıflar arası yoğunluk farklarını görmek için kullanılır.
        Ortalama, standart sapma, aralık ve yayılım grafiklerini içerir.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Ortalama yoğunluk
        sns.boxplot(data=df, x="label_name", y="int_ort", ax=axes[0, 0])
        axes[0, 0].set_title("Ortalama Yoğunluk (Sınıflara Göre)")
        axes[0, 0].set_xlabel("Sınıf")
        axes[0, 0].set_ylabel("Ortalama Yoğunluk")
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Standart sapma
        sns.boxplot(data=df, x="label_name", y="int_std", ax=axes[0, 1])
        axes[0, 1].set_title("Yoğunluk Std. Sapması (Sınıflara Göre)")
        axes[0, 1].set_xlabel("Sınıf")
        axes[0, 1].set_ylabel("Std. Sapma")
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Min-Max range
        df["int_range"] = df["int_max"] - df["int_min"]
        sns.boxplot(data=df, x="label_name", y="int_range", ax=axes[1, 0])
        axes[1, 0].set_title("Yoğunluk Aralığı (Max-Min)")
        axes[1, 0].set_xlabel("Sınıf")
        axes[1, 0].set_ylabel("Aralık")
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Percentile spread
        df["int_spread"] = df["int_p99"] - df["int_p1"]
        sns.boxplot(data=df, x="label_name", y="int_spread", ax=axes[1, 1])
        axes[1, 1].set_title("Yoğunluk Yayılımı (P99-P1)")
        axes[1, 1].set_xlabel("Sınıf")
        axes[1, 1].set_ylabel("Yayılım")
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        self.grafik_kaydet(fig, "3_yogunluk_analizi.png")
    
    def korelasyon_analizi_ciz(self, df: pd.DataFrame):
        """Özellikler arası korelasyon matrisi."""
        numerik_kolonlar = [
            "genislik", "yukseklik", "en_boy_orani",
            "int_ort", "int_std", "int_min", "int_max",
            "int_p1", "int_p99"
        ]
        
        mevcut_kolonlar = [k for k in numerik_kolonlar if k in df.columns]
        korelasyon = df[mevcut_kolonlar].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(korelasyon, annot=True, fmt=".2f", cmap="coolwarm",
                   center=0, square=True, ax=ax)
        ax.set_title("Özellikler Arası Korelasyon Matrisi")
        plt.tight_layout()
        self.grafik_kaydet(fig, "4_korelasyon_matrisi.png")
    
    def pca_analizi_ciz(self, df: pd.DataFrame, n_ornekler: int = 500):
        """PCA görselleştirmesi çiz.
        
        PCA (Principal Component Analysis), çok boyutlu veriyi 2 boyuta indirgeyen
        bir boyut azaltma tekniğidir. Bu grafik, sınıfların birbirinden
        ne kadar ayrılabilir olduğunu gösterir.
        
        İyi ayrılmış kümeler = kolay sınıflandırma
        İç içe geçmiş kümeler = zor sınıflandırma
        
        Args:
            df: Özellik DataFrame'i
            n_ornekler: PCA için kullanılacak maksimum örnek sayısı
        """
        # Özellik matrisi oluştur
        ozellikler = [
            "genislik", "yukseklik", "en_boy_orani",
            "int_ort", "int_std", "int_min", "int_max",
            "int_p1", "int_p99"
        ]
        if len(df) < 2:
            print("PCA atlandı: En az iki örnek gerekiyor.")
            return
        
        df_sample = df.sample(min(n_ornekler, len(df)), random_state=self.tohum)
        X = df_sample[ozellikler].fillna(0).values
        y = df_sample["label_name"].values
        
        # PCA uygula
        pca = PCA(n_components=2, random_state=self.tohum)
        X_pca = pca.fit_transform(X)
        
        # Görselleştir
        fig, ax = plt.subplots(figsize=(10, 7))
        for sinif in np.unique(y):
            mask = y == sinif
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=sinif, alpha=0.6, s=50)
        
        ax.set_xlabel(f"PC1 (Açıklanan Varyans: {pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 (Açıklanan Varyans: {pca.explained_variance_ratio_[1]:.2%})")
        ax.set_title("PCA - İlk 2 Bileşen")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        self.grafik_kaydet(fig, "5_pca_analizi.png")
    
    def ozet_istatistik_raporu(self, df: pd.DataFrame):
        """Özet istatistik raporu oluştur ve kaydet."""
        rapor_yolu = self.cikti_klasoru / "0_ozet_istatistikler.txt"
        
        with open(rapor_yolu, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MRI VERİ SETİ - ÖZET İSTATİSTİKLER\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Toplam Görüntü Sayısı: {len(df)}\n")
            f.write(f"Sınıf Sayısı: {df['label'].nunique()}\n\n")
            
            f.write("Sınıf Dağılımı:\n")
            f.write("-"*70 + "\n")
            for sinif, sayi in df['label_name'].value_counts().items():
                oran = sayi / len(df) * 100
                f.write(f"  {sinif:20s}: {sayi:5d} (%{oran:.1f})\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TEMEL İSTATİSTİKLER\n")
            f.write("="*70 + "\n\n")
            f.write(df.describe().to_string())
            
        print(f"✓ Özet rapor kaydedildi: {rapor_yolu}")
    
    def tam_analiz_yap(self):
        """Tüm EDA analizini çalıştır."""
        print("\n" + "="*70)
        print("MRI VERİ SETİ KEŞİFSEL VERİ ANALİZİ (EDA)")
        print("="*70 + "\n")
        
        # Veri yükle
        print(f"1. Veri yükleniyor... ({self.veri_klasoru})")
        df = self.veri_yukle()
        print(f"   ✓ {len(df)} görüntü yüklendi\n")
        
        # İstatistikleri hesapla
        print("2. Görüntü istatistikleri hesaplanıyor...")
        df = self.goruntu_istatistikleri_hesapla(df)
        print(f"   ✓ İstatistikler hesaplandı\n")
        
        # Özet rapor
        print("3. Özet rapor oluşturuluyor...")
        self.ozet_istatistik_raporu(df)
        
        # Grafikler
        print("\n4. Grafikler oluşturuluyor...")
        print("   - Sınıf dağılımı...")
        self.sinif_dagilimi_ciz(df)
        
        print("   - Boyut analizi...")
        self.boyut_analizi_ciz(df)
        
        print("   - Yoğunluk analizi...")
        self.yogunluk_analizi_ciz(df)
        
        print("   - Korelasyon analizi...")
        self.korelasyon_analizi_ciz(df)
        
        print("   - PCA analizi...")
        self.pca_analizi_ciz(df)
        
        print("\n" + "="*70)
        print("✓ TÜM ANALİZ TAMAMLANDI!")
        print(f"✓ Çıktılar kaydedildi: {self.cikti_klasoru}")
        print("="*70 + "\n")
        
        return df
