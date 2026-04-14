#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eda_araclar.py
--------------
MRI görüntü veri seti için keşifsel veri analizi (EDA) araçları.
İstatistik hesaplama ve görselleştirme fonksiyonları.
"""

import os
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional, Union

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mri_classification_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERI_KLASORU = PROJECT_ROOT / "Veri_Seti" / "OriginalDataset"
DEFAULT_CIKTI_KLASORU = Path(__file__).resolve().parent / "eda_ciktilar"


def _guvenli_print(*args, sep: str = " ", end: str = "\n") -> None:
    """Konsol encoding'i Unicode desteklemese bile yazdırmayı sürdür."""
    metin = sep.join(str(arg) for arg in args)
    try:
        print(metin, end=end)
    except UnicodeEncodeError:
        stdout = sys.stdout
        encoding = getattr(stdout, "encoding", None) or "utf-8"
        tampon = getattr(stdout, "buffer", None)
        guvenli_metin = (metin + end).encode(encoding, errors="replace")
        if tampon is not None:
            tampon.write(guvenli_metin)
            tampon.flush()
        else:
            print(guvenli_metin.decode(encoding, errors="replace"), end="")


def _istatistik_hesapla_wrapper(satir_dict: Dict) -> Optional[Dict]:
    """Paralel istatistik hesaplama için wrapper fonksiyon."""
    try:
        with Image.open(satir_dict["filepath"]) as goruntu:
            if goruntu.mode != "L":
                goruntu = goruntu.convert("L")

            arr = np.array(goruntu)
            genislik, yukseklik = goruntu.size
            p1, p25, p50, p75, p99 = np.percentile(arr, [1, 25, 50, 75, 99])

            istat = {
                "id": satir_dict["id"],
                "genislik": genislik,
                "yukseklik": yukseklik,
                "en_boy_orani": genislik / yukseklik if yukseklik > 0 else 0,
                "int_ort": float(np.mean(arr)),
                "int_std": float(np.std(arr)),
                "int_min": float(np.min(arr)),
                "int_max": float(np.max(arr)),
                "int_p1": float(p1),
                "int_p25": float(p25),
                "int_p50": float(p50),
                "int_p75": float(p75),
                "int_p99": float(p99),
            }
            return istat
    except Exception as e:
        return {
            "__hata__": f"{satir_dict.get('filepath', '?')} ({type(e).__name__}: {e})"
        }


class EDAAnaLiz:
    """MRI görüntü veri seti için EDA sınıfı."""

    SAYISAL_OZELLIKLER = [
        "genislik",
        "yukseklik",
        "en_boy_orani",
        "int_ort",
        "int_std",
        "int_min",
        "int_max",
        "int_p1",
        "int_p99",
    ]  # int_p25/p50/p75 CSV'ye kaydedilir ama korelasyon/PCA'dan çıkarılmıştır (redundancy azaltma).

    def __init__(
        self,
        veri_klasoru: Union[str, Path] = DEFAULT_VERI_KLASORU,
        cikti_klasoru: Union[str, Path] = DEFAULT_CIKTI_KLASORU,
        rastgele_tohum: int = 42,
        n_jobs: Optional[int] = None,
    ):
        """
        EDA analizörünü başlat.

        EDA (Exploratory Data Analysis - Keşifsel Veri Analizi), veri setini
        anlamak ve görselleştirmek için yapılan ilk adımdır. Bu sınıf,
        MRI görüntü veri setini kapsamlı şekilde analiz eder.

        Args:
            veri_klasoru: MRI görüntülerinin bulunduğu klasör
            cikti_klasoru: Grafiklerin kaydedileceği klasör
            rastgele_tohum: Tekrarlanabilirlik için rastgeleliği sabitleme tohumu
            n_jobs: Paralel istatistik hesaplamada kullanılacak çekirdek sayısı
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
            "ModerateDemented",
        ]

        self.sinif_etiketi = {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3,
        }

        np.random.seed(self.tohum)
        if n_jobs is None:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = min(max(1, int(n_jobs)), cpu_count())

    def _mevcut_sinif_sirasi(self, df: pd.DataFrame) -> list[str]:
        """DataFrame icinde bulunan siniflari sabit sirada dondur."""
        mevcut_siniflar = set(df["label_name"].dropna().tolist())
        return [sinif for sinif in self.sinif_klasorleri if sinif in mevcut_siniflar]

    @staticmethod
    def _dosya_siralama_anahtari(dosya: Path) -> str:
        """Dosyalari platformdan bagimsiz ve deterministik sirala."""
        return dosya.name.lower()

    def _veri_klasorunu_dogrula(self):
        """Veri klasörü var mı ve beklenen yapıda mı kontrol et."""
        if not self.veri_klasoru.exists():
            raise FileNotFoundError(f"Veri klasörü bulunamadı: {self.veri_klasoru}")
        self.veri_klasoru = self._veri_klasoru_cozumle(self.veri_klasoru)

    @staticmethod
    def _sinif_klasorleri_var_mi(veri_klasoru: Path, sinif_klasorleri) -> bool:
        """Verilen klasörde sınıf klasörlerinden en az biri var mı?"""
        return any((veri_klasoru / klasor).exists() for klasor in sinif_klasorleri)

    def _veri_klasoru_cozumle(self, giris_klasoru: Path) -> Path:
        """
        Veri klasörünü veri yapısına göre otomatik çöz.

        Desteklenen yapılar:
        1) Veri_Seti/OriginalDataset/<SinifAdi>/
        2) Özel bir klasörde doğrudan <SinifAdi>/ yapısı
        """
        if self._sinif_klasorleri_var_mi(giris_klasoru, self.sinif_klasorleri):
            return giris_klasoru

        adaylar = [giris_klasoru / "OriginalDataset"]
        for aday in adaylar:
            if aday.exists() and self._sinif_klasorleri_var_mi(aday, self.sinif_klasorleri):
                _guvenli_print(f"[BILGI] Veri klasoru otomatik cozuldu: {aday}")
                return aday

        raise FileNotFoundError(
            "Veri klasörü beklenen sınıf klasörlerini içermiyor. "
            f"Verilen yol: {giris_klasoru}. "
            "Beklenen yapılar: Veri_Seti/OriginalDataset/<SinifAdi> "
            "veya verilen klasorde dogrudan <SinifAdi>."
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
        kayitlar = []
        idx = 0

        for sinif_adi in self.sinif_klasorleri:
            sinif_klasoru = self.veri_klasoru / sinif_adi

            if not sinif_klasoru.exists():
                _guvenli_print(f"[UYARI] Klasör bulunamadı: {sinif_klasoru}")
                continue

            dosyalar = sorted(sinif_klasoru.iterdir(), key=self._dosya_siralama_anahtari)
            desteklenen = {".jpg", ".jpeg", ".png"}
            atlanan_uzantilar: set[str] = set()

            for dosya in dosyalar:
                uzanti = dosya.suffix.lower()
                if uzanti in desteklenen:
                    kayitlar.append(
                        {
                            "id": idx,
                            "filepath": str(dosya),
                            "label": self.sinif_etiketi[sinif_adi],
                            "label_name": sinif_adi,
                        }
                    )
                    idx += 1
                elif dosya.is_file() and uzanti:
                    atlanan_uzantilar.add(uzanti)

            if atlanan_uzantilar:
                _guvenli_print(
                    f"[UYARI] {sinif_adi} klasöründe desteklenmeyen uzantılar atlandı: "
                    f"{', '.join(sorted(atlanan_uzantilar))}"
                )

        df = pd.DataFrame(kayitlar)
        if df.empty:
            raise ValueError(
                f"Veri klasöründe desteklenen uzantılarda görüntü bulunamadı: {self.veri_klasoru}"
            )
        return df

    def goruntu_istatistikleri_hesapla(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Her görüntü için temel istatistikleri hesapla.

        Bu fonksiyon, her görüntü için boyut, yoğunluk ve doku özelliklerini
        hesaplayarak DataFrame'e ekler. Bu istatistikler, veri setinin
        genel yapısını anlamamıza yardımcı olur.

        Not:
            Okunamayan veya bozuk görseller için istatistik kolonları NaN olur
            (left join). Downstream analizlerde NaN satırlar şu şekilde
            ele alınır: PCA ``dropna`` ile temizler, korelasyon ``corr()``
            otomatik çıkarır, boxplot/histplot sessizce atlar.

        Args:
            df: Görüntü yollarını içeren DataFrame

        Returns:
            İstatistiklerle genişletilmiş DataFrame
        """
        if df.empty:
            raise ValueError("İstatistik hesaplamak için en az bir satır gerekli.")

        _guvenli_print(f"[BILGI] İstatistikler hesaplanıyor (paralel: {self.n_jobs} çekirdek)...")

        satir_listesi = df.to_dict("records")

        sonuclar = []
        if self.n_jobs > 1:
            try:
                with Pool(processes=self.n_jobs) as pool:
                    sonuclar = list(
                        tqdm(
                            pool.imap(_istatistik_hesapla_wrapper, satir_listesi),
                            total=len(satir_listesi),
                            desc="İstatistikler hesaplanıyor (paralel)",
                        )
                    )
            except Exception as exc:
                _guvenli_print(
                    "[UYARI] Paralel istatistik hesaplama kullanilamadi; "
                    f"tek cekirdege dusuluyor ({type(exc).__name__}: {exc})."
                )
                self.n_jobs = 1

        if not sonuclar:
            for satir in tqdm(
                satir_listesi,
                total=len(satir_listesi),
                desc="İstatistikler hesaplanıyor",
            ):
                sonuclar.append(_istatistik_hesapla_wrapper(satir))

        istatistikler = []
        hatalar = []
        for sonuc in sonuclar:
            if sonuc is None:
                continue
            if "__hata__" in sonuc:
                hatalar.append(sonuc["__hata__"])
            else:
                istatistikler.append(sonuc)

        if not istatistikler:
            raise ValueError(
                "Hiçbir görüntüden istatistik hesaplanamadı. Dosyalar okunabilir mi kontrol edin."
            )
        if len(istatistikler) != len(df):
            _guvenli_print(
                f"[UYARI] {len(df) - len(istatistikler)} görüntüden istatistik alınamadı; dosyalar atlandı."
            )
        if hatalar:
            _guvenli_print("[UYARI] İstatistik hesaplanamayan dosyalar (ilk 5):")
            for hata in hatalar[:5]:
                _guvenli_print(f"   - {hata}")
            if len(hatalar) > 5:
                _guvenli_print(f"   ... ve {len(hatalar) - 5} dosya daha")

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
        fig.savefig(yol, dpi=200, bbox_inches="tight")
        plt.close(fig)
        _guvenli_print(f"[OK] Kaydedildi: {yol}")

    def sinif_dagilimi_ciz(self, df: pd.DataFrame):
        """
        Sınıf dağılımı grafiği çiz.

        Her sınıfta kaç görüntü olduğunu gösteren çubuk grafik.
        Dengesiz veri setlerini tespit etmek için önemlidir.
        """
        if df.empty:
            _guvenli_print("[UYARI] Sınıf dağılımı atlandı: DataFrame boş.")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        sinif_sirasi = self._mevcut_sinif_sirasi(df)
        sns.countplot(data=df, x="label_name", order=sinif_sirasi, ax=ax)
        ax.set_xlabel("Sınıf")
        ax.set_ylabel("Görüntü Sayısı")
        ax.set_title("Sınıf Dağılımı")
        ax.tick_params(axis='x', rotation=45)
        self.grafik_kaydet(fig, "1_sinif_dagilimi.png")

    def boyut_analizi_ciz(self, df: pd.DataFrame):
        """
        Görüntü boyut analizi grafiği çiz.

        Görüntülerin genişlik, yükseklik ve en-boy oranı dağılımlarını gösterir.
        Boyut tutarlılığını ve standartlaştırma ihtiyacını anlamak için kullanılır.
        """
        if df.empty:
            _guvenli_print("[UYARI] Boyut analizi atlandı: DataFrame boş.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        histplot_ayarlari = [
            ("genislik", "Genişlik Dağılımı", "Genişlik (piksel)"),
            ("yukseklik", "Yükseklik Dağılımı", "Yükseklik (piksel)"),
            ("en_boy_orani", "En-Boy Oranı Dağılımı", "En-Boy Oranı"),
        ]
        for ax, (kolon, baslik, xlabel) in zip(axes.flat, histplot_ayarlari):
            sns.histplot(df[kolon], kde=True, ax=ax)
            ax.set_title(baslik)
            ax.set_xlabel(xlabel)

        for sinif in self._mevcut_sinif_sirasi(df):
            alt_df = df[df["label_name"] == sinif]
            axes[1, 1].scatter(
                alt_df["genislik"],
                alt_df["yukseklik"],
                label=sinif,
                alpha=0.6,
                s=20,
            )
        axes[1, 1].set_xlabel("Genişlik")
        axes[1, 1].set_ylabel("Yükseklik")
        axes[1, 1].set_title("Genişlik vs Yükseklik")
        axes[1, 1].legend()

        plt.tight_layout()
        self.grafik_kaydet(fig, "2_boyut_analizi.png")

    def yogunluk_analizi_ciz(self, df: pd.DataFrame):
        """
        Yoğunluk (intensity) analizi grafiği çiz.

        Her sınıf için piksel yoğunluk istatistiklerini karşılaştırır.
        Sınıflar arası yoğunluk farklarını görmek için kullanılır.
        Ortalama, standart sapma, aralık ve yayılım grafiklerini içerir.
        """
        if df.empty:
            _guvenli_print("[UYARI] Yoğunluk analizi atlandı: DataFrame boş.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sinif_sirasi = self._mevcut_sinif_sirasi(df)
        plot_df = df.assign(
            int_range=df["int_max"] - df["int_min"],
            int_spread=df["int_p99"] - df["int_p1"],
        )

        boxplot_ayarlari = [
            ("int_ort", "Ortalama Yoğunluk (Sınıflara Göre)", "Ortalama Yoğunluk"),
            ("int_std", "Yoğunluk Std. Sapması (Sınıflara Göre)", "Std. Sapma"),
            ("int_range", "Yoğunluk Aralığı (Max-Min)", "Aralık"),
            ("int_spread", "Yoğunluk Yayılımı (P99-P1)", "Yayılım"),
        ]
        for ax, (kolon, baslik, ylabel) in zip(axes.flat, boxplot_ayarlari):
            sns.boxplot(data=plot_df, x="label_name", y=kolon, order=sinif_sirasi, ax=ax)
            ax.set_title(baslik)
            ax.set_xlabel("Sınıf")
            ax.set_ylabel(ylabel)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        self.grafik_kaydet(fig, "3_yogunluk_analizi.png")

    def korelasyon_analizi_ciz(self, df: pd.DataFrame):
        """Özellikler arası korelasyon matrisi."""
        mevcut_kolonlar = [k for k in self.SAYISAL_OZELLIKLER if k in df.columns]
        if len(mevcut_kolonlar) < 2:
            _guvenli_print("[UYARI] Korelasyon analizi atlandı: en az iki sayısal özellik gerekiyor.")
            return

        korelasyon = df[mevcut_kolonlar].corr()
        if korelasyon.empty or korelasyon.isna().all().all():
            _guvenli_print("[UYARI] Korelasyon analizi atlandı: korelasyon matrisi hesaplanamadı.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            korelasyon,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
        )
        ax.set_title("Özellikler Arası Korelasyon Matrisi")
        plt.tight_layout()
        self.grafik_kaydet(fig, "4_korelasyon_matrisi.png")

    def pca_analizi_ciz(self, df: pd.DataFrame, n_ornekler: int = 500):
        """
        PCA görselleştirmesi çiz.

        PCA (Principal Component Analysis), çok boyutlu veriyi 2 boyuta indirgeyen
        bir boyut azaltma tekniğidir. Bu grafik, sınıfların birbirinden
        ne kadar ayrılabilir olduğunu gösterir.

        İyi ayrılmış kümeler = kolay sınıflandırma
        İç içe geçmiş kümeler = zor sınıflandırma

        Args:
            df: Özellik DataFrame'i
            n_ornekler: PCA için kullanılacak maksimum örnek sayısı
        """
        ozellikler = self.SAYISAL_OZELLIKLER
        if len(df) < 2:
            _guvenli_print("[UYARI] PCA atlandı: En az iki örnek gerekiyor.")
            return

        eksik_ozellikler = [kolon for kolon in ozellikler if kolon not in df.columns]
        if eksik_ozellikler:
            _guvenli_print(
                "[UYARI] PCA atlandı: gerekli özellikler eksik "
                f"({', '.join(eksik_ozellikler)})."
            )
            return

        temiz_df = df.dropna(subset=ozellikler + ["label_name"])
        if len(temiz_df) < 2:
            _guvenli_print("[UYARI] PCA atlandı: yeterli sayida gecerli ornek yok.")
            return

        df_sample = temiz_df.sample(min(n_ornekler, len(temiz_df)), random_state=self.tohum)
        X = df_sample[ozellikler].values
        y = df_sample["label_name"].values
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=self.tohum)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 7))
        for sinif in self._mevcut_sinif_sirasi(df_sample):
            mask = y == sinif
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=sinif, alpha=0.6, s=50)

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

        with open(rapor_yolu, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("MRI VERİ SETİ - ÖZET İSTATİSTİKLER\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Toplam Görüntü Sayısı: {len(df)}\n")
            f.write(f"Sınıf Sayısı: {df['label'].nunique()}\n\n")

            f.write("Sınıf Dağılımı:\n")
            f.write("-" * 70 + "\n")
            sinif_sayilari = df["label_name"].value_counts().reindex(
                self.sinif_klasorleri,
                fill_value=0,
            )
            for sinif, sayi in sinif_sayilari.items():
                if sayi == 0:
                    continue
                oran = sayi / len(df) * 100
                f.write(f"  {sinif:20s}: {sayi:5d} (%{oran:.1f})\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("TEMEL İSTATİSTİKLER\n")
            f.write("=" * 70 + "\n\n")
            f.write(df.describe().to_string())

        _guvenli_print(f"[OK] Özet rapor kaydedildi: {rapor_yolu}")

    def tam_analiz_yap(self):
        """Tüm EDA analizini çalıştır."""
        _guvenli_print("\n" + "=" * 70)
        _guvenli_print("MRI VERİ SETİ KEŞİFSEL VERİ ANALİZİ (EDA)")
        _guvenli_print("=" * 70 + "\n")

        _guvenli_print(f"1. Veri yükleniyor... ({self.veri_klasoru})")
        df = self.veri_yukle()
        _guvenli_print(f"   [OK] {len(df)} görüntü yüklendi\n")

        _guvenli_print("2. Görüntü istatistikleri hesaplanıyor...")
        df = self.goruntu_istatistikleri_hesapla(df)
        _guvenli_print("   [OK] İstatistikler hesaplandı\n")

        _guvenli_print("3. Özet rapor oluşturuluyor...")
        self.ozet_istatistik_raporu(df)

        _guvenli_print("\n4. Grafikler oluşturuluyor...")
        _guvenli_print("   - Sınıf dağılımı...")
        self.sinif_dagilimi_ciz(df)

        _guvenli_print("   - Boyut analizi...")
        self.boyut_analizi_ciz(df)

        _guvenli_print("   - Yoğunluk analizi...")
        self.yogunluk_analizi_ciz(df)

        _guvenli_print("   - Korelasyon analizi...")
        self.korelasyon_analizi_ciz(df)

        _guvenli_print("   - PCA analizi...")
        self.pca_analizi_ciz(df)

        _guvenli_print("\n" + "=" * 70)
        _guvenli_print("[OK] TUM ANALIZ TAMAMLANDI!")
        _guvenli_print(f"[OK] Ciktilar kaydedildi: {self.cikti_klasoru}")
        _guvenli_print("=" * 70 + "\n")

        return df
