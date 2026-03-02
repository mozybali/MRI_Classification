#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference.py
------------
Eğitilmiş model ile tahmin yapma (inference) scripti.
Yeni MRI görüntüleri için demans seviyesi tahmini yapar.

Kullanım:
    python3 inference.py --model path/to/model.pkl --image path/to/image.jpg
    python3 inference.py --model xgboost_latest.pkl --batch path/to/images/
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Görüntü işleme için
sys.path.insert(0, str(Path(__file__).parent.parent / "goruntu_isleme"))
from goruntu_isleyici import GorselIsleyici
from ayarlar import PROJE_KOK
from goruntu_isleme.ayarlar import SCALING_METODU, CSV_DOSYA_ADI

from ayarlar import MODELS_KLASORU, SCALER_DOSYASI
from scipy import ndimage
from scipy.stats import skew, kurtosis


def _batch_tahmin_wrapper(goruntu_yolu: Path, model_yolu: Path) -> Dict:
    """⚡ Paralel batch tahmin için wrapper fonksiyon."""
    try:
        inference = ModelInference(model_yolu)
        return inference.tahmin_yap(goruntu_yolu, detayli=False)
    except Exception as e:
        return {
            'dosya': str(goruntu_yolu),
            'hata': str(e)
        }


class ModelInference:
    """Eğitilmiş model ile tahmin yapma sınıfı."""
    
    def __init__(self, model_yolu: Union[str, Path]):
        """
        Inference nesnesini başlat.
        
        Args:
            model_yolu: Eğitilmiş model dosyasının yolu (.pkl)
        """
        self.model_yolu = Path(model_yolu)
        self.scaler = None
        self.scaler_columns: List[str] = []
        self.selected_features: Optional[List[str]] = None
        self.feature_names: Optional[List[str]] = None
        
        if not self.model_yolu.exists():
            # MODELS_KLASORU içinde ara
            alternatif = MODELS_KLASORU / self.model_yolu.name
            if alternatif.exists():
                self.model_yolu = alternatif
            else:
                raise FileNotFoundError(f"Model bulunamadı: {model_yolu}")
        
        # Modeli yükle
        self._model_yukle()
        # Ölçekleyiciyi hazırla (eğitim verisinden)
        self._hazirla_scaler()
        
        # Görüntü işleyici
        self.isleyici = GorselIsleyici()
        
        # Sınıf isimleri
        self.sinif_isimleri = {
            0: "NonDemented (Sağlıklı)",
            1: "VeryMildDemented (Çok Hafif Demans)",
            2: "MildDemented (Hafif Demans)",
            3: "ModerateDemented (Orta Seviye Demans)"
        }
        
        # ⚡ Paralel işlem için
        self.n_jobs = max(1, cpu_count() - 1)
    
    def _model_yukle(self):
        """Modeli ve metadata'sını yükle."""
        print(f"\n📦 Model yükleniyor: {self.model_yolu.name}")
        
        # Pickle model yükle
        with open(self.model_yolu, 'rb') as f:
            self.model = pickle.load(f)
        print(f"   ✓ Model yüklendi")
        
        # Metadata yükle (varsa)
        metadata_yolu = self.model_yolu.with_suffix('.json')
        if metadata_yolu.exists():
            with open(metadata_yolu, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"   ✓ Metadata yüklendi")
            print(f"   ℹ️  Model Tipi: {self.metadata.get('model_tipi', 'N/A')}")
            print(f"   ℹ️  Eğitim Tarihi: {self.metadata.get('tarih', 'N/A')}")
            self.feature_names = self.metadata.get('feature_names')
            self.selected_features = self.metadata.get('selected_features')
            
            # Metrikler varsa göster
            if 'metrikler' in self.metadata:
                metriks = self.metadata['metrikler']
                acc = metriks.get('accuracy')
                if isinstance(acc, (int, float)):
                    print(f"   ℹ️  Test Accuracy: {acc:.4f}")
                else:
                    print(f"   ℹ️  Test Accuracy: {acc}")
        else:
            self.metadata = {}
            print(f"   ⚠️  Metadata bulunamadı")

    def _hazirla_scaler(self):
        """Eğitimde kaydedilen ölçekleyiciyi yükle."""
        try:
            if SCALER_DOSYASI.exists():
                with open(SCALER_DOSYASI, 'rb') as f:
                    scaler_bundle = pickle.load(f)
                self.scaler = scaler_bundle.get("scaler")
                self.scaler_columns = scaler_bundle.get("columns", [])
                if not self.feature_names:
                    self.feature_names = self.scaler_columns
                print(f"   ✓ Ölçekleyici yüklendi ({scaler_bundle.get('method', 'bilinmiyor')})")
                return

            raw_csv = PROJE_KOK / "goruntu_isleme" / "cikti" / CSV_DOSYA_ADI
            if not raw_csv.exists():
                print(f"   ⚠️  Ölçekleyici hazırlanamadı (CSV yok): {raw_csv}")
                return
            
            df = pd.read_csv(raw_csv)
            kategorik = ['dosya_adi', 'sinif', 'etiket', 'tam_yol', 'kaynak_id', 'kaynak_grup', 'augmentasyon_mu']
            self.scaler_columns = [c for c in df.columns if c not in kategorik]
            if not self.feature_names:
                self.feature_names = self.scaler_columns
            print(f"   ⚠️  Kaydedilmiş scaler bulunamadı; fallback olarak sütun listesi çıkarıldı")
        except Exception as e:
            print(f"   ⚠️  Ölçekleyici hazırlanamadı: {e}")

    def goruntu_isle(self, goruntu_yolu: Union[str, Path]) -> np.ndarray:
        """
        Görüntüyü eğitim pipeline'ıyla aynı şekilde (goruntu_isleyici.goruntu_isle) işle.
        """
        goruntu = self.isleyici.goruntu_isle(str(goruntu_yolu))
        if goruntu is None:
            raise ValueError(f"Görüntü işlenemedi: {goruntu_yolu}")
        return goruntu

    def _ozellikler_from_array(
        self,
        goruntu: np.ndarray,
        dosya_adi: str,
        boyut_bayt: int,
        tam_yol: str,
    ) -> Dict:
        """
        İşlenmiş görüntüden (np.ndarray) eğitimdeki ile aynı sayısal özellikleri çıkar.
        """
        piksel_array = goruntu.astype(np.float32)
        yukseklik, genislik = piksel_array.shape
        en_boy_orani = genislik / yukseklik if yukseklik > 0 else 0.0

        ort_yogunluk = float(np.mean(piksel_array))
        std_yogunluk = float(np.std(piksel_array))
        min_yogunluk = float(np.min(piksel_array))
        max_yogunluk = float(np.max(piksel_array))

        p1_yogunluk = float(np.percentile(piksel_array, 1))
        p25_yogunluk = float(np.percentile(piksel_array, 25))
        p50_yogunluk = float(np.percentile(piksel_array, 50))
        p75_yogunluk = float(np.percentile(piksel_array, 75))
        p99_yogunluk = float(np.percentile(piksel_array, 99))

        hist, _ = np.histogram(piksel_array.astype(np.uint8), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        hist_nonzero = hist[hist > 0]
        entropi = float(-np.sum(hist_nonzero * np.log2(hist_nonzero))) if hist_nonzero.size > 0 else 0.0

        laplacian = ndimage.laplace(piksel_array)
        kontrast = float(np.var(laplacian))
        homojenlik = 1.0 / (1.0 + std_yogunluk) if std_yogunluk is not None else 0.0
        enerji = float(np.sum(hist ** 2))

        carpiklik = float(skew(piksel_array.flatten()))
        basiklik = float(kurtosis(piksel_array.flatten()))

        grad_y, grad_x = np.gradient(piksel_array)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        ortalama_gradyan = float(np.mean(gradient_magnitude))
        max_gradyan = float(np.max(gradient_magnitude))

        try:
            from skimage.filters import threshold_otsu

            otsu_esik = float(threshold_otsu(piksel_array))
        except Exception:
            otsu_esik = float(np.mean(piksel_array))

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
            "carpiklik": round(carpiklik, 4),
            "basiklik": round(basiklik, 4),
            "ortalama_gradyan": round(ortalama_gradyan, 2),
            "max_gradyan": round(max_gradyan, 2),
            "otsu_esik": round(otsu_esik, 2),
            "tam_yol": tam_yol,
        }
    
    def ozellik_cikar(self, goruntu_yolu: Union[str, Path]) -> pd.DataFrame:
        """
        Görüntüyü eğitim pipeline'ıyla işle ve sayısal özellikleri çıkar.
        """
        goruntu_yolu = Path(goruntu_yolu)
        islenmis = self.goruntu_isle(goruntu_yolu)

        boyut_bayt = os.path.getsize(goruntu_yolu) if goruntu_yolu.exists() else 0
        ozellikler = self._ozellikler_from_array(
            islenmis,
            dosya_adi=goruntu_yolu.name,
            boyut_bayt=boyut_bayt,
            tam_yol=str(goruntu_yolu),
        )

        df = pd.DataFrame([ozellikler])

        # Kategorik kolonları çıkar
        kategorik = ["dosya_adi", "tam_yol"]
        df_ozellikler = df.drop(columns=[c for c in kategorik if c in df.columns])

        # Ölçekleme
        if self.scaler and self.scaler_columns:
            for col in self.scaler_columns:
                if col not in df_ozellikler.columns:
                    df_ozellikler[col] = 0.0
            df_ozellikler = df_ozellikler[self.scaler_columns]
            scaled = self.scaler.transform(df_ozellikler)
            df_ozellikler = pd.DataFrame(scaled, columns=self.scaler_columns)
        elif self.feature_names:
            eksik = [c for c in self.feature_names if c not in df_ozellikler.columns]
            if eksik:
                raise ValueError(f"Eksik özellik kolonları: {eksik}")
            df_ozellikler = df_ozellikler[self.feature_names]

        if self.selected_features:
            eksik = [c for c in self.selected_features if c not in df_ozellikler.columns]
            if eksik:
                raise ValueError(f"Seçilen özellikler veri setinde yok: {eksik}")
            df_ozellikler = df_ozellikler[self.selected_features]

        return df_ozellikler
    
    def tahmin_yap(self, goruntu_yolu: Union[str, Path], 
                   detayli: bool = True) -> Dict:
        """
        Tek bir görüntü için demans seviyesi tahmini yap.
        
        Bu fonksiyon, ham MRI görüntüsünü alır ve şu adımları gerçekleştirir:
        1. Görüntüyü işle (normalizasyon, yeniden boyutlandırma, vb.)
        2. Özellikleri çıkar (20+ sayısal özellik)
        3. Model ile tahmin yap
        4. Olasılıkları ve güven skorunu hesapla
        
        Çıktı örnekleri:
        {
            'tahmin': 'NonDemented (Sağlıklı)',
            'tahmin_kodu': 0,
            'guven': 0.92,
            'olasiliklar': {
                'NonDemented': 0.92,
                'VeryMildDemented': 0.05,
                'MildDemented': 0.02,
                'ModerateDemented': 0.01
            },
            'goruntu_yolu': '/path/to/image.jpg'
        }
        
        Args:
            goruntu_yolu: MRI görüntüsünün dosya yolu
            detayli: True ise tüm olasılıkları da döndür
            
        Returns:
            Dict: Tahmin sonuçları (tahmin, güven, olasılıklar)
        """
        """
        Tek bir görüntü için tahmin yap.
        
        Args:
            goruntu_yolu: Görüntü dosyasının yolu
            detayli: Detaylı çıktı (olasılıklar dahil)
            
        Returns:
            Tahmin sonuçları sözlüğü
        """
        print(f"\n🔍 Tahmin yapılıyor: {Path(goruntu_yolu).name}")
        
        # Özellikleri çıkar
        X = self.ozellik_cikar(goruntu_yolu)
        
        # Tahmin yap
        tahmin = self.model.predict(X)[0]
        sinif_adi = self.sinif_isimleri[tahmin]
        
        sonuc = {
            'dosya': str(goruntu_yolu),
            'tahmin_sinif': int(tahmin),
            'tahmin_adi': sinif_adi
        }
        
        # Olasılıklar (varsa)
        if hasattr(self.model, 'predict_proba'):
            olasiliklar = self.model.predict_proba(X)[0]
            sonuc['olasiliklar'] = {
                self.sinif_isimleri[i]: float(prob) 
                for i, prob in enumerate(olasiliklar)
            }
            sonuc['guven_skoru'] = float(max(olasiliklar))
        
        # Ekrana yazdır
        print(f"\n{'='*60}")
        print(f"📊 TAHMİN SONUCU")
        print(f"{'='*60}")
        print(f"🎯 Tahmin: {sinif_adi}")
        
        if 'guven_skoru' in sonuc:
            print(f"📈 Güven Skoru: {sonuc['guven_skoru']:.2%}")
            
            if detayli:
                print(f"\n📋 Sınıf Olasılıkları:")
                for sinif, prob in sorted(sonuc['olasiliklar'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    bar = '█' * int(prob * 40)
                    print(f"   {sinif:40s}: {prob:6.2%} {bar}")
        
        print(f"{'='*60}\n")
        
        return sonuc
    
    def batch_tahmin(self, goruntu_klasoru: Union[str, Path], 
                     kaydet: bool = True) -> List[Dict]:
        """
        Bir klasördeki tüm görüntüler için toplu tahmin yap.
        
        Bu fonksiyon, klinik kullanım için idealdir:
        - Çok sayıda hasta görüntüsünü tek seferde işler
        - Sonuçları CSV'ye kaydeder (raporlama için)
        - İlerleme çubuğu gösterir
        
        Çıktı CSV formatı:
        | goruntu_adi | tahmin | tahmin_kodu | guven | NonDemented | VeryMildDemented | ... |
        |-------------|--------|-------------|-------|-------------|------------------|-----|
        | img1.jpg    | NonDemented | 0 | 0.92 | 0.92 | 0.05 | ... |
        
        Kullanım senaryosu:
        ```python
        inferencer = ModelInference('xgboost_model.pkl')
        sonuclar = inferencer.batch_tahmin('./yeni_hastalar/')
        # Sonuçlar otomatik CSV'ye kaydedilir
        ```
        
        Args:
            goruntu_klasoru: Görüntülerin bulunduğu klasör yolu
            kaydet: Sonuçları CSV'ye kaydet (varsayılan: True)
            
        Returns:
            List[Dict]: Tüm tahmin sonuçları
        """
        klasor = Path(goruntu_klasoru)
        
        if not klasor.exists():
            raise FileNotFoundError(f"Klasör bulunamadı: {goruntu_klasoru}")
        
        # Görüntüleri bul
        gorseller = list(klasor.glob("*.jpg")) + list(klasor.glob("*.png"))
        
        if not gorseller:
            print(f"⚠️  Klasörde görüntü bulunamadı: {goruntu_klasoru}")
            return []
        
        print(f"\n⚡ Batch tahmin: {len(gorseller)} görüntü (paralel: {self.n_jobs} çekirdek)")
        print(f"{'='*60}")
        
        # ⚡ Paralel batch tahmin
        partial_func = partial(_batch_tahmin_wrapper, model_yolu=self.model_yolu)
        
        with Pool(processes=self.n_jobs) as pool:
            sonuclar = list(tqdm(
                pool.imap(partial_func, gorseller),
                total=len(gorseller),
                desc="Batch tahmin (paralel)"
            ))
        
        # Özet
        print(f"\n{'='*60}")
        print(f"📊 BATCH TAHMİN ÖZETİ")
        print(f"{'='*60}")
        print(f"Toplam: {len(gorseller)}")
        print(f"Başarılı: {len([s for s in sonuclar if 'tahmin_sinif' in s])}")
        print(f"Hatalı: {len([s for s in sonuclar if 'hata' in s])}")
        
        # Sınıf dağılımı
        if sonuclar:
            sinif_sayilari = {}
            for sonuc in sonuclar:
                if 'tahmin_adi' in sonuc:
                    sinif = sonuc['tahmin_adi']
                    sinif_sayilari[sinif] = sinif_sayilari.get(sinif, 0) + 1
            
            print(f"\n📈 Tahmin Dağılımı:")
            for sinif, sayi in sorted(sinif_sayilari.items()):
                print(f"   {sinif:40s}: {sayi:3d}")
        
        # Kaydet
        if kaydet and sonuclar:
            cikti_dosya = klasor / f"tahminler_{Path(self.model_yolu).stem}.csv"
            df = pd.DataFrame(sonuclar)
            df.to_csv(cikti_dosya, index=False, encoding='utf-8')
            print(f"\n💾 Sonuçlar kaydedildi: {cikti_dosya}")
        
        return sonuclar


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="Eğitilmiş model ile MRI tahmin (inference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Tek görüntü tahmin
  python3 inference.py --model xgboost_latest.pkl --image test.jpg
  
  # Batch tahmin (klasördeki tüm görüntüler)
  python3 inference.py --model xgboost_latest.pkl --batch ./test_images/
  
  # En son eğitilmiş model ile tahmin
  python3 inference.py --image test.jpg
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model dosyası yolu (.pkl). Belirtilmezse en son model kullanılır.'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Tahmin yapılacak tek bir görüntü dosyası'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Tahmin yapılacak görüntülerin bulunduğu klasör'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Batch tahmin sonuçlarını kaydetme'
    )
    
    args = parser.parse_args()
    
    # Parametre kontrolü
    if not args.image and not args.batch:
        parser.error("--image veya --batch belirtilmeli")
    
    # Model yolu
    if args.model:
        model_yolu = args.model
    else:
        # En son model ara
        modeller = sorted(MODELS_KLASORU.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        if not modeller:
            print("❌ Hiç model bulunamadı!")
            print(f"   Aranan klasör: {MODELS_KLASORU}")
            print(f"\n💡 Önce model eğitin:")
            print(f"   python3 train.py --auto")
            return 1
        
        model_yolu = modeller[-1]
        print(f"ℹ️  En son model kullanılıyor: {model_yolu.name}")
    
    try:
        # Inference nesnesi oluştur
        inferencer = ModelInference(model_yolu)
        
        # Tek görüntü veya batch
        if args.image:
            inferencer.tahmin_yap(args.image, detayli=True)
        elif args.batch:
            inferencer.batch_tahmin(args.batch, kaydet=not args.no_save)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
