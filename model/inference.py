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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional

# Görüntü işleme için
sys.path.insert(0, str(Path(__file__).parent.parent / "goruntu_isleme"))
from goruntu_isleyici import GorselIsleyici
from ozellik_cikarici import OzellikCikarici

from ayarlar import MODELS_KLASORU


class ModelInference:
    """Eğitilmiş model ile tahmin yapma sınıfı."""
    
    def __init__(self, model_yolu: Union[str, Path]):
        """
        Inference nesnesini başlat.
        
        Args:
            model_yolu: Eğitilmiş model dosyasının yolu (.pkl)
        """
        self.model_yolu = Path(model_yolu)
        
        if not self.model_yolu.exists():
            # MODELS_KLASORU içinde ara
            alternatif = MODELS_KLASORU / self.model_yolu.name
            if alternatif.exists():
                self.model_yolu = alternatif
            else:
                raise FileNotFoundError(f"Model bulunamadı: {model_yolu}")
        
        # Modeli yükle
        self._model_yukle()
        
        # Görüntü işleyicileri
        self.isleyici = GorselIsleyici()
        self.cikarici = OzellikCikarici()
        
        # Sınıf isimleri
        self.sinif_isimleri = {
            0: "NonDemented (Sağlıklı)",
            1: "VeryMildDemented (Çok Hafif Demans)",
            2: "MildDemented (Hafif Demans)",
            3: "ModerateDemented (Orta Seviye Demans)"
        }
    
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
            
            # Metrikler varsa göster
            if 'metrikler' in self.metadata:
                metriks = self.metadata['metrikler']
                print(f"   ℹ️  Test Accuracy: {metriks.get('accuracy', 'N/A'):.4f}")
        else:
            self.metadata = {}
            print(f"   ⚠️  Metadata bulunamadı")
    
    def goruntu_isle(self, goruntu_yolu: Union[str, Path]) -> np.ndarray:
        """
        Görüntüyü işle ve model için hazırla.
        
        Args:
            goruntu_yolu: Ham görüntü dosyasının yolu
            
        Returns:
            İşlenmiş görüntü
        """
        goruntu = self.isleyici.goruntu_yukle(str(goruntu_yolu))
        if goruntu is None:
            raise ValueError(f"Görüntü yüklenemedi: {goruntu_yolu}")
        
        # Tam işleme pipeline'ı
        goruntu = self.isleyici.gurultu_gider(goruntu)
        goruntu = self.isleyici.bias_field_correction(goruntu)
        goruntu = self.isleyici.skull_strip(goruntu)
        goruntu = self.isleyici.center_of_mass_alignment(goruntu)
        goruntu = self.isleyici.yogunluk_normalize(goruntu)
        goruntu = self.isleyici.histogram_esitle(goruntu, adaptive=True)
        goruntu = self.isleyici.boyutlandir(goruntu)
        
        return goruntu
    
    def ozellik_cikar(self, goruntu_yolu: Union[str, Path]) -> pd.DataFrame:
        """
        Görüntüden özellikleri çıkar.
        
        Args:
            goruntu_yolu: Görüntü dosyasının yolu
            
        Returns:
            Özellikler DataFrame'i
        """
        ozellikler = self.cikarici.tek_goruntu_ozellikleri(str(goruntu_yolu))
        
        if ozellikler is None:
            raise ValueError(f"Özellik çıkarılamadı: {goruntu_yolu}")
        
        # DataFrame'e çevir
        df = pd.DataFrame([ozellikler])
        
        # Kategorik kolonları çıkar
        kategorik = ['dosya_adi', 'tam_yol']
        df_ozellikler = df.drop(columns=[c for c in kategorik if c in df.columns])
        
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
        
        print(f"\n📁 Batch tahmin: {len(gorseller)} görüntü")
        print(f"{'='*60}")
        
        sonuclar = []
        
        for i, goruntu_yolu in enumerate(gorseller, 1):
            print(f"\n[{i}/{len(gorseller)}] İşleniyor: {goruntu_yolu.name}")
            
            try:
                sonuc = self.tahmin_yap(goruntu_yolu, detayli=False)
                sonuclar.append(sonuc)
                print(f"   ✓ {sonuc['tahmin_adi']}")
            except Exception as e:
                print(f"   ✗ HATA: {e}")
                sonuclar.append({
                    'dosya': str(goruntu_yolu),
                    'hata': str(e)
                })
        
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
