"""
veri_normalizasyon.py
---------------------
Veri normalizasyon ve ölçekleme fonksiyonları.

Fonksiyonlar:
  - minmax_scaler_uygula(): Min-Max ölçekleme (0-1 aralığı)
  - robust_scaler_uygula(): Robust ölçekleme (istatistiksel aykırı değerlere dayanıklı)
  - StandardScaler vb için sınıflar ve yardımcı fonksiyonlar
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union


class MinMaxScaler:
    """
    Min-Max ölçekleme sınıfı.
    Verileri [0, 1] aralığına normalize eder.
    
    Formula: X_scaled = (X - X_min) / (X_max - X_min)
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        Parametreler:
        - feature_range: Ölçeklemenin yapılacağı aralık (min, max)
        """
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'MinMaxScaler':
        """
        Min ve max değerlerini hesapla.
        
        Parametreler:
        - X: Input veri (numpy array veya pandas DataFrame)
        
        Döndürülen: self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / (self.data_max_ - self.data_min_ + 1e-8)
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Veriyi ölçekle.
        
        Parametreler:
        - X: Input veri
        
        Döndürülen: Ölçeklenmiş veri (aynı type)
        """
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X.values * self.scale_ + self.min_,
                columns=X.columns,
                index=X.index
            )
        else:
            return X * self.scale_ + self.min_
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit et ve transform et (bir adımda).
        
        Parametreler:
        - X: Input veri
        
        Döndürülen: Ölçeklenmiş veri
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Ölçeklenmiş veriyi orijinal aralığına geri dönüştür.
        
        Parametreler:
        - X_scaled: Ölçeklenmiş veri
        
        Döndürülen: Orijinal aralıkta veri
        """
        if isinstance(X_scaled, pd.DataFrame):
            return pd.DataFrame(
                (X_scaled.values - self.min_) / (self.scale_ + 1e-8),
                columns=X_scaled.columns,
                index=X_scaled.index
            )
        else:
            return (X_scaled - self.min_) / (self.scale_ + 1e-8)


class RobustScaler:
    """
    Robust ölçekleme sınıfı (istatistiksel aykırı değerlere dayanıklı).
    Medyan ve interquartile range (IQR) kullanır.
    
    Formula: X_scaled = (X - median) / IQR
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Parametreler:
        - quantile_range: Çeyreklik aralığı (q1, q3)
        """
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'RobustScaler':
        """
        Medyan ve IQR'yi hesapla.
        
        Parametreler:
        - X: Input veri
        
        Döndürülen: self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        q1, q3 = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0)
        self.scale_ = np.nanpercentile(X, q3, axis=0) - np.nanpercentile(X, q1, axis=0)
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Veriyi ölçekle.
        
        Parametreler:
        - X: Input veri
        
        Döndürülen: Ölçeklenmiş veri
        """
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                (X.values - self.center_) / (self.scale_ + 1e-8),
                columns=X.columns,
                index=X.index
            )
        else:
            return (X - self.center_) / (self.scale_ + 1e-8)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit et ve transform et.
        
        Parametreler:
        - X: Input veri
        
        Döndürülen: Ölçeklenmiş veri
        """
        return self.fit(X).transform(X)


def csv_dosyasina_minmax_scaling_uygula(
    csv_dosya_yolu: str,
    ozellik_sutunlari: Optional[List[str]] = None,
    hariç_tutulacak_sutunlar: Optional[List[str]] = None,
    cikti_dosya_adi: str = "goruntu_ozellikleri_scaled.csv"
) -> Tuple[str, MinMaxScaler]:
    """
    CSV dosyasındaki sayısal özelliklere Min-Max scaling uygula.
    
    Parametreler:
    - csv_dosya_yolu: İnput CSV dosyasının yolu
    - ozellik_sutunlari: Ölçeklenecek sütunlar (None ise tüm sayısal sütunlar)
    - hariç_tutulacak_sutunlar: Ölçeklenmeyecek sütunlar (örn: sinif, etiket, dosya_adı)
    - cikti_dosya_adi: Ölçeklenmiş CSV dosyasının adı
    
    Döndürülen: (Ölçeklenmiş CSV dosyasının yolu, Scaler nesnesi)
    """
    import os
    
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
        # Sayısal sütunları otomatik olarak seç
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ozellik_sutunlari = [
            col for col in numeric_cols 
            if col not in hariç_tutulacak_sutunlar
        ]
    
    print(f"[BILGI] Ölçeklenecek sütunlar: {len(ozellik_sutunlari)}")
    print(f"        {ozellik_sutunlari}")
    
    # Min-Max Scaler oluştur ve uygula
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Yalnızca seçilen sütunları ölçekle
    X = df[ozellik_sutunlari].copy()
    X_scaled = scaler.fit_transform(X)
    
    # Orijinal DataFrame'e ölçeklenmiş değerleri ekle
    df_scaled = df.copy()
    df_scaled[ozellik_sutunlari] = X_scaled
    
    # Ölçeklenmiş CSV'yi kaydet
    output_dir = os.path.dirname(csv_dosya_yolu)
    cikti_yolu = os.path.join(output_dir, cikti_dosya_adi)
    
    df_scaled.to_csv(cikti_yolu, index=False, encoding='utf-8')
    
    print(f"[TAMAMLANDI] Ölçeklenmiş CSV kaydedildi: {cikti_yolu}")
    
    # Ölçekleme istatistiklerini göster
    print("\n[ÖLÇEKLEME İSTATİSTİKLERİ]")
    for i, col in enumerate(ozellik_sutunlari):
        min_val = scaler.data_min_[i]
        max_val = scaler.data_max_[i]
        print(f"  {col}: [{min_val:.4f}, {max_val:.4f}] → [0, 1]")
    
    return cikti_yolu, scaler


def csv_dosyasina_robust_scaling_uygula(
    csv_dosya_yolu: str,
    ozellik_sutunlari: Optional[List[str]] = None,
    hariç_tutulacak_sutunlar: Optional[List[str]] = None,
    cikti_dosya_adi: str = "goruntu_ozellikleri_robust_scaled.csv"
) -> Tuple[str, RobustScaler]:
    """
    CSV dosyasındaki sayısal özelliklere Robust scaling uygula.
    Aykırı değerlere karşı daha dayanıklıdır.
    
    Parametreler:
    - csv_dosya_yolu: İnput CSV dosyasının yolu
    - ozellik_sutunlari: Ölçeklenecek sütunlar
    - hariç_tutulacak_sutunlar: Ölçeklenmeyecek sütunlar
    - cikti_dosya_adi: Ölçeklenmiş CSV dosyasının adı
    
    Döndürülen: (Ölçeklenmiş CSV dosyasının yolu, Scaler nesnesi)
    """
    import os
    
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
    
    print(f"[BILGI] Ölçeklenecek sütunlar: {len(ozellik_sutunlari)}")
    
    # Robust Scaler oluştur ve uygula
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    
    X = df[ozellik_sutunlari].copy()
    X_scaled = scaler.fit_transform(X)
    
    # Orijinal DataFrame'e ölçeklenmiş değerleri ekle
    df_scaled = df.copy()
    df_scaled[ozellik_sutunlari] = X_scaled
    
    # Ölçeklenmiş CSV'yi kaydet
    output_dir = os.path.dirname(csv_dosya_yolu)
    cikti_yolu = os.path.join(output_dir, cikti_dosya_adi)
    
    df_scaled.to_csv(cikti_yolu, index=False, encoding='utf-8')
    
    print(f"[TAMAMLANDI] Robust ölçeklenmiş CSV kaydedildi: {cikti_yolu}")
    
    return cikti_yolu, scaler
