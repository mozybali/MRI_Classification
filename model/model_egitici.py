#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model_egitici.py
----------------
Makine öğrenmesi modelleri için birleştirilmiş eğitim ve değerlendirme modülü.

Bu dosya, MRI sınıflandırma projesi için 3 farklı model seçeneği sunar:
1. XGBoost (Gradient Boosting) - Önerilen, yüksek performans
2. LightGBM (Gradient Boosting) - Hızlı, büyük veri setleri için
3. Linear SVM - Basit, hızlı eğitim

MODEL SEÇİMİ:
- Model tipi, program başlatıldığında kullanıcıya sorulur
- Veya kod içinde ModelEgitici(model_tipi="xgboost") ile belirlenebilir
- Ayarlar model/ayarlar.py dosyasında yapılandırılır
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
if not hasattr(np, "int"):  # NumPy 1.24+ uyumluluğu: np.int kaldırıldı
    np.int = int  # type: ignore[attr-defined]
import pandas as pd
import pickle
import json
import inspect
import warnings
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# Scikit-learn modülleri
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Imbalanced-learn (SMOTE için)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError as e:
    SMOTE_AVAILABLE = False
    print("[UYARI] imbalanced-learn yüklenemedi (paket eksik veya scikit-learn uyumsuz). SMOTE kullanılamayacak.")
    print(f"        Detay: {e}")

import matplotlib.pyplot as plt
import seaborn as sns

from ayarlar import *

# LightGBM loglarını tamamen kısmak için global çevre değişkeni
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")


def _lightgbm_silent():
    """LightGBM logger'ını tamamen kapat (fallback ile)."""
    try:
        import lightgbm as lgb
        try:
            lgb.register_logger(lambda msg: None)
        except Exception:
            pass
    except ImportError:
        pass

def _lightgbm_log_period(default: int = 0) -> int:
    """LightGBM log periyodunu ayarlardan al, hata olursa varsayılanda bırak."""
    try:
        return max(0, int(LIGHTGBM_LOG_PERIOD))
    except Exception:
        return default

# Ensure VERI_CSV is imported
try:
    from ayarlar import VERI_CSV
except ImportError:
    VERI_CSV = Path("goruntu_isleme/cikti/goruntu_ozellikleri_scaled.csv")


class ModelEgitici:
    """
    Tüm model eğitim ve değerlendirme işlemleri için birleşik sınıf.
    
    Bu sınıf, 3 farklı makine öğrenmesi algoritmasını destekler:
    - XGBoost: Gradient boosting, yüksek performans
    - LightGBM: Gradient boosting, hızlı eğitim
    - Linear SVM: Doğrusal sınıflandırıcı, basit ve hızlı
    """
    
    def __init__(self, model_tipi: str = "xgboost", 
                 smote_aktif: bool = True,
                 feature_selection_aktif: bool = False):
        """
        Model eğiticiyi başlat.
        
        Args:
            model_tipi: Eğitilecek model türü
                - "xgboost": XGBoost gradient boosting (önerilen)
                - "lightgbm": LightGBM gradient boosting (hızlı)
                - "svm": Linear Support Vector Machine (basit)
            smote_aktif: SMOTE ile veri dengeleme yapılsın mı?
            feature_selection_aktif: Özellik seçimi yapılsın mı?
        """
        self.model_tipi = model_tipi
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.metrikler = {}
        self.smote_aktif = smote_aktif and SMOTE_AVAILABLE
        self.feature_selection_aktif = feature_selection_aktif
        self.cv_scores = None
        
        # Çıktı klasörlerini oluştur (her koşu için ayrı)
        print(f"\n📁 Çıktı klasörleri oluşturuluyor...")
        CIKTI_KLASORU.mkdir(parents=True, exist_ok=True)
        self.run_zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cikti_klasoru = CIKTI_KLASORU / f"{self.model_tipi}_{self.run_zaman_damgasi}"
        self.modeller_klasoru = self.cikti_klasoru / "modeller"
        self.raporlar_klasoru = self.cikti_klasoru / "raporlar"
        self.gorseller_klasoru = self.cikti_klasoru / "gorseller"
        for klasor in [self.cikti_klasoru, self.modeller_klasoru, self.raporlar_klasoru, self.gorseller_klasoru]:
            klasor.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Klasörler hazır: {self.cikti_klasoru}")
    
    def veri_yukle(self, csv_yolu: Path = VERI_CSV) -> Tuple:
        """
        CSV dosyasından veri yükle ve eğitim/doğrulama/test setlerine böl.
        
        Args:
            csv_yolu: Özellik CSV dosyasının yolu
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n📊 Veri yükleniyor: {csv_yolu}")
        
        if not csv_yolu.exists():
            raise FileNotFoundError(
                f"❌ CSV dosyası bulunamadı: {csv_yolu}\n"
                f"   Önce 'goruntu_isleme/ana_islem.py' çalıştırarak CSV oluşturun!"
            )
        
        # CSV'yi oku
        df = pd.read_csv(csv_yolu)
        print(f"   ✓ {len(df)} kayıt yüklendi")
        print(f"   ✓ {df['sinif'].nunique()} sınıf var: {df['sinif'].unique().tolist()}")
        
        # Özellikler ve etiketler
        kategorik = ['dosya_adi', 'sinif', 'tam_yol']
        X = df.drop(columns=[c for c in kategorik if c in df.columns] + ['etiket'])
        y = df['etiket']
        
        self.feature_names = X.columns.tolist()
        print(f"   ✓ {len(self.feature_names)} özellik kullanılacak")
        
        # İlk bölme: eğitim + geçici (doğrulama + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1 - EGITIM_ORANI),
            random_state=RASTGELE_TOHUM,
            stratify=y if STRATIFY_AKTIF else None
        )
        
        # İkinci bölme: doğrulama + test
        val_oran = DOGRULAMA_ORANI / (DOGRULAMA_ORANI + TEST_ORANI)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_oran),
            random_state=RASTGELE_TOHUM,
            stratify=y_temp if STRATIFY_AKTIF else None
        )
        
        print(f"\n📂 Veri seti bölündü:")
        print(f"   • Eğitim: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"   • Doğrulama: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"   • Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
        
        # Sınıf dağılımını göster
        print(f"\n📊 Sınıf dağılımı (Eğitim seti):")
        for sinif, sayi in zip(*np.unique(y_train, return_counts=True)):
            print(f"   Sınıf {sinif}: {sayi} ({sayi/len(y_train)*100:.1f}%)")
        
        # SMOTE uygula (veri dengeleme)
        if self.smote_aktif:
            print(f"\n🔄 SMOTE ile veri dengeleme yapılıyor...")
            smote = SMOTE(random_state=RASTGELE_TOHUM, k_neighbors=3)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"   ✓ SMOTE tamamlandı. Yeni eğitim seti: {len(X_train)} kayıt")
            print(f"\n📊 Dengeli sınıf dağılımı:")
            for sinif, sayi in zip(*np.unique(y_train, return_counts=True)):
                print(f"   Sınıf {sinif}: {sayi} ({sayi/len(y_train)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def feature_selection(self, X_train, y_train, k: int = 15):
        """
        En önemli k özelliği seç (mutual information kullanarak).
        
        Args:
            X_train: Eğitim özellikleri (pandas DataFrame)
            y_train: Eğitim etiketleri
            k: Seçilecek özellik sayısı
            
        Returns:
            Seçilmiş özelliklerle pandas DataFrame
        """
        if not self.feature_selection_aktif:
            return X_train
        
        print(f"\n🔍 Feature Selection: En iyi {k} özellik seçiliyor...")
        
        # Mutual information ile özellik skorları hesapla
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_train, y_train)
        
        # Seçilen özellikleri kaydet
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"   ✓ {len(self.selected_features)} özellik seçildi:")
        scores = selector.scores_[selected_indices]
        for feat, score in sorted(zip(self.selected_features, scores), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"      • {feat}: {score:.4f}")
        
        # DataFrame olarak döndür (feature isimlerini korumak için)
        return X_train[self.selected_features]
    
    def model_olustur(self):
        """
        Seçilen model tipine göre ML modeli oluştur.
        
        3 farklı model tipi desteklenir:
        
        1. XGBoost (xgboost):
           - Gradient boosting ağaçları
           - Yüksek doğruluk, orta hız
           - Hiperparametre: n_estimators, max_depth, learning_rate, vb.
           - class_weight: Otomatik sınıf dengeleme
        
        2. LightGBM (lightgbm):
           - Microsoft'un gradient boosting implementasyonu
           - Çok hızlı eğitim, büyük veri setleri için ideal
           - Histogram tabanlı, bellek verimli
           - class_weight: Otomatik sınıf dengeleme
        
        3. Linear SVM (svm):
           - Doğrusal Support Vector Machine
           - Basit, hızlı eğitim
           - class_weight: 'balanced' - otomatik ağırlıklandırma
        
        Returns:
            Eğitilmemiş model nesnesi (self.model'e atanır)
        """
        print(f"\n🤖 Model oluşturuluyor: {self.model_tipi.upper()}")
        
        if self.model_tipi == "xgboost":
            try:
                import xgboost as xgb
                # XGBoost modelini oluştur
                self.model = xgb.XGBClassifier(**GB_AYARLARI)
                print(f"   ✓ XGBoost modeli hazır")
                print(f"   ℹ️  n_estimators={GB_AYARLARI['n_estimators']}, max_depth={GB_AYARLARI['max_depth']}")
            except ImportError:
                raise ImportError(
                    "❌ XGBoost yüklü değil!\n"
                    "   Kurulum: pip install xgboost"
                )
        
        elif self.model_tipi == "lightgbm":
            try:
                import lightgbm as lgb
                _lightgbm_silent()
                # LightGBM parametrelerini ayarlardan al
                lgb_params = LIGHTGBM_AYARLARI.copy()
                if 'max_depth' in lgb_params and 'num_leaves' not in lgb_params:
                    lgb_params['num_leaves'] = 2 ** lgb_params['max_depth']
                lgb_params['class_weight'] = 'balanced'  # Sınıf dengeleme
                lgb_params['verbose'] = -1  # Uyarıları sustur
                lgb_params['verbosity'] = -1  # Yeni sürümlerde de sessiz mod
                lgb_params['force_col_wise'] = True  # Log spamini ve auto-seçim mesajlarını önle
                # Tüm fit'lerde logları sustur (cross_validate klonlarında da çalışır)
                try:
                    import lightgbm as lgb
                    log_period = _lightgbm_log_period()
                    lgb_params['callbacks'] = [lgb.log_evaluation(period=log_period)]
                except Exception:
                    pass
                lgb_params.pop('max_depth', None)
                
                self.model = lgb.LGBMClassifier(**lgb_params)
                print(f"   ✓ LightGBM modeli hazır")
                print(f"   ℹ️  n_estimators={lgb_params['n_estimators']}, num_leaves={lgb_params['num_leaves']}")
            except ImportError:
                raise ImportError(
                    "❌ LightGBM yüklü değil!\n"
                    "   Kurulum: pip install lightgbm"
                )
        
        elif self.model_tipi == "svm":
            from sklearn.svm import LinearSVC
            # Linear SVM modelini oluştur
            self.model = LinearSVC(**SVM_AYARLARI)
            print(f"   ✓ Linear SVM modeli hazır")
            print(f"   ℹ️  C={SVM_AYARLARI['C']}, class_weight={SVM_AYARLARI['class_weight']}")
        
        else:
            raise ValueError(
                f"❌ Desteklenmeyen model tipi: {self.model_tipi}\n"
                f"   Geçerli seçenekler: 'xgboost', 'lightgbm', 'svm'"
            )
    
    def egit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Modeli eğit.
        
        Bu fonksiyon, hazırlanan veri seti ile makine öğrenmesi modelini eğitir.
        
        Eğitim süreci:
        1. Model oluşturulur (henüz oluşturulmadıysa)
        2. Early stopping için doğrulama seti kontrol edilir
        3. Model fit() metodu ile eğitilir
        4. Eğitim süresi ölçülür
        
        Early Stopping (XGBoost/LightGBM için):
        - Doğrulama seti verilirse early stopping aktif olur
        - Model, doğrulama kaybı artmayı bırakınca eğitimi durdurur
        - Overfitting'i önler ve eğitim süresini kısaltır
        
        Args:
            X_train: Eğitim özellikleri (features) - numpy array veya pandas DataFrame
            y_train: Eğitim etiketleri (labels) - numpy array veya pandas Series
            X_val: Doğrulama özellikleri (opsiyonel, early stopping için gerekli)
            y_val: Doğrulama etiketleri (opsiyonel, early stopping için gerekli)
            
        Returns:
            None (model self.model'de saklanır)
        """
        print(f"\n{'='*60}")
        print(f"🎯 MODEL EĞİTİMİ BAŞLIYOR: {self.model_tipi.upper()}")
        print(f"{'='*60}")
        
        if self.model is None:
            self.model_olustur()
        
        # Eğitim başlat
        print(f"\n⏳ Eğitim devam ediyor...")
        
        if self.model_tipi in ["xgboost", "lightgbm"] and X_val is not None:
            # Gradient boosting için early stopping kullan
            if self.model_tipi == "xgboost":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS
                )
            else:  # lightgbm
                fit_kwargs = {'eval_set': [(X_val, y_val)]}

                # LightGBM 4.x ile early_stopping_rounds fit() imzasından kaldırıldı.
                # Callback tabanlı erken durdurma ekleyip imzaya göre parametre ekle.
                try:
                    import lightgbm as lgb
                    log_period = _lightgbm_log_period()
                    callbacks = [lgb.log_evaluation(period=log_period)]
                    if hasattr(lgb, "early_stopping"):
                        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
                    if callbacks:
                        fit_kwargs["callbacks"] = callbacks
                except Exception:
                    pass

                try:
                    fit_sig = inspect.signature(self.model.fit).parameters
                except (TypeError, ValueError):
                    fit_sig = {}

                if "verbose" in fit_sig:
                    fit_kwargs["verbose"] = -1
                if "early_stopping_rounds" in fit_sig:
                    fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

                self.model.fit(X_train, y_train, **fit_kwargs)
            print(f"   ✓ Early stopping ile eğitim tamamlandı")
        else:
            self.model.fit(X_train, y_train)
            print(f"   ✓ Eğitim tamamlandı")
    
    def tahmin_yap(self, X):
        """Tahmin yap."""
        if self.model is None:
            raise ValueError("❌ Model henüz eğitilmemiş!")
        return self.model.predict(X)
    
    def degerlendir(self, X, y, set_adi: str = "Test") -> Dict:
        """
        Model performansını değerlendir.
        
        Args:
            X: Özellikler
            y: Gerçek etiketler
            set_adi: Veri seti adı (Test, Doğrulama, vb.)
            
        Returns:
            Metrikler sözlüğü
        """
        y_pred = self.tahmin_yap(X)
        
        metrikler = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, zero_division=0),
        }
        
        # Cohen's Kappa (sınıf dengesizliğine robust)
        metrikler['cohen_kappa'] = cohen_kappa_score(y, y_pred)
        
        # ROC-AUC (multi-class için one-vs-rest)
        if hasattr(self.model, 'predict_proba'):
            try:
                y_prob = self.model.predict_proba(X)
                metrikler['roc_auc_ovr'] = roc_auc_score(
                    y, y_prob, 
                    multi_class='ovr', 
                    average='macro'
                )
            except Exception as e:
                metrikler['roc_auc_ovr'] = None
                print(f"   [UYARI] ROC-AUC hesaplanamadı: {e}")
        else:
            metrikler['roc_auc_ovr'] = None
        
        # Ekrana yazdır
        print(f"\n{'='*60}")
        print(f"📊 {set_adi.upper()} SETİ PERFORMANSI")
        print(f"{'='*60}")
        print(f"✓ Doğruluk (Accuracy):    {metrikler['accuracy']:.4f}")
        print(f"✓ Kesinlik (Precision):   {metrikler['precision_macro']:.4f}")
        print(f"✓ Duyarlılık (Recall):    {metrikler['recall_macro']:.4f}")
        print(f"✓ F1 Skoru:               {metrikler['f1_macro']:.4f}")
        print(f"✓ Cohen's Kappa:          {metrikler['cohen_kappa']:.4f}")
        if metrikler['roc_auc_ovr'] is not None:
            print(f"✓ ROC-AUC (OvR):          {metrikler['roc_auc_ovr']:.4f}")
        print(f"\n📋 Detaylı Rapor:\n{metrikler['classification_report']}")
        
        return metrikler
    
    def cross_validate(self, X, y, cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        K-fold cross-validation ile model performansını değerlendir.
        
        Args:
            X: Özellikler
            y: Etiketler
            cv_folds: Cross-validation fold sayısı
            
        Returns:
            Her fold için skorlar sözlüğü
        """
        print(f"\n🔄 {cv_folds}-Fold Cross-Validation yapılıyor...")
        
        if self.model is None:
            self.model_olustur()
        
        # Cross-validation için early stopping olmadan model oluştur
        # (çünkü cross_val_score eval_set desteklemiyor)
        if self.model_tipi == "xgboost":
            import xgboost as xgb
            # Early stopping parametrelerini kaldır
            cv_params = GB_AYARLARI.copy()
            cv_params.pop('early_stopping_rounds', None)
            cv_params.pop('callbacks', None)
            cv_model = xgb.XGBClassifier(**cv_params)
        elif self.model_tipi == "lightgbm":
            import lightgbm as lgb
            _lightgbm_silent()
            # LightGBM için uygun parametreleri ayarla
            cv_params = LIGHTGBM_AYARLARI.copy()
            if 'max_depth' in cv_params and 'num_leaves' not in cv_params:
                cv_params['num_leaves'] = 2 ** cv_params.get('max_depth', 7)
            cv_params['class_weight'] = 'balanced'
            cv_params['verbose'] = -1
            cv_params['verbosity'] = -1
            cv_params['force_col_wise'] = True
            try:
                import lightgbm as lgb
                log_period = _lightgbm_log_period()
                cv_params['callbacks'] = [lgb.log_evaluation(period=log_period)]
            except Exception:
                pass
            cv_params.pop('max_depth', None)
            cv_model = lgb.LGBMClassifier(**cv_params)
        else:
            # SVM veya diğer modeller için mevcut modeli kullan
            cv_model = self.model
        
        # Stratified K-Fold (sınıf oranlarını korur)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RASTGELE_TOHUM)
        
        # Farklı metrikleri tek seferde hesapla (daha az tekrar eğitim)
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
        cv_raw = cross_validate(
            cv_model,
            X,
            y,
            cv=skf,
            scoring=scoring_metrics,
            n_jobs=-1,
            return_train_score=False
        )
        cv_results = {metric: cv_raw[f"test_{metric}"] for metric in scoring_metrics}
        
        for metric, scores in cv_results.items():
            print(f"   {metric:20s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        self.cv_scores = cv_results
        return cv_results
    
    def grid_search(self, X_train, y_train, n_iter: int = 50, search_method: str = "random") -> Dict:
        """
        Geriye donuk uyumluluk icin hyperparameter_tuning kisayolu.
        
        Args:
            X_train: Egitim ozellikleri
            y_train: Egitim etiketleri
            n_iter: RandomizedSearch iterasyon sayisi
            search_method: "random" veya "bayes"
            
        Returns:
            En iyi parametreler
        """
        return self.hyperparameter_tuning(X_train, y_train, n_iter=n_iter, search_method=search_method)
    
    def hyperparameter_tuning(self, X_train, y_train, n_iter: int = 50, search_method: str = "random") -> Dict:
        """
        Hyperparameter tuning ile en iyi parametreleri bul.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            n_iter: Arama iterasyon sayısı
            search_method: "random" (RandomizedSearchCV) veya "bayes" (BayesSearchCV)
            
        Returns:
            En iyi parametreler
        """
        search_method = (search_method or "random").lower()
        print(f"\n🔧 Hyperparameter Tuning başlıyor ({search_method} search, {n_iter} iterasyon)...")
        print(f"   Bu işlem birkaç dakika sürebilir...\n")
        
        # Model tipine göre parametre grid'i belirle
        if self.model_tipi == "xgboost":
            import xgboost as xgb
            param_distributions = {
                'n_estimators': GB_GRID_PARAMS.get('n_estimators', [200, 400, 600, 800]),
                'max_depth': GB_GRID_PARAMS.get('max_depth', [3, 5, 7, 9, 11]),
                'learning_rate': GB_GRID_PARAMS.get('learning_rate', [0.01, 0.03, 0.05, 0.08]),
                'subsample': GB_GRID_PARAMS.get('subsample', [0.6, 0.8, 1.0]),
                'colsample_bytree': GB_GRID_PARAMS.get('colsample_bytree', [0.6, 0.8, 1.0]),
                'gamma': [0, 0.1, 0.2, 0.5],
                'min_child_weight': [1, 3, 5, 7],
            }
            base_model = xgb.XGBClassifier(random_state=RASTGELE_TOHUM)
            
        elif self.model_tipi == "lightgbm":
            import lightgbm as lgb
            param_distributions = {
                # Daha düsük lr + daha fazla iterasyon kombinasyonlarını dene
                'n_estimators': [400, 800, 1200],
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.02, 0.03, 0.05],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [10, 20, 40],
            }
            base_model = lgb.LGBMClassifier(
                random_state=RASTGELE_TOHUM,
                class_weight='balanced'
            )
            
        elif self.model_tipi == "svm":
            from sklearn.svm import LinearSVC
            param_distributions = {
                'C': SVM_GRID_PARAMS.get('C', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                'loss': SVM_GRID_PARAMS.get('loss', ['squared_hinge']),
                'dual': SVM_GRID_PARAMS.get('dual', [False, True]),
                'max_iter': SVM_GRID_PARAMS.get('max_iter', [5000, 20000, 50000, 200000]),
                'tol': SVM_GRID_PARAMS.get('tol', [1e-2, 5e-3, 1e-3]),
            }
            base_model = LinearSVC(random_state=RASTGELE_TOHUM, class_weight='balanced')
        
        # Toplam kombinasyon sayısını hesapla ve gereksiz uyarıları engellemek için n_iter'ı uyumlu hale getir
        n_iter_effective = n_iter
        try:
            combo_sayisi = 1
            for values in param_distributions.values():
                combo_sayisi *= len(values)
        except TypeError:
            combo_sayisi = None
        
        if combo_sayisi is not None and n_iter_effective > combo_sayisi:
            print(f"   [UYARI] Parametre kombinasyonları {combo_sayisi} ile sınırlı. n_iter {combo_sayisi} olarak güncellendi.")
            n_iter_effective = combo_sayisi
        
        use_bayes = search_method == "bayes"
        if use_bayes:
            try:
                from skopt import BayesSearchCV
                from skopt.space import Categorical
            except ImportError:
                print("   [UYARI] BayesSearchCV bulunamadı, RandomizedSearchCV ile devam ediliyor.")
                use_bayes = False
        
        if use_bayes:
            search_spaces = {}
            for param, values in param_distributions.items():
                if isinstance(values, (list, tuple)):
                    search_spaces[param] = Categorical(list(values))
                else:
                    search_spaces[param] = values

            # BayesSearchCV varsayilan n_initial_points=10 degerini,
            # toplam iterasyon sayisini asmamasi icin kisalt.
            bayes_init = min(10, max(1, n_iter_effective))
            optimizer_kwargs = {
                "random_state": RASTGELE_TOHUM,
                "n_initial_points": bayes_init,
            }

            searcher = BayesSearchCV(
                estimator=base_model,
                search_spaces=search_spaces,
                n_iter=n_iter_effective,
                cv=GRID_SEARCH_AYARLARI.get('cv_folds', 5),
                scoring='f1_macro',
                n_jobs=GRID_SEARCH_AYARLARI.get('n_jobs', -1),
                random_state=RASTGELE_TOHUM,
                verbose=GRID_SEARCH_AYARLARI.get('verbose', 1),
                optimizer_kwargs=optimizer_kwargs,
            )
        else:
            searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter_effective,
                cv=GRID_SEARCH_AYARLARI.get('cv_folds', 5),
                scoring='f1_macro',
                n_jobs=GRID_SEARCH_AYARLARI.get('n_jobs', -1),
                random_state=RASTGELE_TOHUM,
                verbose=GRID_SEARCH_AYARLARI.get('verbose', 1)
            )

        # skopt, ayni nokta tekrar geldiyse uyarip rastgele nokta seciyor.
        # Bayes arama kullanirken bu uyarilari gürültüden kaçınmak için gizliyoruz.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The objective has been evaluated at point .* before, using random point .*",
                category=UserWarning,
                module="skopt"
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The objective has been evaluated at this point before.*",
                category=UserWarning,
                module="skopt"
            )
            searcher.fit(X_train, y_train)
        
        print(f"\n✓ Hyperparameter tuning tamamlandı!")
        print(f"\n🏆 En iyi parametreler:")
        for param, value in searcher.best_params_.items():
            print(f"   {param}: {value}")
        print(f"\n📈 En iyi CV skoru: {searcher.best_score_:.4f}")
        
        # En iyi modeli kullan
        self.model = searcher.best_estimator_
        
        return searcher.best_params_
    
    def confusion_matrix_ciz(self, y_true, y_pred, dosya_adi: str = "confusion_matrix.png"):
        """Karışıklık matrisi çizer ve kaydeder."""
        # Sınıf etiketlerini aynı sıra ile göster (görselde isim kullanmak için eşleme yap)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Sayısal etiketleri okunabilir isimlere çevir (gelen veri string ise olduğu gibi kullanılır)
        label_map = {
            0: "NonDemented",
            1: "VeryMildDemented",
            2: "MildDemented",
            3: "ModerateDemented",
        }

        def _etiket_isim(c):
            try:
                c_int = int(c)
                if isinstance(c, (int, np.integer)) or c_int == c:
                    return label_map.get(c_int, str(c))
            except Exception:
                pass
            return label_map.get(c, str(c))

        class_names = [_etiket_isim(c) for c in classes]

        toplam_ornek = int(cm.sum())

        # Hücreleri sade bir şekilde sayım + satır yüzdesi olacak şekilde hazırla
        satir_toplamlari = cm.sum(axis=1, keepdims=True).astype(np.float64)
        annot_labels = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                adet = cm[i, j]
                if satir_toplamlari[i, 0] > 0:
                    yuzde = adet / satir_toplamlari[i, 0] * 100
                    row.append(f"{adet}\n{yuzde:4.1f}%")
                else:
                    row.append(str(adet))
            annot_labels.append(row)

        fig, ax = plt.subplots(figsize=GORSEL_AYARLARI['confusion_matrix_figsize'])
        fig.subplots_adjust(bottom=0.25)
        sns.heatmap(
            cm,
            annot=annot_labels,
            fmt="",
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Sayı'},
            annot_kws={'fontsize': 10}
        )
        ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
        ax.set_ylabel('Gerçek Sınıf', fontsize=12)
        ax.set_title(f'Karışıklık Matrisi - {self.model_tipi.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticklabels(class_names, rotation=30, ha='right')
        ax.set_yticklabels(class_names, rotation=0)

        aciklama = (
            f"Hücredeki üst satır: örnek adedi (toplam: {toplam_ornek}). "
            "Alt satır: satıra göre yüzde. Satır = gerçek sınıf, sütun = tahmin edilen sınıf."
        )
        fig.text(0.5, 0.05, aciklama, va='center', ha='center', fontsize=10, wrap=True)

        kayit_yolu = self.gorseller_klasoru / dosya_adi
        fig.savefig(kayit_yolu, dpi=GORSEL_AYARLARI['dpi'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ Karışıklık matrisi kaydedildi: {kayit_yolu}")

    def ozellik_onemi_ciz(self, top_n: int = 20):
        """Özellik önemini çiz (gradient boosting için)."""
        if self.model_tipi not in ["xgboost", "lightgbm"]:
            print(f"   ⚠️  Özellik önemi sadece gradient boosting modellerde desteklenir")
            return
        
        if not hasattr(self.model, 'feature_importances_'):
            print(f"   ⚠️  Model özellik önemini desteklemiyor")
            return
        
        importances = self.model.feature_importances_
        # Grafik boyutunu eldeki gerçek özellik sayısına göre ayarla
        top_n = min(top_n, len(importances))
        
        # Feature selection yapıldıysa seçilmiş özellikleri, yoksa tüm özellikleri kullan
        feature_list = self.selected_features if self.selected_features else self.feature_names
        if feature_list is None:
            feature_list = []
        if len(feature_list) < len(importances):
            feature_list = feature_list + [f"feature_{i}" for i in range(len(feature_list), len(importances))]
        elif len(feature_list) > len(importances):
            feature_list = feature_list[:len(importances)]
        
        indices = np.argsort(importances)[::-1][:top_n]
        y_pos = np.arange(top_n)
        
        fig, ax = plt.subplots(figsize=GORSEL_AYARLARI['feature_importance_figsize'])
        ax.barh(y_pos, importances[indices], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_list[i] for i in indices])
        ax.set_xlabel('Önem Skoru', fontsize=12)
        ax.set_title(f'En Önemli {top_n} Özellik - {self.model_tipi.upper()}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        kayit_yolu = self.gorseller_klasoru / f"ozellik_onemi_{self.model_tipi}.png"
        fig.savefig(kayit_yolu, dpi=GORSEL_AYARLARI['dpi'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ Özellik önemi grafiği kaydedildi: {kayit_yolu}")
    
    def roc_curve_ciz(self, X_test, y_test, dosya_adi: str = "roc_curves.png"):
        """ROC eğrilerini çiz (multi-class için one-vs-rest)."""
        if not hasattr(self.model, 'predict_proba'):
            print(f"   ⚠️  ROC eğrisi sadece olasılık tahminini destekleyen modeller için çizilebilir")
            return
        
        try:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Tahminler
            y_prob = self.model.predict_proba(X_test)
            n_classes = y_prob.shape[1]
            
            # One-hot encoding
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # Her sınıf için ROC hesapla
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Grafik çiz
            fig, ax = plt.subplots(figsize=GORSEL_AYARLARI['roc_curve_figsize'])
            
            colors = ['blue', 'red', 'green', 'orange']
            class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Rastgele (AUC = 0.5)')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Eğrileri - {self.model_tipi.upper()}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            kayit_yolu = self.gorseller_klasoru / dosya_adi
            fig.savefig(kayit_yolu, dpi=GORSEL_AYARLARI['dpi'], bbox_inches='tight')
            plt.close()
            print(f"   ✓ ROC eğrileri kaydedildi: {kayit_yolu}")
            
        except Exception as e:
            print(f"   ⚠️  ROC eğrisi çizilemedi: {e}")
    
    def precision_recall_curve_ciz(self, X_test, y_test, dosya_adi: str = "precision_recall_curves.png"):
        """Precision-Recall eğrilerini çiz (multi-class için one-vs-rest)."""
        if not hasattr(self.model, 'predict_proba'):
            print(f"   ⚠️  Precision-Recall eğrisi sadece olasılık tahminini destekleyen modeller için çizilebilir")
            return
        
        try:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # Tahminler
            y_prob = self.model.predict_proba(X_test)
            n_classes = y_prob.shape[1]
            
            # One-hot encoding
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # Her sınıf için precision-recall hesapla
            precision = dict()
            recall = dict()
            avg_precision = dict()
            
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
                avg_precision[i] = average_precision_score(y_test_bin[:, i], y_prob[:, i])
            
            # Grafik çiz
            fig, ax = plt.subplots(figsize=GORSEL_AYARLARI['roc_curve_figsize'])
            
            colors = ['blue', 'red', 'green', 'orange']
            class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            
            for i, color in zip(range(n_classes), colors):
                ax.plot(recall[i], precision[i], color=color, lw=2,
                       label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Eğrileri - {self.model_tipi.upper()}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            kayit_yolu = self.gorseller_klasoru / dosya_adi
            fig.savefig(kayit_yolu, dpi=GORSEL_AYARLARI['dpi'], bbox_inches='tight')
            plt.close()
            print(f"   ✓ Precision-Recall eğrileri kaydedildi: {kayit_yolu}")
            
        except Exception as e:
            print(f"   ⚠️  Precision-Recall eğrisi çizilemedi: {e}")
    
    def grafik_ciz(self, X_test, y_test):
        """Tüm grafikleri çiz."""
        print(f"\n📊 Grafikler oluşturuluyor...")
        
        y_test_pred = self.tahmin_yap(X_test)
        
        # 1. Confusion Matrix
        self.confusion_matrix_ciz(y_test, y_test_pred)
        
        # 2. Feature Importance (gradient boosting için)
        self.ozellik_onemi_ciz()
        
        # 3. ROC Curves
        self.roc_curve_ciz(X_test, y_test)
        
        # 4. Precision-Recall Curves
        self.precision_recall_curve_ciz(X_test, y_test)
    
    def rapor_olustur(self):
        """Detaylı performans raporu oluştur."""
        rapor_yolu = self.raporlar_klasoru / f"rapor_{self.model_tipi}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(rapor_yolu, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"MRI SINIFLANDIRMA MODEL RAPORU\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model Tipi: {self.model_tipi.upper()}\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"SMOTE Aktif: {self.smote_aktif}\n")
            f.write(f"Feature Selection Aktif: {self.feature_selection_aktif}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("PERFORMANS METRİKLERİ\n")
            f.write("="*70 + "\n\n")
            
            if self.metrikler:
                for key, value in self.metrikler.items():
                    if key not in ['confusion_matrix', 'classification_report']:
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                if 'classification_report' in self.metrikler:
                    f.write("\n" + "-"*70 + "\n")
                    f.write("DETAYLI SINIF RAPORU\n")
                    f.write("-"*70 + "\n\n")
                    f.write(self.metrikler['classification_report'])
            
            # Cross-validation sonuçları
            if self.cv_scores:
                f.write("\n" + "="*70 + "\n")
                f.write("CROSS-VALIDATION SONUÇLARI\n")
                f.write("="*70 + "\n\n")
                for metric, scores in self.cv_scores.items():
                    f.write(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
        
        print(f"   ✓ Rapor kaydedildi: {rapor_yolu}")
    
    def model_kaydet(self, dosya_adi: Optional[str] = None):
        """Modeli ve metadata'sını kaydet."""
        if dosya_adi is None:
            zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
            dosya_adi = f"{self.model_tipi}_{zaman_damgasi}.pkl"
        
        kayit_yolu = self.modeller_klasoru / dosya_adi
        
        # Modeli kaydet
        with open(kayit_yolu, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n💾 Model kaydedildi: {kayit_yolu}")
        
        # Metadata kaydet
        metadata = {
            'model_tipi': self.model_tipi,
            'tarih': datetime.now().isoformat(),
            'metrikler': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in self.metrikler.items() 
                         if not isinstance(v, (np.ndarray, str))},
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'feature_selection_aktif': self.feature_selection_aktif,
            'veri_kaynagi': str(VERI_CSV),
            'ayarlar': GB_AYARLARI if self.model_tipi in ["xgboost", "lightgbm"] else SVM_AYARLARI
        }
        
        metadata_yolu = kayit_yolu.with_suffix('.json')
        with open(metadata_yolu, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Metadata kaydedildi: {metadata_yolu}")
    
    def tam_egitim_yap(self, hyperparameter_tuning_aktif: bool = False, search_method: str = "random", n_iter: int = 30):
        """
        Tam eğitim pipeline'ı çalıştır.
        
        Args:
            hyperparameter_tuning_aktif: Hyperparameter tuning yapılsın mı?
            search_method: Hyperparameter arama yöntemi ("random" veya "bayes")
            n_iter: Hyperparameter arama iterasyon sayısı
        """
        # 1. Veri yükle ve hazırla
        X_train, X_val, X_test, y_train, y_val, y_test = self.veri_yukle()
        
        # 2. Feature selection (opsiyonel)
        if self.feature_selection_aktif:
            X_train = self.feature_selection(X_train, y_train, k=15)
            # Validation ve test setlerine de uygula
            if self.selected_features:
                X_val = X_val[self.selected_features]
                X_test = X_test[self.selected_features]
        
        # 3. Cross-validation (model eğitiminden önce)
        print(f"\n{'='*60}")
        print(f"📊 CROSS-VALIDATION")
        print(f"{'='*60}")
        self.cross_validate(X_train, y_train, cv_folds=5)
        
        # 4. Hyperparameter tuning (opsiyonel)
        if hyperparameter_tuning_aktif:
            print(f"\n{'='*60}")
            print(f"🔧 HYPERPARAMETER TUNING")
            print(f"{'='*60}")
            best_params = self.hyperparameter_tuning(X_train, y_train, n_iter=n_iter, search_method=search_method)
        else:
            # Normal model oluştur
            self.model_olustur()
        
        # 5. Model eğit
        if not hyperparameter_tuning_aktif:
            # Hyperparameter tuning zaten eğitiyor, tekrar eğitmeye gerek yok
            self.egit(X_train, y_train, X_val, y_val)
        
        # 6. Değerlendir
        print(f"\n{'='*60}")
        print(f"📈 DEĞERLENDİRME")
        print(f"{'='*60}")
        
        # Doğrulama seti
        val_metrikler = self.degerlendir(X_val, y_val, "Doğrulama")
        
        # Test seti
        test_metrikler = self.degerlendir(X_test, y_test, "Test")
        self.metrikler = test_metrikler
        
        # 4. Görselleştir
        print(f"\n{'='*60}")
        print(f"📊 GÖRSELLEŞTİRME")
        print(f"{'='*60}")
        
        self.grafik_ciz(X_test, y_test)
        
        # 5. Rapor oluştur
        print(f"\n{'='*60}")
        print(f"📄 RAPOR OLUŞTURMA")
        print(f"{'='*60}")
        
        self.rapor_olustur()
        
        # 6. Modeli kaydet
        print(f"\n{'='*60}")
        print(f"💾 MODEL KAYDETME")
        print(f"{'='*60}")
        self.model_kaydet()
        
        print(f"\n{'='*60}")
        print(f"✅ EĞİTİM TAMAMLANDI!")
        print(f"{'='*60}")
        print(f"\n📁 Çıktılar: {CIKTI_KLASORU}")


def main():
    """
    Ana program - Model seçimi ve eğitim.
    
    Bu fonksiyon kullanıcıya hangi modeli eğitmek istediğini sorar
    ve seçilen model(ler)i eğitir.
    """
    print(f"\n{'='*70}")
    print(f"🧠 MRI SINIFLANDIRMA - MODEL EĞİTİMİ")
    print(f"{'='*70}")
    
    print(f"\n📋 Mevcut Modeller:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  1️⃣  XGBoost (Gradient Boosting)")
    print(f"      └─ Yüksek doğruluk, güçlü performans")
    print(f"      └─ Önerilen model ⭐")
    print(f"")
    print(f"  2️⃣  LightGBM (Gradient Boosting)")
    print(f"      └─ Hızlı eğitim, büyük veri setleri için")
    print(f"      └─ XGBoost'a alternatif")
    print(f"")
    print(f"  3️⃣  Linear SVM")
    print(f"      └─ Basit ve hızlı")
    print(f"      └─ Test ve karşılaştırma için")
    print(f"")
    print(f"  4️⃣  Tümü (Sırayla hepsini eğit)")
    print(f"      └─ Karşılaştırmalı analiz")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    secim = input(f"\n🎯 Model seçiminiz (1-4): ").strip()
    
    # Gelişmiş özellikler
    print(f"\n⚙️  Gelişmiş Özellikler:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    smote_input = input(f"SMOTE ile veri dengeleme? (E/h) [E]: ").strip().lower()
    smote_aktif = smote_input != 'h'
    
    tuning_input = input(f"Hyperparameter tuning? (e/H) [H]: ").strip().lower()
    hyperparameter_tuning_aktif = tuning_input == 'e'
    
    feature_sel_input = input(f"Feature selection? (e/H) [H]: ").strip().lower()
    feature_selection_aktif = feature_sel_input == 'e'
    
    # Model seçim haritası
    model_map = {
        '1': ['xgboost'],
        '2': ['lightgbm'],
        '3': ['svm'],
        '4': ['xgboost', 'lightgbm', 'svm']
    }
    
    if secim not in model_map:
        print(f"\n❌ Geçersiz seçim! Lütfen 1-4 arası bir sayı girin.")
        return
    
    modeller = model_map[secim]
    
    # Seçilen modelleri eğit
    for i, model_tipi in enumerate(modeller, 1):
        if len(modeller) > 1:
            print(f"\n\n{'#'*70}")
            print(f"# [{i}/{len(modeller)}] {model_tipi.upper()} MODELİ EĞİTİLİYOR")
            print(f"{'#'*70}\n")
        
        try:
            # Model eğiticiyi başlat
            egitici = ModelEgitici(
                model_tipi=model_tipi,
                smote_aktif=smote_aktif,
                feature_selection_aktif=feature_selection_aktif
            )
            
            # Tam eğitim yap
            egitici.tam_egitim_yap(hyperparameter_tuning_aktif=hyperparameter_tuning_aktif)
            
        except FileNotFoundError as e:
            print(f"\n❌ Hata: {e}")
            print(f"\n💡 Çözüm: Önce görüntü işleme adımlarını tamamlayın:")
            print(f"   cd ../goruntu_isleme")
            print(f"   python ana_islem.py")
            break
        except Exception as e:
            print(f"\n❌ {model_tipi} eğitimi başarısız: {e}")
            import traceback
            traceback.print_exc()
    
    if len(modeller) > 1:
        print(f"\n\n{'='*70}")
        print(f"✅ TÜM EĞİTİMLER TAMAMLANDI!")
        print(f"{'='*70}")
        print(f"\n📊 Sonuçları karşılaştırmak için çıktı klasörlerine bakın.")
        print(f"   Ana dizin: {CIKTI_KLASORU}")


if __name__ == "__main__":
    main()
