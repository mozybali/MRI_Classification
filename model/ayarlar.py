#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ayarlar.py
----------
Model eğitimi için merkezi konfigürasyon dosyası.
"""

from pathlib import Path

# ==================== PROJE YOLLARI ====================
# Proje kök dizini - tüm dosya yolları buraya göre belirlenir
PROJE_KOK = Path(__file__).parent.parent

# Veri yolları
# Eğitim için kullanılacak ölçeklendirilmiş özellik CSV'si
VERI_CSV = PROJE_KOK / "goruntu_isleme" / "cikti" / "goruntu_ozellikleri_scaled.csv"

# Model çıktıları için ana klasör
CIKTI_KLASORU = PROJE_KOK / "model" / "ciktilar"

# Alt klasörler
MODELS_KLASORU = CIKTI_KLASORU / "modeller"      # Eğitilmiş modeller buraya kaydedilir
RAPORLAR_KLASORU = CIKTI_KLASORU / "raporlar"    # Performans raporları
GORSELLER_KLASORU = CIKTI_KLASORU / "gorseller"  # Grafikler ve görselleştirmeler

# ==================== VERİ BÖLÜMLEME ====================
# Veri seti eğitim, doğrulama ve test setlerine bölünür
EGITIM_ORANI = 0.70        # %70 eğitim - Model parametrelerini öğrenmek için
DOGRULAMA_ORANI = 0.15     # %15 doğrulama - Hiperparametre ayarlama için
TEST_ORANI = 0.15          # %15 test - Son performans testi için
RASTGELE_TOHUM = 42        # Tekrarlanabilirlik için sabit tohum değeri
STRATIFY_AKTIF = True      # Sınıf dengesini koru (her sete aynı oranda sınıf)

# ==================== GRADIENT BOOSTING (XGBoost) ====================
# Gradient Boosting, güçlü bir topluluk öğrenme algoritmasıdır
# Birden fazla zayıf öğrenci (weak learner) birleştirerek güçlü bir model oluşturur
GB_AYARLARI = {
    'n_estimators': 100,         # Ağaç sayısı (daha fazla = daha iyi öğrenme ama daha yavaş)
    'max_depth': 7,              # Ağaç derinliği (daha derin = daha karmaşık model)
    'learning_rate': 0.1,        # Öğrenme hızı (düşük = daha iyi genelleme ama daha yavaş)
    'random_state': 42,          # Tekrarlanabilirlik için
    'subsample': 0.8,            # Her ağaç için kullanılacak veri oranı (overfitting önler)
    'colsample_bytree': 0.8,     # Her ağaç için kullanılacak özellik oranı
    'reg_lambda': 1.0,           # L2 regularizasyon (ağırlık cezası)
    'reg_alpha': 0.0,            # L1 regularizasyon
}

# Early stopping ayarları (fit() metodunda kullanılır, doğrudan model parametresi değil)
EARLY_STOPPING_ROUNDS = 10       # Doğrulama skorı 10 tur iyileşmezse dur

# Grid Search parametreleri - Otomatik hiperparametre optimizasyonu için
# Bu değerler denenerek en iyi kombinasyon bulunur (işlemci yoğun!)
GB_GRID_PARAMS = {
    'n_estimators': [50, 100, 200],        # Denenenecek ağaç sayıları
    'max_depth': [5, 7, 9],                # Denenenecek derinlikler
    'learning_rate': [0.01, 0.1, 0.2],     # Denenenecek öğrenme hızları
    'subsample': [0.7, 0.8, 0.9],          # Denenenecek örnekleme oranları
}

# ==================== LIGHTGBM ====================
# LightGBM, XGBoost'a alternatif hızlı gradient boosting kütüphanesi
LIGHTGBM_AYARLARI = {
    # Daha düşük öğrenme hızı + daha fazla iterasyon (early stopping ile kontrol)
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.05,
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'class_weight': 'balanced',  # Otomatik sınıf ağırlıklandırma
}

# LightGBM'de log basma periyodu (iterasyon). 0 yaparsanız tamamen sessiz çalışır.
LIGHTGBM_LOG_PERIOD = 50

# ==================== LINEAR SVM ====================
# Support Vector Machine (Destek Vektör Makinesi)
# Veriler arasında optimal bir karar sınırı bulmaya çalışır
# Linear SVM, doğrusal ayrılabilir problemler için hızlı ve etkilidir
SVM_AYARLARI = {
    'C': 0.1,                    # Regularizasyon parametresi (düşük = daha fazla regularizasyon)
    'loss': 'squared_hinge',     # Kayıp fonksiyonu tipi
    'max_iter': 200000,          # Maksimum iterasyon sayısı (LinearSVC'nin daha rahat konverge etmesi için)
    'random_state': 42,          # Tekrarlanabilirlik
    'dual': False,               # Örnek sayısı > özellik sayısı ise primal (daha hızlı konverge)
    'tol': 1e-2,                 # Tolerans (yaklaşım hassasiyeti) - konverjansı hızlandırır
    'class_weight': 'balanced',  # Sınıf dengesizliğini otomatik düzelt
    'verbose': 0,                # Çıktı detay seviyesi
}

SVM_GRID_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1],
    'loss': ['squared_hinge'],
    'dual': [False, True],            # True tercih edilecekse özellik sayısı > örnek sayısı durumunda yararlı
    'max_iter': [5000, 20000, 50000, 200000],
    'tol': [1e-2, 5e-3, 1e-3],
}

# ==================== HİPERPARAMETRE AYARLAMA ====================
GRID_SEARCH_AYARLARI = {
    'cv_folds': 5,
    'n_jobs': -1,
    'verbose': 1,
}

# ==================== MODEL YÖNETİMİ ====================
MODEL_KAYIT_AYARLARI = {
    'save_format': 'pickle',
    'include_metadata': True,
    'include_metrics': True,
    'backup_old_versions': True,
    'max_versions': 5,
}

# ==================== GÖRSELLEŞTİRME ====================
# Grafik çizimi için ayarlar (karışıklık matrisi, özellik önemi, vb.)
GORSEL_AYARLARI = {
    'confusion_matrix_figsize': (10, 8),   # Karışıklık matrisi boyutu (inç)
    'feature_importance_figsize': (12, 6), # Özellik önemi grafiği boyutu
    'roc_curve_figsize': (8, 6),           # ROC eğrisi boyutu
    'dpi': 100,                            # Çözünürlük (dots per inch)
    'style': 'default',                    # Matplotlib stili
}

# ==================== LOGLAMA ====================
# Eğitim sürecini kaydetmek için log ayarları
LOG_AYARLARI = {
    'log_file': CIKTI_KLASORU / 'egitim.log',  # Log dosyası yolu
    'log_level': 'INFO',                        # Log seviyesi (DEBUG, INFO, WARNING, ERROR)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log formatı
}
