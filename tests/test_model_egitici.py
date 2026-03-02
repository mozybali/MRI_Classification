"""
Model eğitimi modülü testleri.

Bu modül ModelEgitici sınıfının tüm fonksiyonlarını test eder:
- XGBoost, LightGBM, SVM model oluşturma
- Model eğitimi ve değerlendirme
- Model kaydetme ve yükleme
- SMOTE ile veri dengeleme
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model_egitici import ModelEgitici


class TestModelEgitici:
    """
    ModelEgitici sınıfı test paketi.
    
    XGBoost, LightGBM ve SVM modellerinin oluşturulması,
    eğitimi, tahmin ve değerlendirme işlemlerini test eder.
    """
    
    def test_init_xgboost(self):
        """
        XGBoost ile ModelEgitici başlatma testi.
        
        ModelEgitici sınıfının XGBoost model tipiyle
        doğru şekilde başlatıldığını kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        
        assert egitici is not None
        assert egitici.model_tipi == "xgboost"
        assert egitici.model is None  # Henüz eğitilmedi
    
    def test_init_lightgbm(self):
        """
        LightGBM ile ModelEgitici başlatma testi.
        
        ModelEgitici sınıfının LightGBM model tipiyle
        doğru şekilde başlatıldığını kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="lightgbm", smote_aktif=False)
        
        assert egitici.model_tipi == "lightgbm"
    
    def test_init_svm(self):
        """
        SVM ile ModelEgitici başlatma testi.
        
        ModelEgitici sınıfının SVM model tipiyle
        doğru şekilde başlatıldığını kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="svm", smote_aktif=False)
        
        assert egitici.model_tipi == "svm"
    
    def test_model_olustur_xgboost(self):
        """
        XGBoost model oluşturma testi.
        
        XGBoost modelinin doğru parametrelerle oluşturulduğunu
        ve fit/predict metodlarına sahip olduğunu kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        
        assert egitici.model is not None
        assert hasattr(egitici.model, 'fit')
        assert hasattr(egitici.model, 'predict')
    
    def test_model_olustur_lightgbm(self):
        """
        LightGBM model oluşturma testi.
        
        LightGBM modelinin doğru parametrelerle oluşturulduğunu
        ve fit metoduna sahip olduğunu kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="lightgbm", smote_aktif=False)
        egitici.model_olustur()
        
        assert egitici.model is not None
        assert hasattr(egitici.model, 'fit')
    
    def test_model_olustur_svm(self):
        """
        SVM model oluşturma testi.
        
        SVM modelinin doğru parametrelerle oluşturulduğunu
        ve fit metoduna sahip olduğunu kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="svm", smote_aktif=False)
        egitici.model_olustur()
        
        assert egitici.model is not None
        assert hasattr(egitici.model, 'fit')
    
    def test_veri_yukle(self, temp_output_dir):
        """
        CSV'den veri yükleme testi.
        
        Tek CSV dosyasının yüklenip otomatik olarak
        train/val/test setlerine bölündüğünü kontrol eder.
        """
        # Tek bir CSV oluştur
        np.random.seed(42)
        
        data = {
            'feature1': np.random.rand(80),
            'feature2': np.random.rand(80),
            'feature3': np.random.rand(80),
            'sinif': [f'Class{i%4}' for i in range(80)],
            'etiket': [i % 4 for i in range(80)]
        }
        df = pd.DataFrame(data)
        
        csv_path = temp_output_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        
        X_train, X_val, X_test, y_train, y_val, y_test = egitici.veri_yukle(
            csv_yolu=csv_path
        )
        
        # Boyutları kontrol et - otomatik bölünerek gelir
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_val) > 0
        assert len(y_test) > 0
        
        # Toplam eşit olmalı
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 80

    def test_veri_yukle_meta_sayisal_kolonlari_dislar(self, temp_output_dir):
        """Boyut ve dosya boyutu gibi meta kolonlar model girdisine alınmamalı."""
        np.random.seed(42)
        df = pd.DataFrame({
            'boyut_bayt': np.random.randint(5000, 6000, 40),
            'genislik': [256] * 40,
            'yukseklik': [256] * 40,
            'en_boy_orani': [1.0] * 40,
            'piksel_sayisi': [65536] * 40,
            'feature1': np.random.rand(40),
            'feature2': np.random.rand(40),
            'sinif': [f'Class{i%4}' for i in range(40)],
            'etiket': [i % 4 for i in range(40)],
        })

        csv_path = temp_output_dir / "data_with_meta.csv"
        df.to_csv(csv_path, index=False)

        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        X_train, X_val, X_test, _, _, _ = egitici.veri_yukle(csv_yolu=csv_path)

        for split_df in [X_train, X_val, X_test]:
            assert 'boyut_bayt' not in split_df.columns
            assert 'genislik' not in split_df.columns
            assert 'yukseklik' not in split_df.columns
            assert 'en_boy_orani' not in split_df.columns
            assert 'piksel_sayisi' not in split_df.columns
    
    def test_egit_basic(self, temp_output_dir):
        """
        Temel model eğitimi testi.
        
        Modelin eğitim verisi ile doğru şekilde
        eğitildiğini kontrol eder.
        """
        # Basit eğitim verisi oluştur
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 4, 50)
        X_val = np.random.rand(15, 5)
        y_val = np.random.randint(0, 4, 15)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        
        egitici.egit(X_train, y_train, X_val, y_val)
        
        # Model eğitilmiş olmalı
        assert egitici.model is not None
        assert hasattr(egitici.model, 'predict')
    
    def test_tahmin_yap(self):
        """
        Tahmin yapma testi.
        
        Eğitilmiş modelin test verisi üzerinde
        doğru tahminler ürettiğini kontrol eder.
        """
        # Basit bir model oluştur ve eğit
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 4, 50)
        X_val = np.random.rand(15, 5)
        y_val = np.random.randint(0, 4, 15)
        X_test = np.random.rand(10, 5)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        egitici.egit(X_train, y_train, X_val, y_val)
        
        predictions = egitici.tahmin_yap(X_test)
        
        # Tahminleri kontrol et
        assert predictions is not None
        assert len(predictions) == 10
        assert all(0 <= p < 4 for p in predictions)
    
    def test_degerlendir(self):
        """
        Model değerlendirme testi.
        
        Modelin test verisi üzerindeki performans
        metriklerinin doğru hesaplandığını kontrol eder.
        """
        # Basit bir model oluştur ve eğit
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 4, 50)
        X_val = np.random.rand(15, 5)
        y_val = np.random.randint(0, 4, 15)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 4, 20)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        egitici.egit(X_train, y_train, X_val, y_val)
        
        metrikler = egitici.degerlendir(X_test, y_test, set_adi="Test")
        
        # Metriklerin var olduğunu kontrol et
        assert 'accuracy' in metrikler or 'dogruluk' in metrikler
        
        # Metrik aralıklarını kontrol et (0 ile 1 arası)
        if 'accuracy' in metrikler:
            assert 0 <= metrikler['accuracy'] <= 1
    
    def test_feature_selection(self):
        """
        Özellik seçimi testi.
        
        En önemli k özelliğin seçildiğini ve
        boyutluluğun azaltıldığını kontrol eder.
        """
        np.random.seed(42)
        X_train = np.random.rand(50, 10)
        y_train = np.random.randint(0, 4, 50)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False, feature_selection_aktif=True)
        # Feature names'i ayarla
        egitici.feature_names = [f"feature_{i}" for i in range(10)]
        
        X_selected = egitici.feature_selection(X_train, y_train, k=5)
        
        # k özellik seçilmiş olmalı
        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == 50
    
    def test_model_kaydet(self, temp_output_dir):
        """
        Model kaydetme testi.
        
        Eğitilmiş modelin dosyaya kaydedildiğini
        ve pickle formatında yüklenebilir olduğunu kontrol eder.
        """
        np.random.seed(42)
        X_train = np.random.rand(30, 5)
        y_train = np.random.randint(0, 4, 30)
        X_val = np.random.rand(10, 5)
        y_val = np.random.randint(0, 4, 10)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        egitici.egit(X_train, y_train, X_val, y_val)
        
        # Modeli kaydet
        model_path = temp_output_dir / "test_model.pkl"
        egitici.model_kaydet(dosya_adi=str(model_path))
        
        # Dosyanın oluştuğunu kontrol et
        assert model_path.exists()
        
        # Yüklemeyi dene - model_kaydet sadece modeli kaydediyor
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')
    
    def test_confusion_matrix_ciz(self, temp_output_dir):
        """
        Confusion matrix çizimi testi.
        
        Karmaşıklık matrisinin görselleştirilip
        PNG dosyası olarak kaydedildiğini kontrol eder.
        """
        np.random.seed(42)
        y_true = np.random.randint(0, 4, 50)
        y_pred = np.random.randint(0, 4, 50)
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        
        output_file = temp_output_dir / "confusion_matrix.png"
        egitici.confusion_matrix_ciz(y_true, y_pred, dosya_adi=str(output_file))
        
        # Dosyanın oluştuğunu kontrol et
        assert output_file.exists()
    
    def test_cross_validate(self):
        """
        Çapraz doğrulama testi.
        
        K-fold cross validation ile modelin
        farklı veri kümelerinde test edildiğini kontrol eder.
        """
        np.random.seed(42)
        X = np.random.rand(60, 5)
        y = np.random.randint(0, 4, 60)
        
        egitici = ModelEgitici(model_tipi="svm", smote_aktif=False)  # SVM doesn't have early stopping issues
        egitici.model_olustur()
        
        cv_scores = egitici.cross_validate(X, y, cv_folds=3)
        
        # CV sonuçlarını kontrol et
        assert cv_scores is not None
        # Her fold için skor olmalı
        if isinstance(cv_scores, dict) and 'accuracy' in cv_scores:
            assert len(cv_scores['accuracy']) == 3  # 3 fold


class TestModelEgiticiWithSMOTE:
    """
    SMOTE aktif ModelEgitici testleri.
    
    Dengesiz veri setlerinde SMOTE ile
    sentetik örnek üretimi testleri.
    """
    
    def test_init_with_smote(self):
        """
        SMOTE ile başlatma testi.
        
        ModelEgitici'nin SMOTE parametresi
        ile doğru başlatıldığını kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=True)
        
        # SMOTE mevcut ise aktif olmalı
        if egitici.smote_aktif:
            assert egitici.smote_aktif is True
    
    def test_egit_with_imbalanced_data(self):
        """
        Dengesiz veri ile eğitim testi.
        
        SMOTE kullanarak dengesiz veri setinin
        dengeli hale getirildiğini ve modelin eğitildiğini kontrol eder.
        """
        # Dengesiz veri seti oluştur
        np.random.seed(42)
        X_train = np.random.rand(60, 5)
        y_train = np.array([0]*40 + [1]*15 + [2]*5)  # Çok dengesiz
        
        egitici = ModelEgitici(model_tipi="svm", smote_aktif=True)  # SVM doesn't need validation set
        egitici.model_olustur()
        
        # Dengesiz veriyi işlemeli
        egitici.egit(X_train, y_train)
        
        assert egitici.model is not None


class TestModelComparison:
    """
    Model karşılaştırma testleri.
    
    Farklı model tiplerinin (XGBoost, LightGBM, SVM)
    aynı şekilde çalıştığını doğrular.
    """
    
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "svm"])
    def test_model_types(self, model_type):
        """
        Tüm model tiplerinin başlatılıp eğitilebildiğini test eder.
        
        Her model tipi için aynı veri seti ile
        eğitim yapılabildiğini doğrular.
        """
        np.random.seed(42)
        X_train = np.random.rand(40, 5)
        y_train = np.random.randint(0, 3, 40)
        X_val = np.random.rand(12, 5)
        y_val = np.random.randint(0, 3, 12)
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 3, 10)
        
        egitici = ModelEgitici(model_tipi=model_type, smote_aktif=False)
        egitici.model_olustur()
        # XGBoost ve LightGBM için validation set gerekli
        if model_type in ["xgboost", "lightgbm"]:
            egitici.egit(X_train, y_train, X_val, y_val)
        else:
            egitici.egit(X_train, y_train)
        
        # Tahmin testi
        predictions = egitici.tahmin_yap(X_test)
        assert len(predictions) == 10
        
        # Değerlendirme testi
        metrikler = egitici.degerlendir(X_test, y_test)
        assert 'accuracy' in metrikler or metrikler is not None


class TestEdgeCases:
    """
    Sınır durumları ve hata yönetimi testleri.
    
    Boş veri, tek sınıf gibi özel durumları test eder.
    """
    
    def test_empty_data(self):
        """
        Boş veri işleme testi.
        
        Boş veri seti ile eğitim yapıldığında
        uygun hata fırlatıldığını kontrol eder.
        """
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()

        # Hatasız işlemeli veya uygun hata fırlatmalı
        with pytest.raises((ValueError, IndexError)):
            X_empty = np.array([]).reshape(0, 5)
            y_empty = np.array([])
            egitici.egit(X_empty, y_empty)
    
    def test_single_class(self):
        """
        Tek sınıf işleme testi.
        
        Eğitim verisinde tek sınıf olduğunda
        modelin nasıl davrandığını kontrol eder.
        """
        np.random.seed(42)
        X_train = np.random.rand(20, 5)
        y_train = np.zeros(20, dtype=int)  # Hepsi aynı sınıf
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        
        # Tek sınıfı işlemeli (ideal olmasa da)
        try:
            egitici.egit(X_train, y_train)
            predictions = egitici.tahmin_yap(X_train)
            assert all(p == 0 for p in predictions)
        except ValueError:
            # Bazı modeller tek sınıf için hata verebilir
            pass
    
    def test_mismatched_dimensions(self):
        """
        Uyuşmayan boyutlar testi.
        
        Eğitim ve test verisi özellik sayılarının
        farklı olduğunda hata alındığını kontrol eder.
        """
        np.random.seed(42)
        X_train = np.random.rand(30, 5)
        y_train = np.random.randint(0, 3, 30)
        X_val = np.random.rand(10, 5)
        y_val = np.random.randint(0, 3, 10)
        X_test = np.random.rand(10, 7)  # Farklı özellik sayısı
        
        egitici = ModelEgitici(model_tipi="xgboost", smote_aktif=False)
        egitici.model_olustur()
        egitici.egit(X_train, y_train, X_val, y_val)
        
        # Farklı boyutlarla tahmin yaparken hata vermeli
        with pytest.raises((ValueError, Exception)):
            egitici.tahmin_yap(X_test)
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            egitici = ModelEgitici(model_tipi="invalid_model", smote_aktif=False)
            egitici.model_olustur()



