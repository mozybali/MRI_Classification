# MRI Sınıflandırma Projesi v2.0

MRI beyin görüntülerinden demans hastalığı teşhisi yapan kapsamlı makine öğrenmesi projesi.

## 📋 Proje Açıklaması

Bu proje, MRI beyin görüntülerini kullanarak 4 farklı demans seviyesini otomatik olarak sınıflandırır:

- **NonDemented** - Sağlıklı (Demans yok)
- **VeryMildDemented** - Çok hafif demans
- **MildDemented** - Hafif demans  
- **ModerateDemented** - Orta seviye demans

## 🏗️ Basitleştirilmiş Proje Yapısı

```
MRI_Classification/
│
├── Veri_Seti/                    # Ham MRI görüntüleri
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
│
├── goruntu_isleme/               # Görüntü işleme modülü
│   ├── ayarlar.py                # Konfigürasyon
│   ├── goruntu_isleyici.py       # İşleme ve veri artırma
│   ├── ozellik_cikarici.py       # Özellik çıkarma ve CSV
│   ├── ana_islem.py              # Ana menü programı
│   ├── pipeline_quick_test.py    # Sistem kontrolü
│   ├── test_pipeline.py          # Pipeline test
│   └── README.md
│
├── eda_analiz/                   # Veri analizi modülü
│   ├── eda_araclar.py            # Analiz araçları
│   ├── eda_calistir.py           # Ana program
│   ├── requirements.txt          # Minimal bağımlılıklar
│   └── README.md
│
├── model/                        # Model eğitimi modülü
│   ├── ayarlar.py                # Konfigürasyon
│   ├── model_egitici.py          # Eğitim ve değerlendirme
│   ├── train.py                  # İnteraktif eğitim scripti
│   ├── inference.py              # Tahmin scripti
│   ├── model_comparison.py       # Model karşılaştırma
│   └── README.md
│
├── tests/                        # Test suite (v2.0)
│   ├── conftest.py               # Test fixtures
│   ├── test_goruntu_isleyici.py  # Görüntü işleme testleri
│   ├── test_ozellik_cikarici.py  # Özellik çıkarma testleri
│   ├── test_model_egitici.py     # Model eğitim testleri (23 test)
│   └── test_eda_araclar.py       # EDA testleri
│
├── pytest.ini                    # Pytest konfigürasyonu
├── requirements.txt              # Tüm bağımlılıklar (prod + dev)
└── README.md                     # Bu dosya
```

## 🚀 Kurulum

### 1. Depoyu klonlayın
```bash
git clone https://github.com/mozybali/MRI_Classification.git
cd MRI_Classification
```

### 2. Gerekli paketleri yükleyin

**Hızlı kurulum (önerilen):**
```bash
pip install -r requirements.txt
```

**⚠️ Python 3.14 Kullanıcıları İçin Önemli Not:**

Python 3.14 çok yeni bir sürüm olduğu için `scikit-image` paketi için derlenmiş binary bulunmayabilir. Bu durumda aşağıdaki komutu kullanın:

```bash
# scikit-image için önceden derlenmiş wheel kullan
pip install --only-binary=:all: scikit-image
```

Eğer hala sorun yaşıyorsanız, tüm paketleri şu şekilde yükleyin:

```bash
# OpenCV'yi yükle
pip install opencv-python

# scikit-image'i binary olarak yükle
pip install --only-binary=:all: scikit-image

# Kalan paketleri yükle
pip install numpy pandas scipy Pillow SimpleITK scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn tqdm
```

**Veya modül bazlı kurulum:**
```bash
# EDA analizi (minimal bağımlılıklar)
cd eda_analiz
pip install -r requirements.txt

# Tüm proje için ana dizinden
cd ..
pip install -r requirements.txt
```

### 3. Sistem kontrolü
```bash
cd goruntu_isleme
python pipeline_quick_test.py
```

**Not:** Komutlarda `python` veya `python3` kullanabilirsiniz. Windows'ta genellikle `python`, Linux/Mac'te `python3` kullanılır.

## 📖 Kullanım

### Adım 1: Görüntü Ön İşleme

```bash
cd goruntu_isleme
python ana_islem.py
```

Menüden seçim yapın:
- **1**: Görüntüleri işle (🆕 bias correction, skull stripping, gelişmiş augmentation)
- **2**: Özellik çıkar ve CSV oluştur
- **3**: CSV'ye ölçeklendirme uygula (🆕 4 farklı metod: minmax/robust/standard/maxabs)
- **4**: Veri setini böl (eğitim/doğrulama/test)
- **6**: Tüm işlemleri otomatik yap (önerilen)

**🆕 Yeni Özellikler (v2.0):**
- ⭐ Bias field correction (MRI yoğunluk düzeltme)
- ⭐ Skull stripping (kafatası çıkarma)
- ⭐ Center of mass alignment (görüntü hizalama)
- ⭐ Adaptive CLAHE (akıllı kontrast iyileştirme)
- 🎯 Medikal-spesifik augmentation (elastic deformation, gaussian noise, vb.)
- 📊 Genişletilmiş scaling seçenekleri

### Adım 2: Veri Analizi (İsteğe Bağlı)

```bash
cd ../eda_analiz
python eda_calistir.py
```

Şunları üretir:
- Sınıf dağılımı grafikleri
- Görüntü boyut analizi
- Yoğunluk istatistikleri
- Korelasyon matrisi
- PCA görselleştirmesi

### Adım 3: Model Eğitimi

**Yeni: Kullanıcı dostu eğitim scripti** 🎯

```bash
cd ../model
python train.py
```

**Hızlı başlatma seçenekleri:**
```bash
# Otomatik mod (varsayılan ayarlar)
python train.py --auto

# Belirli model ile başlat
python train.py --model xgboost
python train.py --model lightgbm
python train.py --model svm
```

Desteklenen modeller:
- **XGBoost** (önerilen) - Yüksek doğruluk
- **LightGBM** - Hızlı eğitim
- **Linear SVM** - Basit ve hızlı

**Gelişmiş özellikler:**
- 🔄 SMOTE ile veri dengeleme
- 🎯 Sınıf ağırlıklandırma
- 📊 Hyperparameter tuning
- 🔍 Feature selection

### Adım 4: Tahmin (Inference)

**Eğitilmiş model ile yeni görüntüleri tahmin et:**

```bash
# Tek görüntü
python inference.py --image test.jpg

# Toplu tahmin (klasör)
python inference.py --batch ./test_images/

# Belirli model ile
python inference.py --model xgboost_latest.pkl --image test.jpg
```

### Adım 5: Model Karşılaştırma

**Birden fazla model eğittiyseniz performansları karşılaştırın:**

```bash
python model_comparison.py
```

Çıktılar:
- 📊 Performans karşılaştırma grafikleri
- 🎯 Radar chart
- 🏆 En iyi model önerisi

## 📊 Özellikler

### Görüntü İşleme (v2.0)
- ✅ Bias field correction (N4ITK)
- ✅ Skull stripping (kafatası çıkarma)
- ✅ Center of mass alignment
- ✅ Adaptif histogram eşitleme (CLAHE)
- ✅ Medikal-spesifik veri artırma
- ✅ Sınıf bazlı dengesiz augmentation
- ✅ Özellik çıkarma (20+ özellik)
- ✅ Çoklu ölçeklendirme metodu

### Model Eğitimi (Güncellenmiş)
- ✅ İnteraktif eğitim arayüzü
- ✅ SMOTE ile veri dengeleme
- ✅ Otomatik veri bölme (70/15/15)
- ✅ Cross-validation desteği
- ✅ Hyperparameter tuning (opsiyonel)
- ✅ Performans metrikleri (accuracy, precision, recall, F1, ROC-AUC, Cohen's Kappa)
- ✅ Karışıklık matrisi
- ✅ ROC eğrileri (multi-class)
- ✅ Precision-Recall eğrileri
- ✅ Özellik önemi analizi
- ✅ Detaylı raporlar
- ✅ Model ve metadata kaydetme
- ✅ Inference scripti (tek/batch tahmin)
- ✅ Model karşılaştırma aracı

### EDA Analizi
- ✅ Kapsamlı istatistiksel analiz
- ✅ Görselleştirme (matplotlib + seaborn)
- ✅ PCA boyut indirgeme
- ✅ Özet raporlar

## 🔧 Konfigürasyon

Her modülün `ayarlar.py` dosyasını düzenleyerek özelleştirin:

**goruntu_isleme/ayarlar.py**
```python
HEDEF_GENISLIK = 256
HEDEF_YUKSEKLIK = 256
VERI_ARTIRMA_AKTIF = True
SINIF_BAZLI_ARTIRMA_AKTIF = True  # Sınıf dengesizliği için
BIAS_FIELD_CORRECTION_AKTIF = True
SKULL_STRIPPING_AKTIF = True
```

**model/ayarlar.py**
```python
GB_AYARLARI = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    'scale_pos_weight': None,  # Otomatik sınıf ağırlığı
    ...
}
```

## 📈 Beklenen Performans

Tipik sonuçlar (33,984 görüntü, XGBoost ile):
- **Accuracy**: ~85-92%
- **F1 Score**: ~0.82-0.88
- **ROC-AUC**: ~0.88-0.93
- **Training Time**: 3-8 dakika (CPU)
- **Inference Time**: ~50-100ms per image

## 📚 Proje Yapısı

```
MRI_Classification/
├── README.md                          # Ana dokümantasyon
├── requirements.txt                   # Tüm bağımlılıklar
├── LICENSE
├── Veri_Seti/                        # Ham MRI görüntüleri (33,984 adet)
│   ├── NonDemented/                  (9,600 görüntü)
│   ├── VeryMildDemented/             (8,960 görüntü)
│   ├── MildDemented/                 (8,960 görüntü)
│   └── ModerateDemented/             (6,464 görüntü)
│
├── goruntu_isleme/                   # Görüntü işleme modülü
│   ├── ana_islem.py                  (Ana çalıştırma scripti)
│   ├── goruntu_isleyici.py           (Core işleme)
│   ├── ozellik_cikarici.py           (Feature extraction)
│   ├── ayarlar.py                    (Konfigürasyon)
│   ├── pipeline_quick_test.py        (Sistem kontrolü)
│   ├── test_pipeline.py              (Pipeline test)
│   └── README.md
│
├── eda_analiz/                       # EDA modülü
│   ├── eda_calistir.py               (Ana çalıştırma scripti)
│   ├── eda_araclar.py                (Analiz araçları)
│   ├── requirements.txt              (Minimal bağımlılıklar)
│   └── README.md
│
└── model/                            # Model eğitim modülü
    ├── train.py                      (Ana eğitim scripti) ⭐
    ├── inference.py                  (Tahmin scripti) ⭐
    ├── model_comparison.py           (Model karşılaştırma) ⭐
    ├── model_egitici.py              (Core eğitim sınıfı)
    ├── ayarlar.py                    (Konfigürasyon)
    └── README.md
```

## 🎯 Özellikler ve İyileştirmeler (v2.0)

### ✅ Yeni Eklenenler
- 🆕 Kullanıcı dostu `train.py` scripti (interaktif + otomatik mod)
- 🆕 `inference.py` - Production-ready tahmin scripti
- 🆕 `model_comparison.py` - Model performans karşılaştırma
- 🆕 `pipeline_quick_test.py` - Sistem ön kontrolü
- 🆕 SMOTE veri dengeleme entegrasyonu
- 🆕 Sınıf bazlı augmentation çarpanları
- 🆕 ROC ve Precision-Recall eğrileri
- 🆕 Kapsamlı README'ler her modül için

### 🔄 İyileştirilenler
- ⬆️ Bias field correction (N4ITK)
- ⬆️ Skull stripping algoritması
- ⬆️ Medikal-spesifik augmentation
- ⬆️ 20+ feature extraction
- ⬆️ Class weights stratejisi
- ⬆️ Detaylı dokümantasyon

## 🎯 Kullanım Senaryoları

### Senaryo 1: Hızlı Başlangıç (5 dakika)
```bash
pip install -r requirements.txt
cd goruntu_isleme ; python ana_islem.py  # Menüden 6
cd ../model ; python train.py --auto
```

### Senaryo 2: Kapsamlı Analiz
```bash
# 1. EDA analizi
cd eda_analiz ; python eda_calistir.py

# 2. Görüntü işleme
cd ../goruntu_isleme ; python ana_islem.py  # Menüden 6

# 3. Model eğitimi (interaktif)
cd ../model ; python train.py

# 4. Model karşılaştırma
python model_comparison.py
```

### Senaryo 3: Production Deployment
```bash
# Model eğit
python train.py --auto --model xgboost

# Yeni görüntüleri tahmin et
python inference.py --batch ./new_patients/

# Sonuçları analiz et
python model_comparison.py
```
- ✅ ASCII klasör isimleri
- ✅ Her modül 2-3 dosyada birleştirildi
- ✅ Tek konfigürasyon dosyası
- ✅ Modüler ve anlaşılır yapı

## 🧪 Test Suite

Proje, kapsamlı bir test suite ile birlikte gelir. Testler pytest framework'ü kullanılarak yazılmıştır.

### Test Kurulumu

```bash
# Test araçları ana requirements.txt içinde dahil
pip install -r requirements.txt
```

### Testleri Çalıştırma

**Tüm testleri çalıştır:**
```bash
pytest
```

**Verbose mode ile:**
```bash
pytest -v
```

**Coverage raporu ile:**
```bash
pytest --cov=goruntu_isleme --cov=model --cov=eda_analiz --cov-report=html
```

**Belirli bir test dosyasını çalıştır:**
```bash
pytest tests/test_goruntu_isleyici.py
pytest tests/test_model_egitici.py
pytest tests/test_ozellik_cikarici.py
pytest tests/test_eda_araclar.py
```

**Belirli bir test fonksiyonunu çalıştır:**
```bash
pytest tests/test_goruntu_isleyici.py::TestGorselIsleyici::test_init
```

**Parallel test execution:**
```bash
pytest -n auto
```

### Test Yapısı

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_goruntu_isleyici.py       # Görüntü işleme testleri (30+ test)
├── test_ozellik_cikarici.py       # Özellik çıkarma testleri (25+ test)
├── test_model_egitici.py          # Model eğitim testleri (30+ test)
└── test_eda_araclar.py            # EDA testleri (20+ test)
```

### Test Kategorileri

Testler şu kategorilere ayrılmıştır:

- **Unit Tests**: Tekil fonksiyon ve metod testleri
- **Integration Tests**: Modüller arası entegrasyon testleri
- **Edge Cases**: Sınır durumları ve hata yönetimi testleri

### Coverage Hedefi

- Hedef: **>80% code coverage**
- Kritik modüller: **>90% coverage**

Coverage raporunu görüntülemek için:
```bash
pytest --cov-report=html
# Sonra htmlcov/index.html dosyasını tarayıcıda açın
```

### Test Fixtures

Proje, test verisi oluşturmak için zengin fixture'lar içerir:

- `test_image`: Test MRI görüntüsü (256x256)
- `test_image_path`: Geçici dosya yolu
- `test_dataset_structure`: Minimal veri seti yapısı
- `sample_features_df`: Örnek özellik DataFrame'i
- `temp_output_dir`: Geçici çıktı klasörü

### Continuous Integration

GitHub Actions ile otomatik test çalıştırma (yakında):

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest
```

## 📝 Notlar

- Veri seti klasörü: `Veri_Seti/` (değiştirilebilir)
- Çıktılar otomatik olarak kaydedilir
- Tüm işlemler terminal üzerinden yönetilir
- İlerleme çubukları ile takip edin

## 🤝 Katkı

Katkılarınızı bekliyoruz! Pull request göndermekten çekinmeyin.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Yazar

- GitHub: [@mozybali](https://github.com/mozybali)

## 🙏 Teşekkürler

MRI veri seti ve ilham için tüm katkıda bulunanlara teşekkürler.
