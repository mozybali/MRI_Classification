# Model Eğitim Modülü

MRI görüntülerinden çıkarılan özelliklerle makine öğrenmesi modelleri eğitir.

## 🆕 v3.0 Performans İyileştirmeleri

### ⚡ Batch Tahmin Optimizasyonu
- **Paralel inference**: Batch tahmin 6-8x daha hızlı
- **Otomatik CPU yönetimi**: Çok çekirdekli işlem
- **Toplu işleme**: Binlerce görüntü dakikalar içinde

📊 **Performans Kazanımları:**
- Batch tahmin (1000 görüntü): 10-15 dk → 1-2 dakika (**6-8x**)

✅ **Geriye Uyumlu**: API değişikliği yok, otomatik çalışır!

---

## 📦 Kurulum

```bash
# Ana dizinden tüm bağımlılıkları yükle
cd ..
pip install -r requirements.txt
```

**Not:** Model modülü için ayrı requirements.txt yok, tüm bağımlılıklar ana `requirements.txt` dosyasında.

## 🚀 Kullanım

**Not:** Komutlarda `python` veya `python3` kullanabilirsiniz. Windows'ta genellikle `python`, Linux/Mac'te `python3` kullanılır.

### 1. Temel Kullanım (Önerilen)

**İnteraktif mod:**
```bash
python train.py
```

**Otomatik mod (hızlı başlangıç):**
```bash
python train.py --auto
```

**Belirli model ile:**
```bash
python train.py --auto --model xgboost
python train.py --auto --model lightgbm
python train.py --auto --model svm
```

### 2. Tahmin (Inference)

**Tek görüntü:**
```bash
python inference.py --model xgboost_latest.pkl --image test.jpg
```

**Toplu tahmin (batch):**
```bash
python inference.py --model xgboost_latest.pkl --batch ./test_images/
```

**En son model ile otomatik:**
```bash
python inference.py --image test.jpg
```

### 3. Model Karşılaştırma

```bash
python model_comparison.py
```

Tüm eğitilmiş modelleri karşılaştırır ve en iyisini seçer.

## 🤖 Desteklenen Modeller

| Model | Özellikler | Kullanım |
|-------|-----------|----------|
| **XGBoost** | Yüksek doğruluk, güçlü performans | Önerilen ⭐ |
| **LightGBM** | Hızlı eğitim, büyük veri setleri | Alternatif |
| **Linear SVM** | Basit, hızlı | Test/karşılaştırma |

## ✨ Özellikler

### Eğitim
- ✅ İnteraktif kullanıcı arayüzü
- ✅ SMOTE ile veri dengeleme
- ✅ Sınıf ağırlıklandırma (class weights)
- ✅ Otomatik veri bölme (70/15/15)
- ✅ K-fold cross-validation
- ✅ Hyperparameter tuning (opsiyonel)
- ✅ Feature selection (opsiyonel)

### Değerlendirme
- ✅ Kapsamlı metrikler (Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's Kappa)
- ✅ Confusion matrix (ısı haritası)
- ✅ ROC curves (multi-class)
- ✅ Precision-Recall curves
- ✅ Feature importance
- ✅ Detaylı raporlar (TXT + JSON)

### Inference
- ✅ Tek görüntü tahmini
- ✅ Batch tahmin (klasör)
- ✅ Olasılık skorları
- ✅ Güven skoru
- ✅ CSV export

### Karşılaştırma
- ✅ Tüm modelleri karşılaştır
- ✅ Performans grafikleri
- ✅ Radar chart
- ✅ En iyi model seçimi

## 📁 Çıktı Yapısı

```
model/ciktilar/
├── modeller/
│   ├── xgboost_20251210_120000.pkl      # Model
│   └── xgboost_20251210_120000.json     # Metadata
├── raporlar/
│   └── rapor_xgboost_20251210_120000.txt
└── gorseller/
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── ozellik_onemi_xgboost.png
    ├── model_karsilastirma.png
    └── model_radar_chart.png
```

## 📊 Örnek Kullanım Senaryosu

```bash
# 1. Model eğit (otomatik mod)
python train.py --auto --model xgboost

# 2. Test görüntüsü ile tahmin yap
python inference.py --image ../Veri_Seti/NonDemented/test.jpg

# 3. Toplu tahmin
python inference.py --batch ../Veri_Seti/NonDemented/

# 4. Birden fazla model eğit ve karşılaştır
python train.py --auto --model xgboost
python train.py --auto --model lightgbm
python model_comparison.py
```

## ⚙️ Yapılandırma

Tüm ayarlar `ayarlar.py` dosyasında:

- **Veri bölme oranları** (train/val/test)
- **Model hiperparametreleri** (XGBoost, LightGBM, SVM)
- **Grid search parametreleri**
- **Görselleştirme ayarları**
- **Dosya yolları**

## 🐛 Sorun Giderme

### CSV bulunamadı hatası:
```bash
cd ../goruntu_isleme
python ana_islem.py
# Menüden 6'yı seç (tüm işlemleri yap)
```

### SMOTE hatası:
```bash
pip install imbalanced-learn
```

### XGBoost/LightGBM yüklü değil:
```bash
pip install xgboost lightgbm
```

## 📚 Dosyalar

- `train.py` - Ana eğitim scripti (kullanıcı dostu)
- `model_egitici.py` - Model eğitim sınıfı (core)
- `inference.py` - Tahmin scripti
- `model_comparison.py` - Model karşılaştırma
- `ayarlar.py` - Yapılandırma dosyası

## 💡 İpuçları

1. İlk eğitimde **otomatik mod** kullanın: `python train.py --auto`
2. **SMOTE** veri dengeleme için önemlidir (varsayılan açık)
3. **Hyperparameter tuning** çok uzun sürer, ilk denemede kapalı tutun
4. **Model karşılaştırma** ile en iyi modeli seçin
5. **Inference** için en son eğitilen model otomatik kullanılır
