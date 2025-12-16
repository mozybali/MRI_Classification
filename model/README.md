# Model Eğitim Modülü

Ön işlenmiş MRI görüntülerinden çıkarılan özelliklerle XGBoost, LightGBM veya Linear SVM modelleri eğitir; metrikleri raporlar ve eğitilmiş modellerle tek/batch tahmin yapar.

## Gereksinimler

Ana dizindeki `requirements.txt` tüm bağımlılıkları içerir. Eğitim için `goruntu_isleme/cikti/goruntu_ozellikleri_scaled.csv` dosyasının hazır olması gerekir.

## Kullanım

### Eğitim
```bash
# Otomatik mod (varsayılan ayarlarla)
python train.py --auto

# İnteraktif mod
python train.py

# Model seçerek otomatik mod
python train.py --auto --model xgboost   # veya lightgbm, svm
```
Eğitim çıktıları `model/ciktilar/` altına kaydedilir (`modeller/`, `raporlar/`, `gorseller/`).

### Tahmin
```bash
# Tek görüntü
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --image /path/to/image.jpg

# Klasör içi batch
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --batch /path/to/folder/
```
Tahmin sırasında görüntü ön işlemesi ve özellik çıkarımı otomatik yapılır; sonuçlar ekrana yazılır ve CSV olarak kaydedilebilir.

### Model karşılaştırma
```bash
python model_comparison.py
```
`modeller/` klasöründeki kayıtlı modellerin performanslarını yan yana raporlar.

## Özellikler

- SMOTE ile dengesiz sınıfları dengeleme, sınıf ağırlıklandırma.  
- İsteğe bağlı özellik seçimi (SelectKBest) ve grid search.  
- Stratified train/val/test bölme ve 5 katlı cross-validation.  
- Değerlendirme metrikleri: accuracy, precision, recall, F1, ROC-AUC, Cohen’s kappa.  
- Rapor ve görseller: confusion matrix, ROC ve precision-recall eğrileri, model destekliyorsa feature importance grafiği.  
- Model + metadata kaydı (`.pkl` + `.json`) ve zaman damgalı dosya adları.

## Yapılandırma

`ayarlar.py` üzerinden:
- Veri yolları ve split oranları (`EGITIM_ORANI`, `DOGRULAMA_ORANI`, `TEST_ORANI`)
- Model hiperparametreleri (`GB_AYARLARI`, `LIGHTGBM_AYARLARI`, `SVM_AYARLARI`)
- Grid search parametreleri (`GB_GRID_PARAMS`, `SVM_GRID_PARAMS`)
- Çıktı klasörleri ve log ayarları

## Sorun Giderme

- **CSV bulunamadı**: `goruntu_isleme/ana_islem.py` ile 7. seçeneği çalıştırıp özellik CSV'lerini oluşturun.  
- **Paket eksik uyarıları**: Ana dizinde `pip install -r requirements.txt`.  
- **LightGBM/XGBoost yok**: Eksik paketleri ayrıca kurabilirsiniz (`pip install xgboost lightgbm`).
