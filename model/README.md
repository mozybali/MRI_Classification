# Model Egitim Modulu

On islenmis MRI goruntulerinden cikarilan ozelliklerle XGBoost, LightGBM veya Linear SVM modelleri egitir; raporlar uretir ve egitilmis modellerle tahmin yapar.

## Gereksinimler

Ana dizindeki `requirements.txt` tum bagimliliklari icerir. Egitim icin su dosyalarin hazir olmasi onerilir:

- `goruntu_isleme/cikti/egitim_scaled.csv`
- `goruntu_isleme/cikti/dogrulama_scaled.csv`
- `goruntu_isleme/cikti/test_scaled.csv`

## Kullanim

### Egitim

```bash
python train.py --auto
python train.py
python train.py --auto --model xgboost
```

### Tahmin

```bash
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --image /path/to/image.jpg
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --batch /path/to/folder/
```

### Model karsilastirma

```bash
python model_comparison.py
```

## Teknik Notlar

- Olcekleme train setine gore yapilir; validation ve test ayni scaler ile donusturulur.
- `boyut_bayt`, `genislik`, `yukseklik`, `en_boy_orani`, `piksel_sayisi` gibi meta sayisal kolonlar model girdisinden cikarilir.
- Split mantigi kaynak gruplarini kullanir; ayni goruntunun turevleri farkli setlere dagilmamaya calisir.

## Ozellikler

- SMOTE ile dengesiz siniflari dengeleme
- Sinif agirliklandirma
- Istege bagli ozellik secimi
- Grid/random search
- 5 katli cross-validation
- Accuracy, precision, recall, F1, ROC-AUC, Cohen's kappa
- Confusion matrix, ROC, precision-recall ve feature importance gorselleri
- Model ve metadata kaydi

## Yapilandirma

`ayarlar.py` uzerinden:
- veri yollari ve split dosyalari
- model hiperparametreleri
- grid search parametreleri
- log ve cikti klasorleri

## Sorun Giderme

- Split CSV'ler yoksa `goruntu_isleme/ana_islem.py` icinden `7` numarali akisi calistirin.
- Paket eksigi varsa ana dizinde `pip install -r requirements.txt`.
- LightGBM veya XGBoost kurulu degilse ilgili paketi ayrica kurun.
