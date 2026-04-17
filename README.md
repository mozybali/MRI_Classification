# MRI Beyin Goruntusu Siniflandirma

Bu repo, MRI beyin goruntulerinden demans seviyesini siniflandirmaya yonelik uctan uca bir Python projesidir. Kod tabani; kesifsel veri analizi (EDA), 2D goruntu on isleme, ozellik cikarma, PyTorch (ResNet) ve XGBoost tabanli model egitimi, Optuna TPE ile Bayesian hiperparametre arama, inference ve pytest altyapisini tek cati altinda toplar.

## Moduller

- [`eda_analiz/README.md`](eda_analiz/README.md): Sinif dagilimi, boyut, yogunluk, korelasyon ve PCA analizleri
- [`goruntu_isleme/README.md`](goruntu_isleme/README.md): On isleme, ozellik cikarma, leak-free bolme ve olceklendirme akisi
- [`model/README.md`](model/README.md): ResNet (derin ogrenme) ve XGBoost (sig ogrenme) egitim, HPO ve inference komutlari
- [`tests/README.md`](tests/README.md): Pytest duzeni, marker'lar, test dosyalari ve calistirma ornekleri
- [`Veri_Seti/README.md`](Veri_Seti/README.md): Beklenen veri klasor yapisi ve varsayilan veri politikasi

## Proje Yapisi

```text
MRI_Classification/
|-- Veri_Seti/
|   |-- OriginalDataset/
|   `-- README.md
|-- eda_analiz/
|   |-- __init__.py
|   |-- __main__.py
|   |-- eda_araclar.py
|   |-- eda_calistir.py
|   `-- README.md
|-- goruntu_isleme/
|   |-- __init__.py
|   |-- ana_islem.py
|   |-- ayarlar.py
|   |-- goruntu_isleyici.py
|   |-- ozellik_cikarici.py
|   `-- README.md
|-- model/
|   |-- __init__.py
|   |-- ayarlar.py
|   |-- train.py
|   |-- hpo.py
|   |-- inference.py
|   |-- training_runner.py
|   |-- common/
|   |   `-- evaluation.py
|   |-- dl/
|   |   |-- dataset.py
|   |   |-- engine.py
|   |   |-- losses.py
|   |   |-- utils.py
|   |   `-- models/
|   |       `-- resnet_classifier.py
|   |-- sl/
|   |   |-- dataset.py
|   |   |-- features.py
|   |   |-- training_runner.py
|   |   `-- xgb_classifier.py
|   `-- README.md
|-- tests/
|   |-- conftest.py
|   |-- pipeline_quick_test.py
|   |-- test_akislari_ve_cli.py
|   |-- test_bugfixes.py
|   |-- test_eda_araclar.py
|   |-- test_goruntu_isleyici.py
|   |-- test_model_altyapi.py
|   |-- test_model_egitici.py
|   |-- test_model_sl.py
|   |-- test_ozellik_cikarici.py
|   |-- test_pipeline.py
|   `-- README.md
|-- LICENSE
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
|-- requirements-torch-cpu.txt
|-- requirements-torch-cu128.txt
`-- README.md
```

Not: `eda_analiz/eda_ciktilar/`, `goruntu_isleme/cikti/` ve `model/ciktilar/` calisma sirasinda uretilen cikti dizinleridir ve `.gitignore` ile repo disinda tutulur.

## Gereksinimler

- Python 3.10+
- `pip`
- Istege bagli olarak CUDA destekli PyTorch ortami (varsayilan: CUDA 12.8)

## Kurulum

GPU (CUDA 12.8) kurulumu:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
pip install -e .[dev] --no-deps
```

CPU-only kurulum:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cpu.txt
pip install -e .[dev] --no-deps
```

Notlar:

- `requirements.txt` PyTorch disindaki ortak bagimliliklari (numpy, pandas, scikit-learn, xgboost, optuna, scikit-image, SimpleITK, matplotlib, seaborn, tqdm, pytest) icerir.
- PyTorch wheel'i ayri bir dosyadan kurulur; boylece `torch` paketinin yanlis backend'e kaymasi engellenir.
- Farkli bir CUDA surumu kullanacaksaniz `requirements-torch-cu128.txt` icindeki PyTorch index URL'sini resmi PyTorch kurulum sayfasindaki komuta gore guncelleyin.

`pip install -e .` veya `pip install -e .[dev]` sonrasinda asagidaki CLI komutlari aktif olur:

- `mri-eda`         → EDA analizleri ([eda_analiz.eda_calistir:main](eda_analiz/eda_calistir.py))
- `mri-preprocess`  → Goruntu on isleme ve ozellik akisi ([goruntu_isleme.ana_islem:main](goruntu_isleme/ana_islem.py))
- `mri-train`       → Model egitimi ([model.train:main](model/train.py))
- `mri-tune`        → Optuna TPE ile hiperparametre arama ([model.hpo:main](model/hpo.py))
- `mri-infer`       → Tek goruntu veya klasor bazli tahmin ([model.inference:main](model/inference.py))

## Hizli Baslangic

### 1. EDA

```bash
mri-eda --interactive
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar
```

### 2. Goruntu on isleme ve ozellik cikarma

Interaktif menu:

```bash
mri-preprocess --action menu
```

Tam 2D akis (`preprocess -> extract -> scale -> report`):

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

### 3. Model egitimi

ResNet (derin ogrenme):

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model resnet --loss focal --lr 3e-4 --pretrained
mri-train --model resnet --full-trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost (sig ogrenme):

```bash
mri-train --model xgboost
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8
mri-train --model xgboost --feature-cache model/ciktilar/sl_ozellikler
```

### 3.1 Bayes search ile hiperparametre optimizasyonu

Desteklenen `--metric` degerleri: `loss`, `accuracy`, `precision`, `recall`, `f1`.

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model resnet --trials 30 --metric loss --search-pretrained --skip-final-train
mri-tune --model xgboost --trials 30 --metric f1
mri-tune --model xgboost --trials 50 --metric accuracy --feature-cache model/ciktilar/sl_ozellikler
```

### 4. Tahmin

Model tipi dosya uzantisindan otomatik algilanir (`.pt` → ResNet, `.json` → XGBoost):

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
```

## Varsayilan Veri Politikasi

- Varsayilan egitim kaynagi: `goruntu_isleme/cikti/trainval`
- Varsayilan test kaynagi: `goruntu_isleme/cikti/test`
- `Veri_Seti/OriginalDataset/<SinifAdi>/`: tek ham/original veri dizini
- `goruntu_isleme/preprocess`, ham veriyi once leak-free `trainval/test` olarak ayirir
- `validation` ve `test` her zaman original-only uretilir; augmentation yalnizca `trainval` tarafinda kalir
- `--full-trainval` modunda validation ayrilmaz, tum `trainval` uzerinde final egitim yapilir ve degerlendirme harici `test` dizini uzerinden tek seferlik tutulur

Proje akisi varsayilan olarak preprocess ciktilari uzerinden ilerler. Ham veriyle dogrudan calismak isterseniz egitim komutunda `--trainval-dir Veri_Seti/OriginalDataset` vererek orijinal klasoru hedefleyebilirsiniz. Ayrintilar icin [`Veri_Seti/README.md`](Veri_Seti/README.md) dosyasina bakin.

Sabit sinif adlari: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`.

## Ciktilar

- `eda_analiz/eda_ciktilar/`: EDA ozet istatistikleri, sinif dagilimi, boyut/yogunluk, korelasyon ve PCA gorselleri
- `goruntu_isleme/cikti/`: `trainval/`, `test/`, ozellik CSV'leri, olceklenmis CSV'ler, split dosyalari ve `feature_scaler.pkl`
- `model/ciktilar/modeller/`: Egitilmis `.pt` (ResNet) ve `.json` (XGBoost) checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari (timestamp'li)
- `model/ciktilar/gorseller/`: Confusion matrix, normalize confusion matrix, sinif bazli performans ozeti, tahmin guven dagilimi, ROC/PR egrileri, egitim dashboard'lari ve XGBoost feature importance
- `model/ciktilar/sl_ozellikler/`: XGBoost ozellik cache (`--feature-cache` ile)
- `model/ciktilar/hiperparametre_arama/<study>/`: Optuna TPE trial history, study ozeti, trial klasorleri, HPO analiz grafikleri ve `best_run/` (final egitim)

## Test

Tum testler:

```bash
pytest
```

Veri veya GPU gerektiren testleri disarida birakmak icin:

```bash
pytest -m "not requires_data and not requires_gpu"
```

Sadece hizli testler:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

`pytest.ini` icinde tanimli marker'lar: `unit`, `integration`, `slow`, `requires_data`, `requires_gpu`. Test duzeni icin [`tests/README.md`](tests/README.md) dosyasina bakin.

## Lisans

MIT. Ayrinti icin [`LICENSE`](LICENSE) dosyasina bakin.
