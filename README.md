# MRI Beyin Goruntusu Siniflandirma

Bu repo, MRI beyin goruntulerinden demans seviyesini siniflandirmaya yonelik uctan uca bir Python projesidir. Kod tabani; kesifsel veri analizi, 2D goruntu on isleme, ozellik cikarma, PyTorch (ResNet) ve XGBoost ile model egitimi, inference ve test altyapisini ayni yerde toplar.

## Moduller

- [`eda_analiz/README.md`](eda_analiz/README.md): Veri seti dagilimi, boyut, yogunluk, korelasyon ve PCA analizleri
- [`goruntu_isleme/README.md`](goruntu_isleme/README.md): On isleme, ozellik cikarma, veri bolme ve olceklendirme akisi
- [`model/README.md`](model/README.md): ResNet (derin ogrenme) ve XGBoost (sig ogrenme) tabanli egitim ve inference komutlari
- [`tests/README.md`](tests/README.md): Pytest duzeni, test dosyalari ve calistirma ornekleri
- [`Veri_Seti/README.md`](Veri_Seti/README.md): Beklenen veri klasor yapisi ve varsayilan veri politikasi

## Proje Yapisi

```text
MRI_Classification/
|-- Veri_Seti/
|   |-- OriginalDataset/
|   `-- README.md
|-- eda_analiz/
|   |-- eda_araclar.py
|   |-- eda_calistir.py
|   `-- README.md
|-- goruntu_isleme/
|   |-- ana_islem.py
|   |-- ayarlar.py
|   |-- goruntu_isleyici.py
|   |-- ozellik_cikarici.py
|   `-- README.md
|-- model/
|   |-- train.py
|   |-- hpo.py
|   |-- inference.py
|   |-- ayarlar.py
|   |-- training_runner.py
|   |-- common/
|   |-- dl/
|   |-- sl/
|   `-- README.md
|-- tests/
|   |-- conftest.py
|   |-- pipeline_quick_test.py
|   |-- test_*.py
|   `-- README.md
|-- pyproject.toml
|-- pytest.ini
|-- requirements-torch-cpu.txt
|-- requirements-torch-cu128.txt
|-- requirements.txt
`-- README.md
```

Not: `goruntu_isleme/cikti/`, `model/ciktilar/` ve `eda_analiz/eda_ciktilar/` calisma sirasinda uretilen cikti dizinleridir.

## Gereksinimler

- Python 3.10+
- `pip`
- Istege bagli olarak CUDA destekli PyTorch ortami

## Kurulum

Onerilen kurulum:

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

Not:

- `requirements.txt` artik PyTorch disindaki ortak bagimliliklari icerir.
- PyTorch wheel'i ayrica kurulur; boylece `torch` paketinin CPU build'e kaymasi engellenir.
- CUDA surumunu degistirmeniz gerekirse `requirements-torch-cu128.txt` icindeki PyTorch index dosyasini, resmi PyTorch kurulum sayfasindaki komuta gore guncelleyin.

`pip install -e .` veya `pip install -e .[dev]` sonrasinda su komutlar aktif olur:

- `mri-eda`
- `mri-preprocess`
- `mri-train`
- `mri-infer`
- `mri-tune`

## Hizli Baslangic

### 1. EDA

```bash
mri-eda --interactive
```

### 2. Goruntu on isleme ve ozellik cikarma

Interaktif menu:

```bash
mri-preprocess --action menu
```

Tam 2D akis:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

### 3. Model egitimi

ResNet (derin ogrenme):

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model resnet
```

XGBoost (sig ogrenme):

```bash
mri-train --model xgboost
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8
```

### 3.1 Bayes search ile hiperparametre optimizasyonu

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model xgboost --trials 30 --metric f1
```

### 4. Tahmin

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

## Varsayilan Veri Politikasi

- Varsayilan egitim kaynagi: `goruntu_isleme/cikti/trainval`
- Varsayilan test kaynagi: `goruntu_isleme/cikti/test`
- `Veri_Seti/OriginalDataset`: ham/original veri dizini
- `goruntu_isleme` ham veriyi leak-free sekilde `trainval/test` olarak ayirir
- `validation` ve `test`: original-only olarak uretilir
- Augmentation train/trainval tarafinda kalir; validation ve test original-only olarak kullanilir

Proje akisi varsayilan olarak preprocess ciktilari uzerinden ilerler. Ham veriyle calismak isterseniz egitim komutunda veri dizinlerini acikca belirtin. Ayrintilar icin [`Veri_Seti/README.md`](Veri_Seti/README.md) dosyasina bakin.

## Ciktilar

- `eda_analiz/eda_ciktilar/`: EDA grafik ve tablo ciktilari
- `goruntu_isleme/cikti/`: `trainval/`, `test/`, ozellik CSV'leri, split dosyalari ve scaler
- `model/ciktilar/modeller/`: Egitilmis `.pt` (ResNet) ve `.json` (XGBoost) checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari
- `model/ciktilar/gorseller/`: Confusion matrix, normalize confusion matrix, sinif bazli performans, guven grafikleri, ROC/PR, egitim dashboard'lari ve XGBoost feature importance
- `model/ciktilar/hiperparametre_arama/`: Optuna TPE tabanli Bayes search trial, study ve HPO analiz grafikleri

## Test

Tum testler:

```bash
pytest
```

Veri veya GPU gerektirenleri disarida birakmak icin:

```bash
pytest -m "not requires_data and not requires_gpu"
```

Test duzeni icin [`tests/README.md`](tests/README.md) dosyasina bakin.

## Lisans

MIT. Ayrinti icin `LICENSE` dosyasina bakin.
