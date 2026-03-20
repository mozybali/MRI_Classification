# MRI Beyin Goruntusu Siniflandirma

Bu repo, MRI beyin goruntulerinden demans seviyesini siniflandirmaya yonelik uctan uca bir Python projesidir. Kod tabani; kesifsel veri analizi, 2D goruntu on isleme, ozellik cikarma, PyTorch ile model egitimi, inference ve test altyapisini ayni yerde toplar.

## Moduller

- [`eda_analiz/README.md`](eda_analiz/README.md): Veri seti dagilimi, boyut, yogunluk, korelasyon ve PCA analizleri
- [`goruntu_isleme/README.md`](goruntu_isleme/README.md): On isleme, ozellik cikarma, veri bolme ve olceklendirme akisi
- [`model/README.md`](model/README.md): ResNet ve U-Net tabanli egitim ve inference komutlari
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
|   |-- goruntu_isleyici.py
|   |-- ozellik_cikarici.py
|   |-- cikti/
|   `-- README.md
|-- model/
|   |-- train.py
|   |-- inference.py
|   |-- dl/
|   |-- ciktilar/
|   `-- README.md
|-- tests/
|   |-- conftest.py
|   |-- test_*.py
|   `-- README.md
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
`-- README.md
```

## Gereksinimler

- Python 3.10+
- `pip`
- Istege bagli olarak CUDA destekli PyTorch ortami

## Kurulum

Onerilen kurulum:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
```

Alternatif:

```bash
pip install -r requirements.txt
```

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

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model unet --epochs 50 --batch-size 16
mri-train --model resnet --use-processed-trainval --trainval-dir goruntu_isleme/cikti
```

### 3.1 Bayes search ile hiperparametre optimizasyonu

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model unet --trials 30 --metric loss --skip-final-train
```

### 4. Tahmin

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Varsayilan Veri Politikasi

- `Veri_Seti/OriginalDataset`: tek kaynak veri dizini
- `train`, `validation` ve `test`: ayni original veri kaynagindan uretilir
- Train tarafinda yalnizca transform tabanli augmentation uygulanir

Proje akisi `OriginalDataset` uzerine sabitlenmistir. Ayrintilar icin [`Veri_Seti/README.md`](Veri_Seti/README.md) dosyasina bakin.

## Ciktilar

- `eda_analiz/eda_ciktilar/`: EDA grafik ve tablo ciktilari
- `goruntu_isleme/cikti/`: Islenmis goruntuler, ozellik CSV'leri, split dosyalari ve scaler
- `model/ciktilar/modeller/`: Egitilmis `.pt` checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari
- `model/ciktilar/gorseller/`: Confusion matrix ve egitim egrileri
- `model/ciktilar/hiperparametre_arama/`: Optuna TPE tabanli Bayes search trial ve study ciktilari

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
