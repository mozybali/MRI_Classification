# MRI Beyin Goruntusu Siniflandirma

Bu repo, MRI beyin goruntulerinden demans seviyesini siniflandirmak icin hazirlanmis uctan uca bir calisma ortami sunar. Proje; kesifsel veri analizi (EDA), goruntu on isleme, ozellik cikarma, derin ogrenme ile egitim ve inference adimlarini tek yerde toplar.

## Moduller

- [`eda_analiz/`](eda_analiz/README.md): Veri seti dagilimi, boyut, yogunluk, korelasyon ve PCA analizleri
- [`goruntu_isleme/`](goruntu_isleme/README.md): On isleme, ozellik cikarma, CSV uretimi, veri bolme ve olceklendirme
- [`model/`](model/README.md): ResNet ve U-Net tabanli egitim ve inference akislari

## Proje Yapisi

```text
MRI_Classification/
|-- Veri_Seti/
|   |-- AugmentedAlzheimerDataset/
|   `-- OriginalDataset/
|-- eda_analiz/
|-- goruntu_isleme/
|-- model/
|-- tests/
|-- pyproject.toml
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
pip install -e .
```

Alternatif olarak:

```bash
pip install -r requirements.txt
```

`pip install -e .` kullanildiginda su komutlar aktif olur:

- `mri-eda`
- `mri-preprocess`
- `mri-train`
- `mri-infer`

## Veri Yapisi

Varsayilan klasor yapisi:

```text
Veri_Seti/
|-- AugmentedAlzheimerDataset/
|   |-- NonDemented/
|   |-- VeryMildDemented/
|   |-- MildDemented/
|   `-- ModerateDemented/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Sinif adlari proje boyunca aynidir:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

Egitim akisinda varsayilan politika:

- `AugmentedAlzheimerDataset`: train + validation
- `OriginalDataset`: test

Bu ayrim, augment edilmis verilerle egitim yaparken gercek performansi original veri uzerinde olcmek icin kullanilir.

## Hizli Baslangic

### 1. EDA

```bash
mri-eda --interactive
```

Alternatif:

```bash
python -m eda_analiz.eda_calistir --interactive
```

### 2. Goruntu on isleme ve ozellik cikarma

Interaktif menu:

```bash
mri-preprocess --action menu
```

Tek komutta tum temel 2D akis:

```bash
mri-preprocess --action all --input-dir Veri_Seti/AugmentedAlzheimerDataset --output-dir goruntu_isleme/cikti --yes
```

### 3. Model egitimi

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model unet --epochs 50 --batch-size 16
```

Alternatif:

```bash
python -m model.train --model resnet --epochs 50 --batch-size 32
```

### 4. Tahmin

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Teknik Ozet

- Framework: PyTorch
- Modeller: ResNet18 ve U-Net encoder tabanli siniflandirici
- Metrikler: Accuracy, precision, recall, F1 (macro)
- Egitim ozellikleri: early stopping, best checkpoint, ReduceLROnPlateau, class weights, focal loss
- Veri guvenligi: augment turevlerini ayni grup icinde tutan split mantigi
- Cihaz secimi: CUDA varsa GPU, yoksa CPU

## Ciktilar

- `eda_analiz/eda_ciktilar/`: EDA raporlari ve grafikler
- `goruntu_isleme/cikti/`: Islenmis goruntuler, ozellik CSV'leri ve scaler dosyalari
- `model/ciktilar/modeller/`: Egitilmis `.pt` checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari
- `model/ciktilar/gorseller/`: Confusion matrix ve egitim egrileri

## Test

```bash
pytest
```

Veri ya da GPU gerektiren testleri filtrelemek icin:

```bash
pytest -m "not requires_data and not requires_gpu"
```

## Lisans

MIT. Ayrinti icin `LICENSE` dosyasina bakin.
