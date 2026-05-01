# MRI Beyin Görüntüsü Sınıflandırma Projesi

Bu depo, 2D beyin MRI görüntülerinden demans seviyesini sınıflandırmak için hazırlanmış uçtan uca bir Python çalışma alanıdır. Akış; keşifsel veri analizi, görüntü ön işleme, leak-free `trainval/test` üretimi, ResNet/PyTorch ve XGBoost eğitimi, Optuna ile hiperparametre araması, inference ve `pytest` testlerini tek proje içinde toplar.

Desteklenen sınıflar:

| Sınıf klasörü | Etiket |
| --- | ---: |
| `NonDemented` | 0 |
| `VeryMildDemented` | 1 |
| `MildDemented` | 2 |
| `ModerateDemented` | 3 |

Sınıf klasörü adları kod içinde sabit kullanılır ve büyük/küçük harfe duyarlıdır.

## İçindekiler

- [Proje Akışı](#proje-akışı)
- [Dizin Yapısı](#dizin-yapısı)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Hızlı Kullanım](#hızlı-kullanım)
- [Modüller](#modüller)
- [Testler](#testler)
- [Notlar](#notlar)
- [Lisans](#lisans)

## Proje Akışı

Önerilen çalışma sırası:

1. Ham görüntüleri `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutun.
2. `eda_analiz/` ile sınıf dağılımı, görüntü boyutu ve yoğunluk istatistiklerini inceleyin.
3. `goruntu_isleme/` ile ham görüntüleri kalite kontrolden geçirip standartlaştırın ve `trainval/test` ayrımı üretin.
4. `model/` ile işlenmiş görüntüler üzerinden ResNet veya XGBoost modeli eğitin.
5. Gerekirse `mri-tune` ile Optuna TPE tabanlı hiperparametre araması çalıştırın.
6. Eğitilmiş `.pt` veya `.json` model dosyasıyla tek görüntü ya da klasör tahmini alın.
7. Değişiklikleri `pytest` testleriyle doğrulayın.

Model eğitimi için önerilen giriş ham veri değil, `mri-preprocess` sonrasında oluşan şu dizinlerdir:

- `goruntu_isleme/cikti/trainval`
- `goruntu_isleme/cikti/test`

## Dizin Yapısı

```text
MRI_Classification/
|-- eda_analiz/
|   |-- __main__.py
|   |-- eda_araclar.py
|   |-- eda_calistir.py
|   `-- README.md
|-- goruntu_isleme/
|   |-- ana_islem.py
|   |-- artirma.py
|   |-- ayarlar.py
|   |-- goruntu_isleyici.py
|   |-- kalite_io.py
|   |-- on_isleme.py
|   |-- temel.py
|   |-- toplu_islem.py
|   |-- veri.py
|   `-- README.md
|-- model/
|   |-- ayarlar.py
|   |-- hpo.py
|   |-- inference.py
|   |-- train.py
|   |-- common/
|   |-- dl/
|   |-- sl/
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
|   |-- test_pipeline.py
|   `-- README.md
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
|-- requirements-torch-cpu.txt
|-- requirements-torch-cu128.txt
|-- LICENSE
`-- README.md
```

Yerel veri ve çalışma çıktıları depoya dahil edilmez. `.gitignore` içinde özellikle şu yollar dışarıda bırakılır:

- `Veri_Seti/`
- `eda_analiz/eda_ciktilar/`
- `goruntu_isleme/cikti/`
- `model/ciktilar/`
- model ağırlıkları ve sayısal cache dosyaları (`*.pt`, `*.pth`, `*.joblib`, `*.pkl`, `*.npy`, `*.npz`)

## Veri Seti

Beklenen ham veri yapısı:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Desteklenen görüntü uzantıları:

- `.jpg`
- `.jpeg`
- `.png`

`Veri_Seti/OriginalDataset` ham görüntü kaynağı olarak korunmalıdır. Ön işlenmiş görüntüler, raporlar, özellik cache'leri ve model çıktıları bu klasöre yazılmaz.

## Kurulum

Python `3.10+` gerekir. Komutları proje kök dizininden çalıştırın.

Sanal ortam:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
python -m pip install -U pip
```

Ortak bağımlılıklar:

```bash
pip install -r requirements.txt
```

PyTorch CPU kurulumu:

```bash
pip install -r requirements-torch-cpu.txt
```

GPU/CUDA ortamı için proje içindeki ayrı PyTorch dosyası:

```bash
pip install -r requirements-torch-cu128.txt
```

CLI komutlarını kullanılabilir yapmak için:

```bash
pip install -e . --no-deps
```

Geliştirme araçları `pyproject.toml` üzerinden kurulacaksa:

```bash
pip install -e ".[dev]" --no-deps
```

Tanımlı komutlar:

| Komut | Görev |
| --- | --- |
| `mri-eda` | Keşifsel veri analizi üretir. |
| `mri-preprocess` | Görüntü ön işleme ve `trainval/test` split üretimi yapar. |
| `mri-train` | ResNet veya XGBoost modeli eğitir. |
| `mri-tune` | Optuna ile hiperparametre araması çalıştırır. |
| `mri-infer` | Eğitilmiş modelle tahmin alır. |

## Hızlı Kullanım

Uçtan uca tipik akış:

```bash
# 1. Veri setini incele
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar

# 2. Görüntüleri işle ve leak-free trainval/test ayrımı üret
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti

# 3. ResNet eğit
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test

# 4. Alternatif olarak XGBoost eğit
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler

# 5. Eğitilmiş modelle tahmin al
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

Paket kurulmadan modül olarak çalıştırma:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
python3 -m goruntu_isleme.ana_islem --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
python3 -m model.train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
python3 -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Modüller

### EDA Analizi

`eda_analiz/` modülü veri setini değiştirmeden rapor, grafik ve CSV çıktıları üretir. Sınıf dağılımı, görüntü boyutları, yoğunluk istatistikleri, korelasyon matrisi ve PCA görünümü sağlar.

Örnek:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar --jobs 1
```

Üretilen ana çıktı:

- `eda_analiz/eda_ciktilar/veri_seti_istatistikler.csv`
- `0_ozet_istatistikler.txt`
- analiz grafikleri (`1_sinif_dagilimi.png`, `2_boyut_analizi.png`, vb.)

Ayrıntılar: [`eda_analiz/README.md`](eda_analiz/README.md)

### Görüntü İşleme

`goruntu_isleme/` modülü ham 2D MRI görüntülerini model eğitimine hazırlar. Varsayılan akışta OpenCV ile görüntüleri okur, kalite kontrol uygular, gri ton ve yoğunluk standartlaştırması yapar, CLAHE ve `256x256` boyutlandırma uygular, ardından `trainval/test` yapısını üretir.

Örnek:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Etkileşimli menü:

```bash
mri-preprocess --action menu
```

Beklenen çıktı:

```text
goruntu_isleme/cikti/
|-- trainval/
|   |-- NonDemented/
|   |-- VeryMildDemented/
|   |-- MildDemented/
|   `-- ModerateDemented/
`-- test/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

`mri-preprocess` için desteklenen aksiyonlar yalnızca `menu` ve `preprocess` değerleridir. CSV, scaler, XGBoost özellik cache'i ve model dosyaları bu modülün görevi değildir.

Ayrıntılar: [`goruntu_isleme/README.md`](goruntu_isleme/README.md)

### Model Eğitimi

`model/` modülü iki model ailesini destekler:

| Model | `--model` değeri | Ana çıktı |
| --- | --- | --- |
| ResNet18/PyTorch | `resnet` | `.pt` checkpoint |
| XGBoost | `xgboost` | `.json` model ve `.meta.json` metadata |

ResNet:

```bash
mri-train --model resnet --epochs 50 --batch-size 32 --lr 1e-4 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

K-fold cross-validation:

```bash
mri-train --model resnet --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Final eğitim modu:

```bash
mri-train --model resnet --full-trainval --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Model çıktıları varsayılan olarak `model/ciktilar/` altında tutulur:

- `model/ciktilar/modeller/`
- `model/ciktilar/raporlar/`
- `model/ciktilar/gorseller/`
- `model/ciktilar/sl_ozellikler/`
- `model/ciktilar/hiperparametre_arama/`

Ayrıntılar: [`model/README.md`](model/README.md)

### Hiperparametre Araması

`mri-tune`, Optuna TPE tabanlı arama yapar. Desteklenen seçim metrikleri arasında `loss`, `accuracy`, `precision`, `recall` ve `f1` bulunur.

ResNet HPO:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost HPO:

```bash
mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Final eğitim atlanacaksa:

```bash
mri-tune --model resnet --trials 20 --skip-final-train
```

### Inference

`mri-infer`, ResNet `.pt` checkpoint'lerini ve XGBoost `.json` modellerini destekler.

Tek görüntü:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

Klasör bazlı tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
```

Ham görüntü üzerinde tahmin öncesi aynı MRI ön işleme hattını uygulamak için:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg --preprocess
```

## Testler

Test altyapısı `pytest` kullanır. Testler EDA, görüntü işleme, CLI davranışı, XGBoost hattı, Torch/ResNet altyapısı ve regresyon kontrollerini kapsar.

Tüm testler:

```bash
pytest
```

Yavaş, gerçek veri veya GPU gerektiren testleri dışarıda bırakmak için:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

Torch bağımlı test dosyaları varsayılan olarak atlanır. Açıkça çalıştırmak için:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Yardımcı pipeline kontrolleri:

```bash
python3 tests/pipeline_quick_test.py
python3 tests/test_pipeline.py
```

Ayrıntılar: [`tests/README.md`](tests/README.md)

## Notlar

- `mri-preprocess` sonrasında oluşan `test` dizinine augmentation uygulanmamalıdır.
- Eğitim sırasında validation ve test ayrımları ayrı tutulmalıdır; `--full-trainval` yalnızca final eğitim için kullanılmalıdır.
- XGBoost özellik cache'i veri dizini ve `image_size` bilgisiyle doğrulanır; farklı veri veya boyutla aynı cache kullanılmamalıdır.
- ResNet hattında yatay çevirme varsayılan olarak kapalıdır; beyin MR görüntülerinde anatomik lateralite bilgi taşıyabilir.
- Bu proje 2D görüntü dosyalarıyla çalışır; doğrudan `.nii` veya `.nii.gz` hacim dosyası akışı ana yol değildir.

## Lisans

Bu proje MIT lisansı ile yayımlanmıştır. Ayrıntılar için [`LICENSE`](LICENSE) dosyasına bakın.
