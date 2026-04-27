# MRI Beyin Görüntüsü Sınıflandırma Projesi

## Proje Amacı

Bu proje, beyin MRI görüntülerinden demans seviyesini sınıflandırmaya yönelik uçtan uca bir Python çalışma alanıdır. Veri seti inceleme, 2D görüntü ön işleme, özellik çıkarımı, ResNet/PyTorch ve XGBoost tabanlı model eğitimi, hiperparametre araması, inference ve test altyapısı aynı depo içinde düzenlenmiştir.

Proje dört sınıflı bir sınıflandırma problemi üzerinde çalışır:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

Bu README dosyası ana giriş noktasıdır. Modül bazlı ayrıntılar için ilgili alt README dosyalarına bakılmalıdır:

- [`Veri_Seti/README.md`](Veri_Seti/README.md)
- [`eda_analiz/README.md`](eda_analiz/README.md)
- [`goruntu_isleme/README.md`](goruntu_isleme/README.md)
- [`model/README.md`](model/README.md)
- [`tests/README.md`](tests/README.md)

## Genel Akış

Önerilen proje akışı aşağıdaki sırayı izler:

1. Ham MRI görüntüleri `Veri_Seti/OriginalDataset/<SınıfAdı>/` yapısında tutulur.
2. İsteğe bağlı olarak EDA modülü ile sınıf dağılımı, görüntü boyutları ve yoğunluk istatistikleri incelenir.
3. Görüntü işleme modülü ham görüntüleri okur, kalite kontrol ve standartlaştırma uygular, ardından `trainval` ve `test` dizinlerini üretir.
4. Özellik çıkarımı ve ölçeklendirme adımları, modelleme için CSV ve scaler çıktıları oluşturur.
5. Model modülü, işlenmiş görüntüler üzerinden ResNet veya XGBoost eğitir.
6. Hiperparametre optimizasyonu gerekiyorsa `mri-tune` ile Optuna TPE tabanlı arama yapılır.
7. Eğitilmiş model dosyaları ile tek görüntü veya klasör bazlı tahmin alınır.
8. `pytest` testleriyle EDA, görüntü işleme, model altyapısı ve CLI akışları doğrulanır.

Varsayılan ve önerilen model eğitim kaynağı ham veri değil, `mri-preprocess` sonrasında oluşan `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` dizinleridir.

## Klasör Yapısı

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
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
|-- requirements-torch-cpu.txt
|-- requirements-torch-cu128.txt
|-- LICENSE
`-- README.md
```

Çalışma sırasında üretilen `eda_analiz/eda_ciktilar/`, `goruntu_isleme/cikti/` ve `model/ciktilar/` dizinleri kaynak kodun parçası değildir; analiz, ön işleme, eğitim ve değerlendirme çıktıları bu dizinlerde tutulur.

## Veri Seti

Ham veri için beklenen yapı şöyledir:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

`Veri_Seti/OriginalDataset` ham ve orijinal görüntü kaynağı olarak korunmalıdır. Ön işlenmiş görüntüler, CSV dosyaları, scaler dosyaları, raporlar veya model çıktıları bu klasörün altına yazılmamalıdır.

Bu çalışma alanında doğrulanan veri dağılımı alt README dosyasında belirtilmiştir: toplam `6400` görüntü, dört sınıf altında `.jpg`, `.jpeg` ve `.png` uzantılarıyla okunur. Sınıf adları büyük/küçük harfe duyarlıdır ve kod içinde sabit olarak kullanılmaktadır.

Ayrıntılı veri politikası ve sınıf-etiket eşleşmeleri için [`Veri_Seti/README.md`](Veri_Seti/README.md) dosyasına bakın.

## Keşifsel Veri Analizi

`eda_analiz/` modülü, eğitimden önce veri setini incelemek için kullanılır. Sınıf dağılımı, görüntü boyutu, piksel yoğunluğu, korelasyon ve PCA çıktıları üretir. Kaynak görüntüleri değiştirmez; rapor, grafik ve CSV çıktıları varsayılan olarak `eda_analiz/eda_ciktilar/` altına yazılır.

Örnek kullanım:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar
```

Paket kurulmadan çalıştırmak için:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
```

Daha ayrıntılı açıklama için [`eda_analiz/README.md`](eda_analiz/README.md) dosyasına bakın.

## Görüntü İşleme

`goruntu_isleme/` modülü, ham 2D MRI görüntülerini modelleme akışına hazırlar. Varsayılan akışta görüntüler `Veri_Seti/OriginalDataset` altından okunur ve çıktılar `goruntu_isleme/cikti/` altına yazılır.

Başlıca adımlar:

- Görüntüleri sınıf klasörlerinden okuma.
- Kalite kontrol uygulama.
- Gri ton yükleme, percentile normalizasyon, CLAHE ve `256x256` yeniden boyutlandırma.
- Gerekirse `trainval/test` ayrımı üretme veya mevcut split yapısını koruma.
- İşlenmiş görüntülerden sayısal özellikler çıkarma.
- Eğitim, doğrulama ve test CSV dosyaları oluşturma.
- NaN doldurma ve scaler fit işlemlerini yalnızca eğitim verisi üzerinden yaparak veri sızıntısı riskini azaltma.

Tam akış:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

Etkileşimli menü:

```bash
mri-preprocess --action menu
```

`mri-preprocess` aksiyonları, varsayılan ayarlar, CSV çıktıları ve leakage-free split politikası için [`goruntu_isleme/README.md`](goruntu_isleme/README.md) dosyasına bakın.

## Model Eğitimi ve Kullanımı

`model/` modülü iki model ailesini destekler:

| Model | Komut değeri | Çıktı |
| --- | --- | --- |
| ResNet/PyTorch | `resnet` | `.pt` checkpoint |
| XGBoost | `xgboost` | `.json` model ve `.meta.json` metadata |

Varsayılan eğitim dizinleri:

- `goruntu_isleme/cikti/trainval`
- `goruntu_isleme/cikti/test`

ResNet eğitimi:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost eğitimi:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Hiperparametre araması:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Tahmin alma:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
```

Model çıktıları varsayılan olarak `model/ciktilar/` altında tutulur. Eğitim parametreleri, HPO seçenekleri, inference davranışı ve çıktı dosyaları için [`model/README.md`](model/README.md) dosyasına bakın.

## Testler

Test altyapısı `pytest` kullanır. Testler EDA, görüntü işleme, özellik çıkarımı, CLI davranışı, XGBoost hattı ve Torch/ResNet altyapısını kapsar. Testlerin önemli bir bölümü sentetik veri ve geçici dosyalarla çalışır; gerçek veri veya GPU isteyen senaryolar marker ve ortam değişkenleriyle ayrılmıştır.

Tüm testleri çalıştırmak için:

```bash
pytest
```

Yavaş, gerçek veri veya GPU gerektiren testleri dışarıda bırakmak için:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

Torch bağımlı testleri açıkça çalıştırmak için:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Test dosyalarının görevleri, marker açıklamaları ve yardımcı pipeline kontrolleri için [`tests/README.md`](tests/README.md) dosyasına bakın.

## Kurulum

Python `3.10+` gereklidir. Komutlar proje kök dizininden çalıştırılmalıdır.

Sanal ortam oluşturma:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Windows için aktivasyon:

```powershell
.venv\Scripts\activate
```

Ortak bağımlılıkları kurma:

```bash
pip install -r requirements.txt
```

CPU-only PyTorch kurulumu:

```bash
pip install -r requirements-torch-cpu.txt
```

CUDA 12.8 PyTorch kurulumu:

```bash
pip install -r requirements-torch-cu128.txt
```

CLI komutlarını kullanılabilir hale getirmek için:

```bash
pip install -e .[dev] --no-deps
```

`pyproject.toml` içinde tanımlı komutlar:

- `mri-eda`
- `mri-preprocess`
- `mri-train`
- `mri-tune`
- `mri-infer`

## Kullanım

Tipik uçtan uca kullanım:

```bash
# 1. Veri setini incele
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar

# 2. Ön işleme, özellik çıkarımı, ölçeklendirme ve rapor adımlarını çalıştır
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes

# 3. ResNet modeli eğit
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test

# 4. Alternatif olarak XGBoost modeli eğit
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler

# 5. Eğitilmiş modelle tahmin al
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

Paket komutları kullanılmadan modül bazlı çalıştırma da mümkündür:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
python3 -m goruntu_isleme.ana_islem --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
python3 -m model.train --model resnet
python3 -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Notlar / Gereksinimler

- Proje Python `3.10+` ile çalışacak şekilde tanımlanmıştır.
- `requirements.txt`, PyTorch dışındaki ortak bağımlılıkları içerir.
- PyTorch kurulumu ortam türüne göre `requirements-torch-cpu.txt` veya `requirements-torch-cu128.txt` üzerinden yapılır.
- Ham veri kaynağı `Veri_Seti/OriginalDataset` altında korunmalıdır.
- Model eğitimi için önerilen giriş `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` dizinleridir.
- Test setinde augmentation kullanılmamalıdır; augmentation etkinleştirilirse eğitim veya `trainval` tarafında kalmalıdır.
- Validation ve test ayrımları model seçimi ve final değerlendirme açısından ayrı tutulmalıdır.
- `--full-trainval` modu tüm `trainval` verisiyle final eğitim yapar ve değerlendirme için harici `test` dizini gerektirir.
- `mri-preprocess --mode 3d` argümanı CLI'da yer alsa da bu repo ağacında opsiyonel 3D modül dosyası bulunmadığından 2D akış varsayılan ve desteklenen ana yoldur.

## Lisans

Bu proje MIT lisansı ile yayımlanmıştır. Ayrıntılar için [`LICENSE`](LICENSE) dosyasına bakın.
