# Model Modülü

`model/` klasörü, Alzheimer/demans MRI görüntülerinden dört sınıflı sınıflandırma yapmak için kullanılan eğitim, hiperparametre optimizasyonu ve tahmin akışlarını içerir. Modül repo kökünden çalışacak şekilde tasarlanmıştır ve iki ayrı model hattı sunar:

- `resnet`: PyTorch ve `torchvision` tabanlı derin öğrenme hattı.
- `xgboost`: görüntülerden çıkarılan klasik özelliklerle çalışan sığ öğrenme hattı.

Beklenen sınıflar şunlardır: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`.

## Dizin Yapısı

```text
model/
|-- README.md
|-- __init__.py
|-- ayarlar.py
|-- train.py
|-- training_runner.py
|-- hpo.py
|-- inference.py
|-- common/
|   |-- __init__.py
|   `-- evaluation.py
|-- dl/
|   |-- __init__.py
|   |-- dataset.py
|   |-- engine.py
|   |-- losses.py
|   |-- utils.py
|   `-- models/
|       |-- __init__.py
|       `-- resnet_classifier.py
|-- sl/
|   |-- __init__.py
|   |-- dataset.py
|   |-- features.py
|   |-- training_runner.py
|   `-- xgb_classifier.py
`-- ciktilar/
    |-- modeller/
    |-- raporlar/
    |-- gorseller/
    |-- sl_ozellikler/
    `-- hiperparametre_arama/
```

`ciktilar/` eğitim, tahmin ve HPO sırasında oluşan çalışma çıktıları için kullanılır. Kaynak kodun parçası olan ana dosyalar `train.py`, `training_runner.py`, `hpo.py`, `inference.py`, `dl/`, `sl/` ve `common/` altındadır.

## Ana Bileşenler

| Yol | Görev |
| --- | --- |
| `ayarlar.py` | Varsayılan veri ve çıktı yollarını tanımlar. |
| `train.py` | `resnet` ve `xgboost` eğitimleri için ortak CLI girişidir. |
| `training_runner.py` | ResNet eğitim döngüsü, validasyon/test değerlendirmesi ve artifact üretimini yönetir. |
| `hpo.py` | Optuna TPE ile ResNet ve XGBoost hiperparametre araması yapar. |
| `inference.py` | `.pt` ve `.json` modelleriyle tek görüntü veya klasör üzerinde tahmin alır. |
| `common/evaluation.py` | Ortak sınıflandırma metrikleri ve detaylı değerlendirme raporları üretir. |
| `dl/` | PyTorch veri seti, augmentasyon, loss, eğitim motoru, görselleştirme ve ResNet modeli. |
| `sl/` | XGBoost için özellik çıkarımı, özellik matrisi, eğitim ve model kaydetme/yükleme araçları. |

## Veri Akışı

Önerilen akış, ham görüntüleri önce `goruntu_isleme` modülünden geçirip leak-free split üretmektir:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

Model modülünün varsayılan veri yolları `model/ayarlar.py` içinde tanımlıdır:

- `goruntu_isleme/cikti/trainval`: eğitim ve validasyon kaynağı.
- `goruntu_isleme/cikti/test`: varsa harici test kaynağı.
- `model/ciktilar`: model, rapor, görsel, özellik cache ve HPO çıktıları.

`--trainval-dir` verilirse eğitim/validasyon kaynağı bu dizinden okunur. `--test-dir` verilirse harici test seti olarak kullanılır. Bu argümanlar verilmezse yukarıdaki varsayılan işlenmiş split dizinleri denenir.

Ham veri üzerinde doğrudan eğitim yapmak mümkündür:

```bash
mri-train --model resnet --trainval-dir Veri_Seti/OriginalDataset
```

Ancak proje için önerilen yöntem, önce ön işleme ile `trainval` ve `test` ayrımını üretmek, sonra eğitimde bu dizinleri kullanmaktır.

## Kurulum

Bağımlılıklar repo kökünden kurulmalıdır. `requirements.txt` ortak paketleri içerir; PyTorch kurulumu CPU/GPU ortamına göre ayrı dosyadan yapılır.

CPU ortamı:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cpu.txt
pip install -e .[dev] --no-deps
```

CUDA 12.8 ortamı:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
pip install -e .[dev] --no-deps
```

Kurulumdan sonra kullanılan CLI komutları:

- `mri-train`: model eğitimi.
- `mri-tune`: hiperparametre optimizasyonu.
- `mri-infer`: eğitilmiş modelle tahmin.

Alternatif olarak komutlar Python modülü olarak da çalıştırılabilir:

```bash
python -m model.train --model resnet
python -m model.hpo --model xgboost --trials 20
python -m model.inference --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

## Model Eğitimleri

### ResNet

ResNet hattı `torchvision` ResNet18 omurgasını kullanır. Checkpoint çıktısı `.pt` dosyasıdır.

Temel eğitim:

```bash
mri-train --model resnet
```

İşlenmiş split ile eğitim:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Daha uzun eğitim ve özel hiperparametreler:

```bash
mri-train --model resnet --epochs 50 --batch-size 32 --lr 3e-4
mri-train --model resnet --loss focal --focal-gamma 2.5
mri-train --model resnet --pretrained
```

Final eğitim modunda validation ayrılmaz; tüm `trainval` kullanılır ve ayrı bir harici test dizini gerekir:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
```

### XGBoost

XGBoost hattı HOG, LBP, GLCM, histogram ve istatistiksel görüntü özellikleriyle çalışır. Model `.json`, yanında metadata `.meta.json` olarak kaydedilir.

Temel eğitim:

```bash
mri-train --model xgboost
```

İşlenmiş split ve özellik cache ile eğitim:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Örnek XGBoost hiperparametreleri:

```bash
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8 --xgb-learning-rate 0.05 --xgb-subsample 0.7 --xgb-colsample-bytree 0.8
```

## Hiperparametre Optimizasyonu

`mri-tune`, Optuna `TPESampler` ile Bayesian-style arama yapar. Desteklenen seçim metrikleri:

- `loss`: minimize edilir.
- `accuracy`, `precision`, `recall`, `f1`: maksimize edilir.

ResNet HPO:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model resnet --trials 30 --metric loss --search-pretrained --skip-final-train
```

XGBoost HPO:

```bash
mri-tune --model xgboost --trials 30 --metric f1
mri-tune --model xgboost --trials 50 --metric accuracy --feature-cache model/ciktilar/sl_ozellikler
```

Varsayılan davranışta HPO bittikten sonra en iyi trial parametreleriyle `trainval` tamamı üzerinde final eğitim çalıştırılır. Bu final eğitim için harici test dizini gerekir. Yalnızca arama sonuçlarını üretmek için `--skip-final-train` kullanılabilir.

HPO çıktıları varsayılan olarak şu dizine yazılır:

```text
model/ciktilar/hiperparametre_arama/<study_name>/
|-- trial_history.csv
|-- study_summary.json
|-- trials/
|-- gorseller/
`-- best_run/
```

## Tahmin Alma

`mri-infer`, model tipini dosya uzantısından algılar:

- `.pt`: ResNet/PyTorch modeli.
- `.json`: XGBoost modeli.

Tek görüntü:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

Klasör üzerinde batch tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --batch ornek_klasor
```

Ham görüntüyle tahmin alırken eğitim dağılımına daha yakın kalmak için ön işleme uygulanabilir:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image Veri_Seti/OriginalDataset/NonDemented/ornek.jpg --preprocess
```

## Önemli CLI Parametreleri

### Ortak Eğitim Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--model` | `resnet` veya `xgboost`. |
| `--image-size` | Model girdi görüntü boyutu. |
| `--trainval-dir` | Train/validation kaynağı veya split kökü. |
| `--test-dir` | Harici test veri dizini. |
| `--val-ratio` | Validation oranı. |
| `--test-ratio` | Harici test yoksa internal test oranı. |
| `--seed` | Rastgelelik tohumu. |
| `--full-trainval` | Validation ayırmadan tüm `trainval` üzerinde final eğitim yapar. |

### ResNet Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--epochs` | Epoch sayısı. |
| `--batch-size` | Batch boyutu. |
| `--lr` | Öğrenme hızı. |
| `--patience` | Early stopping sabır değeri. |
| `--num-workers` | PyTorch `DataLoader` worker sayısı. |
| `--loss` | `ce` veya `focal`. |
| `--pretrained` | ImageNet pretrained ResNet18 ağırlıklarını kullanır. |
| `--weight-decay` | AdamW weight decay değeri. |
| `--scheduler-factor` | `ReduceLROnPlateau` LR çarpanı. |
| `--scheduler-patience` | Scheduler sabır değeri. |
| `--focal-gamma` | Focal loss gamma değeri. |
| `--dropout` | ResNet classifier head dropout oranı. |
| `--label-smoothing` | Cross entropy için label smoothing. |
| `--hflip-p` | Eğitim augmentasyonunda yatay çevirme olasılığı. Varsayılan `0.0`dır. |
| `--rotation-degrees` | Eğitim augmentasyonunda dönüş sınırı. |
| `--color-jitter` | Brightness/contrast jitter şiddeti. |

### XGBoost Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--xgb-n-estimators` | Boosting round sayısı. |
| `--xgb-max-depth` | Maksimum ağaç derinliği. |
| `--xgb-learning-rate` | XGBoost öğrenme hızı. |
| `--xgb-subsample` | Satır örnekleme oranı. |
| `--xgb-colsample-bytree` | Sütun örnekleme oranı. |
| `--xgb-reg-lambda` | L2 regularizasyon değeri. |
| `--xgb-min-child-weight` | Minimum child weight. |
| `--feature-cache` | Özellik matrislerini `.npz` olarak saklayacak dizin. |

### HPO Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--trials` | Hedef toplam trial sayısı. |
| `--timeout` | Saniye cinsinden süre sınırı. |
| `--metric` | Optimize edilecek metrik: `loss`, `accuracy`, `precision`, `recall`, `f1`. |
| `--study-name` | Optuna study adı. |
| `--storage` | Opsiyonel Optuna storage URL değeri. |
| `--output-dir` | Arama çıktılarının yazılacağı klasör. |
| `--batch-size-choices` | ResNet için denenecek batch size adayları. |
| `--image-size-choices` | Denenecek görüntü boyutu adayları. |
| `--search-pretrained` | ResNet için `pretrained` seçeneğini arama uzayına ekler. |
| `--skip-final-train` | Arama sonunda final eğitim yapmaz. |
| `--verbose-trials` | Trial içi eğitim loglarını gösterir. |

## Üretilen Çıktılar

Varsayılan çıktı kökü `model/ciktilar/` dizinidir.

ResNet eğitiminden beklenen ana çıktılar:

```text
model/ciktilar/modeller/best_resnet.pt
model/ciktilar/raporlar/rapor_resnet_<timestamp>.json
model/ciktilar/gorseller/confusion_matrix_resnet.png
model/ciktilar/gorseller/confusion_matrix_normalized_resnet.png
model/ciktilar/gorseller/classification_summary_resnet.png
model/ciktilar/gorseller/prediction_confidence_resnet.png
model/ciktilar/gorseller/roc_pr_curves_resnet.png
model/ciktilar/gorseller/training_curves_resnet.png
```

XGBoost eğitiminden beklenen ana çıktılar:

```text
model/ciktilar/modeller/best_xgboost.json
model/ciktilar/modeller/best_xgboost.meta.json
model/ciktilar/raporlar/rapor_xgboost_<timestamp>.json
model/ciktilar/gorseller/confusion_matrix_xgboost.png
model/ciktilar/gorseller/confusion_matrix_normalized_xgboost.png
model/ciktilar/gorseller/classification_summary_xgboost.png
model/ciktilar/gorseller/prediction_confidence_xgboost.png
model/ciktilar/gorseller/roc_pr_curves_xgboost.png
model/ciktilar/gorseller/training_curves_xgboost.png
model/ciktilar/gorseller/feature_importance_xgboost.png
```

`--feature-cache model/ciktilar/sl_ozellikler` kullanıldığında XGBoost özellik matrisleri örneğin şu dosyalara yazılır:

```text
model/ciktilar/sl_ozellikler/trainval_img224.npz
model/ciktilar/sl_ozellikler/test_img224.npz
```

Tuned final modeller `best_run/` altında saklanır ve adları model tipine göre şu biçimdedir:

- `best_resnet_tuned.pt`
- `best_xgboost_tuned.json`

## Veri Sızıntısı ve Split Politikası

- Eğitim varsayılan olarak `trainval` içinden train/validation ayrımı yapar.
- `goruntu_isleme/cikti/test` varsa harici test seti olarak kullanılır.
- Harici test yoksa `--test-ratio` ile `trainval` içinden internal test ayrılabilir.
- Validation ve test splitleri yalnızca original görüntülerden kurulur; augmentasyon/türev kopyalar train tarafında kalır.
- Dosya adlarından kaynak grup çıkarılabildiğinde group-aware split kullanılır.
- Aynı kaynak görüntüye ait türevlerin farklı splitlere düşmesi engellenmeye çalışılır.
- Harici test kullanıldığında `trainval` ile `test` arasında ortak kaynak grup olup olmadığı kontrol edilir.
- `--full-trainval` final eğitim modudur; validation ayırmaz ve ayrı bir harici test dizini gerektirir.
- HPO’da model seçimi validation metriğiyle yapılır. Test seti yalnızca final değerlendirme için kullanılmalıdır.

## Önerilen Proje Akışı

1. Ham veriyi `Veri_Seti/OriginalDataset` altında beklenen sınıf klasörleriyle hazırla.

2. Ön işleme ve split üret:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

3. Hızlı bir temel eğitim çalıştır:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test

mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

4. Gerekirse hiperparametre optimizasyonu yap:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test

mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

5. Eğitilmiş modelle tahmin al:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

## Test Notu

Model altyapısının hızlı testleri repo kökünden çalıştırılır:

```bash
pytest tests/test_model_sl.py
```

PyTorch ağırlıklı testler varsayılan olarak atlanır. Bu testleri çalıştırmak için ortam değişkeni gerekir:

```powershell
$env:MRI_RUN_TORCH_TESTS = "1"; pytest tests/test_model_altyapi.py
```
