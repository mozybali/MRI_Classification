# Model Modülü

`model/` modülü, MRI beyin görüntülerinden demans seviyesi sınıflandırması için eğitim, hiperparametre optimizasyonu ve inference akışlarını içerir. Modül repo kökünden çalışacak şekilde tasarlanmıştır ve iki model ailesini destekler: ResNet/PyTorch tabanlı derin öğrenme hattı ve görüntüden çıkarılan klasik özelliklerle çalışan XGBoost tabanlı sığ öğrenme hattı.

## Desteklenen Model Türleri

| Model | Komut Değeri | Açıklama |
| --- | --- | --- |
| ResNet/PyTorch | `resnet` | `torchvision` ResNet18 omurgasıyla çalışan derin öğrenme sınıflandırıcısı. `.pt` checkpoint üretir. |
| XGBoost | `xgboost` | HOG, LBP, GLCM ve histogram/istatistik özellikleriyle çalışan sığ öğrenme sınıflandırıcısı. `.json` model ve `.meta.json` metadata üretir. |

## Dizin Yapısı

```text
model/
|-- __init__.py
|-- README.md
|-- ayarlar.py
|-- hpo.py
|-- inference.py
|-- train.py
|-- training_runner.py
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
`-- sl/
    |-- __init__.py
    |-- dataset.py
    |-- features.py
    |-- training_runner.py
    `-- xgb_classifier.py
```

Çalışma çıktıları kaynak ağacının parçası değildir; eğitim ve HPO sırasında `model/ciktilar/` altında oluşturulur.

## Veri Konumu

Varsayılan eğitim akışı, görüntü ön işleme modülünün ürettiği leak-free split yapısını kullanır:

- `goruntu_isleme/cikti/trainval`: varsayılan train/validation kaynağı.
- `goruntu_isleme/cikti/test`: varsa varsayılan harici test kaynağı.
- `Veri_Seti/OriginalDataset`: ham/orijinal veri kaynağı.

Beklenen sınıf klasörleri `NonDemented`, `VeryMildDemented`, `MildDemented` ve `ModerateDemented` adlarıyla yer almalıdır. Eğitim komutlarında `--trainval-dir` veya `--test-dir` verilmezse `model/ayarlar.py` içindeki varsayılan yollar kullanılır.

Ham veri üzerinde doğrudan eğitim yapılabilir, ancak önerilen pratik önce `mri-preprocess` ile `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` dizinlerini üretmektir.

## Kurulum

Bağımlılıklar repo kökündeki dosyalardan kurulmalıdır. `requirements.txt` PyTorch dışındaki ortak bağımlılıkları içerir; PyTorch kurulumu ortamınıza göre ayrı dosyadan yapılır.

GPU/CUDA ortamı için:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
pip install -e .[dev] --no-deps
```

CPU-only ortam için:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cpu.txt
pip install -e .[dev] --no-deps
```

`pip install -e .[dev] --no-deps` sonrasında bu modül için `mri-train`, `mri-tune` ve `mri-infer` komutları kullanılabilir.

## Eğitim Örnekleri

ResNet/PyTorch eğitimi:

```bash
mri-train --model resnet
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model resnet --loss focal --lr 3e-4 --focal-gamma 2.5
mri-train --model resnet --pretrained
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
```

XGBoost eğitimi:

```bash
mri-train --model xgboost
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8
mri-train --model xgboost --xgb-learning-rate 0.05 --xgb-subsample 0.7
mri-train --model xgboost --feature-cache model/ciktilar/sl_ozellikler
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
```

Doğrudan Python modülüyle kullanım:

```bash
python3 -m model.train --model resnet --epochs 50 --batch-size 32
python3 -m model.train --model xgboost --xgb-n-estimators 300 --xgb-max-depth 6
```

## Hiperparametre Optimizasyonu

`mri-tune`, Optuna `TPESampler` ile Bayesian-style arama yapar. Desteklenen seçim metrikleri `loss`, `accuracy`, `precision`, `recall` ve `f1` değerleridir. `loss` minimize edilir; diğer metrikler maksimize edilir.

ResNet arama örnekleri:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model resnet --trials 30 --metric loss --search-pretrained --skip-final-train
mri-tune --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost arama örnekleri:

```bash
mri-tune --model xgboost --trials 30 --metric f1
mri-tune --model xgboost --trials 50 --metric accuracy --feature-cache model/ciktilar/sl_ozellikler
mri-tune --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

`--skip-final-train` verilmezse arama sonunda en iyi trial parametreleriyle `trainval` tamamı üzerinde final eğitim başlatılır. Bu final eğitim için harici bir test dizini gerekir.

## Inference Örnekleri

`mri-infer`, model tipini dosya uzantısından algılar: `.pt` dosyaları ResNet/PyTorch, `.json` dosyaları XGBoost olarak yüklenir.

Tek görüntü:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

Klasör bazlı batch inference:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --batch ornek_klasor
```

Ham görüntü üzerinde inference yaparken eğitim dağılımıyla daha tutarlı olmak için ön işleme uygulanabilir:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image Veri_Seti/OriginalDataset/NonDemented/ornek.jpg --preprocess
python3 -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## CLI Parametreleri

Ortak eğitim parametreleri:

- `--model`: `resnet` veya `xgboost`.
- `--image-size`: model giriş görüntü boyutu.
- `--trainval-dir`: train/validation kaynağı veya split kökü.
- `--test-dir`: harici test veri dizini.
- `--val-ratio`: validation oranı.
- `--test-ratio`: harici test yoksa internal test oranı.
- `--seed`: rastgele tohum.
- `--full-trainval`: validation ayırmadan tüm `trainval` üzerinde final eğitim yapar; harici `--test-dir` gerektirir.

ResNet/PyTorch parametreleri:

- `--epochs`: epoch sayısı.
- `--batch-size`: batch boyutu.
- `--lr`: öğrenme hızı.
- `--patience`: early stopping sabır değeri.
- `--num-workers`: `DataLoader` worker sayısı.
- `--loss`: `ce` veya `focal`.
- `--pretrained`: ImageNet pretrained ResNet18 ağırlıklarını kullanır.
- `--weight-decay`: AdamW weight decay değeri.
- `--scheduler-factor`: `ReduceLROnPlateau` LR çarpanı.
- `--scheduler-patience`: `ReduceLROnPlateau` sabır değeri.
- `--focal-gamma`: `--loss focal` için gamma değeri.
- `--dropout`: ResNet classifier head dropout oranı.
- `--label-smoothing`: `--loss ce` için label smoothing değeri.
- `--hflip-p`: eğitim augmentasyonunda `RandomHorizontalFlip` olasılığı.
- `--rotation-degrees`: eğitim augmentasyonunda `RandomRotation` sınırı.
- `--color-jitter`: eğitim augmentasyonunda brightness/contrast şiddeti.

XGBoost parametreleri:

- `--xgb-n-estimators`: boosting round sayısı.
- `--xgb-max-depth`: maksimum ağaç derinliği.
- `--xgb-learning-rate`: XGBoost öğrenme hızı.
- `--xgb-subsample`: satır örnekleme oranı.
- `--xgb-colsample-bytree`: sütun örnekleme oranı.
- `--xgb-reg-lambda`: L2 regularizasyon değeri.
- `--xgb-min-child-weight`: minimum child weight.
- `--feature-cache`: XGBoost özellik matrislerini `.npz` olarak saklayacak dizin.

HPO parametreleri:

- `--trials`: hedef toplam trial sayısı.
- `--timeout`: saniye cinsinden opsiyonel süre sınırı.
- `--metric`: optimize edilecek validation metriği; `loss`, `accuracy`, `precision`, `recall` veya `f1`.
- `--study-name`: Optuna study adı.
- `--storage`: opsiyonel Optuna storage URL değeri.
- `--output-dir`: arama çıktılarının yazılacağı klasör.
- `--epochs`, `--patience`, `--val-ratio`, `--test-ratio`, `--seed`, `--num-workers`, `--trainval-dir`, `--test-dir`: HPO çalışmasında kullanılan temel ayarlar.
- `--batch-size-choices`, `--image-size-choices`: denenecek batch size ve görüntü boyutu adayları.
- `--lr-min`, `--lr-max`, `--weight-decay-min`, `--weight-decay-max`: ResNet arama aralıkları.
- `--scheduler-factor-min`, `--scheduler-factor-max`, `--scheduler-patience-min`, `--scheduler-patience-max`: scheduler arama aralıkları.
- `--loss-choices`, `--focal-gamma-min`, `--focal-gamma-max`, `--dropout-min`, `--dropout-max`, `--label-smoothing-min`, `--label-smoothing-max`: loss ve classifier head arama seçenekleri.
- `--hflip-p-choices`, `--rotation-degrees-min`, `--rotation-degrees-max`, `--color-jitter-min`, `--color-jitter-max`: augmentasyon arama seçenekleri.
- `--search-pretrained`: ResNet için `pretrained` seçeneğini arama uzayına ekler.
- `--n-startup-trials`, `--pruner-startup-trials`, `--pruner-warmup-epochs`: TPE sampler ve median pruner ayarları.
- `--skip-final-train`: arama sonunda final eğitim yapmaz.
- `--verbose-trials`: trial içi epoch loglarını gösterir.
- `--feature-cache`: XGBoost HPO için özellik cache dizini.

Inference parametreleri:

- `--model-path`: `.pt` veya `.json` model yolu.
- `--image`: tek görüntü yolu.
- `--batch`: görüntü klasörü.
- `--preprocess`: tahmin öncesi `goruntu_isleme` ön işleme akışını uygular.

## Çıktılar

Varsayılan çıktı kökü `model/ciktilar/` dizinidir.

```text
model/ciktilar/
|-- modeller/
|-- raporlar/
|-- gorseller/
|-- sl_ozellikler/
`-- hiperparametre_arama/
```

ResNet/PyTorch eğitim çıktıları:

- `model/ciktilar/modeller/best_resnet.pt`
- `model/ciktilar/raporlar/rapor_resnet_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_resnet.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_resnet.png`
- `model/ciktilar/gorseller/classification_summary_resnet.png`
- `model/ciktilar/gorseller/prediction_confidence_resnet.png`
- `model/ciktilar/gorseller/roc_pr_curves_resnet.png`
- `model/ciktilar/gorseller/training_curves_resnet.png`

XGBoost eğitim çıktıları:

- `model/ciktilar/modeller/best_xgboost.json`
- `model/ciktilar/modeller/best_xgboost.meta.json`
- `model/ciktilar/raporlar/rapor_xgboost_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_xgboost.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_xgboost.png`
- `model/ciktilar/gorseller/classification_summary_xgboost.png`
- `model/ciktilar/gorseller/prediction_confidence_xgboost.png`
- `model/ciktilar/gorseller/roc_pr_curves_xgboost.png`
- `model/ciktilar/gorseller/training_curves_xgboost.png`
- `model/ciktilar/gorseller/feature_importance_xgboost.png`

`--feature-cache model/ciktilar/sl_ozellikler` kullanılırsa XGBoost için aşağıdaki cache dosyaları üretilebilir:

- `model/ciktilar/sl_ozellikler/trainval_img<image_size>.npz`
- `model/ciktilar/sl_ozellikler/test_img<image_size>.npz`

HPO çıktıları varsayılan olarak `model/ciktilar/hiperparametre_arama/<study_name>/` altında tutulur:

- `trial_history.csv`
- `study_summary.json`
- `trials/trial_XXX/trial_summary.json`
- `gorseller/hpo_optimization_history.png`
- `gorseller/hpo_param_importances.png`
- `gorseller/hpo_parallel_coordinate.png`
- `gorseller/hpo_slice.png`
- `best_run/`

HPO final eğitimi `best_run/` altında kendi `modeller/`, `raporlar/` ve `gorseller/` klasörlerini oluşturur. Bu nedenle tuned modellerin adları `best_resnet_tuned.pt` veya `best_xgboost_tuned.json` biçimindedir.

## Validation, Test ve Veri Sızıntısı Politikası

- Eğitim varsayılan olarak `trainval` içinden train/validation ayrımı yapar.
- `goruntu_isleme/cikti/test` mevcutsa harici test dizini olarak kullanılır; yoksa `--test-ratio` ile `trainval` içinden internal test ayrılır.
- Validation ve test split'leri yalnızca original görüntülerden kurulur; augmentation/türev kopyalar train tarafında kalır.
- Dosya adından kaynak grup çıkarılabiliyorsa group-aware split kullanılır. Böylece aynı kaynak görüntünün türevlerinin farklı split'lere düşmesi engellenir.
- Harici test dizini kullanılırken `trainval` ile `test` arasında ortak kaynak grup olup olmadığı kontrol edilir.
- `--full-trainval` final eğitim modudur: validation ayrılmaz, tüm `trainval` ile eğitim yapılır ve değerlendirme için farklı bir harici test dizini gerekir.
- HPO trial seçimleri validation metriğine göre yapılır. Test seti model seçimi için kullanılmamalı, yalnızca final değerlendirmede kullanılmalıdır.

## Önerilen Akış

1. Ham veriyi `Veri_Seti/OriginalDataset` altında beklenen sınıf klasörleriyle hazırlayın.
2. Ön işleme ve leak-free split üretin:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

3. Hızlı bir temel eğitim çalıştırın:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

4. Gerekirse hiperparametre optimizasyonu yapın:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

5. Final modelle tahmin alın:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```
