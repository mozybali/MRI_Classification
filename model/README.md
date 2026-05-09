# Model Modülü

`model/` klasörü, MRI beyin görüntüsü sınıflandırma projesinin eğitim, hiperparametre arama ve tahmin alma katmanıdır. Bu modül ham ya da ön işlenmiş 2D görüntülerden dört demans sınıfı için model üretir; ön işleme ve leak-free `trainval/test` ayrımı ise proje akışında `goruntu_isleme/` tarafından hazırlanır.

Desteklenen model aileleri:

- `resnet`: PyTorch ve `torchvision` ResNet18 tabanlı derin öğrenme hattı.
- `xgboost`: HOG, LBP, GLCM, histogram ve istatistiksel görüntü özellikleriyle çalışan sığ öğrenme hattı.

## Projedeki Yeri

Tipik çalışma akışı:

1. Ham görüntüler `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutulur.
2. İsteğe bağlı EDA çıktıları `eda_analiz/` ile üretilir.
3. `goruntu_isleme/` modülü `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` dizinlerini oluşturur.
4. Bu modül ResNet veya XGBoost modelini eğitir, değerlendirir ve çıktıları `model/ciktilar/` altında saklar.
5. Eğitilmiş `.pt` veya `.json` model dosyaları `mri-infer` ile tek görüntü ya da klasör tahmininde kullanılır.

Ham veri üzerinde doğrudan eğitim yapılabilir, ancak proje için önerilen giriş `mri-preprocess` sonrasında oluşan işlenmiş `trainval/test` klasörleridir.

## Sınıflar

Sınıf klasörü adları kodda sabit kullanılır ve büyük/küçük harfe duyarlıdır.

| Sınıf klasörü | Etiket |
| --- | ---: |
| `NonDemented` | 0 |
| `VeryMildDemented` | 1 |
| `MildDemented` | 2 |
| `ModerateDemented` | 3 |

Desteklenen görüntü uzantıları `.jpg`, `.jpeg` ve `.png` değerleridir.

## Dizin Yapısı

Kaynak dosyalar:

```text
model/
|-- __init__.py
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
|-- sl/
|   |-- __init__.py
|   |-- dataset.py
|   |-- features.py
|   |-- training_runner.py
|   `-- xgb_classifier.py
`-- README.md
```

Çalışma sırasında oluşan dosyalar:

```text
model/
`-- ciktilar/
    |-- modeller/
    |-- raporlar/
    |-- gorseller/
    |-- sl_ozellikler/
    |-- hiperparametre_arama/
    `-- folds/
```

`model/ciktilar/` eğitim, değerlendirme, CV, HPO ve özellik cache çıktıları için kullanılır; kaynak kodun parçası değildir.

## Dosyaların Görevleri

| Yol | Görev |
| --- | --- |
| `ayarlar.py` | Varsayılan veri, çıktı ve seed ayarlarını tanımlar. |
| `train.py` | `mri-train` CLI girişidir; ResNet, XGBoost ve K-fold eğitimlerini başlatır. |
| `training_runner.py` | ResNet eğitim döngüsü, validation/test değerlendirmesi, checkpoint, rapor ve CV akışını yönetir. |
| `hpo.py` | Optuna TPE ile ResNet ve XGBoost hiperparametre araması yapar. |
| `inference.py` | `.pt` ve `.json` modelleriyle tek görüntü veya klasör üzerinde tahmin alır. |
| `common/evaluation.py` | Ortak sınıflandırma metrikleri ve detaylı değerlendirme raporlarını üretir. |
| `dl/dataset.py` | PyTorch dataset/dataloader, transform, split, grup kontrolü ve K-fold veri hazırlığını içerir. |
| `dl/engine.py` | PyTorch epoch eğitim ve değerlendirme fonksiyonlarını içerir. |
| `dl/losses.py` | Focal loss ve class weight yardımcılarını içerir. |
| `dl/utils.py` | Seed, cihaz seçimi, checkpoint yükleme ve görselleştirme yardımcılarını içerir. |
| `dl/models/resnet_classifier.py` | ResNet18 tabanlı sınıflandırıcıyı tanımlar. |
| `sl/dataset.py` | XGBoost için klasör ağacından özellik matrisi ve `.npz` cache üretir. |
| `sl/features.py` | HOG, LBP, GLCM, histogram ve istatistiksel görüntü özelliklerini çıkarır. |
| `sl/training_runner.py` | XGBoost eğitim, validation/test, artifact ve CV akışını yönetir. |
| `sl/xgb_classifier.py` | `XGBClassifier` oluşturma, kaydetme ve metadata ile yükleme yardımcılarını içerir. |

## Kurulum

Komutları proje kök dizininden çalıştırın. Python `3.10+` gerekir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Ortak bağımlılıklar:

```bash
pip install -r requirements.txt
```

PyTorch CPU kurulumu:

```bash
pip install -r requirements-torch-cpu.txt
```

CUDA ortamı için:

```bash
pip install -r requirements-torch-cu128.txt
```

CLI komutlarını kullanılabilir yapmak için:

```bash
pip install -e . --no-deps
```

Kurulumdan sonra bu modülle ilgili komutlar:

| Komut | Görev |
| --- | --- |
| `mri-train` | ResNet veya XGBoost modeli eğitir. |
| `mri-tune` | Optuna ile hiperparametre araması çalıştırır. |
| `mri-infer` | Eğitilmiş modelle tahmin alır. |

Paket kurulmadan modül olarak çalıştırma:

```bash
python3 -m model.train --model resnet
python3 -m model.hpo --model xgboost --trials 20
python3 -m model.inference --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

## Beklenen Veri Yapısı

Önerilen eğitim girişi:

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

Varsayılan yollar `model/ayarlar.py` içinde tanımlıdır:

| Amaç | Varsayılan yol |
| --- | --- |
| Train/validation kaynağı | `goruntu_isleme/cikti/trainval` |
| Harici test kaynağı | `goruntu_isleme/cikti/test` |
| Model dosyaları | `model/ciktilar/modeller` |
| JSON raporlar | `model/ciktilar/raporlar` |
| Grafikler | `model/ciktilar/gorseller` |
| XGBoost özellik cache'i | `model/ciktilar/sl_ozellikler` |
| HPO çıktıları | `model/ciktilar/hiperparametre_arama` |

`--trainval-dir` ve `--test-dir` verilirse bu varsayılanlar geçersiz kılınır. Bir split kökü verilirse kod ilgili `trainval/` veya `test/` alt dizinini otomatik çözmeye çalışır.

## Hızlı Kullanım

Ön işleme sonrasında temel ResNet eğitimi:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost eğitimi:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Eğitilmiş modelle tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

## ResNet Eğitimi

ResNet hattı `torchvision.models.resnet18` omurgasını kullanır. Varsayılan olarak ImageNet ağırlıkları kullanılmaz; `--pretrained` verilirse pretrained ağırlıklar yüklenir. Model checkpoint dosyası `.pt` formatında kaydedilir.

Temel eğitim:

```bash
mri-train --model resnet
```

İşlenmiş split ile eğitim:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Örnek hiperparametreler:

```bash
mri-train --model resnet --epochs 50 --batch-size 32 --lr 3e-4
mri-train --model resnet --loss focal --focal-gamma 2.5
mri-train --model resnet --pretrained --dropout 0.4
```

Final eğitim modunda validation ayrılmaz; tüm `trainval` kullanılır ve harici test dizini gerekir:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
```

K-fold cross-validation:

```bash
mri-train --model resnet --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

## XGBoost Eğitimi

XGBoost hattı görüntüleri gri tona çevirip `image_size x image_size` boyutuna getirir ve özellik matrisi üretir. Model `.json`, metadata ise `.meta.json` olarak kaydedilir.

Temel eğitim:

```bash
mri-train --model xgboost
```

İşlenmiş split ve özellik cache ile eğitim:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Örnek XGBoost parametreleri:

```bash
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8 --xgb-learning-rate 0.05 --xgb-subsample 0.7 --xgb-colsample-bytree 0.8
```

### XGBoost GPU Desteği

`--xgb-device` parametresiyle XGBoost'un hangi cihazda çalışacağı seçilir:

| Değer | Davranış |
| --- | --- |
| `auto` | Hem XGBoost CUDA build'i hem de PyTorch CUDA mevcutsa GPU kullanır; aksi hâlde CPU'ya düşer. Varsayılan. |
| `cuda` | GPU'yu zorla açar. CUDA'lı XGBoost build'i yoksa uyarı verilir ve CPU'ya düşülür. |
| `cpu` | Her zaman CPU kullanır. |

```bash
mri-train --model xgboost --xgb-device cuda --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-tune --model xgboost --xgb-device auto --trials 30 --metric f1
```

Her HPO trial'ı bittikten sonra `gc.collect()` ve `torch.cuda.empty_cache()` çağrısıyla Python referansları ve CUDA önbelleği temizlenir. Bu sayede uzun HPO çalışmalarında VRAM/RAM baskısı birikimi engellenir.

### Augmented Veri Seti ile Eğitim

`mri-preprocess` çıktısında hem orijinal hem de augmented kopyalar aynı `trainval/` klasör yapısında bulunabilir. XGBoost hattı augmented kopyaları dosya adından (`_aug1`, `_aug2` vb.) otomatik tanır.

Veri sızıntısını önlemek için:

- Validation seti yalnızca **orijinal** görüntülerden kurulur; augmented kopyalar train tarafında kalır.
- Group-aware split mevcutsa aynı kaynağın türevleri aynı gruba atanır; aksi hâlde sızıntı riski konusunda uyarı verilir.

```bash
mri-train --model xgboost \
  --trainval-dir goruntu_isleme/cikti/trainval \
  --test-dir goruntu_isleme/cikti/test \
  --feature-cache model/ciktilar/sl_ozellikler
```

K-fold cross-validation:

```bash
mri-train --model xgboost --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

`--feature-cache` kullanıldığında cache dosyaları veri dizini ve `image_size` metadata'sı ile doğrulanır. Aynı cache farklı veri dizini veya farklı görüntü boyutu için kullanılırsa hata verilir.

## Hiperparametre Araması

`mri-tune`, Optuna `TPESampler` ile Bayes tabanlı arama çalıştırır. Desteklenen seçim metrikleri:

| Metrik | Yön |
| --- | --- |
| `loss` | minimize |
| `accuracy` | maximize |
| `precision` | maximize |
| `recall` | maximize |
| `f1` | maximize |

ResNet HPO:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-tune --model resnet --trials 30 --metric loss --search-pretrained --skip-final-train
```

XGBoost HPO:

```bash
mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Trial değerlendirmesini K-fold ile yapmak için:

```bash
mri-tune --model resnet --trials 20 --hpo-folds 5 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Varsayılan davranışta HPO bittikten sonra en iyi trial parametreleriyle `trainval` tamamı üzerinde final eğitim çalıştırılır. Yalnızca arama sonuçlarını üretmek için `--skip-final-train` kullanılabilir.

HPO çıktıları:

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

- `.pt`: ResNet/PyTorch checkpoint.
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

Ham görüntüyle tahmin alırken eğitim dağılımına daha yakın kalmak için görüntü işleme pipeline'ı tahmin öncesi bellek içinde uygulanabilir:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image Veri_Seti/OriginalDataset/NonDemented/ornek.jpg --preprocess
```

Batch ve tek görüntü inference çıktıları terminale yazılır; komutlar varsayılan olarak dosya üretmez.

## Önemli CLI Parametreleri

### Ortak Eğitim Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--model` | `resnet` veya `xgboost`. Varsayılan `resnet`. |
| `--image-size` | Model girdi görüntü boyutu. Varsayılan `224`. |
| `--trainval-dir` | Train/validation kaynağı veya split kökü. |
| `--test-dir` | Harici test veri dizini veya split kökü. |
| `--val-ratio` | Validation oranı. Varsayılan `0.15`. |
| `--test-ratio` | Harici test yoksa internal test oranı. Varsayılan `0.15`. |
| `--seed` | Rastgelelik tohumu. Varsayılan `42`. |
| `--full-trainval` | Validation ayırmadan tüm `trainval` üzerinde final eğitim yapar. |
| `--folds` | `>=2` ise K-fold cross-validation çalıştırır. |

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
| `--hflip-p` | Eğitim augmentasyonunda yatay çevirme olasılığı. Varsayılan `0.0`. |
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
| `--xgb-device` | XGBoost cihaz modu: `auto`, `cpu`, `cuda`. Varsayılan `auto`. |
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
| `--loss-choices` | ResNet HPO için denenecek loss adayları. |
| `--search-pretrained` | ResNet için `pretrained` seçeneğini arama uzayına ekler. |
| `--hpo-folds` | Her trial'i K-fold CV ile değerlendirir. |
| `--feature-cache` | XGBoost HPO için özellik cache dizini. |
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

K-fold eğitiminde her fold ayrı alt dizine yazılır ve özet rapor `raporlar/` altında tutulur:

```text
model/ciktilar/folds/fold_00/
|-- modeller/
|-- raporlar/
`-- gorseller/
model/ciktilar/raporlar/cv_summary_<model>_<timestamp>.json
```

`--feature-cache model/ciktilar/sl_ozellikler` kullanıldığında XGBoost özellik matrisleri örneğin şu dosyalara yazılır:

```text
model/ciktilar/sl_ozellikler/trainval_img224.npz
model/ciktilar/sl_ozellikler/test_img224.npz
```

## Veri Sızıntısı ve Split Politikası

- Eğitim varsayılan olarak `trainval` içinden train/validation ayrımı yapar.
- `goruntu_isleme/cikti/test` varsa harici test seti olarak kullanılır.
- Harici test yoksa `--test-ratio` ile `trainval` içinden internal test ayrılabilir.
- Dosya adından kaynak grup çıkarılabildiğinde group-aware split kullanılır.
- Aynı kaynak görüntüye ait türevlerin farklı splitlere düşmesi engellenmeye çalışılır.
- Validation ve test splitleri yalnızca original görüntülerden kurulur; augmentation/türev kopyalar train tarafında kalır.
- Harici test kullanıldığında `trainval` ile `test` arasında ortak kaynak grup olup olmadığı kontrol edilir.
- `--folds` ve `--hpo-folds` K-fold ayrımlarını grup seviyesinde yapmaya çalışır.
- `--full-trainval` final eğitim modudur; validation ayırmaz ve ayrı bir harici test dizini gerektirir.
- HPO’da model seçimi validation metriğiyle yapılır. Test seti yalnızca final değerlendirme için kullanılmalıdır.

## Test Notu

Model tarafındaki hızlı testler:

```bash
pytest tests/test_model_sl.py
```

PyTorch ağırlıklı testler varsayılan olarak atlanır. Bu testleri çalıştırmak için:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```
