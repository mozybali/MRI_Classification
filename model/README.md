# Model Eğitim Modülü

Bu modül, MRI görüntülerinden demans seviyesi sınıflandırmak için iki eğitim yolu sunar: PyTorch/ResNet tabanlı derin öğrenme ve HOG + LBP + GLCM özellikleriyle XGBoost tabanlı sığ öğrenme. Eğitim, HPO ve inference komutları aynı klasör altında toplanmıştır.

## Desteklenen Modeller

| Model | Tür | Açıklama |
| --- | --- | --- |
| `resnet` | Derin öğrenme | ResNet18 tabanlı PyTorch sınıflandırıcı |
| `xgboost` | Sığ öğrenme | HOG, LBP, GLCM ve histogram/istatistik özellikleriyle XGBoost sınıflandırıcı |

## Dosya Yapısı

```text
model/
|-- __init__.py
|-- ayarlar.py
|-- train.py
|-- hpo.py
|-- inference.py
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

Çalışma çıktıları varsayılan olarak `model/ciktilar` altında oluşturulur; bu dizin kaynak kod yapısının parçası değildir.

## Varsayılan Veri Politikası

- Varsayılan eğitim kaynağı: `goruntu_isleme/cikti/trainval`
- Varsayılan test kaynağı: `goruntu_isleme/cikti/test`
- Ham veri kaynağı: `Veri_Seti/OriginalDataset`
- Validation ve test split'leri original-only tutulur.
- `--full-trainval` final eğitim modudur; validation ayırmadan tüm `trainval` üzerinde eğitir ve harici test dizini gerektirir.

İşlenmiş görüntülerle önerilen sıra:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
mri-train --model resnet
```

## Kurulum Notu

Repo kökünden ortak bağımlılıkları ve uygun PyTorch paketini kurun:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
pip install -e .[dev] --no-deps
```

CPU ortamında ikinci satır yerine `requirements-torch-cpu.txt` kullanın.

## ResNet Eğitimi

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model resnet --loss focal --lr 3e-4 --focal-gamma 2.5
mri-train --model resnet --pretrained
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
```

Doğrudan Python ile:

```bash
python3 -m model.train --model resnet --epochs 50 --batch-size 32
```

## XGBoost Eğitimi

```bash
mri-train --model xgboost
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8
mri-train --model xgboost --xgb-learning-rate 0.05 --xgb-subsample 0.7
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --full-trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --feature-cache model/ciktilar/sl_ozellikler
```

Doğrudan Python ile:

```bash
python3 -m model.train --model xgboost --xgb-n-estimators 300 --xgb-max-depth 6
```

## Eğitim Parametreleri

Ortak parametreler:

- `--model`: `resnet` veya `xgboost`.
- `--image-size`: Model giriş görüntü boyutu.
- `--trainval-dir`: Train/validation kaynağı veya split kökü.
- `--test-dir`: Harici test veri dizini.
- `--val-ratio`: Validation oranı.
- `--test-ratio`: Harici test yoksa internal test oranı.
- `--seed`: Rastgele tohum.
- `--full-trainval`: Final eğitimde validation ayırmadan tüm `trainval` verisini kullanır.

ResNet parametreleri:

- `--epochs`, `--batch-size`, `--lr`, `--patience`, `--num-workers`
- `--loss`: `ce` veya `focal`
- `--pretrained`: ImageNet ağırlıklarını kullanır.
- `--weight-decay`, `--scheduler-factor`, `--scheduler-patience`
- `--focal-gamma`, `--dropout`, `--label-smoothing`
- `--hflip-p`, `--rotation-degrees`, `--color-jitter`

XGBoost parametreleri:

- `--xgb-n-estimators`, `--xgb-max-depth`, `--xgb-learning-rate`
- `--xgb-subsample`, `--xgb-colsample-bytree`
- `--xgb-reg-lambda`, `--xgb-min-child-weight`
- `--feature-cache`: Özellik matrislerini `.npz` olarak önbelleğe alır.

## Inference

Model tipi dosya uzantısından otomatik algılanır: `.pt` ResNet, `.json` XGBoost.

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
python3 -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

XGBoost modelleri kaydedilirken `.meta.json` yan dosyası da üretilir; inference bu dosyadan `image_size` ve sınıf adlarını okuyabilir.

## Hiperparametre Arama

Optuna `TPESampler` ile Bayesian-style arama yapılır. Desteklenen seçim metrikleri: `loss`, `accuracy`, `precision`, `recall`, `f1`.

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
mri-tune --model resnet --trials 30 --metric loss --search-pretrained --skip-final-train
mri-tune --model xgboost --trials 30 --metric f1
mri-tune --model xgboost --trials 50 --metric accuracy --feature-cache model/ciktilar/sl_ozellikler
```

Önerilen HPO akışı:

1. `mri-preprocess` ile `trainval/test` ayrımını üret.
2. Trial seçimlerini yalnızca `trainval` içindeki train/validation ayrımıyla yap.
3. `--skip-final-train` verilmediyse en iyi parametrelerle tüm `trainval` üzerinde final eğitim başlat.
4. Harici `test` dizinini yalnızca en sonda değerlendir.

HPO çıktıları `model/ciktilar/hiperparametre_arama/<study_name>` altına yazılır:

- `trial_history.csv`
- `study_summary.json`
- `trials/trial_XXX/trial_summary.json`
- `gorseller/hpo_optimization_history.png`
- `gorseller/hpo_param_importances.png`
- `gorseller/hpo_parallel_coordinate.png`
- `gorseller/hpo_slice.png`
- `best_run/` final eğitim çıktıları

## Üretilen Model Çıktıları

ResNet:

- `model/ciktilar/modeller/best_resnet.pt`
- `model/ciktilar/raporlar/rapor_resnet_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_resnet.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_resnet.png`
- `model/ciktilar/gorseller/classification_summary_resnet.png`
- `model/ciktilar/gorseller/prediction_confidence_resnet.png`
- `model/ciktilar/gorseller/roc_pr_curves_resnet.png`
- `model/ciktilar/gorseller/training_curves_resnet.png`

XGBoost:

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

## Özellikler

- Group-aware split ile kaynak sızıntısını azaltma.
- Class weights, focal loss ve label smoothing seçenekleri.
- ReduceLROnPlateau, early stopping ve best checkpoint kaydı.
- Accuracy, precision, recall, macro F1, ROC-AUC ve average precision raporları.
- Confusion matrix, normalize confusion matrix, sınıf özeti, güven dağılımı, ROC/PR ve eğitim eğrisi görselleri.
- CUDA varsa GPU, yoksa CPU fallback.
