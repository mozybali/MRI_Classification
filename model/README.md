# Model Egitim Modulu

Bu modul, MRI goruntulerinden demans seviyesi siniflandirmak icin PyTorch tabanli derin ogrenme ve XGBoost tabanli sig ogrenme egitim/inference akisini saglar.

## Desteklenen Modeller

| Model | Tur | Aciklama |
|-------|-----|----------|
| `resnet` | Derin Ogrenme | ResNet18 tabanli siniflandirici |
| `xgboost` | Sig Ogrenme | HOG + LBP + GLCM ozellik vektorleri ile XGBoost siniflandirici |

## Varsayilan Veri Politikasi

- Varsayilan egitim kaynagi: `goruntu_isleme/cikti/trainval`
- Varsayilan test kaynagi: `goruntu_isleme/cikti/test`
- `Veri_Seti/OriginalDataset`: ham/original kaynak veri
- `validation` ve `test`: original-only
- `augmentation`: yalnizca train transform'u uzerinde

## Egitim

Repo kokunden onerilen komutlar:

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu128.txt
pip install -e .[dev] --no-deps
```

CPU ile calisacaksaniz ikinci satir yerine `requirements-torch-cpu.txt` kullanin.

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model resnet --trainval-dir Veri_Seti/OriginalDataset
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Dogrudan Python ile:

```bash
python -m model.train --model resnet --epochs 50 --batch-size 32
```

Ek ornekler:

```bash
mri-train --model resnet --loss focal --lr 3e-4
mri-train --model resnet --loss focal --focal-gamma 2.5
mri-train --model resnet --weight-decay 1e-3 --scheduler-factor 0.3
mri-train --model resnet --pretrained
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --full-trainval
mri-train --model resnet --trainval-dir Veri_Seti/OriginalDataset
mri-train --model resnet --val-ratio 0.2
```

Islenmis goruntulerle egitim icin onerilen akis:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
mri-train --model resnet
```

## Sig Ogrenme (XGBoost) Egitim

XGBoost pipeline, goruntulerden HOG, LBP, GLCM ve histogram istatistik ozelliklerini cikarir ve bir gradient-boosted tree siniflandirici egitir.

```bash
mri-train --model xgboost
mri-train --model xgboost --xgb-n-estimators 500 --xgb-max-depth 8
mri-train --model xgboost --xgb-learning-rate 0.05 --xgb-subsample 0.7
mri-train --model xgboost --trainval-dir Veri_Seti/OriginalDataset
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --full-trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --feature-cache model/ciktilar/sl_ozellikler
```

Dogrudan Python ile:

```bash
python -m model.train --model xgboost --xgb-n-estimators 300 --xgb-max-depth 6
```

## Temel Parametreler

### Ortak Parametreler

- `--model`: `resnet` veya `xgboost`
- `--image-size`: Giris goruntu boyutu
- `--trainval-dir`: Split kaynagi veya train+validation veri dizini
- `--test-dir`: Opsiyonel harici test veri dizini
- `--val-ratio`: Validation orani
- `--test-ratio`: Harici test dizini yoksa internal test orani
- `--seed`: Rastgele tohum
- `--full-trainval`: Validation ayirmadan tum trainval ile final model egitir; harici test dizini gerekir

### ResNet Parametreleri

- `--epochs`: Epoch sayisi
- `--batch-size`: Batch boyutu
- `--lr`: Ogrenme hizi
- `--patience`: Early stopping sabir degeri
- `--use-processed-trainval`: Geriye donuk uyumluluk bayragi
- `--use-processed-test`: Geriye donuk uyumluluk bayragi
- `--loss`: `ce` veya `focal`
- `--num-workers`: DataLoader worker sayisi
- `--pretrained`: ImageNet agirliklarini acar
- `--weight-decay`: AdamW regularizasyon katsayisi
- `--scheduler-factor`: Plateau durumunda LR azaltma carpani
- `--scheduler-patience`: LR scheduler sabir degeri
- `--focal-gamma`: Focal loss gamma parametresi

### XGBoost Parametreleri

- `--xgb-n-estimators`: Agac sayisi (varsayilan: 300)
- `--xgb-max-depth`: Maksimum agac derinligi (varsayilan: 6)
- `--xgb-learning-rate`: Ogrenme hizi (varsayilan: 0.1)
- `--xgb-subsample`: Satir alt-ornekleme orani (varsayilan: 0.8)
- `--xgb-colsample-bytree`: Ozellik alt-ornekleme orani (varsayilan: 0.8)
- `--xgb-reg-lambda`: L2 regularizasyon katsayisi (varsayilan: 1.0)
- `--xgb-min-child-weight`: Yaprakta min agirlik toplami (varsayilan: 1)
- `--feature-cache`: Ozellik cache dizini (.npz)

## Inference

Model tipi dosya uzantisindan otomatik algilanir: `.pt` → ResNet, `.json` → XGBoost.

ResNet ile:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

XGBoost ile:

```bash
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --batch ornek_klasor
```

Klasor bazli batch tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
```

Dogrudan Python ile:

```bash
python -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Bayes Search

Optuna `TPESampler` kullanilarak Bayesian-style hiperparametre aramasi yapilabilir.

ResNet icin:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
python -m model.hpo --model resnet --trials 30 --metric loss --skip-final-train
```

XGBoost icin:

```bash
mri-tune --model xgboost --trials 30 --metric f1
mri-tune --model xgboost --trials 50 --metric accuracy --feature-cache model/ciktilar/sl_ozellikler
python -m model.hpo --model xgboost --trials 20 --skip-final-train
```

Onerilen akis:

1. `preprocess` ile veriyi `trainval/test` olarak ayir.
2. HPO trial'larini sadece `trainval` uzerinde train+validation ile sec.
3. `--skip-final-train` verilmediginde en iyi trial'in `best_epoch` degeriyle tum `trainval` uzerinde yeniden egit.
4. Test degerlendirmesini yalnizca en sonda harici `test` dizini uzerinde bir kez yap.

Aranan baslica hiperparametreler:

### ResNet

- `batch_size`
- `image_size`
- `lr`
- `weight_decay`
- `scheduler_factor`
- `scheduler_patience`
- `loss`
- `focal_gamma` (`loss=focal` ise)
- `pretrained` (`--search-pretrained` ile, sadece ResNet)

### XGBoost

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `reg_lambda`
- `min_child_weight`
- `image_size`

Bayes search ciktilari varsayilan olarak `model/ciktilar/hiperparametre_arama/<study_name>/` altina yazilir:

- `trial_history.csv`
- `study_summary.json`
- `trials/trial_XXX/trial_summary.json`
- `best_run/` (final egitim kapatilmazsa; tum `trainval` uzerinde yeniden egitim + tek seferlik test)

## Dosya Yapisi

```text
model/
|-- train.py
|-- hpo.py
|-- inference.py
|-- ayarlar.py
|-- training_runner.py
|-- common/
|   `-- evaluation.py
|-- dl/
|   |-- dataset.py
|   |-- engine.py
|   |-- losses.py
|   |-- utils.py
|   `-- models/
|       `-- resnet_classifier.py
|-- sl/
|   |-- features.py
|   |-- dataset.py
|   |-- xgb_classifier.py
|   `-- training_runner.py
`-- ciktilar/
    |-- modeller/
    |-- raporlar/
    |-- gorseller/
    `-- sl_ozellikler/
```

## Uretilen Ciktilar

### ResNet

- `model/ciktilar/modeller/best_resnet.pt`
- `model/ciktilar/raporlar/rapor_resnet_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_resnet.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_resnet.png`
- `model/ciktilar/gorseller/classification_summary_resnet.png`
- `model/ciktilar/gorseller/prediction_confidence_resnet.png`
- `model/ciktilar/gorseller/roc_pr_curves_resnet.png`
- `model/ciktilar/gorseller/training_curves_resnet.png`

### XGBoost

- `model/ciktilar/modeller/best_xgboost.json`
- `model/ciktilar/raporlar/rapor_xgboost_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_xgboost.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_xgboost.png`
- `model/ciktilar/gorseller/classification_summary_xgboost.png`
- `model/ciktilar/gorseller/prediction_confidence_xgboost.png`
- `model/ciktilar/gorseller/roc_pr_curves_xgboost.png`
- `model/ciktilar/gorseller/training_curves_xgboost.png`

## Ozellikler

- Early stopping ve best checkpoint kaydi
- ReduceLROnPlateau scheduler
- Class weights veya focal loss ile sinif dengesizligi yonetimi
- Grup bilgisi cikartilabiliyorsa leak-free split, aksi durumda uyari ile fallback stratejisi
- Accuracy, precision, recall ve F1 macro raporlamasi
- Normalize confusion matrix, sinif bazli performans, guven dagilimi ve ROC/PR egirileri
- CUDA varsa GPU, yoksa CPU fallback
