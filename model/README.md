# Model Egitim Modulu

Bu modul, MRI goruntulerinden demans seviyesi siniflandirmak icin PyTorch tabanli egitim ve inference akisini saglar.

## Desteklenen Modeller

| Model | Aciklama |
|-------|----------|
| `resnet` | ResNet18 tabanli siniflandirici |

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

## Temel Parametreler

- `--model`: `resnet`
- `--epochs`: Epoch sayisi
- `--batch-size`: Batch boyutu
- `--lr`: Ogrenme hizi
- `--patience`: Early stopping sabir degeri
- `--image-size`: Giris goruntu boyutu
- `--trainval-dir`: Split kaynagi veya train+validation veri dizini
- `--test-dir`: Opsiyonel harici test veri dizini
- `--use-processed-trainval`: Geriye donuk uyumluluk bayragi; varsayilan train+validation kaynagi zaten `goruntu_isleme/cikti/trainval`
- `--use-processed-test`: Geriye donuk uyumluluk bayragi; varsayilan test kaynagi zaten `goruntu_isleme/cikti/test`
- `--val-ratio`: Validation orani
- `--test-ratio`: Harici test dizini yoksa internal test orani
- `--loss`: `ce` veya `focal`
- `--seed`: Rastgele tohum
- `--num-workers`: DataLoader worker sayisi
- `--pretrained`: Sadece ResNet icin ImageNet agirliklarini acar
- `--weight-decay`: AdamW regularizasyon katsayisi
- `--scheduler-factor`: Plateau durumunda LR azaltma carpani
- `--scheduler-patience`: LR scheduler sabir degeri
- `--focal-gamma`: Focal loss gamma parametresi
- `--full-trainval`: Validation ayirmadan tum trainval ile final model egitir; harici test dizini gerekir

## Inference

Tek goruntu:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
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

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
python -m model.hpo --model resnet --trials 30 --metric loss --skip-final-train
```

Onerilen akis:

1. `preprocess` ile veriyi `trainval/test` olarak ayir.
2. HPO trial'larini sadece `trainval` uzerinde train+validation ile sec.
3. `--skip-final-train` verilmediginde en iyi trial'in `best_epoch` degeriyle tum `trainval` uzerinde yeniden egit.
4. Test degerlendirmesini yalnizca en sonda harici `test` dizini uzerinde bir kez yap.

Aranan baslica hiperparametreler:

- `batch_size`
- `image_size`
- `lr`
- `weight_decay`
- `scheduler_factor`
- `scheduler_patience`
- `loss`
- `focal_gamma` (`loss=focal` ise)
- `pretrained` (`--search-pretrained` ile, sadece ResNet)

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
|-- dl/
|   |-- dataset.py
|   |-- engine.py
|   |-- losses.py
|   |-- utils.py
|   `-- models/
|       `-- resnet_classifier.py
`-- ciktilar/
    |-- modeller/
    |-- raporlar/
    `-- gorseller/
```

## Uretilen Ciktilar

- `model/ciktilar/modeller/best_resnet.pt`
- `model/ciktilar/raporlar/rapor_<model>_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_<model>.png`
- `model/ciktilar/gorseller/confusion_matrix_normalized_<model>.png`
- `model/ciktilar/gorseller/classification_summary_<model>.png`
- `model/ciktilar/gorseller/prediction_confidence_<model>.png`
- `model/ciktilar/gorseller/roc_pr_curves_<model>.png`
- `model/ciktilar/gorseller/training_curves_<model>.png`

## Ozellikler

- Early stopping ve best checkpoint kaydi
- ReduceLROnPlateau scheduler
- Class weights veya focal loss ile sinif dengesizligi yonetimi
- Grup bilgisi cikartilabiliyorsa leak-free split, aksi durumda uyari ile fallback stratejisi
- Accuracy, precision, recall ve F1 macro raporlamasi
- Normalize confusion matrix, sinif bazli performans, guven dagilimi ve ROC/PR egirileri
- CUDA varsa GPU, yoksa CPU fallback
