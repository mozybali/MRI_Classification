# Model Egitim Modulu

Bu modul, MRI goruntulerinden demans seviyesi siniflandirmak icin PyTorch tabanli egitim ve inference akisini saglar.

## Desteklenen Modeller

| Model | Aciklama |
|-------|----------|
| `resnet` | ResNet18 tabanli siniflandirici |
| `unet` | U-Net encoder + classification head |

## Varsayilan Veri Politikasi

- `Veri_Seti/OriginalDataset`: tek train/validation/test split kaynagi
- `validation` ve `test`: original-only
- `augmentation`: yalnizca train transform'u uzerinde

## Egitim

Repo kokunden onerilen komutlar:

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model unet --epochs 50 --batch-size 16
mri-train --model resnet --trainval-dir Veri_Seti/OriginalDataset
mri-train --model resnet --use-processed-trainval --trainval-dir goruntu_isleme/cikti
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
mri-train --model resnet --use-processed-trainval --trainval-dir goruntu_isleme/cikti
mri-train --model resnet --use-processed-trainval --use-processed-test --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model unet --val-ratio 0.2
```

Islenmis goruntulerle egitim icin, original veri uzerinden preprocess alip split'i daha sonra model tarafinda yapmak onerilir:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

## Temel Parametreler

- `--model`: `resnet` veya `unet`
- `--epochs`: Epoch sayisi
- `--batch-size`: Batch boyutu
- `--lr`: Ogrenme hizi
- `--patience`: Early stopping sabir degeri
- `--image-size`: Giris goruntu boyutu
- `--trainval-dir`: Split kaynagi veya train+validation veri dizini
- `--test-dir`: Opsiyonel harici test veri dizini
- `--use-processed-trainval`: Varsayilan train+validation kaynagini islenmis goruntu ciktilarina cevirir
- `--use-processed-test`: Varsayilan test kaynagini islenmis goruntu ciktilarina cevirir
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

## Inference

Tek goruntu:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

Klasor bazli batch tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_unet.pt --batch ornek_klasor
```

Dogrudan Python ile:

```bash
python -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Bayes Search

Optuna `TPESampler` kullanilarak Bayesian-style hiperparametre aramasi yapilabilir.

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1
python -m model.hpo --model unet --trials 30 --metric loss --skip-final-train
```

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
- `best_run/` (final egitim kapatilmazsa)

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
|       |-- resnet_classifier.py
|       `-- unet_classifier.py
`-- ciktilar/
    |-- modeller/
    |-- raporlar/
    `-- gorseller/
```

## Uretilen Ciktilar

- `model/ciktilar/modeller/best_resnet.pt`
- `model/ciktilar/modeller/best_unet.pt`
- `model/ciktilar/raporlar/rapor_<model>_<timestamp>.json`
- `model/ciktilar/gorseller/confusion_matrix_<model>.png`
- `model/ciktilar/gorseller/training_curves_<model>.png`

## Ozellikler

- Early stopping ve best checkpoint kaydi
- ReduceLROnPlateau scheduler
- Class weights veya focal loss ile sinif dengesizligi yonetimi
- Grup bilgisi cikartilabiliyorsa leak-free split, aksi durumda uyari ile fallback stratejisi
- Accuracy, precision, recall ve F1 macro raporlamasi
- CUDA varsa GPU, yoksa CPU fallback
