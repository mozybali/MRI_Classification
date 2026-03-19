# Model Egitim Modulu

Bu modul, MRI goruntulerinden demans seviyesi siniflandirmak icin PyTorch tabanli egitim ve inference akisini saglar.

## Desteklenen Modeller

| Model | Aciklama |
|-------|----------|
| `resnet` | ResNet18 tabanli siniflandirici |
| `unet` | U-Net encoder + classification head |

## Varsayilan Veri Politikasi

- `Veri_Seti/AugmentedAlzheimerDataset`: train + validation
- `Veri_Seti/OriginalDataset`: test

Bu ayrim, augment edilmis goruntulerle egitim yaparken nihai degerlendirmeyi original veri uzerinde tutar.

## Egitim

Repo kokunden onerilen komutlar:

```bash
mri-train --model resnet --epochs 50 --batch-size 32
mri-train --model unet --epochs 50 --batch-size 16
```

Dogrudan Python ile:

```bash
python -m model.train --model resnet --epochs 50 --batch-size 32
```

Ek ornekler:

```bash
mri-train --model resnet --loss focal --lr 3e-4
mri-train --model resnet --pretrained
mri-train --model resnet --trainval-dir Veri_Seti/AugmentedAlzheimerDataset --test-dir Veri_Seti/OriginalDataset
mri-train --model unet --val-ratio 0.2
```

## Temel Parametreler

- `--model`: `resnet` veya `unet`
- `--epochs`: Epoch sayisi
- `--batch-size`: Batch boyutu
- `--lr`: Ogrenme hizi
- `--patience`: Early stopping sabir degeri
- `--image-size`: Giris goruntu boyutu
- `--trainval-dir`: Train + validation veri dizini
- `--test-dir`: Test veri dizini
- `--val-ratio`: Validation orani
- `--loss`: `ce` veya `focal`
- `--seed`: Rastgele tohum
- `--num-workers`: DataLoader worker sayisi
- `--pretrained`: Sadece ResNet icin ImageNet agirliklarini acar

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

## Dosya Yapisi

```text
model/
|-- train.py
|-- inference.py
|-- ayarlar.py
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
