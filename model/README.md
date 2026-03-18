# Model Egitim Modulu

Bu modul, MRI beyin goruntulerinden demans seviyesini siniflandirmak icin PyTorch tabanli egitim ve inference akisini saglar.

## Desteklenen Modeller

| Model | Aciklama |
|-------|----------|
| `resnet` | ResNet18 tabanli siniflandirici |
| `unet` | U-Net encoder + classification head |

## Veri Politikasi

Varsayilan olarak iki ayri veri kaynagi kullanilir:

- `Veri_Seti/AugmentedAlzheimerDataset`: train + validation
- `Veri_Seti/OriginalDataset`: test

Bu ayrim, augment edilmis goruntulerle egitim yaparken nihai degerlendirmeyi original veri uzerinde tutmak icin kullanilir.

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

Diger ornekler:

```bash
mri-train --model resnet --loss focal --lr 3e-4
mri-train --model resnet --pretrained
mri-train --model resnet --trainval-dir Veri_Seti/AugmentedAlzheimerDataset --test-dir Veri_Seti/OriginalDataset
mri-train --model unet --val-ratio 0.2
```

Temel argumanlar:

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

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
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

## Ozellikler

- Early stopping ve best checkpoint kaydi
- ReduceLROnPlateau scheduler
- Class weights veya focal loss ile sinif dengesizligi yonetimi
- Kaynak-grup mantigi ile veri sizintisini azaltan split stratejisi
- Accuracy, precision, recall, F1 (macro) raporlamasi
- Confusion matrix ve egitim egrileri gorselleri
- CUDA varsa GPU, yoksa CPU fallback

## Ciktilar

- `model/ciktilar/modeller/`: `.pt` checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari
- `model/ciktilar/gorseller/`: Confusion matrix ve egitim egrileri
