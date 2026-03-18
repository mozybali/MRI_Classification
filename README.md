# MRI Beyin Goruntusu Siniflandirma

MRI beyin goruntulerinden demans seviyesini tahmin etmek icin uctan uca bir derin ogrenme projesi. Repo; goruntu on isleme ve CNN tabanli siniflandirma modellerini (ResNet, U-Net) tek yerde toplar.

## Proje Yapisi

```text
MRI_Classification/
|-- Veri_Seti/                 # Ham goruntuler (sinif klasorleri)
|-- goruntu_isleme/            # On isleme + ozellik cikarma
|   |-- ana_islem.py           # Menu tabanli ana akis
|   |-- goruntu_isleyici.py    # On isleme pipeline'i
|   |-- ozellik_cikarici.py    # Ozellik cikarma ve CSV uretimi
|   |-- pipeline_quick_test.py # Hizli ortam kontrolu
|   |-- test_pipeline.py       # Tek goruntu pipeline gorsellestirme
|   `-- ayarlar.py             # Goruntu isleme ayarlari
|-- model/                     # Derin ogrenme egitimi ve inference
|   |-- dl/                    # DL modulleri
|   |   |-- dataset.py         # PyTorch Dataset ve DataLoader
|   |   |-- engine.py          # Egitim/degerlendirme dongusu
|   |   |-- losses.py          # FocalLoss, class weights
|   |   |-- utils.py           # Seed, device, gorsellestime
|   |   `-- models/
|   |       |-- resnet_classifier.py  # ResNet18 siniflandirici
|   |       `-- unet_classifier.py    # U-Net encoder + classification head
|   |-- train.py               # Egitim giris noktasi
|   |-- inference.py           # Tahmin scripti
|   `-- ayarlar.py             # Merkezi konfigürasyon
|-- requirements.txt
`-- LICENSE
```

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanim

### 1. Goruntu on isleme (istege bagli)

```bash
cd goruntu_isleme
python ana_islem.py
```

### 2. Model egitimi

Proje iki ayri veri kaynagi kullanir:
- **Augmented veri** (`Veri_Seti/AugmentedAlzheimerDataset`): Yalnizca **train + validation** icin
- **Original veri** (`Veri_Seti/OriginalDataset`): Yalnizca **test** icin

Bu yaklasim, augmente edilmis goruntulerle egitim yaparken modelin gercek performansinin orijinal veriler uzerinde olculmesini saglar.

```bash
# ResNet ile egitim (varsayilan dizinler)
python model/train.py --model resnet --epochs 50 --batch-size 32

# U-Net ile egitim
python model/train.py --model unet --epochs 50 --batch-size 16

# Focal loss ve ozel ogrenme hizi
python model/train.py --model resnet --loss focal --lr 3e-4

# Opsiyonel: ImageNet pretrained agirliklari
python model/train.py --model resnet --pretrained

# Ozel veri dizinleri ve val orani belirtme
python model/train.py --model resnet --trainval-dir Veri_Seti/AugmentedAlzheimerDataset --test-dir Veri_Seti/OriginalDataset
python model/train.py --model unet --val-ratio 0.2
```

Desteklenen argumanlar:
- `--model`: `resnet` veya `unet` (varsayilan: resnet)
- `--epochs`: Epoch sayisi (varsayilan: 50)
- `--batch-size`: Batch boyutu (varsayilan: 32)
- `--lr`: Ogrenme hizi (varsayilan: 1e-4)
- `--loss`: `ce` (CrossEntropy) veya `focal` (FocalLoss) (varsayilan: ce)
- `--patience`: Early stopping sabir degeri (varsayilan: 10)
- `--image-size`: Goruntu boyutu (varsayilan: 224)
- `--trainval-dir`: Train+Val icin augmented veri dizini (varsayilan: `Veri_Seti/AugmentedAlzheimerDataset`)
- `--test-dir`: Test icin original veri dizini (varsayilan: `Veri_Seti/OriginalDataset`)
- `--val-ratio`: Augmented veri icerisindeki validation orani (varsayilan: 0.15)
- `--seed`: Rastgele tohum (varsayilan: 42)
- `--pretrained`: Sadece ResNet icin ImageNet agirliklarini ac (varsayilan: kapali)

### 3. Tahmin (inference)

```bash
# Tek goruntu
python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --image /path/to/image.jpg

# Batch tahmin
python model/inference.py --model-path model/ciktilar/modeller/best_unet.pt --batch /path/to/folder/
```

## Teknik Detaylar

- **Framework**: PyTorch
- **Modeller**: ResNet18 (pretrained ImageNet), U-Net encoder + classification head
- **Siniflar**: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
- **Ozellikler**: Early stopping, best checkpoint, ReduceLROnPlateau scheduler, class weights / focal loss
- **Split**: Augmented veri → train/val, Original veri → test (kaynak-grup leak-free)
- **Donusumler**: Train: augmentation + normalize; Val/Test: sadece resize + normalize
- **Metrikler**: Accuracy, precision, recall, F1 (macro)
- **Gorseller**: Confusion matrix, egitim egrileri (loss/accuracy)
- **GPU**: Otomatik CUDA algilama, yoksa CPU fallback
- **Seed**: Deterministic (varsayilan: 42)

## Ciktilar

- `model/ciktilar/modeller/`: Egitilmis `.pt` checkpoint dosyalari
- `model/ciktilar/raporlar/`: JSON performans raporlari
- `model/ciktilar/gorseller/`: Confusion matrix ve egitim egrileri
- `goruntu_isleme/cikti/`: Islenmis goruntuler ve CSV dosyalari

## Lisans

MIT. Ayrinti icin `LICENSE` dosyasina bakin.
