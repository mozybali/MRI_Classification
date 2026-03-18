# Model Egitim Modulu

MRI beyin goruntulerinden demans seviyesini siniflandirmak icin PyTorch tabanli derin ogrenme modulleri.

## Desteklenen Modeller

| Model | Aciklama |
|-------|----------|
| **ResNet** | Pretrained ResNet18 + classification head |
| **U-Net** | U-Net encoder + GAP + FC classification head (segmentasyon mask'i gerektirmez) |

## Gereksinimler

Ana dizindeki `requirements.txt` tum bagimliliklari icerir (`torch`, `torchvision`, `scikit-learn` vb.).

## Kullanim

### Egitim

Proje iki ayri veri kaynagi kullanir:
- **Augmented** (`Veri_Seti/AugmentedAlzheimerDataset`): train + validation
- **Original** (`Veri_Seti/OriginalDataset`): test

```bash
# ResNet ile egitim (varsayilan dizinler)
python train.py --model resnet --epochs 50 --batch-size 32

# U-Net ile egitim
python train.py --model unet --epochs 50 --batch-size 16

# Focal loss kullan
python train.py --model resnet --loss focal --lr 3e-4

# Opsiyonel: ImageNet pretrained agirliklari
python train.py --model resnet --pretrained

# Ozel veri dizinleri
python train.py --model resnet --trainval-dir ../Veri_Seti/AugmentedAlzheimerDataset --test-dir ../Veri_Seti/OriginalDataset
python train.py --model unet --val-ratio 0.2
```

### Tahmin (Inference)

```bash
# Tek goruntu
python inference.py --model-path ciktilar/modeller/best_resnet.pt --image /path/to/image.jpg

# Batch tahmin
python inference.py --model-path ciktilar/modeller/best_unet.pt --batch /path/to/folder/
```

## Dosya Yapisi

```text
model/
|-- train.py               # Egitim giris noktasi (CLI)
|-- inference.py            # Tahmin scripti (CLI)
|-- ayarlar.py              # Merkezi konfigürasyon
|-- dl/
|   |-- dataset.py          # MRIDataset + DataLoader olusturma
|   |-- engine.py           # train_one_epoch, evaluate, EarlyStopping
|   |-- losses.py           # FocalLoss, compute_class_weights
|   |-- utils.py            # set_seed, get_device, plot_confusion_matrix
|   `-- models/
|       |-- resnet_classifier.py  # ResNet18 wrapper
|       `-- unet_classifier.py    # U-Net encoder + classification head
`-- ciktilar/
    |-- modeller/            # .pt checkpoint dosyalari
    |-- raporlar/            # JSON performans raporlari
    `-- gorseller/           # Confusion matrix, egitim egrileri
```

## Ozellikler

- Early stopping (sabir degeri ayarlanabilir)
- Best checkpoint kaydı (val_loss'a gore)
- ReduceLROnPlateau learning rate scheduler
- Class weights veya Focal Loss ile sinif dengesizligi yonetimi
- Kaynak-grup leak-free split (augment turevleri ayni split'te tutulur)
- Augmented veri → train/val, Original veri → test (veri karisimi yok)
- Train: augmentation + normalize; Val/Test: sadece resize + normalize
- Accuracy, precision, recall, F1 (macro) metrikleri
- Confusion matrix ve egitim egrileri gorselleri
- GPU otomatik algilama, yoksa CPU fallback
- Deterministic seed (varsayilan: 42)

## Yapilandirma

`ayarlar.py` uzerinden:
- Train/Val veri dizini (Augmented)
- Test veri dizini (Original)
- Cikti klasorleri
- Rastgele tohum degeri
