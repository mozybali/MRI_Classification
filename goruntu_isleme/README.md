# Görüntü İşleme Modülü

`goruntu_isleme`, ham 2D beyin MRI görüntülerini modelleme aşamasına hazırlayan ön işleme modülüdür. Bu modülün ana sorumluluğu `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerini okumak, görüntüleri kalite kontrol ve standartlaştırma adımlarından geçirmek, ardından işlenmiş görüntüleri `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` yapısında kaydetmektir.

Özellik matrisi, XGBoost özellik cache'i, model eğitimi ve inference adımları `model/` modülü tarafında yürütülür. Bu nedenle bu klasördeki README, yalnızca görüntü ön işleme akışını anlatır.

## Projedeki Yeri

Tipik proje akışı şu sırayı izler:

1. Ham görüntüler `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutulur.
2. Bu modül `mri-preprocess` ile ham görüntüleri işler ve leak-free `trainval/test` klasörlerini üretir.
3. `model/` modülü bu işlenmiş klasörleri kullanarak ResNet veya XGBoost eğitir.
4. XGBoost için gerekli HOG, LBP, GLCM ve histogram özellikleri `model/sl/features.py` ve `model/sl/dataset.py` tarafında çıkarılır.

## Dosya Yapısı

Güncel kaynak ağacı:

```text
goruntu_isleme/
|-- __init__.py
|-- ana_islem.py
|-- ayarlar.py
|-- goruntu_isleyici.py
`-- README.md
```

Çalışma sırasında oluşabilecek dizinler:

```text
goruntu_isleme/
|-- __pycache__/
`-- cikti/
    |-- trainval/
    `-- test/
```

`__pycache__/` Python tarafından üretilir. `cikti/` ise ön işleme sonucudur; kaynak kodun parçası değildir.

## Python Dosyaları

- `ana_islem.py`: `mri-preprocess` komutunun ve interaktif menünün giriş noktasıdır. Mevcut CLI aksiyonları `menu` ve `preprocess` değerleridir.
- `ayarlar.py`: Proje kökü, veri seti yolları, sınıf adları, etiketler, hedef görüntü boyutu, normalizasyon, kalite kontrol, augmentation ve test split oranını merkezi olarak tanımlar.
- `goruntu_isleyici.py`: `GorselIsleyici` sınıfını içerir. Görüntü listeleme, kaynak grup belirleme, `trainval/test` bölme, kalite kontrol, gri ton yükleme, normalize etme, CLAHE, opsiyonel filtreleme/skull stripping/bias correction/registration, resize, augmentation ve toplu kayıt işlemlerinden sorumludur.
- `__init__.py`: Paket dışına `GorselIsleyici` sınıfını açar.

## Beklenen Girdi Yapısı

Varsayılan girdi klasörü:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Desteklenen uzantılar:

- `.jpg`
- `.jpeg`
- `.png`

`GorselIsleyici`, doğrudan sınıf klasörlerini içeren bir klasörü de okuyabilir. Klasör kökünde `OriginalDataset/` varsa bu alt klasörü otomatik çözmeye çalışır. Girdi zaten `trainval/test` yapısındaysa mevcut split korunur.

## Komut Satırından Çalıştırma

Komutun kullanılabilmesi için paket proje kökünden kurulmuş olmalıdır:

```bash
pip install -e .[dev] --no-deps
```

İnteraktif menü:

```bash
mri-preprocess --action menu
```

Ön işleme akışını doğrudan çalıştırma:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Paket kurulmadan modül olarak çalıştırma:

```bash
python3 -m goruntu_isleme.ana_islem --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

`--action` verilmezse varsayılan olarak interaktif menü açılır.

## CLI Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--action` | Çalıştırılacak işlem. Geçerli değerler: `menu`, `preprocess`. Varsayılan `menu`. |
| `--input-dir` | Girdi klasörü. Verilmezse `Veri_Seti/OriginalDataset` kullanılır. |
| `--output-dir` | Çıktı klasörü. Verilmezse `goruntu_isleme/cikti` kullanılır. |

Bu modülde `all`, `extract`, `scale`, `report`, `clean-nan`, `split` veya `--mode 3d` aksiyonları bulunmaz. CSV/scaler üretimi bu klasörün güncel görev kapsamı dışındadır.

## Ön İşleme Adımları

Tek görüntü için uygulanan temel sıra:

1. Görüntüyü PIL ile aç ve gri tona çevir.
2. Kalite kontrol uygula.
3. Ayara bağlı gürültü giderme uygula.
4. Ayara bağlı bias field correction uygula.
5. Ayara bağlı skull stripping uygula.
6. Ayara bağlı registration/hizalama uygula.
7. Normalizasyon stratejisini uygula.
8. Görüntüyü hedef boyuta getir.
9. İşlenmiş görüntüyü `.png` olarak kaydet.

Varsayılan normalizasyon stratejisi `standard` değeridir:

- `%1-%99` percentile clipping.
- `0-255` aralığına yoğunluk normalizasyonu.
- Sabit `CLAHE_CLIP_LIMIT=2.0` ile CLAHE.
- `256x256` yeniden boyutlandırma.

## Varsayılan Ayarlar

Temel ayarlar `ayarlar.py` içinde tutulur:

| Ayar | Varsayılan |
| --- | --- |
| Girdi klasörü | `Veri_Seti/OriginalDataset` |
| Çıktı klasörü | `goruntu_isleme/cikti` |
| Hedef boyut | `256x256` |
| Test oranı | `0.15` |
| Rastgele tohum | `42` |
| Normalizasyon stratejisi | `standard` |
| Kalite kontrol | Aktif |
| Disk üzerinde augmentation | Kapalı |
| Filtre metodu | `off` |
| Skull stripping | Kapalı |
| Bias field correction | Kapalı |
| Registration | Kapalı |

Sınıf ve etiket eşleşmesi:

| Sınıf | Etiket |
| --- | --- |
| `NonDemented` | `0` |
| `VeryMildDemented` | `1` |
| `MildDemented` | `2` |
| `ModerateDemented` | `3` |

Kalite kontrol varsayılan olarak çok karanlık, çok aydınlık, düşük kontrastlı veya siyah piksel oranı çok yüksek görüntüleri eler.

## Beklenen Çıktı Yapısı

`preprocess` tamamlandığında çıktı şu yapıda olur:

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

Kaydedilen dosyalar `.png` formatındadır. Kaynak dosya köküne orijinal uzantı eklenerek ad çakışması engellenir:

```text
ornek.jpg  -> ornek_jpg.png
ornek.png  -> ornek_png.png
```

Augmentation aktif edilirse ek dosyalar şu biçimde yazılır:

```text
ornek_jpg_aug1.png
ornek_jpg_aug2.png
```

Varsayılan ayarlarda augmentation kapalı olduğu için bu ek dosyalar üretilmez. Test split'i işlenirken augmentation çarpanı sıfırlanır.

## Split ve Veri Sızıntısı Politikası

- Ham veri doğrudan sınıf klasörlerinden geliyorsa `tum_gorselleri_isle_ve_bol`, veriyi kaynak grup bazında `trainval` ve `test` olarak ayırır.
- Kaynak grup, sınıf adı ve dosya kökünden türetilir. `_augN` ve `(1)` gibi türev ekleri temizlenerek aynı kaynaktan gelen görüntüler aynı grupta tutulur.
- Her sınıfta harici test split'i için yeterli kaynak grup bulunmalıdır; aksi durumda işlem hata verir.
- Girdi zaten `trainval/test` klasörlerini içeriyorsa yeniden bölme yapılmaz, mevcut split korunur.
- `trainval` ve `test` klasörlerinin kalite kontrol sonrasında beklenen sınıfları koruduğu doğrulanır.

## Model Modülüne Devam

Ön işleme bittikten sonra model eğitimi için önerilen komutlar:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

XGBoost eğitiminde özellik matrisi `model/sl/dataset.py` tarafından klasör ağacından üretilir ve istenirse `model/ciktilar/sl_ozellikler` altında `.npz` cache olarak tutulur.
