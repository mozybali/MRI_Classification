# Görüntü İşleme Modülü

`goruntu_isleme`, ham 2D beyin MRI görüntülerini modelleme aşamasına hazırlayan ön işleme paketidir. Varsayılan akışta görüntüleri `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerinden okur, kalite kontrol ve standartlaştırma uygular, ardından `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` yapısında `.png` olarak kaydeder.

Bu klasör yalnızca görüntü ön işleme sorumluluğunu taşır. Özellik matrisi üretimi, XGBoost özellik cache'i, model eğitimi, hiperparametre araması ve inference işlemleri `model/` modülünde yürütülür.

## Projedeki Yeri

Tipik proje akışı:

1. Ham MRI görüntüleri `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutulur.
2. İsteğe bağlı EDA adımları `eda_analiz/` ile çalıştırılır.
3. Bu modül `mri-preprocess` komutu ile ham görüntüleri işler ve leak-free `trainval/test` klasörlerini üretir.
4. `model/` modülü, üretilen klasörleri kullanarak ResNet veya XGBoost eğitir.
5. XGBoost için gerekli HOG, LBP, GLCM ve histogram özellikleri model tarafında çıkarılır.

## Dosya Yapısı

Kaynak dosyalar:

```text
goruntu_isleme/
|-- __init__.py
|-- ana_islem.py
|-- artirma.py
|-- ayarlar.py
|-- goruntu_isleyici.py
|-- kalite_io.py
|-- on_isleme.py
|-- temel.py
|-- toplu_islem.py
|-- veri.py
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

`__pycache__/` Python tarafından üretilir. `cikti/` ön işleme çıktısıdır; kaynak kodun parçası değildir.

## Modül Mimarisi

Ana dış API `GorselIsleyici` sınıfıdır. Bu sınıf `goruntu_isleyici.py` içinde, işlevleri ayrı dosyalara bölünmüş mixin sınıflarını birleştirir.

| Dosya | Görev |
| --- | --- |
| `ana_islem.py` | `mri-preprocess` komutunun ve interaktif menünün giriş noktasıdır. Geçerli CLI aksiyonları `menu` ve `preprocess` değerleridir. |
| `ayarlar.py` | Proje kökü, veri yolları, sınıf adları, hedef boyut, normalizasyon, kalite kontrol, augmentation ve split ayarlarını merkezi olarak tanımlar. |
| `goruntu_isleyici.py` | `GorselIsleyici` sınıfını dış API olarak sunar ve eski tek dosya kullanımına dönük uyumluluk katmanı sağlar. |
| `temel.py` | `GorselIsleyici` durum yönetimi, rastgele tohumlama, çıktı dosya adı üretimi ve ortak yardımcıları içerir. |
| `veri.py` | Girdi klasörü çözümleme, görüntü listeleme, kaynak grup belirleme ve leak-free `trainval/test` bölme işlemlerini içerir. |
| `kalite_io.py` | PIL ile görüntü yükleme, gri tona çevirme, kalite kontrol ve görüntü kaydetme işlemlerini içerir. |
| `on_isleme.py` | Gürültü giderme, bias correction, skull stripping, registration, normalizasyon, CLAHE ve resize adımlarını içerir. |
| `artirma.py` | Disk üzerinde augmentation için rotasyon, parlaklık/kontrast, elastik deformasyon, crop, gürültü ve yoğunluk kayması işlemlerini içerir. |
| `toplu_islem.py` | Tekil görüntü kaydı, sınıf bazlı augmentation çarpanları, paralel/toplu işleme ve split çıktı üretimini içerir. |
| `__init__.py` | Paket dışına `GorselIsleyici` sınıfını açar. |

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

Desteklenen dosya uzantıları:

- `.jpg`
- `.jpeg`
- `.png`

`GorselIsleyici` şu girdi biçimlerini okuyabilir:

- Doğrudan sınıf klasörlerini içeren bir klasör.
- Kökünde `OriginalDataset/` bulunan bir klasör.
- Daha önce ayrılmış `trainval/test/<SinifAdi>/` yapısı.

Girdi zaten `trainval/test` yapısındaysa yeniden bölme yapılmaz; mevcut split korunur. Bu modül doğrudan `.nii` veya `.nii.gz` hacim dosyalarını okumaz.

## Sınıflar ve Etiketler

| Sınıf | Etiket |
| --- | --- |
| `NonDemented` | `0` |
| `VeryMildDemented` | `1` |
| `MildDemented` | `2` |
| `ModerateDemented` | `3` |

Sınıf klasörü adları kodda sabit kullanılır; büyük/küçük harf duyarlıdır.

## Komut Satırından Çalıştırma

Proje komutlarını kullanmak için paket proje kökünden kurulabilir:

```bash
pip install -e .
```

Geliştirme bağımlılıklarıyla kurulum:

```bash
pip install -e ".[dev]"
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

Bu modülde `all`, `extract`, `scale`, `report`, `clean-nan`, `split` veya `--mode 3d` aksiyonları bulunmaz. CSV, scaler, özellik cache'i ve model dosyası üretimi bu klasörün görev kapsamı dışındadır.

## Python API Kullanımı

```python
from pathlib import Path

from goruntu_isleme import GorselIsleyici

isleyici = GorselIsleyici()
istatistikler = isleyici.tum_gorselleri_isle_ve_bol(
    cikti_klasoru=Path("goruntu_isleme/cikti"),
    giris_klasoru=Path("Veri_Seti/OriginalDataset"),
)
```

## Ön İşleme Sırası

Tek görüntü için uygulanan temel sıra:

1. Görüntüyü PIL ile aç ve gri tona çevir.
2. Kalite kontrol uygula.
3. Ayara bağlı gürültü giderme uygula.
4. Ayara bağlı bias field correction uygula.
5. Ayara bağlı skull stripping uygula.
6. Ayara bağlı registration/hizalama uygula.
7. Seçili normalizasyon stratejisini uygula.
8. Görüntüyü hedef boyuta getir.
9. İşlenmiş görüntüyü `.png` olarak kaydet.

Normalizasyon stratejileri:

| Strateji | Davranış |
| --- | --- |
| `minimal` | Percentile clipping ve resize uygular. |
| `standard` | Percentile clipping, sabit CLAHE ve resize uygular. Varsayılan stratejidir. |
| `aggressive` | Percentile clipping, sabit CLAHE, z-score normalizasyonu ve resize uygular. |

Varsayılan `standard` stratejisinde `%1-%99` percentile clipping, `0-255` yoğunluk normalizasyonu, `CLAHE_CLIP_LIMIT=2.0` ve `256x256` yeniden boyutlandırma kullanılır.

## Yeniden Boyutlandırma

`BOYUTLANDIRMA_MODU` ayarı görüntülerin hedef boyuta nasıl getirileceğini belirler:

- `pad`: Varsayılan ve medikal olarak önerilen moddur. En-boy oranı korunur, görüntü hedef çerçeveye sığdırılır ve kalan kenarlar `PADDING_DEGERI` ile doldurulur.
- `stretch`: Görüntü doğrudan hedef boyuta gerilir. En-boy oranı korunmaz.

`PADDING_DEGERI` varsayılan olarak `0` değerindedir. Her iki modda da çıktı `(HEDEF_YUKSEKLIK, HEDEF_GENISLIK)` biçimindedir.

## Varsayılan Ayarlar

Temel ayarlar `ayarlar.py` içinde tutulur:

| Ayar | Varsayılan |
| --- | --- |
| Girdi klasörü | `Veri_Seti/OriginalDataset` |
| Çıktı klasörü | `goruntu_isleme/cikti` |
| Hedef boyut | `256x256` |
| Desteklenen uzantılar | `.jpg`, `.jpeg`, `.png` |
| Boyutlandırma modu | `pad` |
| Padding değeri | `0` |
| Test oranı | `0.15` |
| Rastgele tohum | `42` |
| Normalizasyon stratejisi | `standard` |
| Histogram eşitleme | Aktif |
| CLAHE clip limit | `2.0` |
| Filtre metodu | `off` |
| Skull stripping | Kapalı |
| Bias field correction | Kapalı |
| Registration | Kapalı |
| Morfolojik işlemler | Aktif |
| Disk üzerinde augmentation | Kapalı |
| Augmentation çarpanı | `0` |
| Sınıf bazlı augmentation | Kapalı |
| Kalite kontrol | Aktif |

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
- Her sınıfta harici test split'i için en az iki farklı kaynak grup bulunmalıdır; aksi durumda işlem hata verir.
- Girdi zaten `trainval/test` klasörlerini içeriyorsa yeniden bölme yapılmaz, mevcut split korunur.
- `trainval` ve `test` klasörlerinin kalite kontrol sonrasında beklenen sınıfları koruduğu doğrulanır.

## Bağımlılık Notları

- Görüntü yükleme ve kaydetme için `Pillow` kullanılır.
- CLAHE ve resize için OpenCV varsa öncelikli olarak kullanılır.
- OpenCV yoksa bazı CLAHE işlemleri için `scikit-image` fallback'i devreye girebilir.
- `SimpleITK`, sadece bias correction veya gelişmiş registration ayarları aktif edildiğinde anlamlıdır.
- Toplu işlem `multiprocessing.Pool` ile paralel çalışabilir; affine/rigid registration aktifse template tutarlılığı için sequential moda döner.

## Model Modülüne Devam

Ön işleme bittikten sonra model eğitimi için örnek komutlar:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

XGBoost eğitiminde özellik matrisi `model/sl/dataset.py` tarafından klasör ağacından üretilir ve istenirse `model/ciktilar/sl_ozellikler` altında `.npz` cache olarak tutulur.
