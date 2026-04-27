# Görüntü İşleme Modülü

`goruntu_isleme`, MRI görüntülerini modelleme akışına hazırlayan 2D ön işleme ve özellik çıkarma modülüdür. Varsayılan akış `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerini okur, görüntüleri kalite kontrolden geçirip standartlaştırır, işlenmiş görüntüleri `trainval/test` olarak kaydeder, görüntü özelliklerinden CSV üretir ve eğitim/doğrulama/test bölme ile leakage-free ölçeklendirme yapar.

## Dosya Yapısı

Kaynak ağaç şu dosyalardan oluşur:

```text
goruntu_isleme/
|-- __init__.py
|-- ana_islem.py
|-- ayarlar.py
|-- goruntu_isleyici.py
|-- ozellik_cikarici.py
`-- README.md
```

`__pycache__/` Python tarafından çalışma sırasında üretilebilir. `goruntu_isleme/cikti/` ise pipeline çalıştırıldığında oluşan çıktı klasörüdür; kaynak dosya olarak düşünülmemelidir.

## Python Dosyaları

- `ana_islem.py`: Komut satırı giriş noktasıdır. `mri-preprocess` komutunu, etkileşimli menüyü ve `preprocess`, `extract`, `clean-nan`, `scale`, `report`, `split`, `all` aksiyonlarını yönetir.
- `ayarlar.py`: Proje kökü, varsayılan giriş/çıkış yolları, sınıflar, görüntü ön işleme ayarları, augmentation ayarları, split oranları, CSV adları ve varsayılan scaling metodunu tutar.
- `goruntu_isleyici.py`: `GorselIsleyici` sınıfını içerir. Görüntü listeleme, kalite kontrol, gri ton yükleme, percentile normalizasyon, CLAHE, isteğe bağlı filtreleme/skull stripping/bias correction/registration, resize, augmentation ve `trainval/test` görüntü üretiminden sorumludur.
- `ozellik_cikarici.py`: `OzellikCikarici` sınıfını ve `veri_boluntule`, `veri_setini_bol_ve_olceklendir` yardımcılarını içerir. İşlenmiş görüntülerden sayısal özellik çıkarır, CSV yazar, NaN temizler, grup bazlı split yapar, scaler'ı yalnızca eğitim verisine fit eder ve scaler dosyasını kaydeder.
- `__init__.py`: Paket dışına `GorselIsleyici`, `OzellikCikarici`, `veri_boluntule` ve `veri_setini_bol_ve_olceklendir` isimlerini açar.

## Komut Satırından Çalıştırma

`pyproject.toml` içinde tanımlı komut:

```bash
mri-preprocess --action menu
```

Bu komutun kullanılabilmesi için proje paket olarak kurulmuş olmalıdır:

```bash
pip install -e .
```

Paket kurulmadan doğrudan modül olarak da çalıştırılabilir:

```bash
python3 -m goruntu_isleme.ana_islem --action menu
```

Temel bağımlılıklar `requirements.txt` ve `pyproject.toml` içinde tanımlıdır.

## Aksiyonlar

| Aksiyon | Açıklama |
| --- | --- |
| `menu` | Etkileşimli menüyü açar. Varsayılan aksiyondur. |
| `preprocess` | Ham 2D görüntüleri işler. Girdi zaten `trainval/test` yapısındaysa split'i korur; değilse kaynak gruplara göre `trainval/test` ayırır. |
| `extract` | İşlenmiş görüntülerden özellik CSV'si üretir. `trainval/test` yapısı algılanırsa trainval ve test için ayrı CSV yazar. |
| `clean-nan` | Bir CSV'deki NaN değerleri seçilen yöntemle temizler, CSV'yi yerinde günceller ve `.csv.bak` yedeği oluşturur. |
| `scale` | Özellik CSV'sini eğitim/doğrulama/test olarak böler, sayısal NaN'ları eğitim medyanıyla doldurur, scaler'ı eğitim setine fit eder ve tüm split'lere uygular. |
| `report` | CSV için sınıf dağılımı, temel istatistikler ve eksik değer raporu basar. |
| `split` | Ham özellik CSV'sini eğitim/doğrulama/test CSV'lerine böler; scaling uygulamaz. |
| `all` | `preprocess -> extract -> scale -> report` sırasını çalıştırır. `--yes` verilmezse onay ister. |

`--mode 3d` argümanı CLI'da yer alır, ancak bu repo ağacında opsiyonel 3D modül dosyası bulunmadığı için 3D işlem modül eklenmeden çalışmaz.

## Sık Kullanılan Örnekler

Tam 2D akışı çalıştır:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --method robust --yes
```

Sadece ön işleme yap:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

İşlenmiş görüntülerden özellik çıkar:

```bash
mri-preprocess --action extract --input-dir goruntu_isleme/cikti --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv
```

NaN değerleri medyan ile temizle:

```bash
mri-preprocess --action clean-nan --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --method median
```

Train/validation/test split ve robust scaling uygula:

```bash
mri-preprocess --action scale --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti --method robust
```

Harici test CSV yolunu açıkça ver:

```bash
mri-preprocess --action scale --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --test-csv-path goruntu_isleme/cikti/test_goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti
```

Sadece split CSV'lerini üret:

```bash
mri-preprocess --action split --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti
```

Rapor göster:

```bash
mri-preprocess --action report --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv
```

## Önemli CLI Parametreleri

- `--action`: Çalıştırılacak aksiyon. Geçerli değerler: `menu`, `preprocess`, `extract`, `clean-nan`, `scale`, `report`, `split`, `all`.
- `--mode`: Ön işleme modu. Varsayılan `2d`; `3d` yalnızca opsiyonel 3D modül mevcutsa anlamlıdır.
- `--input-dir`: Girdi klasörü. `preprocess`, `extract` ve `all` içinde kullanılır.
- `--output-dir`: Çıktı klasörü. `preprocess`, `scale`, `split` ve `all` içinde kullanılır.
- `--csv-path`: Okunacak veya yazılacak ana CSV yolu.
- `--test-csv-path`: Harici/original test özellik CSV yolu. Verilmezse `--output-dir/test_goruntu_ozellikleri.csv` veya `--csv-path` ile aynı klasörde `test_<csv_adı>` aranır.
- `--method`: `clean-nan` için `drop`, `mean`, `median`, `zero`; `scale` ve `all` için `minmax`, `robust`, `standard`, `maxabs`.
- `--yes`: `all` aksiyonundaki etkileşimli onayı atlar.
- `--volume-path`, `--class-name`, `--model-path`: Sadece opsiyonel 3D akış için ayrılmış parametrelerdir.

## Varsayılan Yollar ve Ayarlar

- Varsayılan girdi klasörü: `Veri_Seti/OriginalDataset`
- Varsayılan çıktı klasörü: `goruntu_isleme/cikti`
- Beklenen sınıf klasörleri: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`
- Sınıf etiketleri: `NonDemented=0`, `VeryMildDemented=1`, `MildDemented=2`, `ModerateDemented=3`
- Desteklenen görüntü uzantıları: `.jpg`, `.jpeg`, `.png`
- Hedef görüntü boyutu: `256x256`
- Split oranları: eğitim `%70`, doğrulama `%15`, test `%15`
- Rastgele tohum: `42`
- Varsayılan scaling metodu: `robust`

Varsayılan 2D ön işleme ayarları:

- Kalite kontrol aktiftir. Çok karanlık, çok aydınlık, düşük kontrastlı veya siyah piksel oranı yüksek görüntüler elenir.
- Normalizasyon stratejisi `standard`: `%1-%99` percentile clipping, `0-255` ölçekleme, sabit `CLAHE_CLIP_LIMIT=2.0` ile CLAHE ve ardından resize uygulanır.
- `Z_SCORE_NORMALIZASYON_AKTIF=True` olsa da varsayılan `standard` stratejisinde z-score uygulanmaz; z-score yalnızca `NORMALIZASYON_STRATEJISI="aggressive"` olduğunda kullanılır.
- `SKULL_STRIPPING_AKTIF=False`, `BIAS_FIELD_CORRECTION_AKTIF=False`, `REGISTRATION_AKTIF=False`.
- `GELISMIS_FILTRE_AKTIF=False`; bu nedenle varsayılan `auto` filtreleme görüntüyü değiştirmez.
- `VERI_ARTIRMA_AKTIF=False` ve `ARTIRMA_CARPANI=0`; disk üzerinde augmentation varsayılan olarak üretilmez.

## Beklenen Çıktı Yapısı

`preprocess` sonrası işlenmiş görüntüler şu yapıda üretilir:

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

İşlenmiş görüntüler `.png` olarak kaydedilir. Kaynak dosya adı ve orijinal uzantı korunarak örneğin `ornek.jpg` için `ornek_jpg.png` üretilir. Augmentation açıksa ek dosyalar `ornek_jpg_aug1.png` biçiminde yazılır.

`extract`, `split`, `scale` ve `all` çalıştırıldığında çıktı klasöründe şu dosyalar oluşabilir:

```text
goruntu_isleme/cikti/
|-- goruntu_ozellikleri.csv
|-- test_goruntu_ozellikleri.csv
|-- egitim.csv
|-- dogrulama.csv
|-- test.csv
|-- egitim_scaled.csv
|-- dogrulama_scaled.csv
|-- test_scaled.csv
`-- feature_scaler.pkl
```

`--csv-path` özel bir dosya adıyla verilirse test özellik CSV adı aynı klasörde `test_<csv_adı>` olarak belirlenir.

## CSV ve Scaler Çıktıları

- `goruntu_ozellikleri.csv`: Train/validation tarafı için çıkarılan ham özellikler.
- `test_goruntu_ozellikleri.csv`: `test/` görüntü klasörü varsa test tarafı için çıkarılan ham özellikler.
- `egitim.csv`, `dogrulama.csv`, `test.csv`: Scaling öncesi split CSV'leri.
- `egitim_scaled.csv`, `dogrulama_scaled.csv`, `test_scaled.csv`: Sayısal özellikleri ölçeklendirilmiş split CSV'leri.
- `feature_scaler.pkl`: Eğitim setine fit edilen scaler, ölçeklenen kolon listesi ve metod bilgisini içeren pickle dosyası.
- `<csv>.csv.bak`: `clean-nan` aksiyonu çalıştırıldığında orijinal CSV yedeği.

Özellik CSV'lerinde dosya ve sınıf bilgileri yanında yoğunluk istatistikleri, histogram/entropi, kontrast, homojenlik, enerji, çarpıklık, basıklık, gradyan ve Otsu eşiği gibi sayısal özellikler bulunur. `model.sl.features.extract_texture_summary` import edilebilirse ek doku özet metrikleri de CSV'ye eklenir.

## Split, Augmentation ve Veri Sızıntısı Notları

- `preprocess`, girdi doğrudan sınıf klasörlerini içeriyorsa ham görüntüleri önce kaynak grup bazında `trainval/test` olarak böler. Girdi zaten `trainval/test` yapısındaysa mevcut split korunur.
- Kaynak grup, sınıf adı ve dosya kökünden türetilir. `_augN` ve `(1)` gibi türev adları aynı kaynak görüntüye bağlamak için temizlenir.
- Harici test seti oluşturulurken her sınıfta yeterli sayıda farklı kaynak grup olması gerekir; aksi durumda işlem hata verir.
- Test split'inde augmentation uygulanmaz. Augmentation aktif edilirse trainval tarafında üretilebilir; doğrulama ve test CSV'leri original-only satırlardan seçilir.
- `scale` ve `all` akışında NaN doldurma değerleri yalnızca eğitim setinden hesaplanan medyanlarla belirlenir.
- Scaler yalnızca `egitim.csv` üzerinden fit edilir; aynı scaler daha sonra `dogrulama.csv` ve `test.csv` üzerine uygulanır.
- Grup ayrımı doğrulanır. Aynı `kaynak_grup` değerinin eğitim, doğrulama ve test arasında paylaşılması veri sızıntısı olarak hata kabul edilir.
