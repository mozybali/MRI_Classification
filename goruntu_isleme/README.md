# Görüntü İşleme Modülü

Bu modül, ham MRI görüntülerini model eğitimine hazırlayan 2D ön işleme, özellik çıkarma, veri bölme ve ölçeklendirme adımlarını yönetir. Varsayılan akış `Veri_Seti/OriginalDataset` içindeki sınıf klasörlerinden başlar ve tüm çıktıları `goruntu_isleme/cikti` altına yazar.

## Dosya Yapısı

```text
goruntu_isleme/
|-- __init__.py
|-- ana_islem.py
|-- ayarlar.py
|-- goruntu_isleyici.py
|-- ozellik_cikarici.py
`-- README.md
```

- `ana_islem.py`: `mri-preprocess` CLI giriş noktası ve menü/action akışı.
- `ayarlar.py`: Varsayılan yollar, sınıflar, ön işleme, augmentation, split ve CSV ayarları.
- `goruntu_isleyici.py`: Görüntü yükleme, kalite kontrol, normalizasyon, skull stripping, hizalama, augmentation ve `trainval/test` üretimi.
- `ozellik_cikarici.py`: Görüntü özellikleri, CSV üretimi, NaN temizleme, split ve scaler işlemleri.

## Çalıştırma

Kurulumdan sonra önerilen komut:

```bash
mri-preprocess --action menu
```

Doğrudan Python modülüyle:

```bash
python -m goruntu_isleme.ana_islem --action menu
```

## Aksiyonlar

- `menu`: Etkileşimli menüyü açar.
- `preprocess`: Ham görüntüleri kaynak grupları koruyarak `trainval/test` yapısında işler.
- `extract`: İşlenmiş `trainval/test` görüntülerinden özellik CSV'leri üretir.
- `clean-nan`: Verilen CSV içindeki NaN değerleri temizler.
- `scale`: CSV'yi eğitim/doğrulama/test olarak böler ve scaler'ı yalnızca eğitim setine fit eder.
- `report`: CSV üzerinden istatistik raporu basar.
- `split`: Ham özellik CSV'sinden split üretir.
- `all`: `preprocess -> extract -> scale -> report` sırasını otomatik çalıştırır.

## Sık Kullanılan Komutlar

Tam 2D akış:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

Sadece ön işleme:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Özellik çıkarma:

```bash
mri-preprocess --action extract --input-dir goruntu_isleme/cikti --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv
```

NaN temizleme:

```bash
mri-preprocess --action clean-nan --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --method median
```

Bölme ve ölçeklendirme:

```bash
mri-preprocess --action scale --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti --method robust
```

Ham CSV üzerinden veri bölme:

```bash
mri-preprocess --action split --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti
```

## Önemli Parametreler

- `--action`: Çalıştırılacak işlem.
- `--mode`: `2d` veya opsiyonel modül mevcutsa `3d`.
- `--input-dir`: Girdi klasörü.
- `--output-dir`: Çıktı klasörü.
- `--csv-path`: İşlem yapılacak CSV yolu.
- `--test-csv-path`: Harici/original test özellik CSV yolu. Verilmezse eşleşen `test_*.csv` veya `test_goruntu_ozellikleri.csv` otomatik aranır.
- `--method`: `clean-nan` için `drop|mean|median|zero`; `scale|all` için `minmax|robust|standard|maxabs`.
- `--yes`: `all` aksiyonunda onay sorusunu atlar.
- `--volume-path`, `--class-name`, `--model-path`: Opsiyonel 3D modül varsa kullanılan ek parametreler.

## Varsayılanlar

- Girdi klasörü: `Veri_Seti/OriginalDataset`
- Çıktı klasörü: `goruntu_isleme/cikti`
- Hedef görüntü boyutu: `256x256`
- Desteklenen uzantılar: `.jpg`, `.jpeg`, `.png`
- Split oranları: eğitim `%70`, doğrulama `%15`, test `%15`
- Ölçeklendirme metodu: `robust`
- Sabit sınıflar: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`

Preprocess sonrası beklenen çıktı yapısı:

```text
goruntu_isleme/cikti/
|-- trainval/<SinifAdi>/
`-- test/<SinifAdi>/
```

## Üretilen Çıktılar

- İşlenmiş `trainval/` ve `test/` görüntüleri.
- `goruntu_ozellikleri.csv`
- `test_goruntu_ozellikleri.csv`
- `egitim.csv`, `dogrulama.csv`, `test.csv`
- `egitim_scaled.csv`, `dogrulama_scaled.csv`, `test_scaled.csv`
- `feature_scaler.pkl`

## Veri Sızıntısı Notu

`preprocess` adımı ham veriyi önce `trainval/test` olarak ayırır. Aynı kaynaktan türeyen görüntüler mümkün olduğunca aynı grupta tutulur. Test tarafında augmentation uygulanmaz; validation ve test CSV'leri original-only satırlardan oluşturulur. Scaler yalnızca eğitim split'ine fit edilir, ardından doğrulama ve test split'lerine uygulanır.

## Yardımcı Komutlar

```bash
python tests/pipeline_quick_test.py
python tests/test_pipeline.py ornek_goruntu.jpg
```
