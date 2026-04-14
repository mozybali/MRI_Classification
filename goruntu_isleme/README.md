# Goruntu Isleme Modulu

Bu modul, ham MRI goruntulerini model egitimine hazir hale getirmek icin 2D on isleme, ozellik cikarma, veri bolme ve olceklendirme adimlarini tek akista toplar.

## Calistirma

Repo kokunden onerilen komut:

```bash
mri-preprocess --action menu
```

Dogrudan Python ile:

```bash
python -m goruntu_isleme.ana_islem --action menu
```

## Aksiyonlar

- `menu`: Interaktif menu
- `preprocess`: Goruntuleri leak-free `trainval/test` yapisinda on isler
- `extract`: Islenmis `trainval/test` goruntulerinden ozellik CSV'leri uretir
- `clean-nan`: CSV icindeki NaN degerleri temizler
- `scale`: Veriyi boler ve scaler'i egitim setine gore fit eder
- `report`: CSV uzerinden istatistik raporu gosterir
- `split`: Ham CSV uzerinden veri bolme yapar
- `all`: `preprocess -> extract -> scale -> report` akisini otomatik calistirir

## Sik Kullanilan Komutlar

Tam 2D akis:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

Sadece on isleme:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Ozellik cikarma:

```bash
mri-preprocess --action extract --input-dir goruntu_isleme/cikti --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv
```

NaN temizleme:

```bash
mri-preprocess --action clean-nan --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --method median
```

Bolme ve olceklendirme:

```bash
mri-preprocess --action scale --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti --method robust
```

Ham CSV uzerinden veri bolme:

```bash
mri-preprocess --action split --csv-path goruntu_isleme/cikti/goruntu_ozellikleri.csv --output-dir goruntu_isleme/cikti
```

## Onemli Parametreler

- `--action`: Calistirilacak islem
- `--mode`: `2d` veya opsiyonel modul varsa `3d`
- `--input-dir`: Girdi klasoru
- `--output-dir`: Cikti klasoru
- `--csv-path`: Islem yapilacak CSV dosyasi
- `--test-csv-path`: Harici/original test ozellik CSV yolu. Belirtilmezse `goruntu_isleme/cikti/test_goruntu_ozellikleri.csv` otomatik aranir
- `--method`: `clean-nan` icin `drop|mean|median|zero`, `scale|all` icin `minmax|robust|standard|maxabs`
- `--yes`: `all` aksiyonunda onayi atlar

3D yolunda ek olarak:

- `--volume-path`: 3D hacim dosyasi veya DICOM klasoru
- `--class-name`: 3D islem icin sinif adi
- `--model-path`: 3D model dosyasi

## Varsayilanlar

- Varsayilan giris klasoru: `Veri_Seti/OriginalDataset`
- Varsayilan cikti klasoru: `goruntu_isleme/cikti`
- Varsayilan scaling metodu: `robust`
- Varsayilan akis 2D'dir; 3D ancak opsiyonel modul mevcutsa kullanilabilir

Desteklenen veri yapilari:

```text
Veri_Seti/OriginalDataset/<SinifAdi>/
```

Preprocess sonrasi uretilen varsayilan goruntu yapisi:

```text
goruntu_isleme/cikti/
|-- trainval/<SinifAdi>/
`-- test/<SinifAdi>/
```

## Uretilen Ciktilar

- Islenmis `trainval/` ve `test/` goruntuleri
- `goruntu_ozellikleri.csv`
- `test_goruntu_ozellikleri.csv`
- `goruntu_ozellikleri_scaled.csv`
- `egitim.csv`, `dogrulama.csv`, `test.csv`
- `egitim_scaled.csv`, `dogrulama_scaled.csv`, `test_scaled.csv`
- `feature_scaler.pkl`

## Veri Sizintisi Notu

Split mantigi, ayni kaynaktan tureyen dosyalari mumkun oldugunca ayni grupta tutar. `preprocess` adimi ham veriyi once `trainval/test` olarak ayirir. Test split'i yalnizca original goruntulerden olusur; augmentasyon yalnizca `trainval` tarafinda kalir. Daha sonra `scale/split` adimlari validation ve test CSV'lerini original-only olarak uretir.

## Yardimci Komutlar

```bash
python tests/pipeline_quick_test.py
python tests/test_pipeline.py ornek_goruntu.jpg
```
