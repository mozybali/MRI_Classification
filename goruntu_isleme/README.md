# Goruntu Isleme Modulu

Bu modul, ham MRI goruntulerini model egitimine hazir hale getirmek icin on isleme, ozellik cikarma, veri bolme ve olceklendirme adimlarini toplar.

## Calistirma

Repo kokunden onerilen komut:

```bash
mri-preprocess --action menu
```

Dogrudan Python ile:

```bash
python -m goruntu_isleme.ana_islem --action menu
```

Etkilesimsiz ornekler:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/AugmentedAlzheimerDataset --output-dir goruntu_isleme/cikti
mri-preprocess --action extract --input-dir goruntu_isleme/cikti
mri-preprocess --action all --input-dir Veri_Seti/AugmentedAlzheimerDataset --output-dir goruntu_isleme/cikti --yes
```

## Desteklenen Veri Yapisi

```text
Veri_Seti/AugmentedAlzheimerDataset/<SinifAdi>/
Veri_Seti/OriginalDataset/<SinifAdi>/
Veri_Seti/<SinifAdi>/                  # Geri uyumluluk
```

Siniflar:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

On isleme varsayilan olarak `AugmentedAlzheimerDataset` klasorunu kullanir. Kok `Veri_Seti` verilirse uygun alt klasor otomatik secilir.

## Is Akisi

Interaktif menudeki temel adimlar:

- `1`: Goruntuleri on isler
- `2`: Ozellik cikarir ve `goruntu_ozellikleri.csv` uretir
- `3`: CSV icindeki `NaN` degerleri temizler (`drop`, `mean`, `median`, `zero`)
- `4`: Veriyi boler ve scaler'i sadece egitim setine fit ederek olceklendirir
- `5`: Istatistik raporu gosterir
- `6`: Ham CSV uzerinden veri boler
- `7`: `1 -> 2 -> 4 -> 5` adimlarini tek akis halinde calistirir

## Varsayilan Ayarlar

- `VERI_ARTIRMA_AKTIF = False`
- `BIAS_FIELD_CORRECTION_AKTIF = False`
- `SKULL_STRIPPING_AKTIF = False`
- `REGISTRATION_AKTIF = False`
- `SCALING_METODU = "robust"`

Kod tabani `bias field correction`, `skull stripping`, `registration` ve 3D islem icin genislemeye aciktir; ancak varsayilan 2D JPG akisinda bu ozellikler kapali gelir.

## Veri Sizintisi Notu

Ayni kaynaktan tureyen dosyalar, ornegin `26.jpg` ve `26 (19).jpg`, ayri split'lere dusmemesi icin grup mantigi ile ele alinmaya calisilir. Bu davranis egitim ve degerlendirme arasinda sizinti riskini azaltir.

## Ciktilar

Varsayilan cikti klasoru: `goruntu_isleme/cikti`

Uretilen baslica dosyalar:

- Islenmis goruntuler
- `goruntu_ozellikleri.csv`
- `goruntu_ozellikleri_scaled.csv`
- `egitim.csv`, `dogrulama.csv`, `test.csv`
- `egitim_scaled.csv`, `dogrulama_scaled.csv`, `test_scaled.csv`
- `feature_scaler.pkl`

## Yardimci Komutlar

```bash
python goruntu_isleme/pipeline_quick_test.py
python goruntu_isleme/test_pipeline.py ornek_goruntu.jpg
```

## Notlar

- Boyut ve dosya boyutu gibi sayisal meta kolonlar modele verilmez.
- Paralel islem sayisi `GorselIsleyici.n_jobs` veya `OzellikCikarici.n_jobs` uzerinden sinirlandirilabilir.
- Tespit edilen `NaN` degerleri icin genelde `3 -> 4 -> 5` adimlarini yeniden calistirmak yeterlidir.
