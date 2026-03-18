# Goruntu Isleme Modulu

Ham MRI goruntulerini kalite kontrolden gecirir, normalize eder, yeniden boyutlandirir ve model egitimine hazir ozellik CSV'leri uretir.

## Kurulum

```bash
pip install -r ../requirements.txt
```

Veri yapisi:

```text
../Veri_Seti/AugmentedAlzheimerDataset/<SinifAdi>/
../Veri_Seti/OriginalDataset/<SinifAdi>/
# (Geri uyumluluk) ../Veri_Seti/<SinifAdi>/
```

Siniflar:
- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

## Is Akisi

### 1) Ana menu

```bash
python ana_islem.py
```

- `1 On isleme`: Varsayilan akış kalite kontrol -> median filtre -> yogunluk normalizasyonu -> adaptif CLAHE -> yeniden boyutlandirma.
- `2 Ozellik cikarma`: Yogunluk, doku ve gradyan tabanli ozelliklerden `goruntu_ozellikleri.csv` uretir.
- `3 NaN temizleme`: `drop`, `mean`, `median`, `zero`.
- `4 Veri bolme + olceklendirme`: Veri once train/validation/test olarak ayrilir; scaler sadece egitim setine fit edilir.
- `5 Istatistik raporu`: CSV ozetlerini gosterir.
- `6 Veri bolme`: Ham CSV'yi split eder.
- `7 Otomatik`: `1 -> 2 -> 4 -> 5` adimlarini guvenli sirada calistirir.

Not: On isleme varsayilan olarak `AugmentedAlzheimerDataset` klasorunu kullanir.
Kok `Veri_Seti` verilirse uygun alt klasor otomatik secilir.

## Varsayilanlar

- `VERI_ARTIRMA_AKTIF = False`
- `BIAS_FIELD_CORRECTION_AKTIF = False`
- `SKULL_STRIPPING_AKTIF = False`
- `REGISTRATION_AKTIF = False`
- `SCALING_METODU = "robust"`

Kodda `bias field correction`, `skull stripping`, `registration` ve augmentasyon desteklenir; ancak 2D JPG veri seti icin varsayilan olarak kapatilidir.

## Veri Sizintisi Notu

Veri setindeki `26.jpg` ve `26 (19).jpg` gibi kopyalar ayni kaynak goruntu altinda gruplanir. Bu sayede ayni kaynaktan tureyen dosyalar farkli split'lere dusmemeye calisir.

## Ciktilar

`cikti/` altinda:
- islenmis goruntuler
- `goruntu_ozellikleri.csv`
- `goruntu_ozellikleri_scaled.csv`
- `egitim.csv`, `dogrulama.csv`, `test.csv`
- `egitim_scaled.csv`, `dogrulama_scaled.csv`, `test_scaled.csv`
- `feature_scaler.pkl`

## Teknik Notlar

- Boyut ve dosya boyutu gibi meta sayisal kolonlar modele sokulmaz.
- Paralel islem sayisi `GorselIsleyici.n_jobs` ile sinirlandirilabilir.
- CSV'de NaN varsa menu `3 -> 4 -> 5` adimlarini yeniden calistirin.

## Yardimci Komutlar

```bash
python pipeline_quick_test.py
python test_pipeline.py /path/to/image.jpg
```
