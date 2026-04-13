# EDA Analiz Modulu

Bu modul, MRI veri seti icin kesifsel veri analizi uretir. Sinif dagilimi, goruntu boyutlari, piksel yogunlugu, korelasyon ve PCA gorsellerini otomatik olarak kaydeder.

## Ne Icin Kullanilir

- Veri setinin dengeli olup olmadigini kontrol etmek
- Boyut ve yogunluk farklarini egitim oncesi incelemek
- Problemli goruntu veya sinif dagilimlarini erken fark etmek
- Rapor ve sunumlar icin hazir grafik ciktilari uretmek

## Calistirma

Repo kokunden onerilen komut:

```bash
mri-eda --interactive
```

Dogrudan Python ile:

```bash
python -m eda_analiz.eda_calistir --interactive
```

Etkilesimsiz ornekler:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar
mri-eda --data-dir Veri_Seti/OriginalDataset --jobs 1
```

## CLI Parametreleri

- `--data-dir`: Analiz edilecek veri klasoru
- `--output-dir`: Ciktilarin yazilacagi klasor
- `--interactive`: Eksik argumanlari soru-cevap ile tamamlar
- `--jobs`: Istatistik hesaplamada kullanilacak cekirdek sayisi

## Varsayilanlar

- Varsayilan veri klasoru: `Veri_Seti/OriginalDataset`
- Varsayilan cikti klasoru: `eda_analiz/eda_ciktilar`
- `--jobs` verilmezse cekirdek sayisi otomatik secilir

Desteklenen veri yapilari:

```text
Veri_Seti/OriginalDataset/<SinifAdi>/
```

### Otomatik Klasor Cozumleme

`--data-dir` olarak ust klasor verildiginde (ornegin `Veri_Seti`), arac
otomatik olarak `OriginalDataset` alt klasorunu arar ve kullanir.
Eger sinif klasorleri dogrudan verilen dizinde bulunursa o dizin kullanilir.

### Interaktif Mod

Interaktif soru-cevap modu yalnizca `--interactive` bayragi ile etkinlesir.
Bayrak verilmezse varsayilan dizinler sessizce kullanilir; CI/CD
ortamlarinda asili kalma riski yoktur.

### Desteklenen Goruntu Uzantilari

Sadece `.jpg`, `.jpeg` ve `.png` dosyalari yuklenir. Diger uzantilar
(`.tif`, `.bmp`, `.gif` vb.) varsa konsola uyari basilir ve atlanir.

## Uretilen Ciktilar

- `0_ozet_istatistikler.txt`
- `1_sinif_dagilimi.png`
- `2_boyut_analizi.png`
- `3_yogunluk_analizi.png`
- `4_korelasyon_matrisi.png`
- `5_pca_analizi.png`
- `veri_seti_istatistikler.csv`

## Notlar

- Arac, cikti klasorunu yoksa otomatik olusturur.
- Istatistik hesaplamalari cok cekirdekli calisma destekler.
- Proje genelindeki sinif adlari `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented` olarak sabittir.
