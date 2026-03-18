# EDA Analiz Modulu

Bu modul, MRI veri seti icin kesifsel veri analizi (EDA) uretir. Sinif dagilimi, goruntu boyutlari, yogunluk istatistikleri, korelasyon ve PCA gorsellerini otomatik olarak kaydeder.

## Ne Icin Kullanilir?

- Veri setinin dengeli olup olmadigini hizlica gormek
- Boyut ve piksel yogunlugu farklarini incelemek
- Egitimden once veri kalitesini kontrol etmek
- Rapor veya sunum icin ozet grafikler uretmek

## Calistirma

Repo kokunden onerilen komut:

```bash
mri-eda --interactive
```

Dogrudan Python ile:

```bash
python -m eda_analiz.eda_calistir --interactive
```

Arguman vererek etkilesimsiz calistirma:

```bash
mri-eda --data-dir Veri_Seti/AugmentedAlzheimerDataset --output-dir eda_analiz/eda_ciktilar
```

Tek cekirdege zorlamak veya paralel sayisini belirlemek icin:

```bash
mri-eda --data-dir Veri_Seti/AugmentedAlzheimerDataset --jobs 1
```

## Varsayilanlar

- Varsayilan veri klasoru: `Veri_Seti/AugmentedAlzheimerDataset`
- Varsayilan cikti klasoru: `eda_analiz/eda_ciktilar`
- `--jobs` verilmezse cekirdek sayisi otomatik secilir; paralel hesaplama kullanilamazsa arac tek cekirdege geri duser
- `Veri_Seti` koku verilirse uygun alt klasor otomatik cozulur
- `Veri_Seti/OriginalDataset` verilirse analiz sadece original veri uzerinde yapilir

Desteklenen veri yapilari:

```text
Veri_Seti/<SinifAdi>/
Veri_Seti/AugmentedAlzheimerDataset/<SinifAdi>/
Veri_Seti/OriginalDataset/<SinifAdi>/
```

## Uretilen Ciktilar

- `0_ozet_istatistikler.txt`: Toplam ornek, sinif dagilimi ve temel ozet
- `1_sinif_dagilimi.png`: Sinif dagilimi grafigi
- `2_boyut_analizi.png`: Genislik, yukseklik ve oran dagilimlari
- `3_yogunluk_analizi.png`: Piksel yogunlugu grafikleri
- `4_korelasyon_matrisi.png`: Sayisal ozellik korelasyonlari
- `5_pca_analizi.png`: Ilk iki bilesen uzerinden PCA gorsellestirmesi
- `veri_seti_istatistikler.csv`: Goruntu bazli temel istatistik tablosu

## Notlar

- Istatistik hesaplamalari cok cekirdekli olarak hizlandirilir.
- Cikti klasoru otomatik olusturulur.
- Modul, proje kokundeki `requirements.txt` veya `pip install -e .` kurulumu ile calisir.
