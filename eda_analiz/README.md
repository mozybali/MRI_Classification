# EDA Analiz Modulu

MRI veri seti icin kesifsel veri analizi (EDA) uretir; sinif dagilimi, boyut ve yogunluk istatistikleri, korelasyon ve PCA gorsellerini otomatik kaydeder. Istatistik hesaplamalari cok cekirdekle hizlandirilir.

## Kurulum

Yalnizca bu modul:
```bash
pip install -r ..\\requirements.txt
```
Tum proje paketleri zaten kuruluysa bu adimi atlayabilirsiniz (`../requirements.txt` yeterli).

## Kullanim

```bash
python eda_calistir.py
```

Komut sirasinda veri klasoru (varsayilan: `Veri_Seti/AugmentedAlzheimerDataset`) ve cikti klasoru (varsayilan: `eda_analiz/eda_ciktilar`) sorulur.

- `Veri_Seti` koku verilirse uygun alt klasor otomatik secilir.
- `Veri_Seti/OriginalDataset` verilirse sadece original veri analiz edilir.

## Uretilenler

- `0_ozet_istatistikler.txt`: Toplam ornek, sinif dagilimi ve temel ozet.
- `1_sinif_dagilimi.png`: Sinif dagilimi grafigi.
- `2_boyut_analizi.png`: Genislik/yukseklik/en-boy orani dagilimlari.
- `3_yogunluk_analizi.png`: Yogunluk histogramlari.
- `4_korelasyon_matrisi.png`: Ozellik korelasyonlari.
- `5_pca_analizi.png`: PCA ilk iki bilesen gorsellestirmesi.
- `veri_seti_istatistikler.csv`: Goruntu bazli temel istatistikler.

## Ne Zaman Calistirilmali?

- Veri setinin icerigini ve dengesini hizlica gormek istediginizde.
- On isleme/augmentasyon stratejisinden once veri kalitesini kontrol ederken.
- Egitim raporlarini desteklemek icin ozet gorseller gerektiginde.
