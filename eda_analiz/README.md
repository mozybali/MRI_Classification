# EDA Analiz Modülü

Bu modül, `Veri_Seti/OriginalDataset` altındaki MRI görüntüleri için keşifsel veri analizi üretir. Sınıf dağılımı, görüntü boyutları, piksel yoğunluğu, korelasyon ve PCA çıktıları eğitimden önce veri setini hızlıca okumayı sağlar.

## Dosya Yapısı

```text
eda_analiz/
|-- __init__.py
|-- __main__.py
|-- eda_araclar.py
|-- eda_calistir.py
`-- README.md
```

- `eda_araclar.py`: `EDAAnaliz` sınıfını ve analiz/görselleştirme fonksiyonlarını içerir.
- `eda_calistir.py`: CLI giriş noktasıdır; `mri-eda` komutu buraya bağlanır.
- `__main__.py`: Modülü `python3 -m eda_analiz` biçiminde çalıştırmayı destekler.

## Ne Zaman Kullanılır

- Sınıf dağılımını ve veri dengesizliğini kontrol etmek.
- Görüntü boyutu, oran ve yoğunluk farklılıklarını eğitimden önce görmek.
- Problemli klasör yapısı veya okunamayan görüntüleri erken yakalamak.
- Rapor veya sunum için temel grafik çıktıları üretmek.

## Çalıştırma

Kurulumdan sonra önerilen komut:

```bash
mri-eda --interactive
```

Doğrudan Python modülüyle:

```bash
python3 -m eda_analiz.eda_calistir --interactive
```

Etkileşimsiz örnekler:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar
mri-eda --data-dir Veri_Seti --jobs 1
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
```

## CLI Parametreleri

- `--data-dir`: Analiz edilecek veri klasörü. Üst klasör olarak `Veri_Seti` verilirse `OriginalDataset` otomatik çözümlenir.
- `--output-dir`: Analiz çıktılarının yazılacağı klasör.
- `--interactive`: Eksik argümanları soru-cevap ile tamamlar.
- `--jobs`: İstatistik hesaplamada kullanılacak çekirdek sayısı. Verilmezse otomatik seçilir.

## Varsayılanlar

- Veri klasörü: `Veri_Seti/OriginalDataset`
- Çıktı klasörü: `eda_analiz/eda_ciktilar`
- Desteklenen görüntü uzantıları: `.jpg`, `.jpeg`, `.png`
- Sabit sınıflar: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`

Beklenen veri yapısı:

```text
Veri_Seti/OriginalDataset/<SinifAdi>/
```

## Üretilen Çıktılar

Varsayılan olarak `eda_analiz/eda_ciktilar` altına şu dosyalar yazılır:

- `0_ozet_istatistikler.txt`
- `1_sinif_dagilimi.png`
- `2_boyut_analizi.png`
- `3_yogunluk_analizi.png`
- `4_korelasyon_matrisi.png`
- `5_pca_analizi.png`
- `veri_seti_istatistikler.csv`

## Notlar

- Çıktı klasörü yoksa otomatik oluşturulur.
- Desteklenmeyen görüntü uzantıları atlanır ve konsola uyarı yazılır.
- İnteraktif mod yalnızca `--interactive` bayrağı verildiğinde açılır; CI/CD ortamlarında komut kendiliğinden girdi beklemez.
