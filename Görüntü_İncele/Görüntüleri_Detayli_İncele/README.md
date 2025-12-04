# JPEG MRI EDA Projesi

Bu proje, 4 seviyeli (veya genel olarak çok sınıflı) bir hastalık için **JPEG/PNG biçiminde**
saklanan MRI görüntülerini model eğitimi öncesinde ayrıntılı olarak incelemen (EDA) için
hazırlanmış modüler bir Python paketidir.

## Özellikler

- Sınıf (label) dağılım grafiği
- Görüntü boyutları (genişlik/yükseklik) histogram ve scatter grafikleri
- Global yoğunluk histogramları (sınıf bazlı KDE)
- Her görüntü için temel özet istatistikler
  - genişlik, yükseklik, en/boy oranı
  - yoğunluk ortalama, std, min, max, p1, p99
- Bu özniteliklerle PCA / t-SNE 2D embedding grafikleri
- Her sınıftan rastgele örnek JPEG görüntüleri (kontrol amaçlı)

## Önerilen Klasör Yapısı

```text
mri_eda_jpg_project/
  mri_eda_jpg/
    __init__.py
    config.py
    io_utils.py
    stats_utils.py
    plot_utils.py
  scripts/
    run_eda.py
  data/
    labels.csv
    jpg/
      hasta_001.jpg
      hasta_002.jpg
      ...
  requirements.txt
  README.md
```

`data/labels.csv` dosyası şu kolonları içermelidir:

```csv
id,filepath,label
hasta_001,data/jpg/hasta_001.jpg,0
hasta_002,data/jpg/hasta_002.jpg,1
hasta_003,data/jpg/hasta_003.jpg,2
hasta_004,data/jpg/hasta_004.jpg,3
```

- `label`: hastalık seviyesi (ör. 0,1,2,3)
- `filepath`: ilgili JPEG/PNG dosyasına göreli veya tam yol

`mri_eda_jpg/config.py` içindeki `METADATA_CSV`, `OUTPUT_DIR` vb. ayarları
ihtiyacına göre güncelleyebilirsin.

Varsayılan olarak görüntüler gri tonlamaya (`CONVERT_TO_GRAYSCALE = True`) çevrilir
ve istatistikler bu gri tonlamalı piksel değerleri üzerinden hesaplanır.

## Kurulum

1. (İsteğe bağlı) Sanal ortam oluştur:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Gerekli paketleri kur:

   ```bash
   pip install -r requirements.txt
   ```

## Çalıştırma

Projenin kök klasöründe:

```bash
python scripts/run_eda.py
```

Çalışmanın sonunda tüm grafikler `config.py` içindeki `OUTPUT_DIR` (varsayılan: `eda_ciktlari`)
klasörüne `.png` dosyaları olarak kaydedilir.

## Notlar

- Global yoğunluk dağılımları için her görüntüden sınırlı sayıda piksel örneklenir
  (detaylar için `config.py` içindeki `N_PIXELS_PER_IMAGE_SAMPLE` değerine bak).
- PCA ve t-SNE embedding için dataset çok büyükse, `N_IMAGES_FOR_EMBEDDING` kadar
  alt örnekleme yapılır.
- Sınıf isimlerini daha anlamlı hale getirmek için `LABEL_NAME_MAP` sözlüğünü
  doldurabilirsin (ör. `{0: "Düşük", 1: "Orta", 2: "Yüksek", 3: "Çok Yüksek"}`). 
