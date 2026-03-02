# MRI Beyin Goruntusu Siniflandirma

MRI beyin goruntulerinden demans seviyesini tahmin etmek icin uctan uca bir makine ogrenmesi projesi. Repo; goruntu on isleme, ozellik cikarma, EDA, klasik ML modelleri ve testleri tek yerde toplar.

## Proje Yapisi

```text
MRI_Classification/
|-- Veri_Seti/                 # Ham goruntuler
|-- goruntu_isleme/            # On isleme + ozellik cikarma
|   |-- ana_islem.py           # Menu tabanli ana akis
|   |-- goruntu_isleyici.py    # On isleme pipeline'i
|   |-- ozellik_cikarici.py    # Ozellik cikarma ve CSV uretimi
|   |-- pipeline_quick_test.py # Hizli ortam kontrolu
|   |-- test_pipeline.py       # Tek goruntu pipeline gorsellestirme
|   `-- ayarlar.py             # Goruntu isleme ayarlari
|-- eda_analiz/                # Kesifsel veri analizi
|-- model/                     # Model egitimi ve inference
|-- tests/                     # Pytest senaryolari
|-- requirements.txt
`-- LICENSE
```

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Goruntu isleme tarafini hizli kontrol etmek icin:

```bash
cd goruntu_isleme
python pipeline_quick_test.py
```

## Kullanim

### 1. Goruntu on isleme

```bash
cd goruntu_isleme
python ana_islem.py
```

Menu:
- `1`: Goruntuleri on isle
- `2`: Ozellik cikar ve CSV olustur
- `3`: NaN degerleri temizle
- `4`: Veri setini bol ve scaler'i sadece egitim setine fit et
- `5`: Istatistik raporu goster
- `6`: Ham CSV'yi bol
- `7`: Tum islemleri otomatik yap

Varsayilan pipeline:
- Kalite kontrol
- Median filtre
- Yogunluk normalizasyonu
- Adaptif CLAHE
- Yeniden boyutlandirma

Notlar:
- `VERI_ARTIRMA_AKTIF = False`
- `BIAS_FIELD_CORRECTION_AKTIF = False`
- `SKULL_STRIPPING_AKTIF = False`
- `REGISTRATION_AKTIF = False`
- `26.jpg` ve `26 (19).jpg` gibi adlar ayni kaynak grup altinda ele alinir; bu split sirasinda veri sizintisi riskini azaltir.

### 2. EDA

```bash
cd ../eda_analiz
python eda_calistir.py
```

### 3. Model egitimi

```bash
cd ../model
python train.py --auto
python train.py
```

Egitim, goruntu isleme tarafinda uretilen `egitim_scaled.csv`, `dogrulama_scaled.csv` ve `test_scaled.csv` dosyalarini kullanir.

### 4. Tahmin

```bash
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --image /path/to/image.jpg
python inference.py --model model/ciktilar/modeller/xgboost_YYYYMMDD_HHMMSS.pkl --batch /path/to/folder/
```

## Onemli Teknik Notlar

- Olcekleme train setine gore yapilir; validation ve test ayni scaler ile donusturulur.
- `boyut_bayt`, `genislik`, `yukseklik`, `en_boy_orani`, `piksel_sayisi` gibi meta sayisal kolonlar model girdisine verilmez.
- Augmentasyon varsayilan olarak kapali tutulur; veri seti zaten turetilmis kopyalar icerebildigi icin bu bilincli bir tercihtir.

## Testler

```bash
python -m pytest
python -m pytest tests/test_goruntu_isleyici.py
python -m pytest tests/test_model_egitici.py
```

## Ciktilar

- `goruntu_isleme/cikti/`: islenmis goruntuler, ham ozellik CSV'si, scaled CSV'ler ve `feature_scaler.pkl`
- `model/ciktilar/`: egitilmis modeller, metadata, raporlar ve gorseller
- `eda_analiz/eda_ciktilar/`: EDA ciktlari

## Lisans

MIT. Ayrinti icin `LICENSE` dosyasina bakin.
