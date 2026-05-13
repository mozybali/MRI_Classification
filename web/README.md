# MRI Classification - Django Web Arayuzu

`web/` dizini, mevcut MRI siniflandirma projesinin Django tabanli web
arayuzudur. Uygulama `model/`, `goruntu_isleme/` ve yerel veri seti ciktilarini
Python paketleri olarak import eder; ML pipeline dosyalarini kopyalamaz.

Web arayuzu su islevleri sunar:

- Tek MRI goruntusu icin ResNet (`.pt`) veya XGBoost (`.json`) tahmini
- ZIP icindeki coklu goruntuler icin toplu tahmin ve CSV disari aktarma
- Tahmin oncesi on isleme adimlarini gorsellestirme
- ResNet tahminleri icin Grad-CAM is haritasi
- Son 20 tahmin kaydini listeleyen gecmis ekrani
- Veri seti dagilimi ve egitim raporlarini gosteren dashboard

## Dizin Yapisi

```text
web/
|-- config/
|   |-- settings.py
|   |-- urls.py
|   |-- asgi.py
|   `-- wsgi.py
|-- dashboard_app/
|   |-- urls.py
|   |-- views.py
|   `-- tests.py
|-- inference_app/
|   |-- migrations/0001_initial.py
|   |-- services/
|   |   |-- gradcam.py
|   |   |-- model_loader.py
|   |   `-- xgb_inference.py
|   |-- batch_views.py
|   |-- models.py
|   |-- preprocess_views.py
|   |-- urls.py
|   |-- views.py
|   `-- xai_views.py
|-- static/
|   |-- css/main.css
|   `-- js/toast.js
|-- templates/
|   |-- base.html
|   |-- dashboard_app/index.html
|   `-- inference_app/
|       |-- history.html
|       `-- index.html
|-- .env.example
|-- Makefile
|-- manage.py
|-- requirements-web.txt
`-- README.md
```

`web/media/`, `web/staticfiles/` ve `web/db.sqlite3` yerel calisma
ciktilaridir. Yanitlarda veya dashboard'da kullanilan model ve rapor dosyalari
ise varsayilan olarak `model/ciktilar/` altindan okunur.

## Kurulum

Komutlari proje kokunden calistirin:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r web/requirements-web.txt
```

Ardindan Django ayar dosyasini hazirlayin:

```bash
cd web
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

Sunucu acildiginda:

- `http://127.0.0.1:8000/` adresi `/infer/` ekranina yonlenir.
- `http://127.0.0.1:8000/infer/` tahmin ekranidir.
- `http://127.0.0.1:8000/infer/history/` son tahminleri gosterir.
- `http://127.0.0.1:8000/dashboard/` veri seti ve rapor dashboard'udur.

Django admin paneli kullanilacaksa ayrica:

```bash
python manage.py createsuperuser
```

## Ortam Degiskenleri

`config/settings.py`, `web/.env` dosyasini `python-dotenv` ile yukler.
`.env.example` icindeki mevcut degiskenler:

| Degisken | Varsayilan | Aciklama |
| --- | --- | --- |
| `SECRET_KEY` | fallback gelistirme anahtari | Uretimde mutlaka degistirilmelidir. |
| `DEBUG` | `True` | `True` ise media dosyalari Django tarafindan serve edilir. |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Virgul ile ayrilmis host listesi. |
| `MODEL_DIR` | `model/ciktilar/modeller` | Proje kokune gore model dizini. |
| `DB_ENGINE` | `sqlite` | `sqlite` veya `postgresql`. |
| `POSTGRES_DB` | `mri_classification` | PostgreSQL veritabani adi. |
| `POSTGRES_USER` | `mri_user` | PostgreSQL kullanicisi. |
| `POSTGRES_PASSWORD` | `change-me` | PostgreSQL parolasi. |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host'u. |
| `POSTGRES_PORT` | `5432` | PostgreSQL port'u. |
| `POSTGRES_CONN_MAX_AGE` | `60` | Persistent connection suresi. |
| `POSTGRES_CONNECT_TIMEOUT` | `10` | Baglanti zaman asimi. |

Varsayilan gelistirme veritabani SQLite'tir ve `web/db.sqlite3` dosyasini
kullanir. PostgreSQL'e gecmek icin `.env` icinde `DB_ENGINE=postgresql`
yapin ve asagidaki gibi veritabani olusturun:

```bash
psql postgres -c "CREATE USER mri_user WITH PASSWORD 'change-me';"
psql postgres -c "CREATE DATABASE mri_classification OWNER mri_user;"
psql postgres -c "ALTER USER mri_user CREATEDB;"
python manage.py migrate
```

## Model Dosyalari

Model secici, `settings.MODEL_DIR` icindeki dosyalari tarar:

- `.pt` dosyalari ResNet modeli olarak listelenir.
- `.json` dosyalari XGBoost modeli olarak listelenir.
- `*.meta.json` dosyalari XGBoost metadata sidecar dosyasi kabul edilir ve
  secilebilir model olarak gosterilmez.

Varsayilan dizin:

```text
model/ciktilar/modeller/
```

Bu dizin yoksa veya icinde desteklenen model yoksa model secicide model
bulunamadigi gosterilir. Model uretmek icin proje kokunden egitim komutlari
kullanilabilir:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost tahmini `requirements-web.txt` ile kurulan paketlerle calisir. ResNet
tahmini ve Grad-CAM icin ek olarak PyTorch ve torchvision gerekir:

```bash
# CPU
python -m pip install -r ../requirements-torch-cpu.txt

# CUDA 12.8
python -m pip install -r ../requirements-torch-cu128.txt
```

Torch kurulu degilse web sunucusu yine acilir. Bu durumda XGBoost akisi
calismaya devam eder; ResNet tahmini ve Grad-CAM endpoint'i HTTP 503 ile
`torch kurulu olmali` hatasi dondurur.

## Web Akislari

### Tekli Tahmin

`/infer/` ekraninda bir goruntu yuklenir, model secilir ve istenirse on isleme
aktif edilir. Backend akisi:

1. `POST /infer/predict/` goruntuyu `MEDIA_ROOT/tmp/` altina kaydeder.
2. Secilen model `inference_app.services.model_loader.registry` uzerinden ilk
   istekte yuklenir ve sonraki istekler icin bellekte tutulur.
3. `.pt` modeli icin `model.inference.predict_image`, `.json` modeli icin
   `services/xgb_inference.py` icindeki XGBoost yardimcisi calisir.
4. Sonuc `PredictionRecord` olarak veritabanina kaydedilir.
5. JSON yaniti sinif, guven skoru, olasiliklar, rozet sinifi ve goruntu URL'i
   icerir.

Kaydedilen alanlar:

- `image`
- `model_type`
- `predicted_class`
- `confidence`
- `probabilities`
- `apply_preprocess`
- `created_at`

### On Isleme Gorseli

`POST /infer/preprocess-steps/`, yuklenen goruntu icin su adimlari base64 PNG
olarak dondurur:

- Ham goruntu
- Gri ton
- CLAHE uygulanmis goruntu
- `224x224` boyutlandirilmis goruntu

### Grad-CAM

`GET /infer/explain/<record_id>/`, sadece `model_type=resnet` olan kayitlar icin
Grad-CAM is haritasi uretir. Kullanim icin ResNet model dosyasi ve torch
bagimliliklari gerekir. Endpoint mevcut uygulamada `MODEL_DIR` icindeki ilk
`.pt` dosyasini kullanir; kayit uzerinde model dosya adi saklanmaz.

### Toplu Tahmin

Toplu modda bir `.zip` dosyasi yuklenir. `POST /infer/batch-predict/` ZIP
icindeki `.png`, `.jpg` ve `.jpeg` dosyalarini isler, her dosya icin tahmin ve
guven skoru dondurur. Gecici dosyalar `MEDIA_ROOT/batch_tmp/` altina yazilir ve
islem sonunda silinir.

`POST /infer/export-csv/`, ekranda uretilen toplu tahmin sonucunu
`mri_batch_results.csv` olarak dondurur.

### Gecmis

`GET /infer/history/`, `PredictionRecord` tablosundaki son 20 kaydi ters tarih
sirasiyla listeler.

## Dashboard

`/dashboard/` sayfasi Chart.js ile iki veri kaynagini gorsellestirir:

- `GET /dashboard/api/eda-stats/`
  - `Veri_Seti/OriginalDataset/<SinifAdi>/` altindaki `.jpg`, `.jpeg` ve
    `.png` dosyalarini sayar.
  - Etiketleri, sinif sayilarini ve toplam goruntu sayisini JSON olarak dondurur.

- `GET /dashboard/api/model-reports/`
  - `model/ciktilar/` altinda recursive olarak `rapor_*.json` dosyalarini arar.
  - Dosyalari degisiklik tarihine gore yeniden eskiye siralar.
  - Model adi, timestamp, accuracy ve macro F1 degerlerini listeler.

- `GET /dashboard/api/report/<filename>/`
  - `model/ciktilar/` altinda dosya adiyla eslesen JSON raporu bulur ve detayini
    dondurur.
  - Frontend, rapordaki `test_detailed_metrics.per_class` alanindan sinif bazli
    precision, recall ve F1 tablosu ile recall bar grafigi olusturur.

Dashboard tarafinda Chart.js CDN'den yuklenir; internet erisimi yoksa grafikler
render edilmeyebilir.

## URL Ozeti

| Yol | Method | Aciklama |
| --- | --- | --- |
| `/` | GET | `/infer/` adresine yonlendirir. |
| `/admin/` | GET | Django admin paneli. |
| `/infer/` | GET | Tekli/toplu tahmin ekrani. |
| `/infer/predict/` | POST | Tek goruntu tahmini. |
| `/infer/history/` | GET | Son 20 tahmin kaydi. |
| `/infer/explain/<record_id>/` | GET | ResNet Grad-CAM aciklamasi. |
| `/infer/preprocess-steps/` | POST | On isleme adimlari. |
| `/infer/batch-predict/` | POST | ZIP ile toplu tahmin. |
| `/infer/export-csv/` | POST | Toplu tahmin sonucunu CSV dondurur. |
| `/dashboard/` | GET | Dashboard ana sayfasi. |
| `/dashboard/api/eda-stats/` | GET | Veri seti sinif dagilimi. |
| `/dashboard/api/model-reports/` | GET | Model rapor listesi. |
| `/dashboard/api/report/<filename>/` | GET | Secili rapor detayi. |

## Dogrulama

Temel Django kontrolleri:

```bash
cd web
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py migrate
```

Aktif veritabani motorunu kontrol etmek:

```bash
python manage.py shell -c "from django.db import connection; print(connection.vendor)"
```

Model dizininde bulunan secilebilir modelleri kontrol etmek:

```bash
python manage.py shell -c "from inference_app.services.model_loader import registry; print(registry.list_available_models())"
```

## Makefile

`web/Makefile` yerel gelistirme icin kisa komutlar icerir:

```bash
make setup     # web/.venv olusturur ve web bagimliliklarini kurar
make migrate   # makemigrations + migrate calistirir
make run       # 8000 portunda runserver baslatir
make shell     # Django shell acar
make clean     # __pycache__, staticfiles ve media/predictions altini temizler
```

`make setup`, ek ML ve analiz paketlerini de kurar. Daha kucuk bir web ortami
isteniyorsa `python -m pip install -r requirements-web.txt` yeterlidir.

## Notlar

- Yuklenen tekli tahmin goruntuleri `MEDIA_ROOT` altinda saklanir; veritabanina
  blob olarak yazilmaz.
- Toplu tahmin sonuclari veritabanina kaydedilmez; yalnizca JSON/CSV olarak
  dondurulur.
- `PredictionRecord` modeli tahminde kullanilan model dosyasinin adini
  saklamaz; yalnizca model tipini (`resnet` veya `xgboost`) tutar.
- `settings.py`, proje kokunu `sys.path` icine ekledigi icin web uygulamasi
  mevcut `model` ve `goruntu_isleme` paketlerini dogrudan import eder.
- Bu web arayuzu egitim ve arastirma amaclidir; klinik tani araci olarak
  kullanilmamalidir.
