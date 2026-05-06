# MRI Classification — Django Web Arayüzü

`web/` dizini, eğitilmiş yerel modellerle (ResNet `.pt`, XGBoost `.json`) tahmin alan
ve EDA dashboard'unu sunan Django uygulamasını barındırır. Bu uygulama, yerel
**PostgreSQL** veritabanını kullanır; geliştirici makinesinde isteğe bağlı SQLite
fallback'i de desteklenir.

ML çıktı dosyaları (`model/ciktilar/modeller/best_xgboost.json` vb.) yerel runtime
artefaktlarıdır; veritabanına yüklenmez. Yüklenen MRI görüntüleri filesystem üzerinde
`MEDIA_ROOT` altında tutulur.

## Bağımlılıklar

```bash
cd web
python -m pip install -r requirements-web.txt
```

`requirements-web.txt`; Django, whitenoise, pillow, python-dotenv, gunicorn,
PostgreSQL sürücüsü (`psycopg[binary]`) ve XGBoost inference'ı için gerekli ML
kütüphanelerini içerir (`xgboost`, `scikit-learn`, `numpy`, `scipy`,
`opencv-python`, `scikit-image`, `tqdm`). `tqdm`, XGBoost + preprocess akışında
`goruntu_isleme/toplu_islem.py` modül seviyesi import'u nedeniyle gereklidir.

### ResNet (.pt) için ek kurulum

ResNet inference ve Grad-CAM uçları `torch` + `torchvision` gerektirir.
Minimal web env'de bunlar kurulmaz; kullanılacaksa proje kökünden:

```bash
# CPU
python -m pip install -r ../requirements-torch-cpu.txt
# veya CUDA
python -m pip install -r ../requirements-torch-cu128.txt
```

Torch kurulu değilken sunucu yine de açılır; ResNet/Grad-CAM uçları HTTP 503
("torch kurulu olmalı") döndürür. XGBoost akışı torch olmadan tam çalışır.

## PostgreSQL Kurulumu

PostgreSQL'in yerel olarak kurulu ve çalışıyor olduğunu varsayar (örn. macOS'ta
Homebrew, Linux'ta sistem paketi). Veritabanı ve kullanıcı oluşturma:

```bash
psql postgres -c "CREATE USER mri_user WITH PASSWORD 'change-me';"
psql postgres -c "CREATE DATABASE mri_classification OWNER mri_user;"
```

Django testleri PostgreSQL üzerinde çalıştırılacaksa, kullanıcıya test
veritabanı oluşturma izni verin:

```bash
psql postgres -c "ALTER USER mri_user CREATEDB;"
```

## Ortam Değişkenleri

`web/.env.example` dosyasını `web/.env` olarak kopyalayıp düzenleyin:

```bash
cp .env.example .env
```

Veritabanıyla ilgili değişkenler:

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `DB_ENGINE` | `postgresql` | `postgresql` veya `sqlite` |
| `POSTGRES_DB` | `mri_classification` | Veritabanı adı |
| `POSTGRES_USER` | `mri_user` | Kullanıcı |
| `POSTGRES_PASSWORD` | — | Parola (yerel geliştirme için `.env` içinde) |
| `POSTGRES_HOST` | `localhost` | Sunucu |
| `POSTGRES_PORT` | `5432` | Port |
| `POSTGRES_CONN_MAX_AGE` | `60` | Saniye cinsinden persistent connection süresi |
| `POSTGRES_CONNECT_TIMEOUT` | `10` | Bağlantı zaman aşımı (saniye) |

## Uygulama Kurulumu

```bash
cd web
python -m pip install -r requirements-web.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

`createsuperuser` yalnızca Django admin paneline erişmek istiyorsanız gereklidir.

## Doğrulama

```bash
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py migrate
python manage.py shell -c "from django.db import connection; print(connection.vendor)"
# Beklenen: postgresql
```

XGBoost modelinin keşfedildiğini doğrulayın:

```bash
python manage.py shell -c "from inference_app.services.model_loader import registry; print(registry.list_available_models())"
# Beklenen: best_xgboost.json girişi (type=xgboost). best_xgboost.meta.json listelenmez.
```

Bağımlılıklar ve yerel model dosyası mevcutsa XGBoost yükleme:

```bash
python manage.py shell -c "from django.conf import settings; from inference_app.services.model_loader import registry; model, meta = registry.get_xgboost(settings.MODEL_DIR / 'best_xgboost.json'); print(meta)"
```

## Mevcut SQLite Verisini PostgreSQL'e Taşıma

Daha önce SQLite ile çalışılmış ve `web/db.sqlite3` mevcutsa, verinin güvenli
aktarımı:

```bash
cd web

# 1) SQLite'tan dump al (geçici olarak DB_ENGINE=sqlite ile)
DB_ENGINE=sqlite python manage.py dumpdata \
    --natural-foreign --natural-primary \
    --exclude contenttypes --exclude auth.Permission \
    --indent 2 > sqlite_data.json

# 2) .env içinde DB_ENGINE=postgresql olduğundan emin olun, ardından:
python manage.py migrate
python manage.py loaddata sqlite_data.json
```

`sqlite_data.json` versiyonlanmamalıdır; yükleme sonrası silinebilir.

## SQLite Fallback (yalnızca yerel)

PostgreSQL'i geçici olarak devre dışı bırakmak için `.env`'de:

```ini
DB_ENGINE=sqlite
```

Bu durumda Django `web/db.sqlite3` kullanır.

## Notlar

- Yüklenen MRI dosyaları `MEDIA_ROOT` altında saklanır; PostgreSQL'e blob
  olarak yazılmaz.
- Inference için yalnızca `MODEL_DIR` (varsayılan
  `model/ciktilar/modeller`) taranır. `model/ciktilar/raporlar` taranmaz.
- `*.meta.json` dosyaları XGBoost modellerinin sidecar metadata dosyalarıdır;
  UI'da seçilebilir model olarak listelenmez.
