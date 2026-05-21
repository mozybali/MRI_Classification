"""
Django settings for MRI Classification Web Application.
Sprint 1: Temel yapılandırma
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── OpenMP runtime çakışması koruması ────────────────────────────────────────
# torch ve xgboost aynı process'te kullanıldığında her ikisi de kendi OpenMP
# runtime'ını (libiomp5 / libomp) yükler. İki runtime aynı anda yüklenince
# "OMP: Error #15" tetiklenir; process abort olur ya da kilitlenir (web tepkisiz
# kalır). Tek model kullanılırken tek framework yüklendiği için sorun görünmez;
# ikinci model (farklı tür) devreye girince ortaya çıkar.
# Bu satır herhangi bir torch/xgboost import'undan ÖNCE çalışmalıdır — settings.py
# Django başlangıcında, model yüklemelerinden çok önce import edildiği için
# burası güvenli noktadır. setdefault: kullanıcı kendi değerini ezmesin diye.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Dizin tanımları ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # web/
PROJECT_ROOT = BASE_DIR.parent                              # MRI_Classification/

# ML modüllerini import edebilmek için proje kökünü sys.path'e ekle
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── .env yükle ───────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

# ── Temel ayarlar ────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-fallback-key-change-in-production",
)
DEBUG = os.environ.get("DEBUG", "True") == "True"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# ── Model dizini (ML checkpoint'leri) ────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / os.environ.get("MODEL_DIR", "model/ciktilar/modeller")

# ── Uygulamalar ──────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "whitenoise.runserver_nostatic",
    "django.contrib.staticfiles",
    # Proje uygulamaları
    "inference_app",
    "dashboard_app",
]

# ── Middleware ───────────────────────────────────────────────────────────────
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

# ── Template yapılandırması ──────────────────────────────────────────────────
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ── Veritabanı ───────────────────────────────────────────────────────────────
# Varsayılan: yerel SQLite. PostgreSQL gerekirse DB_ENGINE=postgresql ile seçilebilir.
DB_ENGINE = os.environ.get("DB_ENGINE", "sqlite").lower()

if DB_ENGINE == "sqlite":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("POSTGRES_DB", "mri_classification"),
            "USER": os.environ.get("POSTGRES_USER", "mri_user"),
            "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
            "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": int(os.environ.get("POSTGRES_CONN_MAX_AGE", "60")),
            "CONN_HEALTH_CHECKS": True,
            "OPTIONS": {
                "connect_timeout": int(os.environ.get("POSTGRES_CONNECT_TIMEOUT", "10")),
            },
        }
    }

# ── Şifre doğrulama ──────────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── Dil & zaman dilimi ───────────────────────────────────────────────────────
LANGUAGE_CODE = "tr-tr"
TIME_ZONE = "Europe/Istanbul"
USE_I18N = True
USE_TZ = True

# ── Static dosyalar ──────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# ── Media (kullanıcı yüklemeleri — MRI görselleri) ──────────────────────────
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ── Dosya yükleme limiti (MRI: 20 MB yeterli) ───────────────────────────────
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024   # 20 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024   # 20 MB

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
