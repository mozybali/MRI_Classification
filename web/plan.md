# 🧠 MRI Classification — Django Web Arayüzü Sprint Planı

> **Proje:** `MRI_Classification` üzerine Django tabanlı web arayüzü eklenmesi  
> **Stack:** Django 5.x · Python 3.10+ · Vanilla CSS · Chart.js · REST (Django views)  
> **Toplam Sprint:** 5  
> **Hedef:** Mevcut ML pipeline'ını bozmadan, `web/` adlı ayrı bir Django uygulaması olarak projeye entegre etmek

---

## 📐 Genel Mimari

```
MRI_Classification/
├── eda_analiz/          ← mevcut (dokunulmaz)
├── goruntu_isleme/      ← mevcut (dokunulmaz)
├── model/               ← mevcut (dokunulmaz)
├── tests/               ← mevcut (dokunulmaz)
│
└── web/                 ← 🆕 Django projesi (bu sprintlerde yapılacak)
    ├── manage.py
    ├── config/          ← Django settings, urls, wsgi
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── inference_app/   ← Tahmin & yükleme ekranı
    ├── dashboard_app/   ← EDA Dashboard & model karşılaştırma
    ├── static/
    │   ├── css/
    │   ├── js/
    │   └── img/
    └── templates/
        ├── base.html
        ├── inference_app/
        └── dashboard_app/
```

> **Kural:** `web/` Django projesi, mevcut modülleri (`model`, `goruntu_isleme`, `eda_analiz`) Python paket olarak import eder. Mevcut dosyalarda değişiklik yapılmaz.

---

## 🏃 Sprint 1 — Django Kurulumu & Proje İskeleti

**Süre:** ~1 gün  
**Hedef:** Çalışan, boş ama doğru yapılandırılmış Django projesini ayağa kaldırmak

### Görevler

| # | Görev | Detay |
|---|-------|-------|
| 1.1 | **Bağımlılık ekleme** | `django`, `django-crispy-forms`, `whitenoise` paketlerini `requirements.txt`'e ekle |
| 1.2 | **Django projesi oluştur** | `web/` klasörünü oluştur, `django-admin startproject config .` ile yapıyı kur |
| 1.3 | **Uygulama oluştur** | `inference_app` ve `dashboard_app` Django uygulamalarını oluştur |
| 1.4 | **`settings.py` yapılandır** | `INSTALLED_APPS`, `STATIC_ROOT`, `MEDIA_ROOT` (yüklenen MRI dosyaları için), `sys.path` ile proje kök dizinini ekle |
| 1.5 | **URL yapısını kur** | Ana `urls.py` → `inference_app.urls` + `dashboard_app.urls` yönlendirmesi |
| 1.6 | **Base template** | `base.html` → Navbar (Tahmin / Dashboard), dark mode CSS değişkenleri, Google Fonts (Inter) |
| 1.7 | **Static dosya yapısı** | `static/css/main.css`, `static/js/`, `static/img/logo.svg` |
| 1.8 | **Sunucu testi** | `python manage.py runserver` ile ana sayfanın açıldığını doğrula |

### Çıktılar
- `web/` klasörü tam yapıda oluşturulmuş
- `http://localhost:8000` → boş ana sayfa açılıyor
- `http://localhost:8000/infer/` ve `/dashboard/` URL'leri 200 döndürüyor (boş view)

### Dikkat Edilecekler
- `PYTHONPATH`'e proje kökü (`MRI_Classification/`) eklenmeli ki `from model.inference import ...` çalışsın
- `MEDIA_ROOT = web/media/` — yüklenen görseller buraya kaydedilir, `.gitignore`'a eklenir
- `SECRET_KEY` → `.env` dosyasından okunacak şekilde yapılandır

---

## 🏃 Sprint 2 — Inference (Tahmin) Ekranı

**Süre:** ~2-3 gün  
**Bağımlılık:** Sprint 1 tamamlanmış olmalı  
**Hedef:** Kullanıcı MRI görüntüsü yükleyip demans sınıfını ve confidence score'ları görebilir

### 2.1 — Backend: `inference_app`

| # | Görev | Detay |
|---|-------|-------|
| 2.1.1 | **`models.py`** | `PredictionRecord` modeli → dosya yolu, tahmin sonucu, zaman damgası, model tipi (resnet/xgboost) |
| 2.1.2 | **Model yükleyici servis** | `services/model_loader.py` → Uygulama başlarken `.pt` / `.json` model dosyasını tek sefer yükle, singleton pattern |
| 2.1.3 | **Inference view** | `POST /infer/predict/` → Yüklenen dosyayı al, `model/inference.py:predict_image()` çağır, JSON döndür |
| 2.1.4 | **Geçmiş view** | `GET /infer/history/` → Son 20 tahmini listele |
| 2.1.5 | **Migration** | `makemigrations` + `migrate` çalıştır |

**`PredictionRecord` alanları:**
```python
class PredictionRecord(models.Model):
    image           = models.ImageField(upload_to='predictions/%Y/%m/%d/')
    model_type      = models.CharField(max_length=20)   # resnet / xgboost
    predicted_class = models.CharField(max_length=50)
    confidence      = models.FloatField()
    probabilities   = models.JSONField()  # {"NonDemented": 0.92, ...}
    created_at      = models.DateTimeField(auto_now_add=True)
```

### 2.2 — Frontend: Tahmin Sayfası

| # | Görev | Detay |
|---|-------|-------|
| 2.2.1 | **Yükleme alanı** | Sürükle-bırak (drag & drop) görüntü yükleme kutusu, önizleme göster |
| 2.2.2 | **Model seçici** | ResNet / XGBoost toggle butonu |
| 2.2.3 | **Ön işleme seçeneği** | `--preprocess` flag için checkbox (ham görüntü mü işlenmiş mi?) |
| 2.2.4 | **Yükleme animasyonu** | Tahmin sürerken spinner / pulse animasyonu |
| 2.2.5 | **Sonuç kartı** | Tahmin edilen sınıf (renkli badge), güven skoru (büyük sayı) |
| 2.2.6 | **Olasılık bar chart** | 4 sınıf için animasyonlu yatay progress bar'lar (vanilla CSS + JS) |
| 2.2.7 | **Geçmiş paneli** | Son tahminler küçük kart olarak listelenir, üzerine tıklanınca detay açılır |

### Sınıf Renk Kodlaması

| Sınıf | Renk | Anlam |
|-------|------|-------|
| NonDemented | `#22c55e` (yeşil) | Sağlıklı |
| VeryMildDemented | `#eab308` (sarı) | Dikkat |
| MildDemented | `#f97316` (turuncu) | Uyarı |
| ModerateDemented | `#ef4444` (kırmızı) | Kritik |

### API Yanıt Formatı
```json
{
  "tahmin_adi": "NonDemented",
  "tahmin_sinif": 0,
  "guven_skoru": 0.9234,
  "olasiliklar": {
    "NonDemented": 0.9234,
    "VeryMildDemented": 0.0512,
    "MildDemented": 0.0198,
    "ModerateDemented": 0.0056
  },
  "model_tipi": "resnet",
  "record_id": 42
}
```

### Çıktılar
- MRI yükleyip tahmin alınabiliyor
- Sonuçlar animasyonlu grafik olarak gösteriliyor
- Her tahmin veritabanına kaydediliyor

---

## 🏃 Sprint 3 — EDA Dashboard & Model Metrik Görselleştirme

**Süre:** ~2-3 gün  
**Bağımlılık:** Sprint 1 tamamlanmış olmalı (Sprint 2 ile paralel geliştirilebilir)  
**Hedef:** Veri seti ve model performans metriklerini interaktif dashboard olarak sunmak

### 3.1 — Backend: `dashboard_app`

| # | Görev | Detay |
|---|-------|-------|
| 3.1.1 | **EDA istatistik endpoint'i** | `GET /dashboard/api/eda-stats/` → `eda_analiz/eda_ciktilar/veri_seti_istatistikler.csv`'yi okur, JSON döndürür |
| 3.1.2 | **Model rapor endpoint'i** | `GET /dashboard/api/model-reports/` → `model/ciktilar/raporlar/` içindeki JSON/CSV raporları okur |
| 3.1.3 | **Görsel listeleme** | `GET /dashboard/api/eda-charts/` → `eda_analiz/eda_ciktilar/` içindeki `.png` dosyalarının URL listesi |
| 3.1.4 | **Confusion matrix endpoint'i** | `GET /dashboard/api/confusion-matrix/<model_id>/` → confusion matrix verisini JSON olarak döndür |
| 3.1.5 | **Training curve endpoint'i** | `GET /dashboard/api/training-curve/<model_id>/` → epoch/loss/accuracy serilerini döndür |

### 3.2 — Frontend: Dashboard Sayfası

#### Bölüm A — Veri Seti Özeti (EDA)

| # | Widget | Görselleştirme |
|---|--------|----------------|
| 3.2.1 | **Sınıf Dağılımı** | Donut chart (Chart.js) — 4 sınıfın görüntü sayısı |
| 3.2.2 | **Görüntü Boyutu Analizi** | Scatter plot — width vs height dağılımı |
| 3.2.3 | **Yoğunluk İstatistikleri** | Bar chart — sınıf başına ortalama piksel yoğunluğu |
| 3.2.4 | **EDA Grafikleri Galerisi** | `eda_ciktilar/` içindeki PNG'leri lightbox ile gösteren ızgara |

#### Bölüm B — Model Performans Karşılaştırması

| # | Widget | Görselleştirme |
|---|--------|----------------|
| 3.2.5 | **Metrik Karşılaştırma** | Grouped bar chart — ResNet vs XGBoost (Accuracy, F1, Precision, Recall) |
| 3.2.6 | **Confusion Matrix** | Isı haritası — Canvas API ile renk geçişli 4x4 matris |
| 3.2.7 | **Training Curve** | Line chart — epoch vs loss / epoch vs accuracy |
| 3.2.8 | **Model Bilgi Kartları** | Checkpoint metadata (tarih, epoch, lr, batch_size, test accuracy) |

### Çıktılar
- EDA grafikleri interaktif olarak gösteriliyor
- ResNet ve XGBoost metrikleri yan yana karşılaştırılıyor
- Confusion matrix ısı haritası render ediliyor

---

## 🏃 Sprint 4 — Gelişmiş Özellikler

**Süre:** ~3-4 gün  
**Bağımlılık:** Sprint 2 ve Sprint 3 tamamlanmış olmalı  
**Hedef:** Projeyi gerçek anlamda gelişmiş bir araç haline getiren ileri özellikler

### 4.1 — Grad-CAM Isı Haritası (XAI)

| # | Görev | Detay |
|---|-------|-------|
| 4.1.1 | **Backend servis** | `services/gradcam.py` → PyTorch hooks ile son conv katmanından aktivasyon haritası |
| 4.1.2 | **Endpoint** | `POST /infer/gradcam/` → Görüntü + model_path alır, ısı haritasını base64 PNG döndürür |
| 4.1.3 | **Frontend overlay** | "Grad-CAM Göster" butonu → MRI üzerine kırmızı-sarı ısı haritası süperpoze |
| 4.1.4 | **Opacity slider** | MRI ile ısı haritası karışım oranını kaydırıcıyla ayarlama |

### 4.2 — Ön İşleme Adımları Görselleştirici

| # | Görev | Detay |
|---|-------|-------|
| 4.2.1 | **Backend endpoint** | `POST /infer/preprocess-steps/` → Her adımın çıktısını (ham → gri ton → CLAHE → resize) base64 döndür |
| 4.2.2 | **Step-by-step UI** | 4 adımlı yatay kart galerisi, animasyonlu geçiş |
| 4.2.3 | **Karşılaştırma modu** | Ham vs. işlenmiş görüntüyü yan yana gösterir |

### 4.3 — Batch Inference

| # | Görev | Detay |
|---|-------|-------|
| 4.3.1 | **ZIP yükleme endpoint'i** | `POST /infer/batch/` → ZIP dosyası al, içindeki tüm görüntüleri işle |
| 4.3.2 | **İşlem durumu** | Batch işlem sırasında progress bar (polling ile) |
| 4.3.3 | **Sonuç tablosu** | Her görüntü için tahmin sonucu tablo olarak göster |
| 4.3.4 | **CSV indirme** | Tahmin sonuçlarını `.csv` olarak indirme butonu |
| 4.3.5 | **Özet pie chart** | Batch sonunda sınıf dağılımını donut chart ile göster |

### 4.4 — Tahmin Geçmişi & Analiz

| # | Görev | Detay |
|---|-------|-------|
| 4.4.1 | **Filtreleme** | Sınıfa / model tipine / tarihe göre filtre |
| 4.4.2 | **Geçmiş istatistikleri** | Kaç tahmin yapıldı, sınıf dağılımı |
| 4.4.3 | **Silme** | Tek veya toplu tahmin kaydı silme |

### Çıktılar
- ResNet için Grad-CAM ısı haritası gösteriliyor
- ZIP ile toplu tahmin yapılabiliyor, CSV indirilebiliyor
- Ön işleme adımları adım adım görselleştiriliyor

---

## 🏃 Sprint 5 — Son Rötuşlar, UX & Deployment Hazırlığı

**Süre:** ~1-2 gün  
**Bağımlılık:** Sprint 1-4 tamamlanmış olmalı  
**Hedef:** Uygulamayı production'a hazır hale getirmek

### 5.1 — UX & Tasarım Polishing

| # | Görev | Detay |
|---|-------|-------|
| 5.1.1 | **Responsive tasarım** | Tablet ve mobil için breakpoint düzenlemeleri |
| 5.1.2 | **Loading skeleton** | İçerik yüklenirken iskelet animasyonu (CSS shimmer) |
| 5.1.3 | **Toast bildirimleri** | Başarı / hata mesajları için toast notification sistemi |
| 5.1.4 | **Hata sayfaları** | 404, 500, model_not_found için özel hata sayfaları |
| 5.1.5 | **Dark/Light mod** | CSS değişkenleri ile toggle |

### 5.2 — Model Yönetim Paneli

| # | Görev | Detay |
|---|-------|-------|
| 5.2.1 | **Model listesi sayfası** | `model/ciktilar/modeller/` içindeki `.pt` ve `.json` dosyaları listelenir |
| 5.2.2 | **Aktif model seçimi** | Hangi modelin tahmin için kullanılacağını session'da sakla |
| 5.2.3 | **Model metadata görüntüle** | Checkpoint'ten okunan eğitim bilgileri (epoch, accuracy, class_names) |

### 5.3 — Deployment Hazırlığı

| # | Görev | Detay |
|---|-------|-------|
| 5.3.1 | **`requirements-web.txt`** | `django`, `whitenoise`, `pillow`, `python-dotenv`, `gunicorn` |
| 5.3.2 | **`.env.example`** | `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `MODEL_DIR` değişkenleri |
| 5.3.3 | **`Makefile` hedefleri** | `make web` → `python web/manage.py runserver`, `make web-setup` → migrate + collectstatic |
| 5.3.4 | **WhiteNoise** | Production'da static dosyaları Django üzerinden serve etmek için |
| 5.3.5 | **`web/README.md`** | Kurulum ve çalıştırma talimatları |

---

## 📊 Sprint Özeti

| Sprint | Konu | Süre | Öncelik |
|--------|------|------|---------|
| **Sprint 1** | Django Kurulumu & İskelet | ~1 gün | 🔴 Kritik |
| **Sprint 2** | Inference Ekranı (Tahmin) | ~2-3 gün | 🔴 Kritik |
| **Sprint 3** | EDA Dashboard & Model Metrikler | ~2-3 gün | 🟠 Yüksek |
| **Sprint 4** | Grad-CAM, Batch, Gelişmiş | ~3-4 gün | 🟡 Orta |
| **Sprint 5** | Polish, Model Yönetimi, Deploy | ~1-2 gün | 🟢 Normal |
| **Toplam** | | **~9-13 gün** | |

---

## 🔧 Teknoloji Seçimleri

| Kategori | Araç | Neden |
|----------|------|-------|
| Backend framework | Django 5.x | ORM, admin, routing, form handling hazır |
| Veritabanı | SQLite (geliştirme) | Kurulum gerektirmez, yeterli |
| Grafik kütüphanesi | Chart.js | Hafif, vanilla JS ile entegre, güzel animasyonlar |
| CSS | Vanilla CSS + CSS Variables | Tam kontrol, bağımlılık yok |
| Görüntü işleme | Pillow + mevcut `goruntu_isleme` modülü | Zaten projede var |
| Grad-CAM | Elle PyTorch hooks | Hafif, ek bağımlılık minimal |
| Static serving | WhiteNoise | Production'da Nginx gerektirmez |
| Font | Google Fonts — Inter | Modern, okunabilir |

---

## ⚠️ Önemli Kısıtlar

> [!IMPORTANT]
> - Mevcut `model/`, `goruntu_isleme/`, `eda_analiz/` modüllerine **hiç dokunulmamalı**
> - Django uygulaması sadece bu modülleri **import ederek** kullanır
> - Eğitilmiş model dosyaları (`*.pt`, `*.json`) kullanıcı tarafından `model/ciktilar/modeller/` içine konulmuş olmalı
> - Model yükleme **uygulama başlarken bir kez** yapılmalı (her request'te değil)

> [!WARNING]
> - `MEDIA_ROOT` dizini `.gitignore`'a eklenmeli (kullanıcı yüklediği MRI dosyaları repoya girmemeli)
> - Tahmin sırasında `torch.no_grad()` kullanıldığından emin ol (inference.py zaten kullanıyor)
