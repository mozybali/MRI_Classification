# MRI Beyin Görüntüsü Sınıflandırma Projesi

Bu depo, 2D beyin MRI görüntülerinden demans seviyesini sınıflandırmak için
hazırlanmış uçtan uca bir Python çalışma alanıdır. Proje; keşifsel veri analizi
(EDA), görüntü ön işleme, veri sızıntısına dikkat eden `trainval/test` üretimi,
ResNet/PyTorch ve XGBoost eğitimi, Optuna ile hiperparametre araması, inference,
pytest testleri ve eğitilmiş modellerle çalışan Django web arayüzünü aynı
çatı altında toplar.

Desteklenen sınıflar:

| Sınıf klasörü | Etiket |
| --- | ---: |
| `NonDemented` | 0 |
| `VeryMildDemented` | 1 |
| `MildDemented` | 2 |
| `ModerateDemented` | 3 |

Sınıf klasörü adları kod içinde sabit kullanılır ve büyük/küçük harfe duyarlıdır.
Desteklenen görüntü uzantıları `.jpg`, `.jpeg` ve `.png` değerleridir.

## İçindekiler

- [Proje Akışı](#proje-akışı)
- [Dizin Yapısı](#dizin-yapısı)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Hızlı Kullanım](#hızlı-kullanım)
- [Modüller](#modüller)
- [Web Arayüzü](#web-arayüzü)
- [Testler](#testler)
- [Notlar](#notlar)
- [Lisans](#lisans)

## Proje Akışı

Önerilen çalışma sırası:

1. Ham görüntüleri `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutun.
2. `eda_analiz/` ile sınıf dağılımı, görüntü boyutu ve yoğunluk istatistiklerini inceleyin.
3. `goruntu_isleme/` ile ham görüntüleri kalite kontrolden geçirip standartlaştırın.
4. `mri-preprocess` çıktısı olan `trainval/test` yapısını model eğitimine verin.
5. `model/` ile ResNet veya XGBoost modeli eğitin.
6. Gerekirse `mri-tune` ile Optuna TPE tabanlı hiperparametre araması çalıştırın.
7. Eğitilmiş `.pt` veya `.json` model dosyasıyla tek görüntü ya da klasör tahmini alın.
8. İsterseniz modeli `web/` altındaki Django arayüzünde kullanın.
9. Değişiklikleri `pytest` testleriyle doğrulayın.

Model eğitimi için önerilen giriş ham veri değil, `mri-preprocess` sonrasında
oluşan şu dizinlerdir:

- `goruntu_isleme/cikti/trainval`
- `goruntu_isleme/cikti/test`

## Dizin Yapısı

```text
MRI_Classification/
|-- eda_analiz/
|   |-- __main__.py
|   |-- eda_araclar.py
|   |-- eda_calistir.py
|   `-- README.md
|-- goruntu_isleme/
|   |-- ana_islem.py
|   |-- artirma.py
|   |-- ayarlar.py
|   |-- goruntu_isleyici.py
|   |-- kalite_io.py
|   |-- on_isleme.py
|   |-- opencv_duzeltmeler.py
|   |-- temel.py
|   |-- toplu_islem.py
|   |-- veri.py
|   `-- README.md
|-- model/
|   |-- ayarlar.py
|   |-- hpo.py
|   |-- inference.py
|   |-- train.py
|   |-- training_runner.py
|   |-- common/
|   |-- dl/
|   |-- sl/
|   `-- README.md
|-- tests/
|   |-- conftest.py
|   |-- pipeline_quick_test.py
|   |-- test_akislari_ve_cli.py
|   |-- test_bugfixes.py
|   |-- test_eda_araclar.py
|   |-- test_goruntu_isleyici.py
|   |-- test_model_altyapi.py
|   |-- test_model_egitici.py
|   |-- test_model_sl.py
|   |-- test_pipeline.py
|   `-- README.md
|-- web/
|   |-- config/
|   |-- dashboard_app/
|   |-- inference_app/
|   |-- static/
|   |-- templates/
|   |-- .env.example
|   |-- Makefile
|   |-- manage.py
|   |-- requirements-web.txt
|   `-- README.md
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
|-- requirements-torch-cpu.txt
|-- requirements-torch-cu128.txt
|-- LICENSE
`-- README.md
```

Yerel veri ve çalışma çıktıları depoya dahil edilmez. `.gitignore` içinde
özellikle şu yollar dışarıda bırakılır:

- `Veri_Seti/`
- `eda_analiz/eda_ciktilar/`
- `goruntu_isleme/cikti/`
- `model/ciktilar/`
- `web/media/`, `web/staticfiles/`, `web/db.sqlite3`, `web/.env`
- model ağırlıkları ve sayısal cache dosyaları (`*.pt`, `*.pth`, `*.joblib`, `*.pkl`, `*.npy`, `*.npz`)

## Veri Seti

Beklenen ham veri yapısı:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

`eda_analiz/` ve `goruntu_isleme/` doğrudan sınıf klasörlerini içeren özel bir
klasörü de okuyabilir. `Veri_Seti` gibi içinde `OriginalDataset` bulunan bir üst
klasör verilirse ilgili alt klasör otomatik çözülebilir.

`Veri_Seti/OriginalDataset` ham görüntü kaynağı olarak korunmalıdır. Ön işlenmiş
görüntüler, raporlar, özellik cache'leri, model çıktıları ve web yüklemeleri bu
klasöre yazılmaz.

## Kurulum

Python `3.10+` gerekir. Komutları proje kök dizininden çalıştırın.

Sanal ortam:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
python -m pip install -U pip
```

Ortak bağımlılıklar:

```bash
pip install -r requirements.txt
```

PyTorch CPU kurulumu:

```bash
pip install -r requirements-torch-cpu.txt
```

GPU/CUDA ortamı için proje içindeki ayrı PyTorch dosyası:

```bash
pip install -r requirements-torch-cu128.txt
```

CLI komutlarını kullanılabilir yapmak için:

```bash
pip install -e . --no-deps
```

Geliştirme araçları `pyproject.toml` üzerinden kurulacaksa:

```bash
pip install -e ".[dev]" --no-deps
```

Tanımlı komutlar:

| Komut | Görev |
| --- | --- |
| `mri-eda` | Keşifsel veri analizi üretir. |
| `mri-preprocess` | Görüntü ön işleme ve `trainval/test` split üretimi yapar. |
| `mri-train` | ResNet veya XGBoost modeli eğitir. |
| `mri-tune` | Optuna ile hiperparametre araması çalıştırır. |
| `mri-infer` | Eğitilmiş modelle tahmin alır. |

## Hızlı Kullanım

Uçtan uca tipik akış:

```bash
# 1. Veri setini incele
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar

# 2. Görüntüleri işle ve leak-free trainval/test ayrımı üret
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti

# 3. ResNet eğit
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test

# 4. Alternatif olarak XGBoost eğit
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler

# 5. Eğitilmiş modelle tahmin al
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

Paket kurulmadan modül olarak çalıştırma:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
python3 -m goruntu_isleme.ana_islem --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
python3 -m model.train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
python3 -m model.inference --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
```

## Modüller

### EDA Analizi

`eda_analiz/` modülü veri setini değiştirmeden rapor, grafik ve CSV çıktıları
üretir. Sınıf dağılımı, görüntü boyutları, yoğunluk istatistikleri, korelasyon
matrisi ve PCA görünümü sağlar.

Örnek:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar --jobs 1
```

Üretilen ana çıktılar:

- `eda_analiz/eda_ciktilar/veri_seti_istatistikler.csv`
- `eda_analiz/eda_ciktilar/0_ozet_istatistikler.txt`
- analiz grafikleri (`1_sinif_dagilimi.png`, `2_boyut_analizi.png`, vb.)

EDA akışı Pillow ile görüntü okur, grafik üretiminde Matplotlib `Agg` backend'ini
kullanır ve okunamayan görüntüleri raporlayarak hesaplanabilen kayıtlarla devam
etmeye çalışır.

Ayrıntılar: [`eda_analiz/README.md`](eda_analiz/README.md)

### Görüntü İşleme

`goruntu_isleme/` modülü ham 2D MRI görüntülerini model eğitimine hazırlar.
Varsayılan akışta OpenCV ile görüntüleri okur, kalite kontrol uygular, kenar
artefaktlarını denetler/temizler, gri ton ve yoğunluk standartlaştırması yapar,
CLAHE uygular, görüntüleri `192x192` hedef boyuta `pad` modu ile getirir ve
`trainval/test` yapısını üretir.

Örnek:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Etkileşimli menü:

```bash
mri-preprocess --action menu
```

Beklenen çıktı:

```text
goruntu_isleme/cikti/
|-- trainval/
|   |-- NonDemented/
|   |-- VeryMildDemented/
|   |-- MildDemented/
|   `-- ModerateDemented/
`-- test/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Ön işleme çıktıları `.png` olarak yazılır. Varsayılan test oranı `0.15`,
rastgele tohum `42`, normalizasyon stratejisi `standard`, disk üzerinde offline
augmentation kapalıdır. Girdi zaten `trainval/test` yapısındaysa yeniden bölme
yapılmaz; mevcut split korunur.

Split üretimi kaynak grup bazında yapılır; aynı kaynaktan gelen orijinal ve
türev görüntülerin farklı splitlere düşmesi engellenmeye çalışılır. Eğim veya
anatomik kalite kontrol aday kaydı açılırsa normal `trainval/test` çıktısına ek
olarak manifest dosyalarıyla birlikte denetim klasörleri üretilebilir.

`mri-preprocess` için desteklenen aksiyonlar yalnızca `menu` ve `preprocess`
değerleridir. CSV özellik matrisi, scaler, XGBoost özellik cache'i ve model
dosyaları bu modülün görevi değildir.

Ayrıntılar: [`goruntu_isleme/README.md`](goruntu_isleme/README.md)

### Model Eğitimi

`model/` modülü eğitim, değerlendirme, hiperparametre arama ve tahmin alma
katmanıdır. İki model ailesini destekler:

| Model | `--model` değeri | Ana çıktı |
| --- | --- | --- |
| ResNet18/PyTorch | `resnet` | `.pt` checkpoint |
| XGBoost | `xgboost` | `.json` model ve `.meta.json` metadata |

ResNet:

```bash
mri-train --model resnet --epochs 50 --batch-size 32 --lr 3e-4 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost:

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

K-fold cross-validation:

```bash
mri-train --model resnet --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
mri-train --model xgboost --folds 5 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Final eğitim modu:

```bash
mri-train --model resnet --full-trainval --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Model çıktıları varsayılan olarak `model/ciktilar/` altında tutulur:

- `model/ciktilar/modeller/`
- `model/ciktilar/raporlar/`
- `model/ciktilar/gorseller/`
- `model/ciktilar/sl_ozellikler/`
- `model/ciktilar/_normalize_istatistikleri/`
- `model/ciktilar/hiperparametre_arama/`
- `model/ciktilar/folds/`

ResNet hattı `torchvision.models.resnet18` omurgasını kullanır; varsayılan
eğitim ImageNet ağırlıkları olmadan başlar, `--pretrained` verilirse pretrained
ağırlıklar kullanılır. XGBoost hattı HOG, LBP, GLCM, histogram ve istatistiksel
görüntü özellikleriyle çalışır; model `.json`, metadata ise `.meta.json` olarak
kaydedilir. Harici test seti varsa `goruntu_isleme/cikti/test` final
değerlendirme için kullanılır; validation ayrımı `trainval` içinden yapılır.

Ayrıntılar: [`model/README.md`](model/README.md)

### Hiperparametre Araması

`mri-tune`, Optuna `TPESampler` ile Bayes tabanlı arama yapar. Desteklenen seçim
metrikleri `loss`, `accuracy`, `precision`, `recall` ve `f1` değerleridir.

ResNet HPO:

```bash
mri-tune --model resnet --trials 20 --epochs 12 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

XGBoost HPO:

```bash
mri-tune --model xgboost --trials 30 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

Trial değerlendirmesini K-fold ile yapmak için:

```bash
mri-tune --model resnet --trials 20 --hpo-folds 5 --metric f1 --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

Varsayılan davranışta HPO bittikten sonra en iyi trial parametreleriyle final
eğitim çalıştırılır. Yalnızca arama sonuçlarını üretmek için:

```bash
mri-tune --model resnet --trials 20 --skip-final-train
```

HPO çıktıları varsayılan olarak
`model/ciktilar/hiperparametre_arama/<study_name>/` altında tutulur. Özet JSON,
trial geçmişi CSV'si, Optuna görselleri ve final eğitim atlanmadıysa `best_run/`
altında en iyi model çıktıları üretilir.

### Inference

`mri-infer`, model tipini dosya uzantısından algılar:

- `.pt`: ResNet/PyTorch checkpoint.
- `.json`: XGBoost modeli.

Tek görüntü:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --image ornek.jpg
```

Klasör bazlı tahmin:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --batch ornek_klasor
mri-infer --model-path model/ciktilar/modeller/best_xgboost.json --batch ornek_klasor
```

Ham görüntü üzerinde tahmin öncesi aynı MRI ön işleme hattını bellekte uygulamak
için:

```bash
mri-infer --model-path model/ciktilar/modeller/best_resnet.pt --image ornek.jpg --preprocess
```

Batch modu klasörün doğrudan içindeki desteklenen görüntüleri işler; alt
klasörleri özyinelemeli taramaz. `.pt` checkpoint'lerde normalize mean/std ve
sınıf adları checkpoint metadata'sından okunur; `.json` XGBoost modellerinde
varsa yan dosya `.meta.json` kullanılır.

## Web Arayüzü

`web/` dizini, eğitilmiş yerel modellerle tahmin alan ve EDA/model raporu
dashboard'unu sunan Django uygulamasını barındırır. Varsayılan geliştirme
veritabanı SQLite'tır (`web/db.sqlite3`); PostgreSQL kullanmak için `.env`
dosyasında `DB_ENGINE=postgresql` seçilir.

Web arayüzünün ana işlevleri:

- Tek MRI görüntüsü için ResNet (`.pt`) veya XGBoost (`.json`) tahmini.
- ZIP içindeki çoklu görüntüler için toplu tahmin ve CSV dışa aktarma.
- Tahmin öncesi ön işleme adımlarını görselleştirme.
- ResNet tahminleri için Grad-CAM ısı haritası.
- Son 20 tahmin kaydını listeleyen geçmiş ekranı.
- Veri seti dağılımı ve `model/ciktilar/` altındaki eğitim raporlarını gösteren dashboard.

Web bağımlılıkları:

```bash
python -m pip install -r web/requirements-web.txt
```

ResNet `.pt` inference ve Grad-CAM uçları için ayrıca `torch` ve `torchvision`
gerekir. XGBoost akışı torch olmadan çalışabilir; torch kurulu değilse ResNet ve
Grad-CAM uçları HTTP 503 ile hata döndürür.

Yerel kurulum özeti:

```bash
cd web
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

`createsuperuser` yalnızca Django admin paneline erişmek istiyorsanız gereklidir.
Sunucu açıldığında `/` adresi `/infer/` ekranına yönlenir; `/dashboard/` veri
seti ve model raporu dashboard'udur.

Model seçici `MODEL_DIR` içindeki dosyaları tarar. Varsayılan dizin
`model/ciktilar/modeller` değeridir; `.pt` dosyaları ResNet, `.json` dosyaları
XGBoost modeli olarak listelenir, `*.meta.json` dosyaları seçilebilir model
olarak gösterilmez. ML çıktı dosyaları veritabanına yüklenmez. Yüklenen tekli
tahmin görüntüleri `MEDIA_ROOT` altında saklanır; toplu tahmin sonuçları
veritabanına kaydedilmez, JSON/CSV olarak döndürülür.

Önemli web yolları:

| Yol | Görev |
| --- | --- |
| `/infer/` | Tekli/toplu tahmin ekranı. |
| `/infer/history/` | Son tahmin kayıtları. |
| `/infer/explain/<record_id>/` | ResNet Grad-CAM açıklaması. |
| `/infer/preprocess-steps/` | Ön işleme adımları. |
| `/dashboard/` | Veri seti ve model raporu dashboard'u. |

Ayrıntılar: [`web/README.md`](web/README.md)

## Testler

Test altyapısı `pytest` kullanır. Testler EDA, görüntü işleme, CLI davranışı,
XGBoost hattı, Torch/ResNet altyapısı, inference ve regresyon kontrollerini
kapsar. Testlerin büyük bölümü sentetik görüntüler ve `tmp_path` altında
oluşturulan geçici dosyalarla çalışır.

Gerçek veri seti gerektirmeyen normal geliştirme koşumu:

```bash
pytest -m "not requires_data"
```

Tüm pytest koleksiyonu:

```bash
pytest
```

Bu komut, gerçek veri isteyen `pipeline_quick_test.py::test_veri_seti` testi
nedeniyle `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını bekleyebilir.
Torch bağımlı test dosyaları varsayılan olarak atlanır. Açıkça çalıştırmak için:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Tek dosya veya tek test örnekleri:

```bash
pytest tests/test_goruntu_isleyici.py
pytest tests/test_model_sl.py::TestBuildFeatureMatrix::test_cache_roundtrip
```

Yardımcı pipeline kontrolleri:

```bash
python3 tests/pipeline_quick_test.py
python3 tests/test_pipeline.py
```

Ayrıntılar: [`tests/README.md`](tests/README.md)

## Notlar

- `mri-preprocess` sonrasında oluşan `test` dizinine augmentation uygulanmamalıdır.
- Eğitim sırasında validation ve test ayrımları ayrı tutulmalıdır; `--full-trainval` yalnızca final eğitim için kullanılmalıdır.
- XGBoost özellik cache'i veri dizini ve `image_size` bilgisiyle doğrulanır; farklı veri veya boyutla aynı cache kullanılmamalıdır.
- Validation ve test splitleri yalnızca orijinal görüntülerden kurulmalıdır; augmented kopyalar train tarafında kalır.
- ResNet hattında yatay çevirme varsayılan olarak kapalıdır; beyin MR görüntülerinde anatomik lateralite bilgi taşıyabilir.
- Bu proje 2D görüntü dosyalarıyla çalışır; doğrudan `.nii` veya `.nii.gz` hacim dosyası akışı ana yol değildir.
- Bu çalışma alanı araştırma/geliştirme amaçlıdır; klinik karar desteği olarak kullanılmadan önce ayrıca doğrulama gerekir.

## Lisans

Bu proje MIT lisansı ile yayımlanmıştır. Ayrıntılar için [`LICENSE`](LICENSE) dosyasına bakın.
