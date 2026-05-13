# Test Klasörü

`tests/` klasörü, MRI beyin görüntüsü sınıflandırma projesinin EDA, görüntü işleme, CLI, XGBoost/SL, PyTorch/ResNet eğitim ve inference katmanlarını `pytest` ile doğrular. Testlerin büyük bölümü sentetik görüntüler ve `tmp_path` altında oluşturulan geçici dosyalarla çalışır. Gerçek veri seti isteyen kontrol ve Torch bağımlı test dosyaları ayrı koşullarla çalıştırılır.

Komutları proje kök dizininden çalıştırın.

## Dizin Yapısı

```text
tests/
|-- __init__.py
|-- conftest.py
|-- pipeline_quick_test.py
|-- test_akislari_ve_cli.py
|-- test_bugfixes.py
|-- test_eda_araclar.py
|-- test_goruntu_isleyici.py
|-- test_model_altyapi.py
|-- test_model_egitici.py
|-- test_model_sl.py
|-- test_pipeline.py
`-- README.md
```

## Kapsam

- `conftest.py`: Proje kökünü ve `model/` klasörünü import yoluna ekler. Sentetik 256x256 gri MRI görüntüsü, geçici dört sınıflı veri yapısı, örnek özellik `DataFrame`'i, geçici çıktı klasörü ve temel test ayarları için ortak fixture'ları sağlar.
- `test_eda_araclar.py`: `eda_analiz/eda_araclar.py` içindeki `EDAAnaliz` sınıfını test eder. BOM kontrolü, headless grafik üretimi, Unicode yazdırma dayanıklılığı, veri klasörü çözümleme, `Veri_Seti/OriginalDataset` otomatik bulma, deterministik dosya tarama, görüntü istatistikleri, paralel istatistik fallback'i, korelasyon/PCA sınır durumları ve tam EDA akışı bu dosyadadır.
- `test_akislari_ve_cli.py`: `goruntu_isleme/ana_islem.py` ve `eda_analiz/eda_calistir.py` CLI katmanlarını doğrular. Argüman ayrıştırma, `mri-preprocess` aksiyon yönlendirmesi, `mri-eda` yol çözümleme, `--jobs` değeri, lazy import davranışı, CSV yazımı ve hata dönüş kodları burada test edilir.
- `test_goruntu_isleyici.py`: `goruntu_isleme/goruntu_isleyici.py` ve bağlı görüntü işleme mixin'leri için ana test dosyasıdır. Görüntü yükleme/kaydetme, tohumlama, yoğunluk normalizasyonu, histogram eşitleme, resize/padding, gürültü giderme, bias correction, maske düzenleme, kalite kontrol, sınıf dosyası listeleme, kaynak grup bazlı split, `trainval/test` üretimi, augmentation sayımı, deterministik çıktı adları ve paralel işlem fallback davranışını kapsar. Ayrıca eğim düzeltme ve eğim kalite kontrolü, anatomik kalite kontrol, kenar artefakt tespit/temizleme, arka planın korunması ve pipeline sonu kalite redleri için regresyon testleri içerir.
- `test_model_sl.py`: `model/sl/` XGBoost hattını test eder. `extract_features`, gri/RGB giriş dönüştürme, histogram istatistikleri, özellik grup dilimleri, `build_feature_matrix`, feature cache kullanımı, grup anahtarları, XGBoost model oluşturma/kaydetme/yükleme, minimal SL eğitim koşumu, `mri-train --model xgboost` argümanları, seçim metriği wiring'i, best-iteration grafiği ve k-fold SL eğitim orkestrasyonu bu dosyada doğrulanır.
- `test_model_altyapi.py`: `model/dl/dataset.py`, `model/dl/engine.py` ve `model/inference.py` için Torch bağımlı altyapı testleridir. Dataset, transform, DataLoader splitleri, leak-free grup ayrımı, seeded generator kullanımı, harici test loader'ı, k-fold loader, eğitim/değerlendirme yardımcıları, checkpoint metadata kullanımı ve inference yardımcılarını kapsar.
- `test_model_egitici.py`: Derin öğrenme eğitim katmanını, CLI seçeneklerini ve HPO akışını test eder. Grup/sınıf dengeli splitler, focal loss ve class weight hesapları, ayrıntılı değerlendirme raporları ve grafikler, ResNet model oluşturma, güvenli checkpoint yükleme, `model/train.py` argümanları, veri dizini çözümleme, `model/training_runner.py`, final training, CV eğitim, dataset stats cache'i, deterministik/TF32 runtime ayarları, CUDA bellek temizliği, NaN loss korumaları ve `model/hpo.py` davranışları bu dosyadadır.
- `test_bugfixes.py`: Raporlanan regresyonlar için koruma testlerini içerir. XGBoost inference metadata, feature cache `image_size`/veri dizini uyumu, validation/test loss hesapları, SL split güvenliği, augmented örneklerin val/test dışı tutulması, HPO özetleri, final XGBoost `best_iteration`, HPO görselleştirmeleri, SL config validasyonu, gereksiz `num_class` temizliği ve NaN/Inf özellik kontrollerini doğrular.
- `pipeline_quick_test.py`: Paket/import kontrolü, gerçek veri seti yapısı kontrolü ve `goruntu_isleme` modülü smoke testleri için yardımcı script'tir. Pytest tarafından da kolekte edilir; gerçek veri seti isteyen `test_veri_seti` testi `requires_data` marker'ı taşır.
- `test_pipeline.py`: Tek görüntü üzerinde görüntü işleme pipeline aşamalarını ve augmentation örneklerini görselleştiren yardımcı script'tir. Pytest koşumunda yalnızca script importlarının ve temel sabitlerin çözülebildiğini kontrol eden hafif test çalışır. Script olarak çalıştırıldığında çıktıları `cikti/test/pipeline_test.png` ve `cikti/test/augmentation_test.png` altına yazar.

> Not: Özellik çıkarma testleri güncel yapıda `test_model_sl.py` içinde, ilgili regresyonlar ise `test_bugfixes.py` içinde tutulur.

## Testleri Çalıştırma

`pytest.ini`, varsayılan olarak `tests/` klasörünü kullanır ve `-v -l -ra --strict-markers` seçeneklerini ekler.

Gerçek veri seti gerektirmeyen normal geliştirme koşumu:

```bash
pytest -m "not requires_data"
```

Tüm pytest koleksiyonunu çalıştırmak için:

```bash
pytest
```

Bu komut, `pipeline_quick_test.py::test_veri_seti` nedeniyle `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını bekler. Torch bağımlı iki dosya ise ortam değişkeni verilmediği sürece modül seviyesinde atlanır.

Tek bir test dosyası çalıştırmak için:

```bash
pytest tests/test_goruntu_isleyici.py
```

Belirli bir test fonksiyonu çalıştırmak için:

```bash
pytest tests/test_model_sl.py::TestBuildFeatureMatrix::test_cache_roundtrip
```

Torch bağımlı testleri çalıştırmak için PyTorch kurulu bir ortamda `MRI_RUN_TORCH_TESTS=1` verin:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Windows PowerShell için:

```powershell
$env:MRI_RUN_TORCH_TESTS = "1"
pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Yardımcı pipeline scriptlerini doğrudan çalıştırmak için:

```bash
python3 tests/pipeline_quick_test.py
python3 tests/test_pipeline.py
```

`pipeline_quick_test.py` script olarak çalıştırıldığında paket, veri seti ve modül kontrollerinin tamamını yapar. `test_pipeline.py`, görüntü yolu verilmezse `Veri_Seti/OriginalDataset/<SinifAdi>/` altında ilk uygun görüntüyü arar. Belirli bir görüntüyü denemek için komutun sonuna görüntü yolunu ekleyin:

```bash
python3 tests/test_pipeline.py Veri_Seti/OriginalDataset/NonDemented/ornek.jpg
```

## Veri ve Ortam Beklentileri

- Testler repo kökünden çalıştırılacak şekilde tasarlanmıştır; CLI ve göreli yol kontrolleri buna göre yapılır.
- Sentetik veri kullanan testler gerçek veri setini değiştirmez. Geçici görüntüler, raporlar, cache dosyaları ve modeller çoğunlukla `tmp_path` altında üretilir.
- Gerçek veri bekleyen kontroller `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını arar. Beklenen sınıflar `NonDemented`, `VeryMildDemented`, `MildDemented` ve `ModerateDemented` adlarıyla bulunmalıdır.
- Görüntü işleme ve CLI testleri `cv2` import eder; `opencv-python` kurulmadan bu dosyalar koleksiyon aşamasında bile çalışmaz.
- Torch testleri için PyTorch bağımlılıkları ayrıca kurulmuş olmalıdır. CPU kurulumu için `requirements-torch-cpu.txt`, CUDA ortamı için `requirements-torch-cu128.txt` kullanılabilir.
- SL/XGBoost testleri `model/sl/` modüllerini ve `xgboost` bağımlılığını kullanır. Ortak bağımlılıklar `requirements.txt` ve `pyproject.toml` içinde tanımlıdır.
- HPO görselleştirme regresyonu `optuna` kuruluysa çalışır; ilgili test Optuna yoksa kendi içinde atlanır.

## Pytest Marker'ları

`pytest.ini` içinde tanımlı marker'lar:

- `unit`: Küçük ve izole birim testleri için ayrılmış marker.
- `integration`: Modüller arası akış testleri için ayrılmış marker.
- `slow`: Uzun sürebilecek testler için ayrılmış marker.
- `requires_data`: Gerçek veri seti gerektiren testler.
- `requires_gpu`: GPU gerektiren testler için ayrılmış marker.

Marker filtreleri `-m` ile kullanılabilir:

```bash
pytest -m "not requires_data"
```

## Proje Modülleriyle Eşleşme

```text
eda_analiz/                    -> test_eda_araclar.py, test_akislari_ve_cli.py
goruntu_isleme/                -> test_goruntu_isleyici.py, test_akislari_ve_cli.py, test_pipeline.py, pipeline_quick_test.py
model/sl/features.py           -> test_model_sl.py, test_bugfixes.py
model/sl/dataset.py            -> test_model_sl.py, test_bugfixes.py
model/sl/xgb_classifier.py     -> test_model_sl.py, test_bugfixes.py
model/sl/training_runner.py    -> test_model_sl.py, test_bugfixes.py
model/dl/dataset.py            -> test_model_altyapi.py, test_model_egitici.py
model/dl/engine.py             -> test_model_altyapi.py, test_model_egitici.py
model/dl/losses.py             -> test_model_egitici.py
model/dl/utils.py              -> test_model_egitici.py
model/dl/models/               -> test_model_egitici.py
model/train.py                 -> test_model_sl.py, test_model_egitici.py
model/training_runner.py       -> test_model_egitici.py, test_bugfixes.py
model/hpo.py                   -> test_model_egitici.py, test_bugfixes.py
model/inference.py             -> test_model_altyapi.py, test_bugfixes.py
```
