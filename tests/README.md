# Test Klasörü

`tests/` klasörü, MRI beyin görüntüsü sınıflandırma projesinin EDA, görüntü işleme, CLI, XGBoost/SL ve PyTorch/ResNet katmanlarını `pytest` ile doğrular. Testlerin çoğu sentetik görüntüler ve `tmp_path` altında üretilen geçici dosyalarla çalışır; gerçek veri seti ve Torch bağımlı senaryolar ayrı koşullarla açılır.

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

- `conftest.py`: Proje kökünü ve `model/` klasörünü import yoluna ekler. Sentetik MRI görüntüsü, geçici sınıf klasörü yapısı, örnek özellik `DataFrame`'i, çıktı klasörü ve temel test ayarları için ortak fixture'ları sağlar.
- `test_eda_araclar.py`: `eda_analiz/eda_araclar.py` içindeki `EDAAnaliz` sınıfını test eder. Veri klasörü çözümleme, `Veri_Seti/OriginalDataset` otomatik bulma, deterministik dosya tarama, headless grafik üretimi, görüntü istatistikleri, korelasyon/PCA sınır durumları ve tam EDA akışı bu dosyadadır.
- `test_akislari_ve_cli.py`: `goruntu_isleme/ana_islem.py` ve `eda_analiz/eda_calistir.py` CLI katmanlarını doğrular. Argüman ayrıştırma, `mri-preprocess` aksiyon yönlendirmesi, `mri-eda` yol çözümleme, lazy import davranışı, CSV yazımı ve hata dönüş kodları burada test edilir.
- `test_goruntu_isleyici.py`: `goruntu_isleme/goruntu_isleyici.py` için ana test dosyasıdır. Görüntü yükleme, yoğunluk normalizasyonu, histogram eşitleme, resize/padding modları, gürültü giderme, maskeleme, kalite kontrol, sınıf dosyası listeleme, kaynak grup bazlı split, `trainval/test` üretimi, augmentation politikası, deterministik çıktı adları ve paralel işlem fallback davranışını kapsar.
- `test_model_sl.py`: `model/sl/` XGBoost hattını test eder. `extract_features`, `build_feature_matrix`, feature cache kullanımı, grup anahtarları, XGBoost model oluşturma/kaydetme/yükleme, minimal SL eğitim koşumu, `mri-train --model xgboost` argümanları ve k-fold SL eğitim orkestrasyonu bu dosyada doğrulanır.
- `test_model_altyapi.py`: `model/dl/dataset.py`, `model/dl/engine.py` ve `model/inference.py` için Torch bağımlı altyapı testleridir. Dataset, transform, DataLoader splitleri, leak-free grup ayrımı, k-fold loader, eğitim/değerlendirme yardımcıları, checkpoint metadata kullanımı ve inference yardımcılarını kapsar.
- `test_model_egitici.py`: Derin öğrenme eğitim katmanını ve HPO akışını test eder. ResNet model oluşturma, focal loss ve class weight hesapları, grafik yardımcıları, `model/train.py` argümanları, veri dizini çözümleme, `model/training_runner.py`, final training, CV eğitim ve `model/hpo.py` davranışları bu dosyadadır.
- `test_bugfixes.py`: Raporlanan regresyonlar için koruma testlerini içerir. XGBoost inference metadata, feature cache `image_size`/veri dizini uyumu, validation/test loss hesapları, SL split güvenliği, HPO özetleri, final XGBoost `best_iteration`, HPO görselleştirmeleri, SL config validasyonu, `num_class` temizliği ve NaN/Inf özellik kontrollerini doğrular.
- `pipeline_quick_test.py`: Paket/import kontrolü, gerçek veri seti yapısı kontrolü ve `goruntu_isleme` modülü smoke testleri için yardımcı script'tir. Pytest tarafından da kolekte edilir; gerçek veri seti isteyen `test_veri_seti` testi `requires_data` marker'ı taşır.
- `test_pipeline.py`: Tek görüntü üzerinde görüntü işleme pipeline aşamalarını ve augmentation örneklerini görselleştiren yardımcı script'tir. Pytest koşumunda yalnızca script importlarının ve temel sabitlerin çözülebildiğini kontrol eden hafif test çalışır.

> Not: Özellik çıkarma testleri güncel yapıda `test_model_sl.py` ve ilgili regresyonlar için `test_bugfixes.py` içinde tutulur.

## Testleri Çalıştırma

Tüm varsayılan testleri çalıştırmak için:

```bash
pytest
```

`pytest.ini` varsayılan olarak `tests/` klasörünü kullanır ve `-v -l -ra --strict-markers` seçenekleriyle çalışır.

Gerçek veri seti gerektiren kontrolleri dışarıda bırakan hızlı koşum:

```bash
pytest -m "not requires_data and not requires_gpu and not slow"
```

Tek bir test dosyası çalıştırmak için:

```bash
pytest tests/test_goruntu_isleyici.py
```

Belirli bir test fonksiyonu çalıştırmak için:

```bash
pytest tests/test_model_sl.py::TestBuildFeatureMatrix::test_cache_roundtrip
```

Torch bağımlı iki dosya varsayılan olarak modül seviyesinde atlanır. Bu testleri çalıştırmak için ortam değişkenini verin:

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

`test_pipeline.py`, görüntü yolu verilmezse `Veri_Seti/OriginalDataset/<SinifAdi>/` altında ilk uygun görüntüyü arar. Belirli bir görüntüyü denemek için komutun sonuna görüntü yolunu ekleyin:

```bash
python3 tests/test_pipeline.py Veri_Seti/OriginalDataset/NonDemented/ornek.jpg
```

## Veri ve Ortam Beklentileri

- Testler repo kökünden çalıştırılacak şekilde tasarlanmıştır; CLI ve göreli yol kontrolleri buna göre yapılır.
- Sentetik veri kullanan testler gerçek veri setini değiştirmez. Geçici görüntüler, raporlar, cache dosyaları ve modeller çoğunlukla `tmp_path` altında üretilir.
- Gerçek veri bekleyen kontroller `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını arar. Beklenen sınıflar `NonDemented`, `VeryMildDemented`, `MildDemented` ve `ModerateDemented` adlarıyla bulunmalıdır.
- Torch testleri için PyTorch bağımlılıkları ayrıca kurulmuş olmalıdır. CPU kurulumu için `requirements-torch-cpu.txt`, CUDA 12.8 ortamı için `requirements-torch-cu128.txt` kullanılabilir.
- SL/XGBoost testleri `model/sl/` modüllerini ve `xgboost` bağımlılığını kullanır. Ortak bağımlılıklar `requirements.txt` içinde tanımlıdır.

## Pytest Marker'ları

`pytest.ini` içinde tanımlı marker'lar:

- `unit`: Küçük ve izole birim testleri.
- `integration`: Modüller arası akış testleri.
- `slow`: Uzun sürebilecek testler.
- `requires_data`: Gerçek veri seti gerektiren testler.
- `requires_gpu`: GPU gerektiren testler.

Marker filtreleri `-m` ile kullanılabilir:

```bash
pytest -m "not requires_data"
```

## Proje Modülleriyle Eşleşme

```text
eda_analiz/        -> test_eda_araclar.py, test_akislari_ve_cli.py
goruntu_isleme/    -> test_goruntu_isleyici.py, test_akislari_ve_cli.py, test_pipeline.py, pipeline_quick_test.py
model/sl/          -> test_model_sl.py, test_bugfixes.py
model/dl/          -> test_model_altyapi.py, test_model_egitici.py
model/train.py     -> test_model_sl.py, test_model_egitici.py
model/hpo.py       -> test_model_egitici.py, test_bugfixes.py
model/inference.py -> test_model_altyapi.py, test_bugfixes.py
```
