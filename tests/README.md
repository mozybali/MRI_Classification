# Test Klasörü

`tests/` klasörü, MRI beyin görüntüsü sınıflandırma projesinin EDA, görüntü işleme, özellik çıkarma, CLI, XGBoost ve PyTorch/ResNet katmanlarını `pytest` ile doğrular. Testlerin büyük bölümü sentetik görüntüler, geçici CSV'ler ve `tmp_path` kullanır; gerçek veri seti, GPU veya daha ağır Torch senaryoları ayrı koşullarla çalıştırılır.

Komutları repo kök dizininden çalıştırın.

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
|-- test_ozellik_cikarici.py
|-- test_pipeline.py
`-- README.md
```

## Dosyaların Görevleri

- `conftest.py`: Proje kökünü import yoluna ekler; sentetik görüntü, geçici veri seti, örnek özellik DataFrame'i, çıktı klasörü ve temel test ayarları için ortak fixture'ları sağlar.
- `test_eda_araclar.py`: `EDAAnaliz` sınıfını, veri klasörü çözümlemeyi, deterministik dosya taramayı, görüntü istatistiklerini, grafik üretimini, PCA/korelasyon sınır durumlarını ve tam EDA akışını test eder.
- `test_goruntu_isleyici.py`: `GorselIsleyici` için görüntü yükleme, normalizasyon, histogram eşitleme, resize, kalite kontrol, sınıf dosyalarını listeleme, kaynak grup bazlı split, `trainval/test` üretimi, augmentation politikası ve paralel işlem fallback davranışlarını doğrular.
- `test_ozellik_cikarici.py`: `OzellikCikarici` ile tek görüntü özellikleri, CSV üretimi, `.jpeg` desteği, NaN temizliği, scaling yöntemleri, grup/kaynak kolonları, train/validation/test bölme, original-only validation/test politikası ve scaler çıktısını test eder.
- `test_akislari_ve_cli.py`: Görüntü işleme ve EDA CLI katmanında argüman ayrıştırma, `preprocess`, `extract`, `scale`, `report`, `all` aksiyonları, EDA çalıştırıcı davranışı ve hata durumlarını test eder.
- `test_model_sl.py`: XGBoost hattında özellik çıkarma, özellik matrisi oluşturma, cache kullanımı, model kaydetme/yükleme, minimal shallow-learning eğitim koşumu ve `mri-train --model xgboost` argümanlarını doğrular.
- `test_model_altyapi.py`: `MRI_RUN_TORCH_TESTS=1` ile açılan Torch bağımlı altyapı testleridir. Dataset, DataLoader splitleri, eğitim/değerlendirme yardımcıları, checkpoint yükleme ve inference yardımcılarını kapsar.
- `test_model_egitici.py`: `MRI_RUN_TORCH_TESTS=1` ile açılan Torch bağımlı derin öğrenme testleridir. ResNet model oluşturma, loss fonksiyonları, görselleştirme yardımcıları, eğitim argümanları, HPO argümanları ve training runner seçim mantığını doğrular.
- `test_bugfixes.py`: Raporlanan kritik/yüksek öncelikli hatalar için regresyon testlerini içerir. XGBoost metadata, feature cache uyumluluğu, test loss, split güvenliği, HPO özetleri ve SL config validasyonları burada kontrol edilir.
- `pipeline_quick_test.py`: Paket/import kontrolü, `Veri_Seti/OriginalDataset/<SinifAdi>/` veri yapısı kontrolü ve görüntü işleme modülü smoke testlerini yapan yardımcı script'tir. Pytest içinde de kolekte edilir; veri seti kontrolü `requires_data` marker'ı taşır.
- `test_pipeline.py`: Tek görüntü üzerinde görüntü işleme adımlarını ve augmentation örneklerini görselleştiren yardımcı script'tir. Pytest tarafında script importlarının çözülebildiğini kontrol eden hafif bir test içerir.

## Testleri Çalıştırma

Tüm testleri çalıştırmak için:

```bash
pytest
```

Yavaş, gerçek veri veya GPU gerektiren testleri dışarıda bırakan hızlı koşum:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

Tek bir test dosyasını çalıştırmak için:

```bash
pytest tests/test_akislari_ve_cli.py
```

Torch bağımlı test dosyaları varsayılan olarak modül seviyesinde atlanır. Bu testleri çalıştırmak için ortam değişkenini verin:

```bash
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Windows PowerShell kullanıyorsanız:

```powershell
$env:MRI_RUN_TORCH_TESTS = "1"
pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Yardımcı pipeline kontrollerini doğrudan script olarak çalıştırmak için:

```bash
python3 tests/pipeline_quick_test.py
python3 tests/test_pipeline.py
```

`test_pipeline.py`, görüntü yolu verilmezse `Veri_Seti/OriginalDataset/<SinifAdi>/` altında ilk uygun görüntüyü arar. Belirli bir görüntüyü denemek için aynı komutun sonuna görüntü yolunu ekleyebilirsiniz.

## Pytest Marker'ları

`pytest.ini` içinde tanımlı marker'lar:

- `unit`: Küçük ve izole birim testleri.
- `integration`: Modüller arası akış testleri.
- `slow`: Uzun sürebilecek testler.
- `requires_data`: Gerçek veri seti gerektiren testler.
- `requires_gpu`: GPU gerektiren testler.

Marker filtreleri `-m` ile kullanılabilir:

```bash
pytest -m "not requires_data and not requires_gpu"
```

## Notlar

- Testleri repo kökünden çalıştırın; göreli yollar ve CLI kontrolleri buna göre tasarlanmıştır.
- Sentetik veri kullanan testler gerçek veri setini değiştirmez ve çoğunlukla `tmp_path` altında geçici dosyalar üretir.
- Gerçek veri bekleyen kontroller `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını arar. Beklenen sınıflar: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`.
- `pytest` ayarları `pytest.ini` içindedir; test kökü `tests`, minimum pytest sürümü `7.0`, varsayılan seçenekler `-v -l -ra --strict-markers` olarak tanımlıdır.
