# Test Klasörü

Bu klasör, projenin EDA, görüntü işleme, özellik çıkarma, CLI, derin öğrenme ve XGBoost akışlarını `pytest` ile doğrular. Testlerin önemli bölümü sentetik veri ve `tmp_path` kullanır; gerçek veri veya ağır Torch senaryoları varsayılan koşuldan ayrılmıştır.

## Dosya İçeriği

- `conftest.py`: Ortak fixture ve test yardımcıları.
- `test_eda_araclar.py`: EDA sınıfı, veri çözümleme ve analiz çıktıları.
- `test_goruntu_isleyici.py`: Görüntü yükleme, normalizasyon, kalite kontrol, augmentation ve leak-free split kontrolleri.
- `test_ozellik_cikarici.py`: Tek görüntü özellikleri, CSV üretimi, NaN temizleme, split ve ölçeklendirme.
- `test_akislari_ve_cli.py`: `mri-eda` ve `mri-preprocess` tarafındaki CLI/action akışları.
- `test_model_sl.py`: XGBoost özellik matrisi, model kaydetme/yükleme ve shallow-learning smoke testleri.
- `test_model_altyapi.py`: Torch tabanlı dataset, DataLoader, inference ve eğitim yardımcıları.
- `test_model_egitici.py`: ResNet, loss, HPO ve eğitim yardımcıları için Torch bağımlı testler.
- `test_bugfixes.py`: Daha önce raporlanan kritik/yüksek öncelikli hatalar için regresyon testleri.
- `pipeline_quick_test.py`: Paket kurulumu, veri seti yapısı ve görüntü işleme modülü için hızlı kontrol script'i.
- `test_pipeline.py`: Tek görüntü üzerinde ön işleme aşamalarını görselleştiren yardımcı script.

## Testleri Çalıştırma

Tüm varsayılan testleri çalıştırmak için:

```bash
pytest
```

Veri, GPU ve yavaş testleri dışarıda bırakan hızlı koşum:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

Belirli bir dosyayı çalıştırmak için:

```bash
pytest tests/test_akislari_ve_cli.py
```

Torch bağımlı test dosyaları varsayılan olarak modül seviyesinde atlanır. Bu testleri açmak için ortam değişkeni verin:

```bash
# Windows PowerShell
$env:MRI_RUN_TORCH_TESTS = "1"
pytest tests/test_model_altyapi.py tests/test_model_egitici.py

# macOS / Linux
MRI_RUN_TORCH_TESTS=1 pytest tests/test_model_altyapi.py tests/test_model_egitici.py
```

Görüntü işleme yardımcı kontrolleri:

```bash
python3 tests/pipeline_quick_test.py
python3 tests/test_pipeline.py ornek_goruntu.jpg
```

## Marker'lar

`pytest.ini` içinde tanımlı marker'lar:

- `unit`: Küçük ve izole birim testleri.
- `integration`: Modüller arası akış testleri.
- `slow`: Uzun sürebilecek testler.
- `requires_data`: Gerçek veri seti gerektiren testler.
- `requires_gpu`: GPU gerektiren testler.

## Notlar

- Testler repo kökünden çalıştırılmalıdır; paket kurulumu yapılmışsa CLI giriş noktaları da kullanılabilir.
- Sentetik veri kullanan testler proje veri setini değiştirmez.
- Gerçek veriyle çalışan yardımcı script'ler `Veri_Seti/OriginalDataset/<SinifAdi>/` yapısını bekler.
