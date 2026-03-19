# Test Yapisi

Bu klasor, projenin EDA, goruntu isleme, CLI ve model altyapisini dogrulayan `pytest` testlerini icerir.

## Icerik

- `conftest.py`: Ortak fixture'lar ve test yardimcilari
- `test_eda_araclar.py`: EDA sinifi ve analiz ciktilari
- `test_goruntu_isleyici.py`: On isleme ve goruntu donusumleri
- `test_ozellik_cikarici.py`: Ozellik cikarma, split ve olceklendirme
- `test_model_altyapi.py`: Dataset, loss, utility ve split altyapisi
- `test_model_egitici.py`: Egitim akisina yonelik birim kontroller
- `test_akislari_ve_cli.py`: CLI action'lari ve uctan uca akis kontrolleri
- `_runtime_verify/`: Calisma sirasinda olusabilecek gecici dogrulama klasoru

## Calistirma

Tum testler:

```bash
pytest
```

Sadece hizli testler:

```bash
pytest -m "not slow and not requires_data and not requires_gpu"
```

Belirli dosya:

```bash
pytest tests/test_akislari_ve_cli.py
```

## Marker'lar

`pytest.ini` icinde tanimli marker'lar:

- `unit`
- `integration`
- `slow`
- `requires_data`
- `requires_gpu`

## Notlar

- Testlerin buyuk bolumu sentetik veri ve gecici dizinler kullanir.
- Gercek veri veya GPU bagimli senaryolari ayirmak icin marker'lar kullanilir.
- Paket kurulu degilse testler repo kokunden calistirilmalidir.
