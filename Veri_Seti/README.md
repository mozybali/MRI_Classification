# Veri Seti

`Veri_Seti/` klasörü, projenin ham ve orijinal MRI görüntüleri için beklenen kaynak dizindir. EDA, görüntü ön işleme ve isteğe bağlı doğrudan eğitim akışları varsayılan olarak `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerini okur.

Bu klasör veri kaynağı olarak korunmalıdır; çalışma sırasında üretilen görüntü, CSV, scaler, rapor veya model çıktıları `Veri_Seti/` altına yazılmamalıdır.

## Beklenen Klasör Yapısı

```text
Veri_Seti/
|-- README.md
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Sınıf klasör adları sabittir ve büyük/küçük harf duyarlılığı nedeniyle birebir bu şekilde kullanılmalıdır. Desteklenen görüntü uzantıları `.jpg`, `.jpeg` ve `.png` dosyalarıdır.

## Sınıflar ve Etiketler

| Sınıf klasörü | Etiket | Açıklama | Mevcut görüntü sayısı |
| --- | ---: | --- | ---: |
| `NonDemented` | 0 | Demans yok | 3200 |
| `VeryMildDemented` | 1 | Çok hafif demans | 2240 |
| `MildDemented` | 2 | Hafif demans | 896 |
| `ModerateDemented` | 3 | Orta seviye demans | 64 |

Bu sayılar, bu repo çalışma alanında `Veri_Seti/OriginalDataset` altında doğrulanan `.jpg`, `.jpeg` ve `.png` dosyalarına göre verilmiştir. Toplam doğrulanan görüntü sayısı `6400` dosyadır.

## Proje Akışındaki Yeri

- EDA modülü ham veri analizini `Veri_Seti/OriginalDataset` üzerinden yapar.
- Görüntü ön işleme modülü ham görüntüleri buradan okur ve işlenmiş çıktıları `goruntu_isleme/cikti` altına yazar.
- Model eğitimi için önerilen normal akış, ön işleme sonrası oluşan `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` klasörlerini kullanmaktır.
- Ham veriyle doğrudan eğitim mümkündür; bunun için eğitim komutunda `--trainval-dir Veri_Seti/OriginalDataset` verilebilir. Ancak veri sızıntısı kontrolü, test ayrımı ve tekrar üretilebilirlik açısından önce `mri-preprocess` çalıştırılması önerilir.

## Veri Politikası

- `OriginalDataset` değiştirilemez ham veri kaynağı olarak kabul edilir.
- Ön işlenmiş görüntüler, artırılmış örnekler, özellik CSV'leri, raporlar ve model çıktıları `Veri_Seti/` içine yazılmamalıdır.
- Test verisi original-only kalmalıdır; test tarafında augmentation kullanılmamalıdır.
- Augmentation etkinleştirilirse yalnızca eğitim veya `trainval` tarafında kalmalıdır.
- Split işlemleri, aynı kaynaktan türeyen ilişkili görüntüleri mümkün olduğunca aynı tarafta tutarak veri sızıntısı riskini azaltacak şekilde yapılmalıdır.

## Yararlı Komutlar

EDA çalıştırma:

```bash
mri-eda --data-dir Veri_Seti/OriginalDataset
```

Tam ön işleme, özellik çıkarma, ölçekleme ve rapor akışı:

```bash
mri-preprocess --action all --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti --yes
```

Hızlı pipeline kontrolü:

```bash
python3 tests/pipeline_quick_test.py
```

Paket komutları kullanılmıyorsa aynı akışlar `python3 -m eda_analiz` ve `python3 -m goruntu_isleme.ana_islem` üzerinden de çalıştırılabilir.
