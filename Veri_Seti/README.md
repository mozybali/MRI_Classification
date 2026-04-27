# Veri Seti Klasörü

Bu klasör, projenin ham MRI görüntülerini aradığı yerdir. Kod tabanı tek kaynak veri dizini olarak `Veri_Seti/OriginalDataset` yapısını bekler; EDA, ön işleme ve eğitim adımları varsayılan olarak bu yerleşimle uyumludur.

## Beklenen Yapı

```text
Veri_Seti/
|-- README.md
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Sınıf klasör adları birebir bu şekilde yazılmalıdır. Projedeki etiket eşlemesi sabittir:

| Klasör | Etiket | Anlam |
| --- | ---: | --- |
| `NonDemented` | 0 | Demans yok |
| `VeryMildDemented` | 1 | Çok hafif demans |
| `MildDemented` | 2 | Hafif demans |
| `ModerateDemented` | 3 | Orta seviye demans |

## Kullanım Akışı

- EDA modülü varsayılan olarak `Veri_Seti/OriginalDataset` içeriğini analiz eder.
- Görüntü işleme modülü ham veriyi önce kaynak grupları karışmayacak şekilde `trainval` ve `test` olarak böler.
- İşlenmiş görüntüler ve özellik CSV'leri bu klasöre değil, `goruntu_isleme/cikti` altına yazılır.
- Model eğitimi varsayılan olarak `goruntu_isleme/cikti/trainval` ve `goruntu_isleme/cikti/test` dizinlerini kullanır.
- Ham veriyle doğrudan eğitim yapılacaksa eğitim komutunda `--trainval-dir Veri_Seti/OriginalDataset` verilebilir.

## Veri Politikası

- `OriginalDataset` ham/orijinal veri kaynağıdır; bu klasördeki dosyalar çalışma sırasında değiştirilmez.
- Desteklenen görüntü uzantıları `.jpg`, `.jpeg` ve `.png` dosyalarıdır.
- `preprocess` adımı test verisini original-only tutar; augmentation yalnızca `trainval` tarafına uygulanır.
- Validation ayrımı eğitim sırasında `trainval` içinden yapılır ve original-only örneklerle kurulur.
- Aynı kaynaktan türeyen dosyalar mümkün olduğunca aynı split içinde tutulur; bu sayede veri sızıntısı riski azaltılır.

## Hızlı Kontrol

Veri yapısını ve paket kurulumunu hızlıca kontrol etmek için repo kökünden şu komut çalıştırılabilir:

```bash
python3 tests/pipeline_quick_test.py
```

EDA veya ön işleme başlatmadan önce dört sınıf klasörünün de mevcut olduğundan ve dosya adlarının aynı kaynaktan türeyen kopyaları ayırt edilebilir bıraktığından emin olun.
