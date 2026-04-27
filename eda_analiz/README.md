# EDA Analiz Modülü

Bu modül, MRI sınıflandırma projesindeki görüntü veri seti için keşifsel veri analizi (EDA, Exploratory Data Analysis) üretir. Amaç; eğitimden önce sınıf dağılımını, görüntü boyutlarını, piksel yoğunluğu istatistiklerini, temel korelasyonları ve PCA görünümünü incelemektir.

Modül veri setini değiştirmez; görüntüleri okuyarak rapor, grafik ve CSV çıktıları oluşturur.

## Dosya Yapısı

```text
eda_analiz/
|-- __init__.py
|-- __main__.py
|-- eda_araclar.py
|-- eda_calistir.py
`-- README.md
```

`eda_ciktilar/` klasörü kaynak dosya yapısının parçası değildir; analiz çalıştırıldığında varsayılan çıktı klasörü olarak otomatik oluşturulabilir.

## Dosyaların Görevleri

- `__init__.py`: Paket dışına `EDAAnaliz` sınıfını ve geriye dönük uyumluluk için `EDAAnaLiz` adını açar. Ağır analiz bağımlılıklarını doğrudan import sırasında değil, ilgili öznitelik istendiğinde yükler.
- `__main__.py`: `python3 -m eda_analiz` komutunu destekler ve çalışmayı `eda_calistir.main()` fonksiyonuna yönlendirir.
- `eda_araclar.py`: EDA işlemlerinin ana uygulamasını içerir. `EDAAnaliz` sınıfı veri klasörünü doğrular, sınıf klasörlerini tarar, desteklenen görüntü dosyalarını yükler, temel görüntü istatistiklerini hesaplar ve rapor/grafik çıktılarını kaydeder.
- `eda_calistir.py`: Komut satırı arayüzüdür. Argümanları ayrıştırır, varsayılan veya kullanıcı tarafından verilen yolları belirler, `EDAAnaliz` sınıfını çalıştırır ve analiz DataFrame'ini CSV olarak kaydeder.
- `README.md`: Bu modülün kullanımı ve beklenen davranışı için dokümantasyon dosyasıdır.

## Ne Zaman Kullanılır?

Bu modül özellikle model eğitimi veya veri ön işleme kararlarından önce kullanılmalıdır:

- Sınıflar arasında veri dengesizliği olup olmadığını görmek için.
- MRI görüntülerinin genişlik, yükseklik ve en-boy oranı dağılımlarını kontrol etmek için.
- Piksel yoğunluğu istatistiklerini sınıflar arasında karşılaştırmak için.
- Okunamayan dosyaları, desteklenmeyen uzantıları veya eksik sınıf klasörlerini erken fark etmek için.
- Rapor, proje sunumu veya deney takibi için temel EDA görselleri üretmek için.

## Komut Satırından Çalıştırma

Komutları proje kök dizininden çalıştırmak önerilir.

Doğrudan Python modülü olarak:

```bash
python3 -m eda_analiz
```

Alternatif olarak çalıştırma modülünü doğrudan çağırabilirsiniz:

```bash
python3 -m eda_analiz.eda_calistir
```

Script dosyası olarak:

```bash
python3 eda_analiz/eda_calistir.py
```

Proje paket olarak kurulduysa `pyproject.toml` içinde tanımlı komut da kullanılabilir:

```bash
mri-eda
```

Örnek kullanımlar:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
python3 -m eda_analiz --data-dir Veri_Seti --output-dir eda_analiz/eda_ciktilar
python3 -m eda_analiz --interactive
mri-eda --data-dir Veri_Seti/OriginalDataset --jobs 1
```

## CLI Parametreleri ve Varsayılanlar

| Parametre | Varsayılan | Açıklama |
| --- | --- | --- |
| `--data-dir` | `Veri_Seti/OriginalDataset` | Analiz edilecek veri klasörü. Verilen klasör doğrudan sınıf klasörlerini içerebilir. `Veri_Seti` gibi içinde `OriginalDataset` bulunan bir üst klasör verilirse `OriginalDataset` otomatik çözümlenir. |
| `--output-dir` | `eda_analiz/eda_ciktilar` | Rapor, grafik ve CSV çıktılarının yazılacağı klasör. Klasör yoksa otomatik oluşturulur. |
| `--interactive` | `False` | Verilirse veri ve çıktı klasörleri soru-cevap şeklinde istenir. Enter tuşuna basılırsa ilgili varsayılan değer kullanılır. |
| `--jobs` | `None` | Görüntü istatistiklerini hesaplamak için kullanılacak işlemci çekirdeği sayısı. Verilmezse otomatik olarak `max(1, cpu_count() - 1)` seçilir. Verilen değer en az `1`, en fazla mevcut CPU sayısı olacak şekilde sınırlandırılır. |

`--interactive` yalnızca veri klasörü ve çıktı klasörü için soru sorar; `--jobs` değeri interaktif olarak sorulmaz.

## Beklenen Veri Seti Yapısı

Varsayılan veri yapısı:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Doğrudan sınıf klasörlerini içeren özel bir veri dizini de desteklenir:

```text
benim_veri_klasorum/
|-- NonDemented/
|-- VeryMildDemented/
|-- MildDemented/
`-- ModerateDemented/
```

Sınıf etiketleri kod içinde şu şekilde tanımlıdır:

| Sınıf klasörü | Etiket |
| --- | ---: |
| `NonDemented` | `0` |
| `VeryMildDemented` | `1` |
| `MildDemented` | `2` |
| `ModerateDemented` | `3` |

Analiz, sınıf klasörlerinin doğrudan içindeki dosyaları tarar; alt klasörleri özyinelemeli olarak taramaz.

## Üretilen Çıktılar

Varsayılan olarak çıktılar `eda_analiz/eda_ciktilar/` klasörüne yazılır. Farklı bir klasör için `--output-dir` kullanılabilir.

| Dosya | İçerik |
| --- | --- |
| `0_ozet_istatistikler.txt` | Toplam görüntü sayısı, mevcut sınıf sayısı, sınıf dağılımı ve `pandas.DataFrame.describe()` çıktısı. |
| `1_sinif_dagilimi.png` | Sınıf başına görüntü sayısını gösteren çubuk grafik. |
| `2_boyut_analizi.png` | Genişlik, yükseklik, en-boy oranı dağılımları ve genişlik-yükseklik saçılım grafiği. |
| `3_yogunluk_analizi.png` | Sınıflara göre ortalama yoğunluk, standart sapma, yoğunluk aralığı ve P99-P1 yayılımı kutu grafikleri. |
| `4_korelasyon_matrisi.png` | Sayısal görüntü özellikleri arasındaki korelasyon matrisi. Sabit özellikler varsa analizden çıkarılır. |
| `5_pca_analizi.png` | Sayısal özelliklerle oluşturulan iki bileşenli PCA saçılım grafiği. En fazla 500 geçerli örnek kullanılır. |
| `veri_seti_istatistikler.csv` | CLI çalıştırması sonunda kaydedilen görüntü kayıtları ve hesaplanan istatistikler. |

CSV dosyasında temel olarak şu alanlar bulunur:

```text
id, filepath, label, label_name,
genislik, yukseklik, en_boy_orani,
int_ort, int_std, int_min, int_max,
int_p1, int_p25, int_p50, int_p75, int_p99
```

## Notlar ve Uyarılar

- Desteklenen görüntü uzantıları `.jpg`, `.jpeg` ve `.png` dosyalarıdır. Diğer uzantılar atlanır ve konsola uyarı yazılır.
- Görüntüler Pillow ile okunur. Gri tonlamalı olmayan görüntüler istatistik hesaplama sırasında bellekte gri tonlamaya çevrilir; kaynak dosyalar değiştirilmez.
- Veri klasörü bulunamazsa veya beklenen sınıf klasörlerinden hiçbiri çözümlenemezse analiz hata verir.
- Beklenen sınıflardan bazıları eksikse konsola uyarı yazılır ve mevcut sınıflarla devam edilir.
- Desteklenen uzantıya sahip hiç görüntü bulunamazsa analiz hata verir.
- Okunamayan veya bozuk görüntüler için uyarı yazılır. Bu dosyaların istatistik kolonları boş kalabilir; hiçbir görüntüden istatistik hesaplanamazsa analiz durur.
- Paralel istatistik hesaplama kullanılamazsa kod tek çekirdeğe düşerek devam etmeyi dener.
- Korelasyon ve PCA grafikleri için yeterli sayıda geçerli/değişken özellik yoksa ilgili grafik atlanabilir.
- Komut `--interactive` verilmeden çalıştırıldığında kullanıcıdan girdi beklemez; doğrudan varsayılan veya argümanla verilen yolları kullanır.
- Analiz sırasında hata oluşursa CLI hata mesajı ve traceback yazdırır, çıkış kodu `1` döndürür. Başarılı çalışmada çıkış kodu `0` olur.
