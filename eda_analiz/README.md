# EDA Analiz Modülü

`eda_analiz`, MRI sınıflandırma projesindeki ham görüntü veri seti için keşifsel veri analizi üretir. Modül veri setini değiştirmez; sınıf klasörlerini tarar, desteklenen görüntüleri okur, temel boyut ve piksel yoğunluğu istatistiklerini hesaplar, ardından rapor, grafik ve CSV çıktıları oluşturur.

Analiz kodu komut satırından, paket modülü olarak veya doğrudan `EDAAnaliz` sınıfı üzerinden kullanılabilir.

## Dosya Yapısı

```text
eda_analiz/
|-- __init__.py
|-- __main__.py
|-- eda_araclar.py
|-- eda_calistir.py
`-- README.md
```

`eda_ciktilar/` kaynak dosya yapısının parçası değildir. Analiz çalıştığında varsayılan çıktı klasörü olarak otomatik oluşturulur.

## Dosyaların Görevleri

| Dosya | Görev |
| --- | --- |
| `__init__.py` | Paket dışına `EDAAnaliz` ve geriye dönük uyumluluk için `EDAAnaLiz` adlarını açar. Ağır EDA bağımlılıklarını paket import edildiği anda değil, ilgili öznitelik istendiğinde yükler. |
| `__main__.py` | `python3 -m eda_analiz` komutunu destekler ve çalışmayı `eda_calistir.main()` fonksiyonuna yönlendirir. |
| `eda_calistir.py` | Komut satırı arayüzünü içerir. Argümanları ayrıştırır, veri/çıktı yollarını çözer, `EDAAnaliz` sınıfını lazy import ile yükler, tam analizi çalıştırır ve DataFrame'i CSV olarak kaydeder. |
| `eda_araclar.py` | Ana analiz sınıfını ve yardımcı fonksiyonları içerir. Veri klasörünü doğrular, sınıf klasörlerini tarar, görüntü istatistiklerini paralel veya tek çekirdekli hesaplar, özet raporu ve grafikleri üretir. |
| `README.md` | Bu modülün güncel kullanım ve davranış dokümantasyonudur. |

## Bağımlılıklar

EDA akışı için temel bağımlılıklar şunlardır:

- `numpy`, `pandas`
- `Pillow`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `tqdm`

Bağımlılıklar proje kökünden kurulabilir:

```bash
pip install -r requirements.txt
```

Bu modül PyTorch gerektirmez. Grafikler sunucusuz/headless ortamlarda da üretilebilmesi için Matplotlib `Agg` backend'i ile çalışır.

## Veri Seti Yapısı

Varsayılan veri klasörü:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Doğrudan sınıf klasörlerini içeren özel bir klasör de desteklenir:

```text
benim_veri_klasorum/
|-- NonDemented/
|-- VeryMildDemented/
|-- MildDemented/
`-- ModerateDemented/
```

`--data-dir Veri_Seti` gibi içinde `OriginalDataset` bulunan bir üst klasör verilirse kod bu klasörü otomatik çözer. Boş veya beklenen sınıf klasörlerini içermeyen özel bir klasör verilirse varsayılana düşmez; hata verir.

Sınıf etiketleri:

| Sınıf klasörü | Etiket |
| --- | ---: |
| `NonDemented` | `0` |
| `VeryMildDemented` | `1` |
| `MildDemented` | `2` |
| `ModerateDemented` | `3` |

Analiz yalnızca sınıf klasörlerinin doğrudan içindeki dosyaları tarar; alt klasörleri özyinelemeli taramaz. Dosyalar platformdan bağımsız ve deterministik olması için dosya adına göre sıralanır.

## Desteklenen Görüntüler

- Desteklenen uzantılar: `.jpg`, `.jpeg`, `.png`
- Uzantı kontrolü küçük/büyük harfe duyarsızdır.
- Diğer dosya uzantıları atlanır ve ilgili sınıf için konsola uyarı yazılır.
- Görüntüler Pillow ile okunur.
- Gri tonlamalı olmayan görüntüler istatistik hesaplama sırasında bellekte `L` moduna çevrilir; kaynak dosyalar değiştirilmez.
- Okunamayan veya bozuk görüntüler raporlanır. Bu satırlar CSV'de kalır, ancak istatistik kolonları boş (`NaN`) olabilir.

## Komut Satırından Çalıştırma

Komutları proje kök dizininden çalıştırmak önerilir.

```bash
python3 -m eda_analiz
```

Alternatif kullanımlar:

```bash
python3 -m eda_analiz.eda_calistir
python3 eda_analiz/eda_calistir.py
```

Proje paket olarak kurulduysa `pyproject.toml` içinde tanımlı komut kullanılabilir:

```bash
mri-eda
```

Örnekler:

```bash
python3 -m eda_analiz --data-dir Veri_Seti/OriginalDataset
python3 -m eda_analiz --data-dir Veri_Seti --output-dir eda_analiz/eda_ciktilar
python3 -m eda_analiz --interactive
mri-eda --data-dir Veri_Seti/OriginalDataset --output-dir eda_analiz/eda_ciktilar --jobs 1
```

## CLI Parametreleri

| Parametre | Varsayılan | Açıklama |
| --- | --- | --- |
| `--data-dir` | `Veri_Seti/OriginalDataset` | Analiz edilecek veri klasörü. Doğrudan sınıf klasörlerini içerebilir veya içinde `OriginalDataset` bulunan bir üst klasör olabilir. |
| `--output-dir` | `eda_analiz/eda_ciktilar` | Rapor, grafik ve CSV çıktılarının yazılacağı klasör. Yoksa otomatik oluşturulur. |
| `--interactive` | `False` | Veri ve çıktı klasörlerini soru-cevap ile ister. Enter tuşu, CLI'da değer verilmişse o değeri; verilmemişse varsayılan yolu kullanır. |
| `--jobs` | `None` | Görüntü istatistikleri için kullanılacak çekirdek sayısı. Verilmezse `max(1, cpu_count() - 1)` seçilir. Verilen değer `1` ile mevcut CPU sayısı arasında sınırlandırılır. |

`--interactive` yalnızca veri ve çıktı klasörlerini sorar; `--jobs` interaktif olarak sorulmaz.

## Python İçinden Kullanım

```python
from eda_analiz import EDAAnaliz

analizci = EDAAnaliz(
    veri_klasoru="Veri_Seti/OriginalDataset",
    cikti_klasoru="eda_analiz/eda_ciktilar",
    n_jobs=1,
)

df = analizci.tam_analiz_yap()
```

`tam_analiz_yap()` rapor ve grafik çıktıları üretir, ardından hesaplanan kayıtları `pandas.DataFrame` olarak döndürür. CLI kullanıldığında bu DataFrame ayrıca `veri_seti_istatistikler.csv` dosyasına yazılır.

## Analiz Akışı

1. Veri klasörü doğrulanır ve gerekiyorsa `OriginalDataset` otomatik çözülür.
2. Beklenen sınıf klasörleri sabit sırayla taranır.
3. Her desteklenen görüntü için `id`, `filepath`, `label`, `label_name` kayıtları oluşturulur.
4. Genişlik, yükseklik, en-boy oranı ve piksel yoğunluğu istatistikleri hesaplanır.
5. Özet metin raporu oluşturulur.
6. Sınıf dağılımı, boyut, yoğunluk, korelasyon ve PCA grafikleri kaydedilir.
7. CLI ile çalıştırıldıysa tüm kayıtlar CSV olarak yazılır.

Paralel istatistik hesaplama kullanılamazsa kod uyarı yazıp tek çekirdeğe düşerek devam etmeyi dener.

## Üretilen Çıktılar

Varsayılan çıktı klasörü:

```text
eda_analiz/eda_ciktilar/
```

| Dosya | İçerik |
| --- | --- |
| `0_ozet_istatistikler.txt` | Toplam görüntü sayısı, mevcut sınıf sayısı, sınıf dağılımı ve `DataFrame.describe()` çıktısı. |
| `1_sinif_dagilimi.png` | Sınıf başına görüntü sayısını gösteren çubuk grafik. |
| `2_boyut_analizi.png` | Genişlik, yükseklik, en-boy oranı histogramları ve genişlik-yükseklik saçılım grafiği. |
| `3_yogunluk_analizi.png` | Sınıflara göre ortalama yoğunluk, standart sapma, `max-min` aralığı ve `P99-P1` yayılımı kutu grafikleri. |
| `4_korelasyon_matrisi.png` | Sayısal özellikler arasındaki korelasyon matrisi. Sabit özellikler analizden çıkarılır; yeterli değişken özellik yoksa grafik atlanır. |
| `5_pca_analizi.png` | Sayısal özelliklerle oluşturulan iki bileşenli PCA saçılım grafiği. En fazla 500 geçerli örnek kullanılır; yeterli geçerli/değişken özellik yoksa grafik atlanır. |
| `veri_seti_istatistikler.csv` | CLI çalıştırması sonunda kaydedilen görüntü kayıtları ve hesaplanan istatistikler. |

CSV kolonları:

```text
id, filepath, label, label_name,
genislik, yukseklik, en_boy_orani,
int_ort, int_std, int_min, int_max,
int_p1, int_p25, int_p50, int_p75, int_p99
```

Korelasyon ve PCA analizinde kullanılan sayısal özellikler:

```text
genislik, yukseklik, en_boy_orani,
int_ort, int_std, int_min, int_max,
int_p1, int_p99
```

`int_p25`, `int_p50` ve `int_p75` CSV'ye yazılır; korelasyon/PCA özellik setine dahil edilmez.

## Hata ve Uyarı Davranışı

- Veri klasörü yoksa `FileNotFoundError` oluşur.
- Verilen klasörde beklenen sınıf klasörlerinden hiçbiri bulunamazsa analiz durur.
- Beklenen sınıflardan bazıları eksikse uyarı yazılır ve mevcut sınıflarla devam edilir.
- Desteklenen uzantıya sahip hiç görüntü bulunamazsa analiz durur.
- Hiçbir görüntüden istatistik hesaplanamazsa analiz durur.
- Bazı görüntüler okunamazsa bu dosyalar uyarı olarak listelenir; hesaplanabilen görüntülerle devam edilir.
- CLI hata durumunda hata mesajı ve traceback yazdırır, çıkış kodu `1` döndürür.
- Başarılı CLI çalışması çıkış kodu `0` ile biter.

## Ne Zaman Kullanılır?

Bu modül model eğitimi veya ön işleme kararlarından önce yararlıdır:

- Sınıflar arasında dengesizlik olup olmadığını görmek için.
- Görüntü genişliği, yüksekliği ve en-boy oranı dağılımlarını kontrol etmek için.
- Piksel yoğunluğu istatistiklerini sınıflar arasında karşılaştırmak için.
- Okunamayan dosyaları, desteklenmeyen uzantıları veya eksik sınıf klasörlerini erken fark etmek için.
- Deney takibi, raporlama veya sunum için temel EDA çıktıları üretmek için.
