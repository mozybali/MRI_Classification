# Görüntü İşleme Modülü

`goruntu_isleme`, ham 2D beyin MRI görüntülerini model eğitimine hazır hale getiren ön işleme paketidir. Varsayılan akışta görüntüleri `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerinden okur, kalite kontrol ve standartlaştırma uygular, kaynak grup bazlı `trainval/test` ayrımı yapar ve işlenmiş görüntüleri `goruntu_isleme/cikti/` altında `.png` olarak kaydeder.

Bu klasör yalnızca görüntü ön işleme sorumluluğunu taşır. Özellik matrisi üretimi, XGBoost özellik cache'i, model eğitimi, hiperparametre araması ve inference işlemleri `model/` modülündedir.

## Projedeki Yeri

Tipik akış:

1. Ham MRI görüntüleri `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutulur.
2. İsteğe bağlı EDA işlemleri `eda_analiz/` ile çalıştırılır.
3. Bu modül `mri-preprocess` komutu ile ham görüntüleri işler ve `trainval/test` klasörlerini üretir.
4. `model/` modülü, üretilen klasörleri kullanarak ResNet veya XGBoost eğitir.
5. XGBoost için gerekli HOG, LBP, GLCM ve histogram özellikleri model tarafında çıkarılır.

## Dosya Yapısı

Kaynak dosyalar:

```text
goruntu_isleme/
|-- __init__.py
|-- ana_islem.py
|-- artirma.py
|-- ayarlar.py
|-- goruntu_isleyici.py
|-- kalite_io.py
|-- on_isleme.py
|-- opencv_duzeltmeler.py
|-- temel.py
|-- toplu_islem.py
|-- veri.py
`-- README.md
```

Çalışma sırasında oluşabilecek dizinler:

```text
goruntu_isleme/
|-- __pycache__/
`-- cikti/
    |-- trainval/
    |-- test/
    |-- kalite_kontrol_adaylari/      # opsiyonel
    `-- anatomik_kontrol_adaylari/    # opsiyonel
```

`__pycache__/` Python tarafından üretilir. `cikti/` ön işleme çıktısıdır; kaynak kodun parçası değildir.

## Modül Mimarisi

Ana dış API `GorselIsleyici` sınıfıdır. `goruntu_isleyici.py`, işlevleri ayrı dosyalara bölünmüş mixin sınıflarını birleştirir ve eski tek dosya kullanımına dönük uyumluluk katmanı sağlar.

| Dosya | Görev |
| --- | --- |
| `ana_islem.py` | `mri-preprocess` komutunun ve interaktif menünün giriş noktasıdır. Geçerli CLI aksiyonları `menu` ve `preprocess` değerleridir. |
| `ayarlar.py` | Proje kökü, veri yolları, sınıf adları, hedef boyut, normalizasyon, kalite kontrol, augmentation ve split ayarlarını merkezi olarak tanımlar. |
| `goruntu_isleyici.py` | `GorselIsleyici` sınıfını dış API olarak sunar; module-level monkeypatch uyumluluğunu mixin modüllerine yayar. |
| `temel.py` | İşleyici durum yönetimi, rastgele tohumlama, çıktı dosya adı üretimi, kaynak grup belirleme ve ortak yardımcıları içerir. |
| `veri.py` | Girdi klasörü çözümleme, görüntü listeleme, kaynak grup belirleme ve leak-free `trainval/test` bölme işlemlerini içerir. |
| `kalite_io.py` | OpenCV ile görüntü yükleme/kaydetme, gri ton okuma, temel kalite kontrol ve kenar artefakt metriklerini içerir. |
| `on_isleme.py` | Tekil görüntü pipeline'ını içerir: kenar temizliği, eğim analizi/düzeltme, gürültü giderme, bias correction, skull stripping, registration, normalizasyon, CLAHE ve resize. |
| `opencv_duzeltmeler.py` | Kenar artefakt temizliği, eğim analizi, maske ve PCA işlemleri için stateless OpenCV yardımcılarını toplar. |
| `artirma.py` | Disk üzerinde offline augmentation için ayna, rotasyon, parlaklık/kontrast, elastik deformasyon, crop, gürültü ve yoğunluk kayması işlemlerini içerir. |
| `toplu_islem.py` | Tekil görüntü kaydı, sınıf bazlı augmentation çarpanları, paralel/toplu işleme, splitli/düz çıktı üretimi ve aday manifestlerini yönetir. |
| `__init__.py` | Paket dışına `GorselIsleyici` sınıfını açar. |

## Beklenen Girdi Yapısı

Varsayılan girdi klasörü:

```text
Veri_Seti/
`-- OriginalDataset/
    |-- NonDemented/
    |-- VeryMildDemented/
    |-- MildDemented/
    `-- ModerateDemented/
```

Desteklenen görüntü uzantıları:

- `.jpg`
- `.jpeg`
- `.png`

`GorselIsleyici` şu girdi biçimlerini okuyabilir:

- Doğrudan sınıf klasörlerini içeren bir klasör.
- Kökünde `OriginalDataset/` bulunan bir klasör, örneğin `Veri_Seti/`.
- Daha önce ayrılmış `trainval/test/<SinifAdi>/` yapısı.

Girdi zaten `trainval/test` yapısındaysa yeniden bölme yapılmaz; mevcut split korunur. Bu modül doğrudan `.nii` veya `.nii.gz` hacim dosyalarını okumaz.

## Sınıflar ve Etiketler

| Sınıf | Etiket |
| --- | --- |
| `NonDemented` | `0` |
| `VeryMildDemented` | `1` |
| `MildDemented` | `2` |
| `ModerateDemented` | `3` |

Sınıf klasörü adları kodda sabit kullanılır ve büyük/küçük harf duyarlıdır.

## Komut Satırından Çalıştırma

Proje komutlarını kullanmak için paket proje kökünden kurulabilir:

```bash
pip install -e .
```

Geliştirme bağımlılıklarıyla kurulum:

```bash
pip install -e ".[dev]"
```

İnteraktif menü:

```bash
mri-preprocess --action menu
```

Ön işleme akışını doğrudan çalıştırma:

```bash
mri-preprocess --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

Paket kurulmadan modül olarak çalıştırma:

```bash
python3 -m goruntu_isleme.ana_islem --action preprocess --input-dir Veri_Seti/OriginalDataset --output-dir goruntu_isleme/cikti
```

`--action` verilmezse varsayılan olarak interaktif menü açılır.

## CLI Parametreleri

| Parametre | Açıklama |
| --- | --- |
| `--action` | Çalıştırılacak işlem. Geçerli değerler: `menu`, `preprocess`. Varsayılan `menu`. |
| `--input-dir` | Girdi klasörü. Verilmezse `Veri_Seti/OriginalDataset` kullanılır. |
| `--output-dir` | Çıktı klasörü. Verilmezse `goruntu_isleme/cikti` kullanılır. |

Bu modülde `all`, `extract`, `scale`, `report`, `clean-nan`, `split` veya `--mode 3d` aksiyonları bulunmaz. CSV özellik matrisi, scaler, model dosyası ve feature cache üretimi `model/` kapsamındadır.

## Python API Kullanımı

```python
from pathlib import Path

from goruntu_isleme import GorselIsleyici

isleyici = GorselIsleyici()
istatistikler = isleyici.tum_gorselleri_isle_ve_bol(
    cikti_klasoru=Path("goruntu_isleme/cikti"),
    giris_klasoru=Path("Veri_Seti/OriginalDataset"),
)
```

Sık kullanılan API metotları:

| Metot | Amaç |
| --- | --- |
| `gorselleri_listele(giris_klasoru)` | Desteklenen görüntüleri sınıf, etiket, kaynak ID ve kaynak grup bilgisiyle listeler. |
| `veri_dosyalarini_bol(dosyalar)` | Görüntüleri kaynak grup bazında `trainval` ve `test` olarak böler. |
| `goruntu_isle(dosya_yolu)` | Tek görüntüyü işler; normal çıktıya alınacaksa `numpy.ndarray`, reddedilecekse `None` döndürür. |
| `goruntu_isle_sonuc(dosya_yolu)` | İşlenmiş görüntüyle birlikte `quality_rejected`, `quality_reason`, `tilt_angle`, `tilt_reliable`, `tilt_analysis` ve `quality_analysis` alanlarını döndürür. |
| `tum_gorselleri_isle(cikti_klasoru, ...)` | Verilen görüntü listesini işler ve çıktı klasöründe doğrudan sınıf klasörlerine kaydeder. |
| `tum_gorselleri_isle_ve_bol(cikti_klasoru, giris_klasoru)` | Varsayılan CLI akışıdır; split üretir veya mevcut split'i korur. |

## Ön İşleme Sırası

`goruntu_isle_sonuc` ile tek görüntü için uygulanan sıra:

1. Görüntüyü OpenCV ile gri ton olarak aç.
2. Boş, siyah, geçersiz boyutlu veya geçersiz pikselli görüntüleri erken ele.
3. Ayara bağlı kenar artefakt tespiti yap ve sayaçları güncelle.
4. Ayara bağlı kenar artefakt temizliğini normalizasyon ve CLAHE'den önce uygula.
5. Strict kalite kontrol uygula: ortalama yoğunluk, standart sapma ve siyah piksel oranı.
6. Temizleme sonrası görüntünün boşa düşmediğini tekrar kontrol et.
7. Eğim düzeltme veya eğim kalite kontrolü açıksa eğim analizi yap; kalite reddi veya otomatik düzeltme uygula.
8. Gürültü giderme uygula.
9. Açık ise bias field correction uygula.
10. Açık ise skull stripping uygula.
11. Açık ise registration/hizalama uygula.
12. Seçili normalizasyon stratejisini uygula.
13. Görüntüyü hedef boyuta getir.
14. Pipeline sonu kalite kontrolüyle siyahlaşmış, foreground'u zayıf veya kontrastı düşmüş çıktıları ele.
15. Açık ise anatomik kalite kontrolü uygula.
16. Açık ise çıktı sonrası eğim kalite kontrolünü uygula.

Kayıt işlemi `toplu_islem.py` içindeki toplu akışta yapılır. `goruntu_isle_sonuc` doğrudan dosya yazmaz. `goruntu_isle`, geriye dönük API olarak kalite reddi durumunda `None` döndürür.

## Normalizasyon Stratejileri

| Strateji | Davranış |
| --- | --- |
| `minimal` | Foreground maskesi üzerinden percentile clipping uygular; ardından genel pipeline resize yapar. |
| `standard` | Percentile clipping ve sabit CLAHE uygular; ardından genel pipeline resize yapar. Varsayılan stratejidir. |
| `aggressive` | Percentile clipping, sabit CLAHE ve foreground tabanlı z-score normalizasyonu uygular; ardından genel pipeline resize yapar. |

Varsayılan `standard` stratejisinde foreground/beyin aday maskesi içinde `KIRPMA_YUZDELERI=(0.5, 99.5)` percentile clipping, `0-255` yoğunluk normalizasyonu, `CLAHE_CLIP_LIMIT=2.0` ve `192x192` yeniden boyutlandırma kullanılır. Arka plan pikselleri normalizasyon istatistiklerini belirlemez ve normalizasyon çıkışında arka plan olarak korunur.

## Yeniden Boyutlandırma

`BOYUTLANDIRMA_MODU` ayarı görüntülerin hedef boyuta nasıl getirileceğini belirler:

- `pad`: Varsayılan moddur. En-boy oranı korunur, görüntü hedef çerçeveye sığdırılır ve kalan kenarlar otomatik arka plan tahminiyle doldurulur; tahmin güvenilir değilse `PADDING_DEGERI` kullanılır.
- `stretch`: Görüntü doğrudan hedef boyuta gerilir. En-boy oranı korunmaz.

`PADDING_OTOMATIK_ARKAPLAN=True` varsayılanıyla, CLAHE/normalizasyon sonrası siyah arka planın küçük nonzero değerlere taşındığı durumlarda padding ile görüntü arka planı arasında keskin yapay sınır oluşması azaltılır. Her iki modda da çıktı `(HEDEF_YUKSEKLIK, HEDEF_GENISLIK)` biçimindedir.

## Varsayılan Ayarlar

Temel ayarlar `ayarlar.py` içinde tutulur:

| Ayar | Varsayılan |
| --- | --- |
| Girdi klasörü | `Veri_Seti/OriginalDataset` |
| Çıktı klasörü | `goruntu_isleme/cikti` |
| Hedef boyut | `192x192` |
| Desteklenen uzantılar | `.jpg`, `.jpeg`, `.png` |
| Boyutlandırma modu | `pad` |
| Padding değeri | `0` |
| Otomatik padding arka planı | Aktif |
| Test oranı | `0.15` |
| Rastgele tohum | `42` |
| Normalizasyon stratejisi | `standard` |
| Histogram eşitleme | Aktif |
| CLAHE clip limit | `2.0` |
| Filtre metodu | `bilateral` |
| Gaussian blur sigma | `0.5` |
| Skull stripping | Kapalı |
| Bias field correction | Kapalı |
| Registration | Aktif, `simple` center-of-mass |
| Morfolojik işlemler | Aktif |
| Morfolojik kernel boyutu | `3` |
| Disk üzerinde augmentation | Kapalı |
| Augmentation çarpanı | `0` |
| Sınıf bazlı augmentation | Kapalı |
| Kalite kontrol | Aktif |
| Minimum ortalama yoğunluk | `5` |
| Maksimum ortalama yoğunluk | `200` |
| Minimum standart sapma | `15` |
| Maksimum siyah piksel oranı | `0.80` |
| Siyah piksel eşiği | `10` |
| Kenar artefakt kontrol | Aktif |
| Kenar artefakt temizleme | Aktif |
| Kenar şerit oranı | `0.10` |
| Kenar parlaklık eşiği | `225` |
| Kenar çok parlak eşiği | `245` |
| Kenar parlak piksel oran eşiği | `0.01` |
| Kenar bileşen oran eşiği | `0.003` |
| Kenar anatomi koruma oranı | `0.5` |
| Eğim düzeltme | Kapalı |
| Eğim kalite kontrol | Kapalı |
| Eğim kalite red eşiği | `12.5` |
| Eğim kalite adaylarını kaydetme | Aktif, kontrol açılırsa etkili |
| Parlak doku eğim kalite fallback'i | Kapalı |
| Anatomik kalite kontrol | Kapalı |
| Anatomik adaylarını kaydetme | Kapalı |
| Anatomik merkez boşluk red eşiği | `0.35` |

Kalite kontrol varsayılan olarak çok karanlık, çok aydınlık, düşük kontrastlı veya siyah piksel oranı çok yüksek görüntüleri eler. Kenar artefakt temizliği strict kalite kontrolünden önce çalışır; parlak kenar bantları yüzünden reddedilecek ama temizlenebilir görüntüler kurtarılabilir.

## Kenar Artefakt Kontrolü ve Temizliği

Bazı 2D MRI dilimlerinde özellikle üst ve alt kenarlarda parlak/saturasyona yakın artefaktlar görülebilir. Bu artefaktlar global ortalama, standart sapma ve siyah piksel oranı kontrollerinden geçebilir; CLAHE ise bu bantları güçlendirerek model girdisini bozabilir.

`goruntu_isle_sonuc`, normalize ve CLAHE adımlarından önce opsiyonel kenar artefakt tespit ve temizleme adımı çalıştırır. Tespit ile temizleme bağımsızdır: `KENAR_ARTEFAKT_KONTROL_AKTIF` sayaç toplar, `KENAR_ARTEFAKT_TEMIZLEME_AKTIF` ise parlak kenar bileşenlerini temizler.

Temizleme, parlak bağlantılı bileşenlerin kenar şeritlerindeki payına bakar. Çoğunlukla kenarda kalan bileşenler tamamen silinir; merkezi anatomik yapıya bağlı görünen bileşenlerde yalnızca şerit içindeki pikseller temizlenir. Görüntünün yarısından büyük tek parlak bileşen anatomik kabul edilip korunur.

Toplu işlem özetinde `kenar_artefakt_tespit`, `kenar_artefakt_temizlendi` ve varsa `kaydetme_hatasi` sayaçları raporlanır.

## Eğim Analizi, Düzeltme ve Kalite Kontrolü

Eğim analizi, Otsu maskesi ve en büyük geçerli foreground bileşeni üzerinden PCA açısı hesaplar; aynı bileşenin `minAreaRect` açısıyla uyumunu kontrol ederek güvenilirlik kararı verir. Yakın dairesel maskeler, zayıf foreground, düşük major eksen uzanımı veya uyumsuz açı ölçümleri güvenilmez sayılır.

Varsayılan politika otomatik döndürme yapmaz:

- `EGIM_DUZELTME_AKTIF=False`
- `EGIM_KALITE_KONTROL_AKTIF=False`
- `EGIM_PARLAK_DOKU_KALITE_KONTROL_AKTIF=False`

Bu kontroller açılırsa:

- `EGIM_MIN_ACI=1.0` altındaki küçük açılar düzeltilmez.
- `EGIM_MAKS_ACI=5.0` üstündeki güvenilir açılar görsel kontrol adayı sayılabilir.
- `EGIM_KALITE_RED_ESIGI=12.5` ve üstündeki kalite açıları normal çıktıya yazılmayabilir.
- `EGIM_KALITE_ADAYLARI_KAYDET=True` ise reddedilen görüntüler denetim için `kalite_kontrol_adaylari/` altına yazılır ve `egim_kalite_kontrol_manifest.csv` oluşturulur.

Eğim kalite adayları yalnızca ilgili kontrol açıldığında ve görüntü reddedildiğinde üretilir.

## Anatomik Kalite Kontrolü

Anatomik kalite kontrol varsayılan olarak kapalıdır:

- `ANATOMIK_KALITE_KONTROL_AKTIF=False`
- `ANATOMIK_ADAYLARI_KAYDET=False`

Açılırsa, işlenmiş çıktıdaki merkezi karanlık boşluk/ventrikül benzeri alan oranı `ANATOMIK_MERKEZ_BOSLUK_RED_ESIGI` eşiğini aşan görüntüler normal `trainval` veya `test` çıktılarına yazılmaz. `ANATOMIK_ADAYLARI_KAYDET=True` ise aday kopyalar `anatomik_kontrol_adaylari/` altında tutulur ve `anatomik_kontrol_manifest.csv` dosyasına yazılır.

## Beklenen Çıktı Yapısı

Varsayılan `preprocess` çıktısı:

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

Eğim veya anatomik kalite aday kaydı açılır ve reddedilen görüntü oluşursa ek dizinler üretilebilir:

```text
goruntu_isleme/cikti/
|-- kalite_kontrol_adaylari/
|   |-- egim_kalite_kontrol_manifest.csv
|   |-- trainval/
|   `-- test/
`-- anatomik_kontrol_adaylari/
    |-- anatomik_kontrol_manifest.csv
    |-- trainval/
    `-- test/
```

`tum_gorselleri_isle` doğrudan çağrılırsa split klasörleri oluşturulmaz; çıktı `cikti/<SinifAdi>/` yapısında üretilir. CLI ve önerilen Python akışı `tum_gorselleri_isle_ve_bol` kullandığı için `trainval/test` yapısını üretir.

Kaydedilen dosyalar `.png` formatındadır. Kaynak dosya köküne orijinal uzantı eklenerek ad çakışması engellenir:

```text
ornek.jpg -> ornek_jpg.png
ornek.png -> ornek_png.png
```

Augmentation aktif edilirse ek dosyalar şu biçimde yazılır:

```text
ornek_jpg_aug1.png
ornek_jpg_aug2.png
```

Varsayılan ayarlarda offline augmentation kapalıdır. Test split'i işlenirken augmentation çarpanı sıfırlanır.

## Split ve Veri Sızıntısı Politikası

- Ham veri doğrudan sınıf klasörlerinden geliyorsa `tum_gorselleri_isle_ve_bol`, veriyi kaynak grup bazında `trainval` ve `test` olarak ayırır.
- Kaynak grup, sınıf adı ve dosya kökünden türetilir. `_augN` ve `(1)` gibi türev ekleri temizlenerek aynı kaynaktan gelen görüntüler aynı grupta tutulur.
- Her sınıfta harici test split'i için en az iki farklı kaynak grup bulunmalıdır; aksi durumda işlem hata verir.
- Girdi zaten `trainval/test` klasörlerini içeriyorsa yeniden bölme yapılmaz, mevcut split korunur.
- `trainval` ve `test` klasörlerinin kalite kontrol sonrasında beklenen sınıfları koruduğu doğrulanır.

## Offline Augmentation

`artirma.py` disk üzerinde augmentation üretmek için vardır, ancak varsayılan proje politikası offline augmentation kullanmaz:

- `VERI_ARTIRMA_AKTIF=False`
- `ARTIRMA_CARPANI=0`
- Sınıf bazlı çarpanlar `0`
- Ayna, rotasyon, elastik deformasyon, random crop, Gaussian noise ve intensity shift alt anahtarları kapalı

Bu politika, veri dengesizliğinin model eğitiminde class weights ve DL tarafındaki online augmentation ile ele alınmasını hedefler. Offline augmentation açılırsa yalnızca trainval tarafında ek `.png` dosyaları yazılır; test tarafında çarpan sıfırdır.

## Bağımlılık Notları

- Görüntü yükleme, kaydetme, CLAHE, resize, temel filtreler, morfoloji, Otsu maskeleme, bağlantılı bileşenler ve augmentation adımları OpenCV ile çalışır.
- `SimpleITK`, sadece `BIAS_FIELD_METHOD="n4itk"` veya `REGISTRATION_METHOD` olarak `affine`/`rigid` seçildiğinde anlamlıdır.
- `scikit-image` bu modülde fallback olarak kullanılmaz; bağımlılıklarda model/EDA tarafındaki ihtiyaçlar için kalır.
- Toplu işlem `multiprocessing.Pool` ile paralel çalışabilir. Affine/rigid registration aktifse template tutarlılığı için sequential moda döner.

## Model Modülüne Devam

Ön işleme bittikten sonra model eğitimi için örnek komutlar:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

XGBoost eğitiminde özellik matrisi `model/sl/dataset.py` tarafından klasör ağacından üretilir ve istenirse `model/ciktilar/sl_ozellikler` altında `.npz` cache olarak tutulur.
