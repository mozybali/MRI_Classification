# Görüntü İşleme Modülü

`goruntu_isleme`, ham 2D beyin MRI görüntülerini modelleme aşamasına hazırlayan ön işleme paketidir. Varsayılan akışta görüntüleri `Veri_Seti/OriginalDataset` altındaki sınıf klasörlerinden okur, kalite kontrol ve standartlaştırma uygular, leak-free `trainval/test` ayrımı yapar ve çıktıları `goruntu_isleme/cikti/` altında `.png` olarak kaydeder.

Bu klasör yalnızca görüntü ön işleme sorumluluğunu taşır. Özellik matrisi üretimi, XGBoost özellik cache'i, model eğitimi, hiperparametre araması ve inference işlemleri `model/` modülünde yürütülür.

## Projedeki Yeri

Tipik proje akışı:

1. Ham MRI görüntüleri `Veri_Seti/OriginalDataset/<SinifAdi>/` altında tutulur.
2. İsteğe bağlı EDA adımları `eda_analiz/` ile çalıştırılır.
3. Bu modül `mri-preprocess` komutu ile ham görüntüleri işler ve leak-free `trainval/test` klasörlerini üretir.
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
    |-- kalite_kontrol_adaylari/
    |-- trainval/
    `-- test/
```

`__pycache__/` Python tarafından üretilir. `cikti/` ön işleme çıktısıdır; kaynak kodun parçası değildir.

## Modül Mimarisi

Ana dış API `GorselIsleyici` sınıfıdır. Bu sınıf `goruntu_isleyici.py` içinde, işlevleri ayrı dosyalara bölünmüş mixin sınıflarını birleştirir.

| Dosya | Görev |
| --- | --- |
| `ana_islem.py` | `mri-preprocess` komutunun ve interaktif menünün giriş noktasıdır. Geçerli CLI aksiyonları `menu` ve `preprocess` değerleridir. |
| `ayarlar.py` | Proje kökü, veri yolları, sınıf adları, hedef boyut, normalizasyon, kalite kontrol, augmentation ve split ayarlarını merkezi olarak tanımlar. |
| `goruntu_isleyici.py` | `GorselIsleyici` sınıfını dış API olarak sunar ve eski tek dosya kullanımına dönük uyumluluk katmanı sağlar. |
| `temel.py` | `GorselIsleyici` durum yönetimi, rastgele tohumlama, çıktı dosya adı üretimi ve ortak yardımcıları içerir. |
| `veri.py` | Girdi klasörü çözümleme, görüntü listeleme, kaynak grup belirleme ve leak-free `trainval/test` bölme işlemlerini içerir. |
| `kalite_io.py` | OpenCV ile görüntü yükleme/kaydetme, gri tona çevirme ve kalite kontrol işlemlerini içerir. |
| `on_isleme.py` | Gürültü giderme, bias correction, skull stripping, registration, percentile/z-score normalizasyonu, CLAHE ve resize adımlarını içerir. |
| `artirma.py` | Disk üzerinde augmentation için rotasyon, parlaklık/kontrast, elastik deformasyon, crop, gürültü ve yoğunluk kayması işlemlerini içerir. |
| `toplu_islem.py` | Tekil görüntü kaydı, sınıf bazlı augmentation çarpanları, paralel/toplu işleme ve splitli ya da düz çıktı üretimini içerir. |
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

Desteklenen dosya uzantıları:

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

Sınıf klasörü adları kodda sabit kullanılır; büyük/küçük harf duyarlıdır.

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

Bu modülde `all`, `extract`, `scale`, `report`, `clean-nan`, `split` veya `--mode 3d` aksiyonları bulunmaz. CSV, scaler, özellik cache'i ve model dosyası üretimi bu klasörün görev kapsamı dışındadır.

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
| `gorselleri_listele(giris_klasoru)` | Desteklenen görüntüleri sınıf, etiket ve kaynak grup bilgisiyle listeler. |
| `veri_dosyalarini_bol(dosyalar)` | Görüntüleri kaynak grup bazında `trainval` ve `test` olarak böler. |
| `goruntu_isle(dosya_yolu)` | Tek görüntüyü yükler, kalite kontrolden geçirir ve normal çıktıya alınacaksa ön işleme sonrası `numpy.ndarray` döndürür. |
| `goruntu_isle_sonuc(dosya_yolu)` | İşlenmiş görüntüyle birlikte `quality_rejected`, `quality_reason` ve `tilt_angle` alanlarını döndürür. |
| `tum_gorselleri_isle(cikti_klasoru, ...)` | Verilen görüntü listesini işler ve çıktı klasöründe doğrudan sınıf klasörlerine kaydeder. |
| `tum_gorselleri_isle_ve_bol(cikti_klasoru, giris_klasoru)` | Varsayılan CLI akışıdır; split üretir veya mevcut split'i korur. |

## Ön İşleme Sırası

`goruntu_isle` ile tek görüntü için uygulanan temel sıra:

1. Görüntüyü OpenCV ile aç ve gri tona çevir.
2. Temel boş/bozuk görüntü ön kontrolü uygula.
3. Ayara bağlı kenar artefakt tespiti (raporlama).
4. Ayara bağlı kenar artefakt temizliği (normalize/CLAHE'den önce).
5. Strict kalite kontrol uygula.
6. Ayara bağlı eğim analizi yap; aşırı güvenilir eğimleri kalite kontrol adayı olarak işaretle, izinli aralıktaki güvenilir eğimleri düzeltebilir.
7. Ayara bağlı gürültü giderme uygula.
8. Ayara bağlı bias field correction uygula.
9. Ayara bağlı skull stripping uygula.
10. Ayara bağlı registration/hizalama uygula.
11. Seçili normalizasyon stratejisini uygula.
12. Görüntüyü hedef boyuta getir.

Kayıt işlemi `toplu_islem.py` içindeki toplu akışta yapılır. `goruntu_isle` doğrudan dosya yazmaz; işlenmiş görüntüyü dizi olarak döndürür.

Normalizasyon stratejileri:

| Strateji | Davranış |
| --- | --- |
| `minimal` | Percentile clipping uygular; ardından genel pipeline resize yapar. |
| `standard` | Percentile clipping ve sabit CLAHE uygular; ardından genel pipeline resize yapar. Varsayılan stratejidir. |
| `aggressive` | Percentile clipping, sabit CLAHE ve z-score normalizasyonu uygular; ardından genel pipeline resize yapar. |

Varsayılan `standard` stratejisinde foreground/beyin aday maskesi içinde `KIRPMA_YUZDELERI=(0.5, 99.5)` percentile clipping, `0-255` yoğunluk normalizasyonu, `CLAHE_CLIP_LIMIT=2.0` ve `256x256` yeniden boyutlandırma kullanılır. Arka plan pikselleri normalizasyon istatistiklerini belirlemez ve normalizasyon çıkışında arka plan olarak korunur.

Bias field correction açılırsa `simple` yöntem de aynı foreground maskesiyle düşük frekanslı bias alanını tahmin eder; siyah padding/arka plan bölgeleri bias tahminine katılmaz ve düzeltme sonrası sabit tutulur. `n4itk` yolu SimpleITK maskesini açıkça kullanır.

## Yeniden Boyutlandırma

`BOYUTLANDIRMA_MODU` ayarı görüntülerin hedef boyuta nasıl getirileceğini belirler:

- `pad`: Varsayılan ve medikal olarak önerilen moddur. En-boy oranı korunur, görüntü hedef çerçeveye sığdırılır ve kalan kenarlar güvenilir kose arka plan tahminiyle doldurulur; tahmin güvenilir değilse `PADDING_DEGERI` kullanılır.
- `stretch`: Görüntü doğrudan hedef boyuta gerilir. En-boy oranı korunmaz.

`PADDING_OTOMATIK_ARKAPLAN=True` varsayılanıyla, CLAHE/normalizasyon sonrası siyah arka planın küçük nonzero değerlere taşındığı durumlarda padding ile görüntü arka planı arasında keskin yapay sınır oluşması azaltılır. `PADDING_DEGERI` varsayılan olarak `0` değerindedir ve otomatik tahmin güvenilir değilse fallback olarak kullanılır. Her iki modda da çıktı `(HEDEF_YUKSEKLIK, HEDEF_GENISLIK)` biçimindedir.

## Varsayılan Ayarlar

Temel ayarlar `ayarlar.py` içinde tutulur:

| Ayar | Varsayılan |
| --- | --- |
| Girdi klasörü | `Veri_Seti/OriginalDataset` |
| Çıktı klasörü | `goruntu_isleme/cikti` |
| Hedef boyut | `256x256` |
| Desteklenen uzantılar | `.jpg`, `.jpeg`, `.png` |
| Boyutlandırma modu | `pad` |
| Padding değeri | `0` |
| Otomatik padding arka planı | Aktif |
| Test oranı | `0.15` |
| Rastgele tohum | `42` |
| Normalizasyon stratejisi | `standard` |
| Histogram eşitleme | Aktif |
| CLAHE clip limit | `2.0` |
| Filtre metodu | `off` |
| Gaussian blur sigma | `0.5` |
| Skull stripping | Kapalı |
| Bias field correction | Kapalı |
| Registration | Kapalı |
| Morfolojik işlemler | Aktif |
| Morfolojik kernel boyutu | `3` |
| Disk üzerinde augmentation | Kapalı |
| Augmentation çarpanı | `0` |
| Sınıf bazlı augmentation | Kapalı |
| Kalite kontrol | Aktif |
| Minimum ortalama yoğunluk | `10` |
| Maksimum ortalama yoğunluk | `180` |
| Minimum standart sapma | `15` |
| Maksimum siyah piksel oranı | `0.75` |
| Kenar artefakt kontrol | Aktif |
| Kenar artefakt temizleme | Aktif |
| Kenar şerit oranı | `0.12` |
| Kenar parlaklık eşiği | `220` |
| Kenar anatomi koruma oranı | `0.5` |
| Eğim düzeltme | Kapalı |
| Eğim minimum açı | `1.0` |
| Eğim maksimum açı | `6.0` |
| Eğim kalite kontrol | Aktif |
| Eğim kalite red eşiği | `6.0` |
| Eğim kalite aday klasörü | `kalite_kontrol_adaylari` |

Kalite kontrol varsayılan olarak çok karanlık, çok aydınlık, düşük kontrastlı veya siyah piksel oranı çok yüksek görüntüleri eler. Kenar artefakt temizliği strict kalite kontrolünden önce çalışır; böylece parlak kenar bantları yüzünden reddedilecek ama temizlenebilir görüntüler kurtarılabilir. Açıkça boş/bozuk görüntüler yine erken elenir.

## Kenar Artefakt Kontrolü ve Temizliği

Bazı 2D MRI dilimlerinde özellikle üst ve alt (daha az sıklıkla sol ve sağ) kenarlarda parlak/saturasyona yakın artefaktlar görülür. Bu artefaktlar global ortalama, standart sapma ve siyah piksel oranı kontrollerinden kolayca geçtiği için klasik kalite kontrol yakalayamaz; ancak CLAHE bu yerel parlak bantları çoğaltarak sınıflandırma performansını bozabilir.

`goruntu_isle` bu nedenle normalize ve CLAHE adımlarından **önce** opsiyonel bir kenar artefakt tespit ve temizleme adımı çalıştırır. Adım konservatiftir: skull stripping yerine geçmez, beyni kırpmaz, merkezi anatomik yapılara dokunmaz.

Tespit mantığı: görüntünün üst, alt, sol ve sağ kenar şeritleri (`KENAR_SERIT_ORANI` ile oranlanır) için ortalama, p95, p99, parlak piksel oranı ve en büyük bağlantılı parlak bileşenin alan oranı hesaplanır. Bir kenar şeridi şu üç koşuldan herhangi biri sağlandığında suspicious sayılır:

- parlak piksel oranı >= `KENAR_PARLAK_PIXEL_ORANI_ESIGI`,
- en büyük parlak bileşen oranı >= `KENAR_BILESEN_ORANI_ESIGI`,
- p99 >= `KENAR_COK_PARLAKLIK_ESIGI`.

Bir veya daha fazla kenar suspicious işaretlenirse görüntü için `artefakt_var` True olur ve sayaç artırılır. Görüntü yine de pipeline'da kalır; varsayılan davranış kenar artefaktı yüzünden görüntüyü reddetmek değil, temizlemektir.

Temizleme aşamasında `KENAR_PARLAKLIK_ESIGI` ile parlak maske üretilir ve OpenCV bağlantılı bileşen analizi uygulanır. Piksellerinin çoğu kenar şeritlerinde kalan bileşenler `KENAR_TEMIZLEME_DEGERI` (varsayılan `0`) ile tamamen silinir. Merkez anatomisine bağlanan ama kenar şeridinde anlamlı parlak payı olan bileşenlerde yalnızca şerit içindeki pikseller temizlenir; merkezdeki parlak yapı korunur. Görüntünün yarısından büyük tek bir parlak bileşen anatomik kabul edilir ve dokunulmaz.

Adım deterministiktir, paralel işleme ile uyumludur ve çıktı `uint8` dtype'ını korur.

İlgili ayarlar:

| Ayar | Varsayılan | Açıklama |
| --- | --- | --- |
| `KENAR_ARTEFAKT_KONTROL_AKTIF` | `True` | Kenar şerit tespiti ve sayaç toplama. |
| `KENAR_ARTEFAKT_TEMIZLEME_AKTIF` | `True` | Kenar artefaktlarını sil. |
| `KENAR_SERIT_ORANI` | `0.12` | Şerit kalınlığı (yükseklik/genişlik oranı). |
| `KENAR_PARLAKLIK_ESIGI` | `220` | Parlak piksel eşiği (uint8). |
| `KENAR_COK_PARLAKLIK_ESIGI` | `245` | Çok parlak (saturasyon) eşiği. |
| `KENAR_PARLAK_PIXEL_ORANI_ESIGI` | `0.01` | Şerit içinde parlak piksel oranı eşiği. |
| `KENAR_BILESEN_ORANI_ESIGI` | `0.003` | Şerit içinde en büyük parlak bileşen oranı eşiği. |
| `KENAR_BILESEN_SERIT_PAY_ESIGI` | `0.6` | Bileşenin çoğunlukla kenar şeridinde sayılması için gereken piksel payı. |
| `KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI` | `0.01` | Kısmi şerit temizliği için gereken minimum şerit piksel oranı. |
| `KENAR_KISMI_TEMIZLEME_MIN_PIXEL` | `50` | Kısmi şerit temizliği için gereken minimum mutlak şerit piksel sayısı. |
| `KENAR_TEMIZLEME_DEGERI` | `0` | Silinen artefakt piksellerine yazılan değer. |
| `KENAR_ANATOMI_KORUMA_ORANI` | `0.5` | Bu orandan büyük parlak bileşenler anatomik kabul edilip korunur. |
| `KENAR_ARTEFAKT_RAPORLA` | `True` | Toplu işleme özetinde sayaçları yazdır. |

Toplu işlem özetinde `kenar_artefakt_tespit` (kenar artefakt görülen görüntü sayısı), `kenar_artefakt_temizlendi` (gerçekten piksel silinen görüntü sayısı) ve varsa `kaydetme_hatasi` sayaçları raporlanır; mevcut `kalite_istatistikleri` anahtarları geriye dönük uyumlu kalır.

Adım skull stripping'in yerine geçmez. Skull stripping varsayılan olarak kapalı kalır; OASIS Kaggle 2D dilimlerinde görüntü zaten kabaca beyin-kırpılmış olduğu için yalnızca açık kenar artefaktlarını sustururuz.

## Eğim Düzeltme

Eğim düzeltme, kenar artefakt temizliğinden ayrı bir adımdır. Kenar artefakt temizliği yalnızca parlak sınır bantlarını bastırır; sola/sağa hafif dönmüş MRI dilimlerini hizalama amacı taşımaz. Eğim düzeltme de registration değildir: görüntüyü bir atlasa kaydetmez, yalnızca aynı dilimin güvenilir bulunan küçük açısını düzeltmeye çalışır. Data augmentation da değildir; eğitim çeşitliliği üretmek için kullanılmamalıdır.

Otomatik rotasyon medikal görüntülerde riskli olabileceği için eğim düzeltme konservatif eşiklerle çalışır. Bu ayar veri setinden örnekler görsel olarak incelendikten ve hafif, sistematik eğim gerçekten sorun olarak doğrulandıktan sonra kullanılmalıdır.

Yöntem PCA açısını tek başına kullanmaz. Görüntü güvenli `uint8` aralığına alınır, Otsu eşikleme ile kaba beyin maskesi üretilir, maske morfolojik open/close işlemleriyle temizlenir ve en büyük geçerli bağlantılı bileşen seçilir. Bu bileşenin piksel koordinatlarından PCA ana eksen açısı hesaplanır; aynı bileşenin ana konturu için `cv2.minAreaRect` açısı da çıkarılır. Dönüş açısı olarak hassas PCA tahmini kullanılır, fakat yalnızca PCA ve contour/minAreaRect açıları uyumluysa sonuç güvenilir kabul edilir.

Güvenilirlik konservatiftir: küçük gürültü bileşenleri yok sayılır, ana foreground çok küçükse tahmin reddedilir, minor/major eksen oranı yüksek olan yakın dairesel veya simetrik maskeler güvenilmez sayılır, PCA ile minAreaRect açısı `EGIM_RMSE_MAKS` dereceden fazla ayrışırsa otomatik düzeltme yapılmaz. Bu nedenle zaten hizalı veya anatomik ekseni belirsiz görüntülerde over-rotation riski azaltılır.

Başlangıç için önerilen eşikler (özellik manuel olarak açıldığında):

| Ayar | Önerilen başlangıç |
| --- | --- |
| `EGIM_MIN_ACI` | `1.0` |
| `EGIM_MAKS_ACI` | `6.0` |

`EGIM_MIN_ACI` altındaki açılar otomatik düzeltilmez. Güvenilir olmayan tahminler de değiştirilmeden bırakılır. `EGIM_MAKS_ACI` üstündeki güvenilir açılar otomatik döndürülmez; `egim_gorsel_kontrol_adayi` sayacına eklenir ve manuel/görsel inceleme adayı olarak ele alınmalıdır. Küçük ve güvenilir eğimler konservatif biçimde düzeltilebilir; aşırı güvenilir eğimler ayrıca kalite kontrol kuralıyla normal çıktılardan ayrılır.

İlgili ayarlar:

| Ayar | Varsayılan | Açıklama |
| --- | --- | --- |
| `EGIM_DUZELTME_AKTIF` | `False` | Eğim düzeltmeyi aç/kapat (varsayılan kapalı; opt-in). |
| `EGIM_DUZELTME_RAPORLA` | `False` | Özellik aktifken toplu işlem özetinde eğim sayaçlarını yazdır. |
| `EGIM_ONIZLEME_URET` | `False` | Varsayılan akışta dosya yazmaz; preview üretilirse bu bayrakla yönetilir. |
| `EGIM_MIN_ACI` | `2.0` | Bu açı altındaki tahminleri otomatik düzeltme. |
| `EGIM_MAKS_ACI` | `20.0` | Bu açı üstündeki güvenilir tahminleri manuel kontrol adayı say. |
| `EGIM_RMSE_MAKS` | `2.5` | PCA ve minAreaRect açıları arasındaki maksimum fark (derece). Açı-duyarlı tolerans `max(EGIM_RMSE_MAKS, abs(pca_aci))` olarak uygulanır; küçük açılarda minAreaRect kuantizasyonu yüzünden statik eşik aşılırsa bile sonuç güvenilir sayılabilir. |
| `EGIM_MIN_SATIR_SAYISI` | `45` | Ana maskenin gereken minimum dikey bbox yüksekliği. |
| `EGIM_MIN_X_SPAN` | `8.0` | PCA major eksen uzanımı için minimum piksel eşiği. |
| `EGIM_MIN_FOREGROUND_ORANI` | `0.04` | Ana foreground bileşeninin minimum görüntü alanı oranı. |
| `EGIM_MIN_EKSEN_ORANI` | `0.95` | Minor/major eksen oranı bu değerin üstündeyse maske belirsiz sayılır. OASIS/Kaggle 2D dilimleri nispeten izotropik olduğu için yüksek eğimli adayları yakalamak amacıyla konservatif ama kullanılabilir başlangıç eşiğidir. |
| `EGIM_DOLDURMA_DEGERI` | `0` | Rotasyonda oluşan boş alanların doldurma değeri. |
| `EGIM_ROTASYON_PADDING_ORANI` | `0.12` | Rotasyondan önce kırpmayı azaltmak için eklenen geçici padding oranı. |

Toplu işlemde `egim_tespit`, `egim_duzeltildi`, `egim_gorsel_kontrol_adayi` ve `egim_kalite_red` sayaçları, kenar artefakt sayaçları gibi görüntü bazlı delta olarak toplanır ve multiprocessing ile uyumludur. `EGIM_DUZELTME_AKTIF = False` iken görüntüler döndürülmez, preview dosyası yazılmaz; `EGIM_KALITE_KONTROL_AKTIF = False` iken aşırı eğimli görüntüler eski davranıştaki gibi normal çıktıya yazılabilir.

## Eğim Kalite Kontrolü

`EGIM_KALITE_KONTROL_AKTIF = True` iken kalite açısı `abs(angle) >= EGIM_KALITE_RED_ESIGI` olan görüntüler normal `trainval` veya `test` çıktılarına yazılmaz. Varsayılan eşik `3.0` derecedir ve aynı sabit eşik hem `trainval` hem de `test` için kullanılır; test performansına göre eşik seçilmez veya ayarlanmaz.

Ana dış kontur belirsizse parlak iç doku ölçümü (`EGIM_PARLAK_DOKU_KALITE_KONTROL_AKTIF`) kalite kararı için fallback olarak kullanılır. `EGIM_KALITE_GUVENILIRLIK_ZORUNLU = False` olduğunda güvenilirlik filtresine takılan ama eşik üstü kalan şüpheli dilimler de normal çıktılardan ayrılır.

Bu görüntüler ham veri klasöründen silinmez ve kaynak dosyalar değiştirilmez. Bunun yerine `EGIM_KALITE_ADAYLARI_KAYDET = True` ise ön işleme çıktı kökü altında ayrı bir kontrol adayı olarak kaydedilir:

```text
goruntu_isleme/cikti/
|-- kalite_kontrol_adaylari/
|   |-- egim_kalite_kontrol_manifest.csv
|   |-- trainval/
|   |   `-- <SinifAdi>/
|   `-- test/
|       `-- <SinifAdi>/
|-- trainval/
`-- test/
```

Aday dosya adları normal çıktıdaki deterministik şemayı kullanır; örneğin `ornek.jpg` için `ornek_jpg.png`, `ornek.png` için `ornek_png.png` yazılır. Böylece aynı köke sahip farklı uzantılı kaynaklar birbirini ezmez.

Manifest CSV her reddedilen görüntü için en az şu alanları içerir: `original_path`, `split`, `class_name`, `detected_angle`, `reason` ve `candidate_path`. `reason` değeri aşırı eğim için `excessive_tilt` olur. Ham görüntülerin fiziksel olarak silinmemesinin nedeni denetlenebilirlik ve geri dönüş imkanıdır: kalite kararı manifestte izlenir, aday kopya görsel kontrol için ayrılır, fakat orijinal veri seti kaynak doğrulama veya farklı eşiklerle yeniden işleme için aynen kalır.

Aday klasör yerleşimi `split_adi`'na göre belirlenir: `tum_gorselleri_isle_ve_bol` (önerilen akış) `trainval` ve `test` için `split_adi`'yı otomatik geçirir; aday klasörü `cikti/kalite_kontrol_adaylari/<split>/<SinifAdi>/` olarak `cikti/<split>/` ile kardeş üretilir. `tum_gorselleri_isle` doğrudan no-split bir çıktı klasörüyle çağrıldığında (örneğin `cikti_klasoru="cikti"` ve `split_adi=None`) aday klasörü deterministik biçimde aynı çıktının içine `cikti/kalite_kontrol_adaylari/<SinifAdi>/` olarak yuvalanır; manifest CSV bu nested kökte yer alır. Konfigürasyon hatasına yol açan örtük bir fallback değildir, belgelenmiş ikinci bir yerleşimdir.

## Anatomik Kalite Kontrolü

`ANATOMIK_KALITE_KONTROL_AKTIF = True` iken merkezi karanlık boşluk/ventrikül oranı `ANATOMIK_MERKEZ_BOSLUK_RED_ESIGI` eşiğini aşan görüntüler de normal `trainval` veya `test` çıktılarına yazılmaz. Bu kontrol eğimden bağımsızdır; reason değeri `anatomik_merkez_bosluk` olur.

Reddedilen görüntüler, denetim için ayrı aday klasörüne ve manifest dosyasına yazılır:

```text
goruntu_isleme/cikti/
|-- anatomik_kontrol_adaylari/
|   |-- anatomik_kontrol_manifest.csv
|   |-- trainval/
|   |   `-- <SinifAdi>/
|   `-- test/
|       `-- <SinifAdi>/
|-- trainval/
`-- test/
```

Manifest alanları: `original_path`, `split`, `class_name`, `reason`, `score`, `central_dark_ratio`, `central_hole_ratio` ve `candidate_path`. Bu adaylar model eğitim/test klasörlerinde tutulmaz; yalnızca görsel denetim ve eşik ayarı için saklanır.

## Beklenen Çıktı Yapısı

`preprocess` tamamlandığında çıktı şu yapıda olur:

```text
goruntu_isleme/cikti/
|-- anatomik_kontrol_adaylari/
|   |-- anatomik_kontrol_manifest.csv
|   |-- trainval/
|   `-- test/
|-- kalite_kontrol_adaylari/
|   |-- egim_kalite_kontrol_manifest.csv
|   |-- trainval/
|   `-- test/
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

`tum_gorselleri_isle` doğrudan çağrılırsa split klasörleri oluşturulmaz; çıktı `cikti/<SinifAdi>/` yapısında üretilir. CLI ve önerilen Python akışı `tum_gorselleri_isle_ve_bol` kullandığı için `trainval/test` yapısını üretir.

Kaydedilen dosyalar `.png` formatındadır. Kaynak dosya köküne orijinal uzantı eklenerek ad çakışması engellenir:

```text
ornek.jpg  -> ornek_jpg.png
ornek.png  -> ornek_png.png
```

Augmentation aktif edilirse ek dosyalar şu biçimde yazılır:

```text
ornek_jpg_aug1.png
ornek_jpg_aug2.png
```

Varsayılan ayarlarda augmentation kapalı olduğu için bu ek dosyalar üretilmez. Test split'i işlenirken augmentation çarpanı sıfırlanır.

## Split ve Veri Sızıntısı Politikası

- Ham veri doğrudan sınıf klasörlerinden geliyorsa `tum_gorselleri_isle_ve_bol`, veriyi kaynak grup bazında `trainval` ve `test` olarak ayırır.
- Kaynak grup, sınıf adı ve dosya kökünden türetilir. `_augN` ve `(1)` gibi türev ekleri temizlenerek aynı kaynaktan gelen görüntüler aynı grupta tutulur.
- Her sınıfta harici test split'i için en az iki farklı kaynak grup bulunmalıdır; aksi durumda işlem hata verir.
- Girdi zaten `trainval/test` klasörlerini içeriyorsa yeniden bölme yapılmaz, mevcut split korunur.
- `trainval` ve `test` klasörlerinin kalite kontrol sonrasında beklenen sınıfları koruduğu doğrulanır.

## Bağımlılık Notları

- Görüntü yükleme, kaydetme, CLAHE, resize, temel filtreler, morfoloji, Otsu maskeleme ve bazı augmentation adımları OpenCV ile çalışır.
- `Pillow`, `SciPy` ve `scikit-image` ön işleme için yedek yol olarak kullanılmaz; DL/SL model katmanlarındaki görüntü okuma ve klasik özellik çıkarımı ihtiyaçları için bağımlılıklarda kalır.
- `scikit-learn`, kaynak grup bazlı stratified `trainval/test` bölmesi için kullanılır.
- `SimpleITK`, sadece bias correction veya gelişmiş registration ayarları aktif edildiğinde anlamlıdır.
- Toplu işlem `multiprocessing.Pool` ile paralel çalışabilir; affine/rigid registration aktifse template tutarlılığı için sequential moda döner.

## Model Modülüne Devam

Ön işleme bittikten sonra model eğitimi için örnek komutlar:

```bash
mri-train --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
```

```bash
mri-train --model xgboost --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test --feature-cache model/ciktilar/sl_ozellikler
```

XGBoost eğitiminde özellik matrisi `model/sl/dataset.py` tarafından klasör ağacından üretilir ve istenirse `model/ciktilar/sl_ozellikler` altında `.npz` cache olarak tutulur.
