# MRI Görüntü Ön İşleme Projesi (JPEG/PNG)

Bu proje, siyah veya gri arka plana sahip 2B MRI görüntülerini (JPEG/PNG)
**derin öğrenme / makine öğrenmesi model eğitimi için en iyi şekilde hazırlamak**
amacıyla tasarlanmıştır.

Ana hedef:
- Arka planı (siyah veya gri) mümkün olduğunca temizleyip
- Sadece ilgi bölgesini (vücut dokusu) içeren, normalize edilmiş,
  tekdüze boyutlu görüntüler üretmek
- İsteğe bağlı olarak veri artırma (augmentation) ile modele daha zengin bir eğitim seti sağlamak

## Klasör Yapısı

Önerilen yapı:

```text
mri_on_isleme_projesi/
  goruntu_isleme_mri/
    __init__.py
    ayarlar.py
    io_araclari.py
    arka_plan_isleme.py
    on_isleme_adimlari.py
    artirma.py
  scripts/
    toplu_on_isleme.py
    tek_goruntuyu_incele.py
  veri/
    girdi/
      ... burada ham (orijinal) MRI JPEG/PNG dosyaların ...
    cikti/
      ... otomatik oluşturulacak, ön işlenmiş görüntüler ...
  requirements.txt
  README.md
```

Sen sadece:
- Ham görüntülerini `veri/girdi/` klasörüne koy
- Gerekirse `goruntu_isleme_mri/ayarlar.py` dosyasındaki ayarları düzenle
- Aşağıdaki komutları çalıştır

## Kurulum

1. (Opsiyonel ama önerilir) Sanal ortam oluştur:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Gerekli paketleri kur:

   ```bash
   pip install -r requirements.txt
   ```

## Toplu Ön İşleme

Tüm görüntüleri ön işlemek için:

```bash
python scripts/toplu_on_isleme.py
```

Bu script:

- `veri/girdi/` içindeki tüm `.jpg`, `.jpeg`, `.png` dosyalarını bulur
- Her biri için:
  - Arka plan tipini tahmin eder (siyah/gri/diger)
  - Otsu eşiği ile ikili maske çıkarır
  - Maskeye göre sınır kutusunu bulup kenar payı ile genişletir
  - Bu sınır kutusuna göre görüntüyü kırpar
  - Yoğunluk normalizasyonu yapar (örn. %1-%99 arasında kırpıp 0-255 aralığına çeker)
  - İsteğe bağlı CLAHE benzeri adaptif histogram eşitleme uygular
  - Sabit boyuta (örn. 256x256) yeniden boyutlandırır
  - Sonucu `veri/cikti/` altına kaydeder (klasör yapısı korunur)
- `veri/cikti/on_isleme_log.csv` dosyasında tüm adımların özetini tutar
  (orijinal boyutlar, kırpma kutusu, arka plan tipi vb.)

Ayrıca `ayarlar.py` içinde `VERI_ARTIRMA_AKTIF = True` ise,
her görüntü için belirttiğin sayıda (`EKSTRA_KOPYA_SAYISI`) artırılmış kopya üretir:
- Yatay/dikey ayna
- 90/180/270 derece döndürme
- Küçük parlaklık/kontrast oynamaları

## Tek Görüntüyü İnceleme

Bir görüntü üzerinde ön işleme adımlarının etkisini hızlıca görmek için:

```bash
python scripts/tek_goruntuyu_incele.py veri/girdi/ornek.jpg
```

Bu komut:
- Orijinal ve ön işlenmiş görüntüyü yan yana gösterir
- Konsola kırpma kutusu ve arka plan tipi gibi meta bilgileri yazar

## Ayarları Özelleştirme

`goruntu_isleme_mri/ayarlar.py` dosyasında:

- `HEDEF_GENISLIK`, `HEDEF_YUKSEKLIK` → modeline uygun hedef boyut
- `KIRPMA_YUZDELERI` → yoğunluk normalizasyonunda uç değer kırpma
- `HISTOGRAM_ESITLEME_AKTIF` → CLAHE kullanmak isteyip istemediğin
- `MASKE_KENAR_PAYI` → kırpma kutusunun etrafına eklenilecek güvenlik payı
- `VERI_ARTIRMA_AKTIF`, `EKSTRA_KOPYA_SAYISI` → veri artırma ayarları

gibi parametreleri kolayca değiştirebilirsin.

## Notlar

- Arka plan tespiti, görüntülerin çoğunda siyah arka plan olduğu varsayımıyla tasarlanmıştır.
  Yine de kenar piksellere bakarak gri arka planı da ayırt etmeye çalışır.
- Otsu eşiği ile maske çıkarımında, ilgi bölgesinin arka plandan **daha parlak** olduğu
  senaryo varsayılmıştır. Eğer senin verinde tam tersi durum baskınsa,
  `arka_plan_isleme.py` içindeki `ikili_maske_olustur` fonksiyonunda
  `goruntu > esik` yerine `goruntu < esik` kullanabilirsin.
- Proje tamamen modüler yazılmıştır; ihtiyaç duyduğun kısma kolayca müdahale edebilirsin.
