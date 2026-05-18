"""
ayarlar.py
----------
MRI görüntü işleme için merkezi konfigürasyon dosyası.
"""

from pathlib import Path

__all__ = [
    # Genel
    "PROJE_KOK", "VERI_SETI_KLASORU", "ORIGINAL_VERI_SETI_KLASORU",
    "ON_ISLEME_VARSAYILAN_GIRIS_KLASORU", "CIKTI_KLASORU",
    "ISLENMIS_TRAINVAL_DIZINI", "ISLENMIS_TEST_DIZINI",
    # Sınıf
    "SINIF_KLASORLERI", "SINIF_ETIKETI",
    # Görüntü işleme
    "HEDEF_GENISLIK", "HEDEF_YUKSEKLIK", "GORUNTU_UZANTILARI",
    "BOYUTLANDIRMA_MODU", "PADDING_DEGERI",
    "KIRPMA_YUZDELERI", "NORMALIZASYON_STRATEJISI",
    "HISTOGRAM_ESITLEME_AKTIF", "CLAHE_CLIP_LIMIT",
    "FILTRE_METODU", "GAUSSIAN_BLUR_SIGMA",
    "MASKE_KENAR_PAYI",
    "REGISTRATION_AKTIF",
    "MORFOLOJIK_OPERASYONLAR_AKTIF", "MORFOLOJIK_KERNEL_BOYUTU",
    # Veri bölümleme
    "TEST_ORANI", "RASTGELE_TOHUM",
    # Kalite kontrol
    "KALITE_KONTROL_AKTIF", "MIN_MEAN_INTENSITY", "MAX_MEAN_INTENSITY",
    "MIN_STD_INTENSITY", "MAX_BLACK_RATIO", "SIYAH_PIKSEL_ESIGI",
    # Kenar artefakt kontrol
    "KENAR_ARTEFAKT_KONTROL_AKTIF", "KENAR_ARTEFAKT_TEMIZLEME_AKTIF",
    "KENAR_SERIT_ORANI", "KENAR_PARLAKLIK_ESIGI", "KENAR_COK_PARLAKLIK_ESIGI",
    "KENAR_SERIT_PERCENTILE_P95", "KENAR_SERIT_PERCENTILE_P99",
    "KENAR_PARLAK_PIXEL_ORANI_ESIGI", "KENAR_BILESEN_ORANI_ESIGI",
    "KENAR_BILESEN_SERIT_PAY_ESIGI",
    "KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI", "KENAR_KISMI_TEMIZLEME_MIN_PIXEL",
    "KENAR_TEMIZLEME_DEGERI", "KENAR_ARTEFAKT_RAPORLA",
    "KENAR_ANATOMI_KORUMA_ORANI",
    # Padding davranisi
    "PADDING_OTOMATIK_ARKAPLAN",
    # Goruntu minimum boyut
    "GORUNTU_MIN_BOYUT",
]

# ==================== GENEL AYARLAR ====================
# Proje kök dizini - tüm dosya yolları buraya göre belirlenir
PROJE_KOK = Path(__file__).parent.parent

# Ham (orijinal) MRI görüntülerinin bulunduğu klasör
VERI_SETI_KLASORU = PROJE_KOK / "Veri_Seti"

# Dataset alt klasörleri
ORIGINAL_VERI_SETI_KLASORU = VERI_SETI_KLASORU / "OriginalDataset"

# Ön işleme için varsayılan giriş:
# - Proje akisi yalnizca OriginalDataset uzerinden calisir
ON_ISLEME_VARSAYILAN_GIRIS_KLASORU = ORIGINAL_VERI_SETI_KLASORU

# İşlenmiş görüntülerin ve CSV dosyalarının kaydedileceği klasör
CIKTI_KLASORU = PROJE_KOK / "goruntu_isleme" / "cikti"
ISLENMIS_TRAINVAL_DIZINI = CIKTI_KLASORU / "trainval"
ISLENMIS_TEST_DIZINI = CIKTI_KLASORU / "test"

# ==================== SINIF AYARLARI ====================
# MRI veri setindeki demans seviye sınıfları (klasör isimleri)
SINIF_KLASORLERI = [
    "NonDemented",       # 0 - Sağlıklı (Demans yok)
    "VeryMildDemented",  # 1 - Çok hafif demans
    "MildDemented",      # 2 - Hafif demans
    "ModerateDemented",  # 3 - Orta seviye demans
]

# Sınıf adlarından sayısal etiketlere dönüşüm haritasi
# Makine öğrenmesi modelleri sayısal etiketlerle çalışır
SINIF_ETIKETI = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}

# ==================== GÖRÜNTÜ İŞLEME AYARLARI ====================
# Hedef boyut - Tüm görüntüler bu boyuta getirilir (standartlaştırma)
HEDEF_GENISLIK = 192   # Piksel cinsinden genişlik
HEDEF_YUKSEKLIK = 192  # Piksel cinsinden yükseklik

# İzin verilen görüntü dosya uzantıları
GORUNTU_UZANTILARI = [".jpg", ".jpeg", ".png"]

# Yeniden boyutlandirma modu - hedef boyuta getirirken en-boy oraninin
# korunup korunmayacagini belirler.
# "pad"     : En-boy orani korunur; goruntu hedef cerceveye sigdirilir ve
#             bos kenarlar PADDING_DEGERI ile doldurulur (medikal olarak
#             daha guvenli, anatomik distorsiyon yaratmaz).
# "stretch" : Goruntu dogrudan hedef boyuta gerilir (eski davranis).
BOYUTLANDIRMA_MODU = "pad"  # "pad" veya "stretch"

# "pad" modunda eklenen kenar piksellerinin doldurulacagi yogunluk degeri.
PADDING_DEGERI = 0

# Padding canvas'ini doldururken otomatik arka plan tahmini kullanilsin mi?
PADDING_OTOMATIK_ARKAPLAN = True

# Normalizasyon ayarları
# Kırpma yüzdeleri: Aşırı karanlık ve aydınlık pikselleri temizler.
KIRPMA_YUZDELERI = (0.5, 99.5)

# Normalizasyon stratejisi - z-score yalnizca "aggressive" stratejide uygulanir.
# "minimal": Sadece percentile clipping + resize
# "standard": percentile + CLAHE + resize (önerilen)
# "aggressive": percentile + CLAHE + z-score + resize
NORMALIZASYON_STRATEJISI = "standard"  # "minimal", "standard", "aggressive"

# Histogram eşitleme (CLAHE) - Kontrast iyileştirme.
HISTOGRAM_ESITLEME_AKTIF = True
CLAHE_CLIP_LIMIT = 2.0  # Sabit clip limit (orta seviye kontrast iyilestirme)

# Gurultu giderme metodu (gurultu_gider auto modu icin tek kaynak).
# "off"       : filtre uygulanmaz
# "bilateral" : kenar koruyan bilateral filtre (cv2 gerekli)
# "gaussian"  : Gaussian blur (sigma=GAUSSIAN_BLUR_SIGMA)
FILTRE_METODU = "bilateral"
GAUSSIAN_BLUR_SIGMA = 0.5  # "gaussian" modunda kullanilir

# Arka plan işleme
MASKE_KENAR_PAYI = 2

# Registration/Hizalama (center-of-mass tabanli basit oteleme)
# Girdi dilimleri merkezden kayabiliyorsa basit center-of-mass hizalama
# uygulanir. Yalnızca kaydırma yapar; ölçekleme/kırpma yapmadan beyin
# dokusunu hedef çerçevenin merkezine taşır.
REGISTRATION_AKTIF = True

# Morfolojik işlemler (foreground maskesi temizliginde kullanilir)
MORFOLOJIK_OPERASYONLAR_AKTIF = True
MORFOLOJIK_KERNEL_BOYUTU = 3

# ==================== VERİ BÖLÜMLEME AYARLARI ====================
# Trainval/test kaynak grubu bolmesi icin kullanilir
# (goruntu_isleyici.veri_dosyalarini_bol). Egitim/dogrulama orani egitim
# tarafinda model/dl/dataset.py icindeki splitter tarafindan belirlenir.
TEST_ORANI = 0.15          # %15 test
RASTGELE_TOHUM = 42        # Tekrarlanabilirlik için sabit tohum

# ==================== KALİTE KONTROL AYARLARI ====================
# Görüntü kalite kontrol eşikleri
KALITE_KONTROL_AKTIF = True
MIN_MEAN_INTENSITY = 5       # Minimum ortalama yoğunluk (çok karanlık kontrol)
MAX_MEAN_INTENSITY = 200     # Maksimum ortalama yoğunluk (çok aydınlık kontrol)
MIN_STD_INTENSITY = 15        # Minimum standart sapma (düz görüntü kontrol)
MAX_BLACK_RATIO = 0.80        # Maksimum siyah piksel oranı (boş görüntü kontrol)
SIYAH_PIKSEL_ESIGI = 10      # Siyah piksel sayımında kullanılan yoğunluk eşiği

# ==================== KENAR ARTEFAKT AYARLARI ====================
# Bazi MRI gorsellerinde ust/alt/sol/sag kenarlarda parlak/saturasyona
# yakin artefaktlar bulunur. Bu artefaktlar global mean/std/black-ratio
# kontrollerinden gecer ancak CLAHE ile yerel olarak guclendirildiginde
# beyin dokusunun siniflandirmasini bozabilir. Asagidaki ayarlar yalnizca
# kenar seritlerine bakarak konservatif bir tespit ve temizlik adimini
# yonetir; merkezi beyin dokusuna mudahale etmez.

# Kenar artefakt tespit/raporlama aktif mi?
KENAR_ARTEFAKT_KONTROL_AKTIF = True

# Tespit edilen kenar artefaktlarini temizle.
KENAR_ARTEFAKT_TEMIZLEME_AKTIF = True

# Kenar serit kalinligi (yukseklik/genisligin orani). 0.10 -> %10.
KENAR_SERIT_ORANI = 0.10

# "Parlak" piksel esigi (uint8, 0-255).
KENAR_PARLAKLIK_ESIGI = 225

# "Cok parlak" piksel esigi (uint8). p99 bu esigin ustundeyse kenar
# suspicious kabul edilir ve temizleme cagrildiginda hedef olur.
KENAR_COK_PARLAKLIK_ESIGI = 245

# Kenar serit istatistiklerinde hesaplanan percentile degerleri.
KENAR_SERIT_PERCENTILE_P95 = 95
KENAR_SERIT_PERCENTILE_P99 = 99

# Bir kenar seridinde parlak piksel orani >= bu deger ise kenar suspicious.
KENAR_PARLAK_PIXEL_ORANI_ESIGI = 0.01

# Kenar seridi icindeki en buyuk parlak baglantili bilesenin tum kenar
# serit alanina orani >= bu deger ise kenar suspicious sayilir.
KENAR_BILESEN_ORANI_ESIGI = 0.003

# Temizleme kriteri: bir parlak baglantili bilesenin piksellerinin
# kenar seritlerine dusen pay'i >= bu deger ise bilesen artefakt sayilir.
KENAR_BILESEN_SERIT_PAY_ESIGI = 0.7

# Kismi temizleme kriteri.
KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI = 0.015
KENAR_KISMI_TEMIZLEME_MIN_PIXEL = 50

# Temizleme sirasinda artefakt piksellerine yazilacak deger (uint8).
KENAR_TEMIZLEME_DEGERI = 0

# Bir parlak baglantili bilesenin alani toplam goruntu alaninin bu oranindan
# buyukse anatomik kabul edilir ve dokunulmaz.
KENAR_ANATOMI_KORUMA_ORANI = 0.5

# Kalite raporlamada kenar artefakt sayaclari yazdirilsin mi?
KENAR_ARTEFAKT_RAPORLA = True

# ==================== GENEL GORUNTU ISLEM SINIRLARI ====================
# Foreground maske olusumunda morfolojik adimlarin uygulandigi minimum
# goruntu boyutu (piksel). Bu boyutun altindaki goruntuler icin yalnizca
# basit Otsu maskesi kullanilir.
GORUNTU_MIN_BOYUT = 8
