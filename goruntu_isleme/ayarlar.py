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
    "KIRPMA_YUZDELERI", "NORMALIZASYON_STRATEJISI",
    "HISTOGRAM_ESITLEME_AKTIF", "CLAHE_CLIP_LIMIT",
    "FILTRE_METODU", "GAUSSIAN_BLUR_SIGMA",
    "MASKE_KENAR_PAYI",
    "SKULL_STRIPPING_AKTIF", "SKULL_STRIPPING_METHOD",
    "BIAS_FIELD_CORRECTION_AKTIF", "BIAS_FIELD_METHOD",
    "REGISTRATION_AKTIF", "REGISTRATION_METHOD",
    "MORFOLOJIK_OPERASYONLAR_AKTIF", "MORFOLOJIK_KERNEL_BOYUTU",
    # Veri artırma
    "VERI_ARTIRMA_AKTIF", "ARTIRMA_CARPANI",
    "SINIF_BAZLI_ARTIRMA_AKTIF", "SINIF_BAZLI_CARPANLAR",
    "YATAY_AYNA_AKTIF", "YATAY_AYNA_OLASILIK",
    "ROTASYON_AKTIF", "ROTASYON_MAKS_ACI",
    "PARLAKLIK_ARALIK", "KONTRAST_ARALIK",
    "ELASTIC_DEFORMATION_AKTIF", "ELASTIC_ALPHA", "ELASTIC_SIGMA",
    "RANDOM_CROP_AKTIF", "RANDOM_CROP_RATIO",
    "GAUSSIAN_NOISE_AKTIF", "GAUSSIAN_NOISE_MEAN", "GAUSSIAN_NOISE_SIGMA",
    "INTENSITY_SHIFT_AKTIF", "INTENSITY_SHIFT_LIMIT",
    # Veri bölümleme
    "TEST_ORANI", "RASTGELE_TOHUM",
    # Kalite kontrol
    "KALITE_KONTROL_AKTIF", "MIN_MEAN_INTENSITY", "MAX_MEAN_INTENSITY",
    "MIN_STD_INTENSITY", "MAX_BLACK_RATIO",
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
HEDEF_GENISLIK = 256   # Piksel cinsinden genişlik
HEDEF_YUKSEKLIK = 256  # Piksel cinsinden yükseklik

# İzin verilen görüntü dosya uzantıları
GORUNTU_UZANTILARI = [".jpg", ".jpeg", ".png"]

# Normalizasyon ayarları
# Kırpma yüzdeleri: Aşırı karanlık ve aydınlık pikselleri temizler
# (%1 en düşük ve %1 en yüksek değerler kırpılır)
KIRPMA_YUZDELERI = (1, 99)

# Normalizasyon stratejisi - z-score yalnizca "aggressive" stratejide uygulanir.
# "minimal": Sadece percentile clipping + resize
# "standard": percentile + CLAHE + resize (önerilen)
# "aggressive": percentile + CLAHE + z-score + resize
NORMALIZASYON_STRATEJISI = "standard"  # "minimal", "standard", "aggressive"

# Histogram eşitleme (CLAHE) - Kontrast iyileştirme.
# Train ve test arasinda deterministik olmasi icin sabit clip_limit kullanilir
# (adaptive mod, std-bazli esiklerden dolayi splitler arasi tutarsizlik yaratir).
HISTOGRAM_ESITLEME_AKTIF = True
CLAHE_CLIP_LIMIT = 2.0  # Sabit clip limit (orta seviye kontrast iyilestirme)

# Gurultu giderme metodu (gurultu_gider auto modu icin tek kaynak).
# "off"       : filtre uygulanmaz (default)
# "bilateral" : kenar koruyan bilateral filtre (cv2 gerekli)
# "gaussian"  : Gaussian blur (sigma=GAUSSIAN_BLUR_SIGMA)
FILTRE_METODU = "off"
GAUSSIAN_BLUR_SIGMA = 0.5  # "gaussian" modunda kullanilir

# Arka plan işleme
MASKE_KENAR_PAYI = 2

# Skull stripping (kafatası çıkarma)
# Kaggle Alzheimer 2D slice veri setinde goruntuler zaten beyin-kirpilmis durumdadir;
# Otsu tabanli basit skull strip kortikal dokuyu kismen silebilir. Default kapali.
# Ham hacim (NIfTI) ile calisirken HD-BET / SynthStrip onerilir.
SKULL_STRIPPING_AKTIF = False
SKULL_STRIPPING_METHOD = "simple"  # "simple" veya "advanced" (morfolojik işlemlerle)

# Bias field correction (MRI yoğunluk düzensizliği düzeltme)
BIAS_FIELD_CORRECTION_AKTIF = False
BIAS_FIELD_METHOD = "n4itk"  # "n4itk" (profesyonel) veya "simple" (hızlı)

# Registration/Hizalama
# Kaggle 2D slice veri seti onceden kabaca hizalanmistir; basit center-of-mass
# kaydirma fazla deger katmaz ve kenar kayiplari yaratabilir. Default kapali.
# Ham veri icin REGISTRATION_AKTIF=True ve REGISTRATION_METHOD="affine" + sabit
# atlas onerilir.
REGISTRATION_AKTIF = False
REGISTRATION_METHOD = "simple"  # "simple" (center-of-mass), "affine" (gelişmiş), "rigid"

# Morfolojik işlemler
MORFOLOJIK_OPERASYONLAR_AKTIF = True
MORFOLOJIK_KERNEL_BOYUTU = 3

# ==================== VERİ ARTIRMA AYARLARI ====================
# Veri artırma (Data Augmentation) - Yapay veri üretimi
# Disk üzerinde çoğaltma yerine eğitimde online augmentasyon ve
# class-weighted loss (compute_class_weights) kullanılır.
VERI_ARTIRMA_AKTIF = False
ARTIRMA_CARPANI = 0  # Her orijinal görüntüden kaç artırılmış versiyon üretilecek

# Sınıf bazlı dengesiz augmentation - artık devre dışı; dengesizlik
# eğitimde class weights ile telafi edilir.
SINIF_BAZLI_ARTIRMA_AKTIF = False
SINIF_BAZLI_CARPANLAR = {
    "NonDemented": 0,
    "VeryMildDemented": 0,
    "MildDemented": 0,
    "ModerateDemented": 0,
}

# Artırma parametreleri (basit)
YATAY_AYNA_AKTIF = False
YATAY_AYNA_OLASILIK = 0.15
ROTASYON_AKTIF = True
ROTASYON_MAKS_ACI = 7.0       # Anatomik yapıyı korumak için daha küçük açı
PARLAKLIK_ARALIK = (-20, 20)     # Parlaklık değişimi aralığı (piksel)
KONTRAST_ARALIK = (0.9, 1.1)     # Kontrast çarpanı aralığı

# Gelişmiş medikal-spesifik artırma parametreleri
ELASTIC_DEFORMATION_AKTIF = True
ELASTIC_ALPHA = 15               # Daha yumuşak deformasyon
ELASTIC_SIGMA = 8                # Deformasyon yumuşaklığı

RANDOM_CROP_AKTIF = True
RANDOM_CROP_RATIO = 0.90         # Anlamli regulasyon icin daha belirgin kirpma

GAUSSIAN_NOISE_AKTIF = False
GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_SIGMA = 5         # Gürültü şiddeti

INTENSITY_SHIFT_AKTIF = True
INTENSITY_SHIFT_LIMIT = 0.03      # Yogunluk kaymasi limiti (%3)

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
MAX_MEAN_INTENSITY = 245      # Maksimum ortalama yoğunluk (çok aydınlık kontrol)
MIN_STD_INTENSITY = 5         # Minimum standart sapma (düz görüntü kontrol)
MAX_BLACK_RATIO = 0.8         # Maksimum siyah piksel oranı (boş görüntü kontrol)
