"""
ayarlar.py
----------
MRI görüntü işleme için merkezi konfigürasyon dosyası.
"""

from pathlib import Path

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

# Normalizasyon stratejisi
# "minimal": Sadece percentile clipping + resize
# "standard": percentile + CLAHE + resize (önerilen)
# "aggressive": percentile + CLAHE + z-score + resize
NORMALIZASYON_STRATEJISI = "standard"  # "minimal", "standard", "aggressive"

# Z-score normalizasyonu: Ortalama=0, Std=1 yapma (isteğe bağlı)
Z_SCORE_NORMALIZASYON_AKTIF = True

# Histogram eşitleme (CLAHE) - Kontrast iyileştirme
HISTOGRAM_ESITLEME_AKTIF = True
CLAHE_CLIP_LIMIT = 5.0  # Kırpma sınırı (yüksek = daha fazla kontrast)

# Filtreler - Gelişmiş görüntü filtreleme seçenekleri
GELISMIS_FILTRE_AKTIF = False
BILATERAL_FILTRE_AKTIF = True  # Bilateral filtreleme (kenar koruma)

GAUSSIAN_BLUR_AKTIF = False  # Gaussian bulanıklaştırma (gürültü azaltma)
GAUSSIAN_BLUR_SIGMA = 0.5    # Bulanıklaştırma şiddeti

# Arka plan işleme
MASKE_KENAR_PAYI = 5

# Skull stripping (kafatası çıkarma)
SKULL_STRIPPING_AKTIF = True
SKULL_STRIPPING_METHOD = "simple"  # "simple" veya "advanced" (morfolojik işlemlerle)

# Bias field correction (MRI yoğunluk düzensizliği düzeltme)
BIAS_FIELD_CORRECTION_AKTIF = False
BIAS_FIELD_METHOD = "n4itk"  # "n4itk" (profesyonel) veya "simple" (hızlı)

# Registration/Hizalama
REGISTRATION_AKTIF = True
REGISTRATION_METHOD = "simple"  # "simple" (center-of-mass), "affine" (gelişmiş), "rigid"

# Morfolojik işlemler
MORFOLOJIK_OPERASYONLAR_AKTIF = True
MORFOLOJIK_KERNEL_BOYUTU = 3

# ==================== VERİ ARTIRMA AYARLARI ====================
# Veri artırma (Data Augmentation) - Yapay veri üretimi
# Mevcut görüntülerden döndürme, aynalama vb. ile yeni varyasyonlar oluşturur
VERI_ARTIRMA_AKTIF = True
ARTIRMA_CARPANI = 2  # Her orijinal görüntüden kaç artırılmış versiyon üretilecek

# Sınıf bazlı dengesiz augmentation - Az örnekli sınıfları daha fazla artır
SINIF_BAZLI_ARTIRMA_AKTIF = True
SINIF_BAZLI_CARPANLAR = {
    "NonDemented": 0,        # Sadece orijinal örnekleri kullan
    "VeryMildDemented": 0,   # Mevcut sayi yeterli, ek augmentation uygulama
    "MildDemented": 1,       # Orta seviyede artır
    "ModerateDemented": 11,  # Ciddi azinlik sinifi - kontrollu sekilde artır
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
RANDOM_CROP_RATIO = 0.97         # Beyin dokusunu korumak için daha hafif kırpma

GAUSSIAN_NOISE_AKTIF = False
GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_SIGMA = 5         # Gürültü şiddeti

INTENSITY_SHIFT_AKTIF = True
INTENSITY_SHIFT_LIMIT = 0.03      # Yogunluk kaymasi limiti (%3)

# ==================== VERİ BÖLÜMLEME AYARLARI ====================
# Veri seti üç parçaya bölünür:
# - Eğitim (Training): Modeli eğitmek için
# - Doğrulama (Validation): Hiperparametre ayarlama ve erken durdurma için
# - Test: Son performans değerlendirmesi için (model hiç görmemiş)
EGITIM_ORANI = 0.70        # %70 eğitim
DOGRULAMA_ORANI = 0.15     # %15 doğrulama
TEST_ORANI = 0.15          # %15 test
RASTGELE_TOHUM = 42        # Tekrarlanabilirlik için sabit tohum

# ==================== CSV AYARLARI ====================
# Özelliklerin kaydedileceği CSV dosya isimleri
CSV_DOSYA_ADI = "goruntu_ozellikleri.csv"          # Ham özellikler
CSV_SCALED_DOSYA_ADI = "goruntu_ozellikleri_scaled.csv"  # Ölçeklendirilmiş özellikler
EGITIM_CSV_DOSYA_ADI = "egitim.csv"
DOGRULAMA_CSV_DOSYA_ADI = "dogrulama.csv"
TEST_CSV_DOSYA_ADI = "test.csv"
EGITIM_SCALED_CSV_DOSYA_ADI = "egitim_scaled.csv"
DOGRULAMA_SCALED_CSV_DOSYA_ADI = "dogrulama_scaled.csv"
TEST_SCALED_CSV_DOSYA_ADI = "test_scaled.csv"
SCALER_DOSYA_ADI = "feature_scaler.pkl"

# Ölçeklendirme (Scaling) metodu
# "minmax": Tüm değerleri 0-1 aralığına sıkıştırır
# "robust": Aykırı değerlere karşı daha dayanıklı, medyan ve IQR kullanır
# "standard": Z-score normalizasyonu (mean=0, std=1)
# "maxabs": [-1, 1] aralığına ölçeklendirir
SCALING_METODU = "robust"

# ==================== KALİTE KONTROL AYARLARI ====================
# Görüntü kalite kontrol eşikleri
KALITE_KONTROL_AKTIF = True
MIN_MEAN_INTENSITY = 5       # Minimum ortalama yoğunluk (çok karanlık kontrol)
MAX_MEAN_INTENSITY = 245      # Maksimum ortalama yoğunluk (çok aydınlık kontrol)
MIN_STD_INTENSITY = 5         # Minimum standart sapma (düz görüntü kontrol)
MAX_BLACK_RATIO = 0.8         # Maksimum siyah piksel oranı (boş görüntü kontrol)
