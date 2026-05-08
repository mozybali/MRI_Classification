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
    "EGIM_KALITE_KONTROL_AKTIF", "EGIM_KALITE_RED_ESIGI",
    "EGIM_KALITE_GUVENILIRLIK_ZORUNLU",
    "EGIM_KALITE_GUVENILMEZ_UYUMLU_RMSE_ESIGI",
    "EGIM_KALITE_GUVENILMEZ_BUYUK_ACI_RED_ESIGI",
    "EGIM_KALITE_ADAYLARI_KAYDET",
    "EGIM_KALITE_ADAYLARI_KLASOR_ADI", "EGIM_KALITE_MANIFEST_DOSYA_ADI",
    "EGIM_PARLAK_DOKU_KALITE_KONTROL_AKTIF",
    "EGIM_PARLAK_DOKU_PERCENTILE", "EGIM_PARLAK_DOKU_FOREGROUND_ESIGI",
    "EGIM_PARLAK_DOKU_MIN_PIXEL_ORANI", "EGIM_PARLAK_DOKU_MAKS_EKSEN_ORANI",
    "ANATOMIK_KALITE_KONTROL_AKTIF", "ANATOMIK_ADAYLARI_KAYDET",
    "ANATOMIK_ADAYLARI_KLASOR_ADI", "ANATOMIK_MANIFEST_DOSYA_ADI",
    "ANATOMIK_MERKEZ_BOSLUK_RED_ESIGI", "ANATOMIK_MERKEZ_ROI_ORANI",
    "ANATOMIK_KARANLIK_ESIGI", "ANATOMIK_FOREGROUND_ESIGI",
    "ANATOMIK_MIN_FOREGROUND_ORANI",
    # Kenar artefakt kontrol
    "KENAR_ARTEFAKT_KONTROL_AKTIF", "KENAR_ARTEFAKT_TEMIZLEME_AKTIF",
    "KENAR_SERIT_ORANI", "KENAR_PARLAKLIK_ESIGI", "KENAR_COK_PARLAKLIK_ESIGI",
    "KENAR_PARLAK_PIXEL_ORANI_ESIGI", "KENAR_BILESEN_ORANI_ESIGI",
    "KENAR_BILESEN_SERIT_PAY_ESIGI",
    "KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI", "KENAR_KISMI_TEMIZLEME_MIN_PIXEL",
    "KENAR_TEMIZLEME_DEGERI", "KENAR_ARTEFAKT_RAPORLA",
    "KENAR_ANATOMI_KORUMA_ORANI",
    # Padding davranisi
    "PADDING_OTOMATIK_ARKAPLAN",
    # Egim duzeltme guvenilirlik
    "EGIM_MIN_EKSEN_ORANI",
    # Eğim düzeltme
    "EGIM_DUZELTME_AKTIF", "EGIM_DUZELTME_RAPORLA",
    "EGIM_ONIZLEME_URET", "EGIM_MIN_ACI", "EGIM_MAKS_ACI",
    "EGIM_RMSE_MAKS", "EGIM_MIN_SATIR_SAYISI", "EGIM_MIN_X_SPAN",
    "EGIM_MIN_FOREGROUND_ORANI", "EGIM_DOLDURMA_DEGERI",
    "EGIM_ROTASYON_PADDING_ORANI",
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

# Yeniden boyutlandirma modu - hedef boyuta getirirken en-boy oraninin
# korunup korunmayacagini belirler.
# "pad"     : En-boy orani korunur; goruntu hedef cerceveye sigdirilir ve
#             bos kenarlar PADDING_DEGERI ile doldurulur (medikal olarak
#             daha guvenli, anatomik distorsiyon yaratmaz).
# "stretch" : Goruntu dogrudan hedef boyuta gerilir (eski davranis).
BOYUTLANDIRMA_MODU = "pad"  # "pad" veya "stretch"

# "pad" modunda eklenen kenar piksellerinin doldurulacagi yogunluk degeri.
# MRI 2D dilimlerinde arka plan tipik olarak siyah oldugu icin 0 onerilir.
# PADDING_OTOMATIK_ARKAPLAN True ise bu deger yalnizca tahmin basarisiz
# oldugunda fallback olarak kullanilir.
PADDING_DEGERI = 0

# Padding canvas'ini doldururken otomatik arka plan tahmini kullanilsin mi?
# True ise olceklenmis goruntunun dort kose yamasinin medyani padding
# degeri olarak kullanilir. Boylece CLAHE gibi normalizasyon sonrasi
# "0" arka plan ile padding "0"'i arasinda olusan gorunur sinir bandi
# engellenir. False ise yukaridaki PADDING_DEGERI sabiti kullanilir.
PADDING_OTOMATIK_ARKAPLAN = True

# Normalizasyon ayarları
# Kırpma yüzdeleri: Aşırı karanlık ve aydınlık pikselleri temizler.
# Kenar/anatomik detay kaybını azaltmak için konservatif tutulur.
KIRPMA_YUZDELERI = (0.5, 99.5)

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
# "off"       : filtre uygulanmaz
# "bilateral" : kenar koruyan bilateral filtre (cv2 gerekli)
# "gaussian"  : Gaussian blur (sigma=GAUSSIAN_BLUR_SIGMA)
FILTRE_METODU = "bilateral"
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
# Girdi dilimleri merkezden kayabiliyorsa basit center-of-mass hizalama
# uygulanir. "simple" mod yalnızca kaydırma yapar; ölçekleme/kırpma yapmadan
# beyin dokusunu hedef çerçevenin merkezine taşır.
REGISTRATION_AKTIF = True
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
ROTASYON_AKTIF = False
ROTASYON_MAKS_ACI = 5.0       # Anatomik yapıyı korumak için daha küçük açı
PARLAKLIK_ARALIK = (-20, 20)     # Parlaklık değişimi aralığı (piksel)
KONTRAST_ARALIK = (0.9, 1.1)     # Kontrast çarpanı aralığı

# Gelişmiş medikal-spesifik artırma parametreleri
ELASTIC_DEFORMATION_AKTIF = True
ELASTIC_ALPHA = 15               # Daha yumuşak deformasyon
ELASTIC_SIGMA = 8                # Deformasyon yumuşaklığı

RANDOM_CROP_AKTIF = False
RANDOM_CROP_RATIO = 0.97         # Acilirse kenarlari koruyan hafif kirpma

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
MAX_MEAN_INTENSITY = 200     # Maksimum ortalama yoğunluk (çok aydınlık kontrol)
MIN_STD_INTENSITY = 15        # Minimum standart sapma (düz görüntü kontrol)
MAX_BLACK_RATIO = 0.80        # Maksimum siyah piksel oranı (boş görüntü kontrol)

# Guvenilir egim tahmini kalite kontrolu. Esik trainval ve test icin aynidir;
# test performansina gore degistirilmez. Reddedilen goruntuler ham veriden
# silinmez, cikti klasorunde ayri bir aday klasorune kopya olarak yazilir.
#
# Politika: yalnizca egim *kontrolu* yapilir, otomatik egim *duzeltmesi*
# uygulanmaz (bkz. EGIM_DUZELTME_AKTIF = False). Bu nedenle 15 derece ve
# uzeri guvenilir egimler kalite adayi olarak ayrilir; daha kucuk egimler
# normal ciktiya birakilir (duzeltilmez).
#
# EGIM_KALITE_GUVENILIRLIK_ZORUNLU False ise guvenilirlik filtresinden
# gecmeyen ama aci esigini asan goruntuler de kalite adayi olarak ayrilir.
# Bu agresif mod daha cok goruntu ayirabilir.
EGIM_KALITE_KONTROL_AKTIF = True
EGIM_KALITE_RED_ESIGI = 15.0
EGIM_KALITE_GUVENILIRLIK_ZORUNLU = True
# Ana maske izotropik gorunse bile PCA ve minAreaRect acilari cok iyi
# uyusuyorsa, red esigi ve ustundeki egimler normal ciktiya kacmasin.
EGIM_KALITE_GUVENILMEZ_UYUMLU_RMSE_ESIGI = 1.0
# Ana egim tahmini guvenilmez olsa bile cok buyuk acilar normal ciktiya
# kacmasin. Bu esik 15 derecelik ana red esiginden kasitli olarak yuksektir;
# maske_isotropik/kontur_ambiguous gibi belirsiz ama bariz yatmis adaylari
# ayirirken kucuk guvenilmez acilarda fazla red uretmez.
EGIM_KALITE_GUVENILMEZ_BUYUK_ACI_RED_ESIGI = 25.0
EGIM_KALITE_ADAYLARI_KAYDET = True
EGIM_KALITE_ADAYLARI_KLASOR_ADI = "kalite_kontrol_adaylari"
EGIM_KALITE_MANIFEST_DOSYA_ADI = "egim_kalite_kontrol_manifest.csv"

# Ic/parlak doku egimi kalite kontrolu. Dis beyin konturu yuvarlak veya
# ambiguous gorundugunde ana PCA olcumu kucuk aci verebilir; bu fallback
# yalnizca kalite adayi kararinda kullanilir, otomatik dondurme yapmaz.
EGIM_PARLAK_DOKU_KALITE_KONTROL_AKTIF = True
EGIM_PARLAK_DOKU_PERCENTILE = 85.0
EGIM_PARLAK_DOKU_FOREGROUND_ESIGI = 10
EGIM_PARLAK_DOKU_MIN_PIXEL_ORANI = 0.03
EGIM_PARLAK_DOKU_MAKS_EKSEN_ORANI = 0.95

# Anatomik/gorsel kalite kontrolu. Büyük merkezi karanlik bosluk/ventrikul
# gorunumu olan dilimler normal train/test ciktilarina alinmaz; denetim icin
# ayri aday klasorune kopyalanir.
ANATOMIK_KALITE_KONTROL_AKTIF = True
ANATOMIK_ADAYLARI_KAYDET = True
ANATOMIK_ADAYLARI_KLASOR_ADI = "anatomik_kontrol_adaylari"
ANATOMIK_MANIFEST_DOSYA_ADI = "anatomik_kontrol_manifest.csv"
ANATOMIK_MERKEZ_BOSLUK_RED_ESIGI = 0.35
ANATOMIK_MERKEZ_ROI_ORANI = 0.60
ANATOMIK_KARANLIK_ESIGI = 15
ANATOMIK_FOREGROUND_ESIGI = 10
ANATOMIK_MIN_FOREGROUND_ORANI = 0.05

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
# Tespit ile temizleme bagimsiz yonetilir; tespit acik ama temizleme
# kapali ise sadece istatistik toplanir. Varsayilan aktif oldugunda
# normalize/CLAHE oncesi konservatif kenar temizligi uygulanir.
KENAR_ARTEFAKT_TEMIZLEME_AKTIF = True

# Kenar serit kalinligi (yukseklik/genisligin orani). 0.10 -> %10.
# Artefakt temizligi aktifken anatomik kenar kaybini azaltmak icin dar tutulur.
KENAR_SERIT_ORANI = 0.10

# "Parlak" piksel esigi (uint8, 0-255). Bu esigin ustundeki pikseller
# kenar serit istatistiklerinde parlak sayilir.
KENAR_PARLAKLIK_ESIGI = 225

# "Cok parlak" piksel esigi (uint8). p99 bu esigin ustundeyse kenar
# suspicious kabul edilir ve temizleme cagrildiginda hedef olur.
KENAR_COK_PARLAKLIK_ESIGI = 245

# Bir kenar seridinde parlak piksel orani >= bu deger ise kenar
# suspicious sayilir (varsayilan: %1).
KENAR_PARLAK_PIXEL_ORANI_ESIGI = 0.01

# Kenar seridi icindeki en buyuk parlak baglantili bilesenin tum kenar
# serit alanina orani >= bu deger ise kenar suspicious sayilir.
KENAR_BILESEN_ORANI_ESIGI = 0.003

# Temizleme kriteri: bir parlak baglantili bilesenin piksellerinin
# kenar seritlerine dusen pay'i >= bu deger ise bilesen artefakt sayilir
# ve silinir. 0.7 -> bilesenin %70'inden fazlasi serit icindeyse temizle.
# Boylece serit kalinligini biraz asarak merkeze tasan parlak bantlar
# da yakalanir; merkezdeki anatomik parlak bolgeler (serit pay'i dusuk)
# korunur.
KENAR_BILESEN_SERIT_PAY_ESIGI = 0.7

# Kismi temizleme kriteri: bilesen merkeze bagli oldugu icin tamamen
# silinmeyecekse, serit icindeki parlak piksel sayisi hem asagidaki oran
# hem de mutlak piksel esigi ile kontrol edilir. Bu iki esikten buyugu
# kullanilir. 0.015 -> goruntu alaninin %1.5'i.
KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI = 0.015
KENAR_KISMI_TEMIZLEME_MIN_PIXEL = 50

# Temizleme sirasinda artefakt piksellerine yazilacak deger (uint8).
# MRI arka plani genelde 0 oldugundan varsayilan 0.
KENAR_TEMIZLEME_DEGERI = 0

# Bir parlak baglantili bilesenin alani toplam goruntu alaninin bu oranindan
# buyukse anatomik kabul edilir ve dokunulmaz. 0.5 -> goruntunun yarisindan
# buyuk parlak yapilar (ornegin saturasyonlu beyin dokusu) korunur. Deger
# [0.0, 1.0] araligi disinda verilirse uyarilir ve guvenli sinirlara klempelenir.
KENAR_ANATOMI_KORUMA_ORANI = 0.5

# Kalite raporlamada kenar artefakt sayaclari yazdirilsin mi?
KENAR_ARTEFAKT_RAPORLA = True

# ==================== EGIM DUZELTME AYARLARI ====================
# Hafif sola/saga yatmis 2D MRI dilimleri icin konservatif egim duzeltme.
# Mevcut politika: egim *kontrolu* yapilir (bkz. EGIM_KALITE_KONTROL_AKTIF)
# ancak otomatik egim *duzeltmesi* yapilmaz. Asagidaki esikler yalnizca
# duzeltme yeniden devreye alinirsa (EGIM_DUZELTME_AKTIF = True) etkindir.
EGIM_DUZELTME_AKTIF = False
EGIM_DUZELTME_RAPORLA = False
EGIM_ONIZLEME_URET = False
EGIM_MIN_ACI = 1.0
EGIM_MAKS_ACI = 5.0
# PCA ve minAreaRect acilari arasindaki izin verilen maksimum fark (derece).
EGIM_RMSE_MAKS = 2.5
EGIM_MIN_SATIR_SAYISI = 45
EGIM_MIN_X_SPAN = 8.0
# Ana foreground bileseni goruntu alaninin en az bu orani kadar olmali.
EGIM_MIN_FOREGROUND_ORANI = 0.04
EGIM_DOLDURMA_DEGERI = 0
EGIM_ROTASYON_PADDING_ORANI = 0.18

# Egim tahmini PCA tabanli olarak yapilir; minor/major eksen orani bu
# esikten buyukse maske neredeyse izotropik (dairesel) kabul edilir ve
# tahmin guvenilmez sayilir. OASIS/Kaggle 2D Alzheimer dilimleri beyin
# kirpilmis ve nispeten izotropik gorundugu icin 0.80 tum gercek dilimleri
# "maske_isotropik" sayabiliyordu. 0.95, yuksek egimli adaylari egitim
# ciktilarindan ayirmak icin daha kullanisli ama hala konservatif bir
# baslangic esigidir; daha agresif ayrim icin veri setinden orneklerle
# 0.98 ayrica degerlendirilmelidir.
EGIM_MIN_EKSEN_ORANI = 0.88
