"""
ayarlar.py
----------
MRI JPEG/PNG görüntülerinin model eğitimi öncesi ön işlenmesi için temel ayarlar.
Bu dosyadaki değerleri kendi proje yapına göre rahatça değiştirebilirsin.
"""

# Girdi ve çıktı klasörleri
GIRDİ_KLASORU = "veri/girdi"   # Ham (orijinal) görüntülerin olduğu klasör
CIKTI_KLASORU = "veri/cikti"   # Ön işlenmiş görüntülerin kaydedileceği klasör

# İzin verilen görüntü uzantıları
GORUNTU_UZANTILARI = [".jpg", ".jpeg", ".png"]

# Model eğitimi için hedef boyut (genişlik, yükseklik)
HEDEF_GENISLIK = 256
HEDEF_YUKSEKLIK = 256

# Yoğunluk normalizasyonu için kullanılacak yüzdelikler (alt, üst)
# Örn: (1, 99) => en düşük %1 ve en yüksek %1 uç değerleri kırp
KIRPMA_YUZDELERI = (1, 99)

# Adaptif histogram eşitleme (CLAHE benzeri) kullanılsın mı?
HISTOGRAM_ESITLEME_AKTIF = True

# Maskeden kırpma yaparken etrafına eklenecek kenar payı (piksel cinsinden)
MASKE_KENAR_PAYI = 5

# Rastgelelik için sabit tohum (reproducible olsun diye)
RASTGELE_TOHUM = 42

# Veri artırma (augmentation) ayarları
VERI_ARTIRMA_AKTIF = True
# Her orijinal görüntü için kaç ekstra artırılmış kopya üretilecek
EKSTRA_KOPYA_SAYISI = 2

# Sınıf etiketin varsa ve isimlendirmek istersen (opsiyonel):
# Örnek: {0: "Seviye 1", 1: "Seviye 2", 2: "Seviye 3", 3: "Seviye 4"}
ETIKET_ISIM_HARITASI = None
