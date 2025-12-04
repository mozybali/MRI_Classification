"""
config.py
---------
JPEG (ve benzeri 2B) MRI görüntüleri için EDA ayarları.
"""

# Etiket bilgilerini içeren CSV dosyasının yolu
METADATA_CSV = "data/labels.csv"

# Tüm grafiklerin kaydedileceği çıktı klasörü
OUTPUT_DIR = "eda_ciktlari"

# Tekrarlanabilirlik için rastgelelik tohumu
RANDOM_SEED = 42

# Global yoğunluk histogramları için kaç görüntü kullanılacak
N_IMAGES_INTENSITY_SAMPLE = 40

# Her görüntüden kaç piksel örneklenecek
N_PIXELS_PER_IMAGE_SAMPLE = 5000

# PCA / t-SNE gömme için maksimum kaç görüntü kullanılacak
N_IMAGES_FOR_EMBEDDING = 200

# Sınıf etiketlerini daha anlamlı isimlere çevirmek istersen:
# Örnek: {0: "Seviye 1", 1: "Seviye 2", 2: "Seviye 3", 3: "Seviye 4"}
LABEL_NAME_MAP = None

# Görüntüleri gri tonlamaya mı çevirelim? (Önerilen: True)
CONVERT_TO_GRAYSCALE = True
