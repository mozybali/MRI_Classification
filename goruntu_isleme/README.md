# Görüntü İşleme Modülü

MRI görüntülerini işlemek ve özellik çıkarmak için gelişmiş modül.

## 🆕 v3.0 Performans İyileştirmeleri

### ⚡ Multiprocessing ile 4-10x Hızlanma
- **Paralel görüntü işleme**: CPU çekirdeğinizi tam kullanır
- **Paralel özellik çıkarma**: 8-10x daha hızlı CSV oluşturma
- **Otomatik CPU yönetimi**: (n-1) çekirdek otomatik kullanılır
- **Akıllı önbellekleme**: Tekrar eden işlemler cache'lenir

📊 **Performans Kazanımları:**
- Görüntü işleme: 2-3 saat → 20-40 dakika (**4-8x**)
- Özellik çıkarma: 30-45 dk → 3-5 dakika (**8-10x**)

✅ **Geriye Uyumlu**: Tüm eski kodunuz aynen çalışır, hiçbir değişiklik gerekmez!

---

## 📦 Kurulum

```bash
# Ana dizinden tüm bağımlılıkları yükle
cd ..
pip install -r requirements.txt
```

**Not:** Görüntü işleme modülü için ayrı requirements.txt yok, tüm bağımlılıklar ana `requirements.txt` dosyasında.

## 🚀 Kullanım

**Not:** Komutlarda `python` veya `python3` kullanabilirsiniz. Windows'ta genellikle `python`, Linux/Mac'te `python3` kullanılır.

### 1. Sistem Kontrolü (Önerilen)
```bash
python pipeline_quick_test.py
```
Paket ve veri seti kontrolü yapar.

### 2. Ana İşleme Pipeline
```bash
python ana_islem.py
```

**Menü seçenekleri:**
```
1. Görüntüleri ön işle          → Normalize, CLAHE, bias correction
2. Özellik çıkar ve CSV oluştur → 20+ özellik çıkarma
3. CSV'ye ölçeklendirme uygula  → MinMax/Robust/Standard scaling
4. Veri setini böl              → Train/Val/Test split
5. İstatistik raporu göster     → Özet istatistikler
6. TÜM İŞLEMLERİ OTOMATIK YAP   → ⭐ Önerilen
```

### 3. Pipeline Test (Tek Görüntü)
```bash
python test_pipeline.py [goruntu_yolu]
```
Tek görüntü üzerinde tüm adımları görselleştirir.

## 📁 Modül Yapısı

```
goruntu_isleme/
├── ayarlar.py                 # Merkezi konfigürasyon
├── goruntu_isleyici.py        # Core işleme sınıfı
├── ozellik_cikarici.py        # Özellik çıkarma
├── ana_islem.py               # Ana menü (⭐ buradan başla)
├── pipeline_quick_test.py     # Sistem kontrolü
├── test_pipeline.py           # Pipeline test
└── requirements.txt           # Bağımlılıklar
```

## ✨ Özellikler (v2.0)

### Gelişmiş Ön İşleme
- ✅ **Bias field correction** (N4ITK) - MRI yoğunluk düzeltme
- ✅ **Skull stripping** - Kafatası çıkarma
- ✅ **Center of mass alignment** - Görüntü hizalama
- ✅ **Adaptive CLAHE** - Akıllı kontrast iyileştirme
- ✅ **Gürültü giderme** - Median/Gaussian filtreleme
- ✅ **Z-score normalizasyonu** - Standardizasyon

### Medikal-Spesifik Augmentation
- ✅ **Elastic deformation** - Doku benzeri deformasyon
- ✅ **Gaussian noise** - Gerçekçi gürültü ekleme
- ✅ **Random crop & resize** - Rastgele kırpma
- ✅ **Intensity shift** - Yoğunluk kayması
- ✅ **Flip (horizontal/vertical)** - Aynalama
- ✅ **Sınıf bazlı dengesiz augmentation** - Az örnekli sınıflar için daha fazla artırma

### Özellik Çıkarma
**20+ özellik:**
- Boyut özellikleri (genişlik, yükseklik, en-boy oranı)
- Yoğunluk istatistikleri (mean, std, min, max, percentiles)
- Doku özellikleri (entropi, kontrast, homojenlik, enerji)
- Gelişmiş özellikler (skewness, kurtosis, gradient, Otsu threshold)

### Ölçeklendirme
- ✅ MinMax (0-1 aralığı)
- ✅ Robust (outlier'lara dayanıklı)
- ✅ Standard (Z-score)
- ✅ MaxAbs ([-1, 1] aralığı)

## 📊 Çıktılar

```
goruntu_isleme/cikti/
├── NonDemented/                      # İşlenmiş görüntüler
├── VeryMildDemented/
├── MildDemented/
├── ModerateDemented/
├── goruntu_ozellikleri.csv           # Ham özellikler
├── goruntu_ozellikleri_scaled.csv    # Ölçeklendirilmiş (model için)
├── train/                            # Eğitim seti
├── validation/                       # Doğrulama seti
└── test/                             # Test seti
```

## ⚙️ Konfigürasyon

`ayarlar.py` dosyasından tüm parametreler ayarlanabilir:

```python
# Görüntü boyutu
HEDEF_GENISLIK = 256
HEDEF_YUKSEKLIK = 256

# Veri artırma
VERI_ARTIRMA_AKTIF = True
SINIF_BAZLI_ARTIRMA_AKTIF = True
SINIF_BAZLI_CARPANLAR = {
    "NonDemented": 1,
    "ModerateDemented": 3,  # En az örnek - en çok artır
}

# Gelişmiş işleme
BIAS_FIELD_CORRECTION_AKTIF = True
SKULL_STRIPPING_AKTIF = True
```

## 🐛 Sorun Giderme

### OpenCV/scikit-image/tqdm yüklü değil:
```bash
pip install opencv-python scikit-image tqdm
```

### SimpleITK eksik (opsiyonel):
```bash
pip install SimpleITK
```
SimpleITK yoksa bias correction çalışmaz ama diğer özellikler çalışır.

### Veri seti bulunamadı:
```bash
# Veri setinin doğru konumda olduğunu kontrol edin
ls -la ../Veri_Seti/
```

## 💡 İpuçları

1. **İlk kullanımda** `pipeline_quick_test.py` çalıştırın
2. **Hızlı başlangıç** için ana_islem.py'de "6" seçin
3. **Tek görüntü test** için test_pipeline.py kullanın
4. **Augmentation çarpanlarını** sınıf dengesine göre ayarlayın
5. **SimpleITK** kurarak daha iyi bias correction elde edin
