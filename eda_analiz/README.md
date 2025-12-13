# EDA Analiz Modülü

MRI veri seti için keşifsel veri analizi (Exploratory Data Analysis).

## 🆕 v3.0 Performans İyileştirmeleri

### ⚡ Paralel İstatistik Hesaplama
- **Multiprocessing**: İstatistik hesaplama 4-6x daha hızlı
- **Otomatik CPU yönetimi**: Tüm çekirdekler kullanılır
- **Toplu işleme**: Binlerce görüntü hızlıca analiz edilir

📊 **Performans Kazanımları:**
- EDA analizi: 15-20 dk → 3-4 dakika (**4-6x**)

✅ **Geriye Uyumlu**: Aynı API, otomatik hızlanma!

---

## 📦 Kurulum

**Minimal kurulum (sadece EDA için):**
```bash
pip install -r requirements.txt
```

**Tam kurulum (tüm proje için):**
```bash
cd ..
pip install -r requirements.txt
```

## 🚀 Kullanım

**Not:** Komutlarda `python` veya `python3` kullanabilirsiniz. Windows'ta genellikle `python`, Linux/Mac'te `python3` kullanılır.

**Interaktif mod:**
```bash
python eda_calistir.py
```

Program şunları soracak:
- Veri seti klasörü yolu (varsayılan: ../../Veri_Seti)
- Çıktı klasörü yolu (varsayılan: eda_ciktilar)

## 📊 Özellikler

### Analiz Türleri
- ✅ **Sınıf dağılımı** - Her sınıfta kaç görüntü var?
- ✅ **Görüntü boyut analizi** - Genişlik, yükseklik, en-boy oranı
- ✅ **Yoğunluk istatistikleri** - Piksel yoğunluk dağılımları (mean, std, percentiles)
- ✅ **Korelasyon matrisi** - Özellikler arası ilişkiler
- ✅ **PCA görselleştirmesi** - 2D boyut indirgeme, sınıf ayrılabilirliği

### Çıktılar
- 📈 **Grafikler** (PNG formatında):
  - `1_sinif_dagilimi.png`
  - `2_boyut_analizi.png`
  - `3_yogunluk_analizi.png`
  - `4_korelasyon_matrisi.png`
  - `5_pca_analizi.png`
- 📄 **Özet rapor** (TXT):
  - `0_ozet_istatistikler.txt`
- 💾 **Veri seti CSV**:
  - `veri_seti_istatistikler.csv`

## 📁 Çıktı Yapısı

```
eda_ciktilar/
├── 0_ozet_istatistikler.txt
├── 1_sinif_dagilimi.png
├── 2_boyut_analizi.png
├── 3_yogunluk_analizi.png
├── 4_korelasyon_matrisi.png
├── 5_pca_analizi.png
└── veri_seti_istatistikler.csv
```

## 💡 Ne Zaman Kullanılır?

- ✓ Veri setini ilk kez keşfetmek istediğinizde
- ✓ Sınıf dengesizliği kontrolü için
- ✓ Görüntü kalitesi ve tutarlılık analizi için
- ✓ Model eğitiminden önce veri anlayışı için

## 🐛 Sorun Giderme

### Veri seti bulunamadı:
```powershell
# Veri seti yolunu kontrol edin (PowerShell)
Get-ChildItem ..\..\Veri_Seti\
```

```bash
# Veya bash/Linux için
ls -la ../../Veri_Seti/
```

### Eksik paket:
```bash
pip install -r requirements.txt
```

## 📚 Dosyalar

- `eda_araclar.py` - Ana analiz sınıfı ve fonksiyonlar
- `eda_calistir.py` - Çalıştırılabilir script
- `requirements.txt` - Gerekli Python paketleri
