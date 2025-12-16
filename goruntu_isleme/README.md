# Görüntü İşleme Modülü

Ham MRI görüntülerini kalite kontrolünden geçirir, normalize eder, hizalar, sınıf bazlı augmentasyon uygular ve model eğitimine hazır özellik CSV’leri üretir. Tüm ağır işler çok çekirdekli çalışır.

## Kurulum

Ana dizinden bağımlılıkları yükleyin:
```bash
pip install -r ../requirements.txt
```
Veri yapısı: `../Veri_Seti/<SınıfAdı>/` (NonDemented, VeryMildDemented, MildDemented, ModerateDemented).

## Modüller ve İş Akışı

### 1) Ana menü (ana_islem.py)
```bash
python ana_islem.py
```
- **1 Ön işleme**: Kalite kontrol → median filtre → bias field correction (SimpleITK varsa N4ITK, yoksa hızlı yöntem) → skull stripping → hizalama (center-of-mass/affine) → yoğunluk normalizasyonu + adaptif CLAHE → yeniden boyutlandırma → sınıf bazlı augmentasyon. Çıkış: `cikti/<sınıf>/`.
- **2 Özellik çıkarma**: `ozellik_cikarici.py` ile 20+ öznitelik (boyut, yoğunluk istatistikleri, entropi, kontrast, gradyan, Otsu eşiği) hesaplanır, `goruntu_ozellikleri.csv` oluşturulur.
- **3 NaN temizleme**: CSV’deki eksik değerleri düşürme veya doldurma (drop/mean/median/zero).
- **4 Ölçeklendirme**: `SCALING_METODU` (minmax/robust/standard/maxabs) ile `goruntu_ozellikleri_scaled.csv`.
- **5 Veri bölme**: Stratified train/val/test CSV’leri (`egitim.csv`, `dogrulama.csv`, `test.csv`).
- **6 İstatistik raporu**: CSV özetlerini terminalde gösterir.
- **7 Otomatik**: 1→2→3→4→5 adımlarını sırayla çalıştırır (önerilen).

### 2) Hızlı kontrol (pipeline_quick_test.py)
```bash
python pipeline_quick_test.py
```
Paket ve veri dizini kontrolü yapar.

### 3) Tek görüntü görselleştirme (test_pipeline.py)
```bash
python test_pipeline.py /path/to/image.jpg
```
Pipeline adımlarını tek bir görüntü üzerinde görselleştirir; argüman verilmezse veri setinden örnek arar.

## Çıktılar

`cikti/` altında:
- İşlenmiş görüntüler (`<sınıf>/<dosya>.png`, augmentasyon dahil)
- `goruntu_ozellikleri.csv` (ham özellikler)
- `goruntu_ozellikleri_scaled.csv` (ölçekli özellikler)
- `egitim.csv`, `dogrulama.csv`, `test.csv` (stratified bölünmüş setler)

## Ayarlar (ayarlar.py)

- Boyut ve normalizasyon: `HEDEF_GENISLIK`, `HEDEF_YUKSEKLIK`, `NORMALIZASYON_STRATEJISI`
- Görüntü iyileştirme: `BIAS_FIELD_CORRECTION_AKTIF`, `SKULL_STRIPPING_AKTIF`, `REGISTRATION_AKTIF`
- Augmentasyon: `VERI_ARTIRMA_AKTIF`, `SINIF_BAZLI_ARTIRMA_AKTIF`, `SINIF_BAZLI_CARPANLAR`
- Ölçekleme: `SCALING_METODU`
- Bölme oranları: `EGITIM_ORANI`, `DOGRULAMA_ORANI`, `TEST_ORANI`

## İpuçları

- SimpleITK yoksa bias correction otomatik hızlı metoda düşer (uyarı görürsünüz).  
- Paralel işlem sayısı `GorselIsleyici.n_jobs` ile sınırlanabilir.  
- CSV’de NaN varsa menü 3 → 4 → 5 adımlarını yeniden çalıştırın.
