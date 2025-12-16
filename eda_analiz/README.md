# EDA Analiz Modülü

MRI veri seti için keşifsel veri analizi (EDA) üretir; sınıf dağılımı, boyut ve yoğunluk istatistikleri, korelasyon ve PCA görsellerini otomatik kaydeder. İstatistik hesaplamaları çok çekirdekle hızlandırılır.

## Kurulum

Yalnızca bu modül:
```bash
pip install -r requirements.txt
```
Tüm proje paketleri zaten kuruluysa bu adımı atlayabilirsiniz (`../requirements.txt` yeterli).

## Kullanım

```bash
python eda_calistir.py
```
Komut sırasında veri klasörü (varsayılan: depo içindeki `Veri_Seti`) ve çıktı klasörü
(varsayılan: `eda_analiz/eda_ciktilar`) sorulur. Varsayılanları kabul etmek için Enter
tuşuna basmanız yeterli; yollar otomatik olarak gerçek konumlara çözümlenir.

## Üretilenler

- `0_ozet_istatistikler.txt`: Toplam örnek, sınıf dağılımı ve temel özet.
- `1_sinif_dagilimi.png`: Sınıf dağılımı grafiği.
- `2_boyut_analizi.png`: Genişlik/yükseklik/en-boy oranı dağılımları.
- `3_yogunluk_analizi.png`: Yoğunluk histogramları.
- `4_korelasyon_matrisi.png`: Özellik korelasyonları.
- `5_pca_analizi.png`: PCA ilk iki bileşen görselleştirmesi.
- `veri_seti_istatistikler.csv`: Görüntü bazlı temel istatistikler.

## Ne Zaman Çalıştırılmalı?

- Veri setinin içeriğini ve dengesini hızlıca görmek istediğinizde.  
- Ön işleme/augmentasyon stratejisinden önce veri kalitesini kontrol ederken.  
- Eğitim raporlarını desteklemek için özet görseller gerektiğinde.
