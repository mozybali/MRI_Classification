#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_eda.py
----------
JPEG/PNG MRI görüntüleri için uçtan uca EDA pipeline'ını çalıştırır.

Kullanım:
    1) data/labels.csv dosyasını oluştur (kolonlar: id, filepath, label)
    2) Gerekirse mri_eda_jpg/config.py içindeki yolları güncelle
    3) Proje kökünden şu komutu çalıştır:

        python scripts/run_eda.py
"""

from mri_eda_jpg.config import METADATA_CSV
from mri_eda_jpg.io_utils import rastgele_tohum_ayarla, cikti_klasorunu_olustur, etiket_tablosu_yukle
from mri_eda_jpg.stats_utils import tum_gorseller_icin_istatistik_hesapla
from mri_eda_jpg.plot_utils import (
    sinif_dagilimi_ciz,
    boyut_dagilimlari_ciz,
    yogunluk_ozellik_kutu_grafikleri_ciz,
    global_yogunluk_histogramlari_ciz,
    rastgele_ornek_gorseller_ciz,
    pca_gomuleme_ciz,
    tsne_gomuleme_ciz,
)


def main():
    rastgele_tohum_ayarla()
    cikti_klasorunu_olustur()

    print("[BILGI] Etiket tablosu yükleniyor...")
    df = etiket_tablosu_yukle(METADATA_CSV)

    print("[BILGI] Görüntü istatistikleri hesaplanıyor...")
    df_stats = tum_gorseller_icin_istatistik_hesapla(df)

    print("[BILGI] Sınıf dağılımı çiziliyor...")
    sinif_dagilimi_ciz(df_stats)

    print("[BILGI] Görüntü boyut dağılımları çiziliyor...")
    boyut_dagilimlari_ciz(df_stats)

    print("[BILGI] Yoğunluk tabanlı öznitelik kutu grafikleri çiziliyor...")
    yogunluk_ozellik_kutu_grafikleri_ciz(df_stats)

    print("[BILGI] Sınıfa göre global yoğunluk histogramları çiziliyor...")
    global_yogunluk_histogramlari_ciz(df_stats)

    print("[BILGI] Her sınıftan rastgele örnek görüntüler çiziliyor...")
    rastgele_ornek_gorseller_ciz(df_stats, n_per_label=4)

    print("[BILGI] PCA gömme grafiği çiziliyor...")
    pca_gomuleme_ciz(df_stats)

    print("[BILGI] t-SNE gömme grafiği çiziliyor (biraz sürebilir)...")
    tsne_gomuleme_ciz(df_stats)

    print("[TAMAMLANDI] Tüm grafikler çıktı klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
