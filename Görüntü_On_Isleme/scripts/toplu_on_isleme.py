#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toplu_on_isleme.py
------------------
Girdi klasöründeki tüm MRI görüntülerine ön işleme pipeline'ını uygular.

Adımlar:
  1) Girdi klasöründeki tüm JPEG/PNG dosyalarını bul
  2) Her görüntüyü gri tonlamada oku
  3) Arka plan tespiti, maskeleme, kırpma, normalizasyon, histogram eşitleme, yeniden boyutlandırma
  4) Ön işlenmiş görüntüyü çıktı klasörüne kaydet
  5) (İsteğe bağlı) Veri artırma ile ekstra kopyalar üret
  6) Tüm işlemleri CSV log dosyasında özetle

Çalıştırma:
    python scripts/toplu_on_isleme.py
"""

import os
import csv
from pathlib import Path

from goruntu_isleme_mri.ayarlar import (
    GIRDİ_KLASORU,
    CIKTI_KLASORU,
    VERI_ARTIRMA_AKTIF,
    EKSTRA_KOPYA_SAYISI,
)
from goruntu_isleme_mri.io_araclari import (
    rastgele_tohum_ayarla,
    klasor_olustur_yoksa,
    girdi_gorsellerini_listele,
    goruntu_oku_gri,
    goruntu_kaydet,
    cikti_yolu_olustur,
)
from goruntu_isleme_mri.on_isleme_adimlari import tek_goruntu_on_isle
from goruntu_isleme_mri.artirma import rastgele_artirma_uygula


def ana():
    rastgele_tohum_ayarla()
    klasor_olustur_yoksa(GIRDİ_KLASORU)
    klasor_olustur_yoksa(CIKTI_KLASORU)

    print(f"[BILGI] Girdi klasörü: {GIRDİ_KLASORU}")
    print(f"[BILGI] Çıktı klasörü: {CIKTI_KLASORU}")

    girdi_listesi = girdi_gorsellerini_listele(GIRDİ_KLASORU)
    print(f"[BILGI] Toplam {len(girdi_listesi)} adet görüntü bulundu.")

    log_kayitlari = []

    for i, girdi_yolu in enumerate(girdi_listesi, start=1):
        print(f"[{i}/{len(girdi_listesi)}] İşleniyor: {girdi_yolu}")

        goruntu_gri = goruntu_oku_gri(girdi_yolu)
        on_islenmis, meta = tek_goruntu_on_isle(goruntu_gri)

        # Ana çıktı dosyasının yolunu üret ve kaydet
        cikti_yolu = cikti_yolu_olustur(girdi_yolu)
        goruntu_kaydet(cikti_yolu, on_islenmis)

        # Log kaydı
        kayit = {
            "girdi_yolu": girdi_yolu,
            "cikti_yolu": cikti_yolu,
            **meta,
        }
        log_kayitlari.append(kayit)

        # Veri artırma aktif ise ekstra kopyalar üret
        if VERI_ARTIRMA_AKTIF and EKSTRA_KOPYA_SAYISI > 0:
            ana_govde, uzanti = os.path.splitext(cikti_yolu)
            for k in range(EKSTRA_KOPYA_SAYISI):
                artirilmis = rastgele_artirma_uygula(on_islenmis)
                artirilmis_yol = f"{ana_govde}_aug{k+1}{uzanti}"
                goruntu_kaydet(artirilmis_yol, artirilmis)
                # Augmented örnekler için de basit bir log ekleyebiliriz
                log_kayitlari.append({
                    "girdi_yolu": girdi_yolu,
                    "cikti_yolu": artirilmis_yol,
                    **meta,
                    "aug_kopya": k + 1,
                })

    # Log dosyasını yaz
    log_dosyasi_yolu = Path(CIKTI_KLASORU) / "on_isleme_log.csv"
    alanlar = sorted({anahtar for kayit in log_kayitlari for anahtar in kayit.keys()})

    with open(log_dosyasi_yolu, "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=alanlar)
        yazici.writeheader()
        for kayit in log_kayitlari:
            yazici.writerow(kayit)

    print(f"[TAMAMLANDI] Ön işleme bitti. Log dosyası: {log_dosyasi_yolu}")


if __name__ == "__main__":
    ana()
