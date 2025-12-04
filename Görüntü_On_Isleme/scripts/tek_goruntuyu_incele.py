#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tek_goruntuyu_incele.py
-----------------------
Tek bir görüntü üzerinde ön işleme adımlarının sonucunu hızlıca görmek için
kullanılan yardımcı script.

Kullanım:
    python scripts/tek_goruntuyu_incele.py veri/girdi/ornek.jpg
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from goruntu_isleme_mri.io_araclari import goruntu_oku_gri
from goruntu_isleme_mri.on_isleme_adimlari import tek_goruntu_on_isle


def ana(girdi_yolu: str):
    goruntu_gri = goruntu_oku_gri(girdi_yolu)
    on_islenmis, meta = tek_goruntu_on_isle(goruntu_gri)

    print("[BILGI] Meta bilgi:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    fig, eksenler = plt.subplots(1, 2, figsize=(8, 4))
    eksenler[0].imshow(goruntu_gri, cmap="gray")
    eksenler[0].set_title("Orijinal")
    eksenler[0].axis("off")

    eksenler[1].imshow(on_islenmis, cmap="gray")
    eksenler[1].set_title("Ön İşlenmiş")
    eksenler[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Lütfen bir görüntü yolu verin. Örnek:")
        print("  python scripts/tek_goruntuyu_incele.py veri/girdi/ornek.jpg")
        sys.exit(1)

    girdi_yolu = sys.argv[1]
    if not Path(girdi_yolu).exists():
        print(f"Girdi yolu bulunamadı: {girdi_yolu}")
        sys.exit(1)

    ana(girdi_yolu)
