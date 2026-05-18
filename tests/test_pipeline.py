#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pipeline.py
----------------
Guncellenmis goruntu isleme pipeline'ini test eden script.
Tek bir goruntu uzerinde aktif tum adimlari gosterir.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goruntu_isleme.ayarlar import (
    CIKTI_KLASORU,
    GORUNTU_UZANTILARI,
    ON_ISLEME_VARSAYILAN_GIRIS_KLASORU,
    SINIF_KLASORLERI,
)
from goruntu_isleme.goruntu_isleyici import GorselIsleyici


def pipeline_test(goruntu_yolu: str):
    """
    Pipeline'in tum aktif adimlarini test et ve gorsellestir.

    Args:
        goruntu_yolu: Test edilecek goruntunun yolu
    """
    print("\n" + "=" * 70)
    print("GORUNTU ISLEME PIPELINE TEST")
    print("=" * 70)
    print(f"\nTest goruntusu: {goruntu_yolu}")

    isleyici = GorselIsleyici()

    goruntu_ham = isleyici.goruntu_yukle(goruntu_yolu)
    if goruntu_ham is None:
        print("\n[HATA] Goruntu yuklenemedi!")
        return

    print(f"\n[OK] Goruntu yuklendi: {goruntu_ham.shape}")

    asamalar = {"1. Orijinal": goruntu_ham}

    print("\n" + "-" * 70)
    print("PIPELINE ASAMALARI")
    print("-" * 70)

    print("\n1. Kenar artefakt temizligi...")
    g1 = isleyici.kenar_artefakt_temizle(goruntu_ham.copy())
    asamalar["2. Kenar Temizlik"] = g1

    print("2. Gurultu giderme (bilateral)...")
    g2 = isleyici.gurultu_gider(g1.copy(), metod="auto")
    asamalar["3. Gurultu Giderme"] = g2

    print("3. Center of mass alignment (hizalama)...")
    g3 = isleyici.center_of_mass_alignment(g2.copy())
    asamalar["4. Alignment"] = g3

    print("4. Yogunluk normalizasyonu (percentile)...")
    g4 = isleyici.yogunluk_normalize(g3.copy())
    asamalar["5. Yogunluk Norm."] = g4

    print("5. CLAHE...")
    g5 = isleyici.histogram_esitle(g4.copy(), adaptive=False)
    asamalar["6. CLAHE"] = g5

    print("6. Boyutlandirma (192x192)...")
    g6 = isleyici.boyutlandir(g5.copy())
    asamalar["7. Boyutlandirma"] = g6

    print("\n[OK] Tum asamalar tamamlandi!")

    print("\n" + "-" * 70)
    print("GORSELLESTIRME")
    print("-" * 70)
    print("\nGrafik olusturuluyor...")

    sayi = len(asamalar)
    cols = 4
    rows = (sayi + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for idx, (baslik, goruntu) in enumerate(asamalar.items()):
        axes[idx].imshow(goruntu, cmap="gray")
        axes[idx].set_title(baslik, fontsize=10, weight="bold")
        axes[idx].axis("off")
        ort = float(np.mean(goruntu))
        std = float(np.std(goruntu))
        axes[idx].text(
            0.05,
            0.95,
            f"mu={ort:.1f}, sigma={std:.1f}",
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

    for idx in range(sayi, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    cikti_klasoru = CIKTI_KLASORU / "test"
    cikti_klasoru.mkdir(parents=True, exist_ok=True)
    cikti_yolu = cikti_klasoru / "pipeline_test.png"
    plt.savefig(cikti_yolu, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Grafik kaydedildi: {cikti_yolu}")

    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI!")
    print("=" * 70)
    print(f"\nCikti: {cikti_yolu}")
    print()


def test_pipeline_script_imports():
    """Script bagimliliklari pytest koleksiyonunda cozulmeli."""
    assert callable(pipeline_test)
    assert CIKTI_KLASORU.name == "cikti"
    assert ON_ISLEME_VARSAYILAN_GIRIS_KLASORU.name == "OriginalDataset"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_goruntu = sys.argv[1]
    else:
        print("\nVeri setinden test goruntusu araniyor...")

        aday_kokler = [ON_ISLEME_VARSAYILAN_GIRIS_KLASORU]
        bulunan = None
        for kok in aday_kokler:
            for sinif in SINIF_KLASORLERI:
                sinif_klasoru = kok / sinif
                if not sinif_klasoru.exists():
                    continue
                dosyalar = []
                for uzanti in GORUNTU_UZANTILARI:
                    dosyalar.extend(sorted(sinif_klasoru.glob(f"*{uzanti}")))
                if dosyalar:
                    bulunan = dosyalar[0]
                    break
            if bulunan is not None:
                break

        if bulunan is None:
            print("\n[HATA] Veri setinde goruntu bulunamadi!")
            print("Beklenen yapi: Veri_Seti/OriginalDataset/<Sinif>")
            print("Kullanim: python tests/test_pipeline.py [goruntu_yolu]")
            sys.exit(1)
        test_goruntu = str(bulunan)

    pipeline_test(test_goruntu)
