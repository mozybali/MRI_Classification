#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hizli pipeline testi."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover - dogrudan script olarak calistirma destegi
    pytest = None


def _requires_data(func):
    if pytest is None:
        return func
    return pytest.mark.requires_data(func)


def _check_imports():
    """Gerekli paketlerin yuklu olup olmadigini kontrol et."""
    print("\n" + "=" * 70)
    print("PAKET KONTROLU")
    print("=" * 70)

    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "PIL": "Pillow",
        "scipy": "scipy",
        "skimage": "scikit-image",
        "sklearn": "scikit-learn",
        "tqdm": "tqdm",
    }

    optional_packages = {
        "cv2": "opencv-python",
        "SimpleITK": "SimpleITK",
    }

    all_ok = True

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"[OK] {package:20s} - Yuklu")
        except ImportError:
            print(f"[HATA] {package:20s} - Eksik (pip install {package})")
            all_ok = False

    print("\nOpsiyonel Paketler:")
    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"[OK] {package:20s} - Yuklu")
        except ImportError:
            print(f"[UYARI] {package:20s} - Yok (bazi ozellikler kullanilamaz)")

    return all_ok


def _check_veri_seti():
    """Veri setinin varligini kontrol et."""
    print("\n" + "=" * 70)
    print("VERI SETI KONTROLU")
    print("=" * 70)

    veri_klasoru = Path(__file__).resolve().parent.parent / "Veri_Seti"
    if not veri_klasoru.exists():
        print(f"[HATA] Veri seti bulunamadi: {veri_klasoru.absolute()}")
        return False

    siniflar = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    aday_kokler = [
        veri_klasoru / "OriginalDataset",
    ]

    bulunan = 0
    toplam = 0
    for kok in aday_kokler:
        if not kok.exists():
            continue

        sinif_var = any((kok / sinif).exists() for sinif in siniflar)
        if not sinif_var:
            continue

        bulunan += 1
        print(f"\nKaynak klasor: {kok}")
        kaynak_toplam = 0
        for sinif in siniflar:
            sinif_klasoru = kok / sinif
            if sinif_klasoru.exists():
                dosyalar = []
                for uzanti in (".jpg", ".jpeg", ".png"):
                    dosyalar.extend(sinif_klasoru.glob(f"*{uzanti}"))
                sayi = len(dosyalar)
                kaynak_toplam += sayi
                print(f"[OK] {sinif:20s}: {sayi:5d} goruntu")
            else:
                print(f"[HATA] {sinif:20s}: Klasor bulunamadi")
        print(f"  Alt toplam: {kaynak_toplam}")
        toplam += kaynak_toplam

    if bulunan == 0:
        print("[HATA] Bilinen veri yapilarinda sinif klasoru bulunamadi.")
        print("  Beklenen: Veri_Seti/OriginalDataset/<Sinif>")
        return False

    print(f"\nToplam: {toplam} goruntu")
    return toplam > 0


def _check_modul():
    """Modul import'unu test et."""
    print("\n" + "=" * 70)
    print("MODUL KONTROLU")
    print("=" * 70)

    try:
        from goruntu_isleme import ayarlar
        from goruntu_isleme.goruntu_isleyici import GorselIsleyici

        print("[OK] ayarlar.py yuklendi")
        print("[OK] goruntu_isleyici.py yuklendi")

        isleyici = GorselIsleyici()
        print("[OK] GorselIsleyici nesnesi olusturuldu")

        # Referansi canli tutarak import zincirini dogruladigimizi belirtiyoruz.
        _ = ayarlar, isleyici
        return True
    except Exception as exc:
        print(f"[HATA] Modul yukleme hatasi: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_imports():
    """Gerekli paketlerin yuklu oldugunu pytest ile dogrula."""
    assert _check_imports()


@_requires_data
def test_veri_seti():
    """Veri setinin varligini pytest ile dogrula."""
    assert _check_veri_seti()


def test_modul():
    """Modul import'unu pytest ile dogrula."""
    assert _check_modul()


def main():
    """Ana test fonksiyonu."""
    print("\n" + "=" * 70)
    print("MRI GORUNTU ISLEME - HIZLI TEST")
    print("=" * 70)

    results = {
        "Paket Kontrolu": _check_imports(),
        "Veri Seti Kontrolu": _check_veri_seti(),
        "Modul Kontrolu": _check_modul(),
    }

    print("\n" + "=" * 70)
    print("TEST SONUCLARI")
    print("=" * 70)

    for test_adi, sonuc in results.items():
        durum = "[OK] BASARILI" if sonuc else "[HATA] BASARISIZ"
        print(f"{test_adi:25s}: {durum}")

    if all(results.values()):
        print("\n[OK] Tum testler basarili. Pipeline hazir.")
        print("\nBir sonraki adim:")
        print("  python -m goruntu_isleme.ana_islem")
        return 0

    print("\n[HATA] Bazi testler basarisiz. Lutfen eksikleri giderin.")
    print("\nEksik paketleri yuklemek icin:")
    print("  pip install -r requirements.txt")
    print("  pip install -r requirements-torch-cu128.txt  # veya requirements-torch-cpu.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
