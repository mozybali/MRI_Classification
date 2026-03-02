#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ana_islem.py
------------
MRI goruntu isleme ana menu ve islem yoneticisi.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ayarlar import *
from goruntu_isleyici import GorselIsleyici
from ozellik_cikarici import OzellikCikarici, veri_boluntule, veri_setini_bol_ve_olceklendir


def ana_menu():
    """Ana menuyu goster."""
    print("\n" + "=" * 60)
    print("MRI GORUNTU ISLEME SISTEMI")
    print("=" * 60)
    print("\n1. Goruntuleri on isle (2D/3D)")
    print("2. Ozellik cikar ve CSV olustur")
    print("3. CSV'deki NaN degerleri temizle")
    print("4. Veri setini bol + egitim setine gore olceklendir")
    print("5. Istatistik raporu goster")
    print("6. Veri setini bol (ham CSV)")
    print("7. TUM ISLEMLERI OTOMATIK YAP")
    print("0. Cikis")
    print("\n" + "=" * 60)


def goruntu_on_isleme():
    """Ham goruntuleri on isle."""
    print("\n[1] GORUNTU ON ISLEME")
    print("-" * 60)

    mod = input("\nIsleme modu (2d/3d, Enter=2d): ").strip().lower()
    if mod == "3d":
        try:
            import importlib
            islem3d = importlib.import_module("3d_isleme")
        except Exception as e:
            print(f"[HATA] 3D islem modulu yuklenemedi: {e}")
            return

        hacim_yolu = input("3D hacim dosyasi (nii/nii.gz/npy) veya DICOM klasoru: ").strip()
        if not hacim_yolu:
            print("[HATA] Hacim yolu gerekli.")
            return

        sinif_adi = input(f"Sinif adi ({'/'.join(SINIF_KLASORLERI)}): ").strip()
        if sinif_adi and sinif_adi not in SINIF_KLASORLERI:
            print(f"[HATA] Gecersiz sinif: {sinif_adi}")
            return
        if not sinif_adi:
            sinif_adi = SINIF_KLASORLERI[0]

        model_yolu_girdi = input(f"3D model yolu (varsayilan: {islem3d.DEFAULT_MODEL_PATH}): ").strip()
        model_yolu = Path(model_yolu_girdi) if model_yolu_girdi else islem3d.DEFAULT_MODEL_PATH

        cikti3d_girdi = input(f"3D cikti klasoru (varsayilan: {CIKTI_KLASORU / '3d'}): ").strip()
        cikti3d = Path(cikti3d_girdi) if cikti3d_girdi else CIKTI_KLASORU / "3d"

        try:
            kayit = islem3d.calistir_3d_infer(
                Path(hacim_yolu),
                model_path=model_yolu,
                output_dir=cikti3d,
                sinif_adi=sinif_adi,
            )
            print(f"\n3D islem tamamlandi. Maske: {kayit}")
        except Exception as e:
            print(f"[HATA] 3D islem basarisiz: {e}")
        return

    isleyici = GorselIsleyici()

    giris = input(f"\nGirdi klasoru (varsayilan: {VERI_SETI_KLASORU}): ").strip()
    giris_klasoru = Path(giris) if giris else VERI_SETI_KLASORU

    cikis = input(f"Cikti klasoru (varsayilan: {CIKTI_KLASORU}): ").strip()
    cikti_klasoru = Path(cikis) if cikis else CIKTI_KLASORU

    istatistikler = isleyici.tum_gorselleri_isle(cikti_klasoru)

    if istatistikler:
        print("\n[BASARILI] Goruntu isleme tamamlandi.")
    else:
        print("\n[HATA] Goruntu isleme basarisiz.")


def ozellik_cikar():
    """Islenmis goruntulerden ozellik cikar."""
    print("\n[2] OZELLIK CIKARMA VE CSV OLUSTURMA")
    print("-" * 60)

    cikarici = OzellikCikarici()
    giris = input(f"\nIslenmis goruntuler klasoru (varsayilan: {CIKTI_KLASORU}): ").strip()
    giris_klasoru = Path(giris) if giris else CIKTI_KLASORU

    df = cikarici.csv_olustur(giris_klasoru)
    if not df.empty:
        print("\n[BASARILI] Ozellik cikarimi tamamlandi.")
    else:
        print("\n[HATA] Ozellik cikarimi basarisiz.")


def nan_temizle():
    """CSV'deki NaN degerleri temizle."""
    print("\n[3] NaN DEGERLERI TEMIZLEME")
    print("-" * 60)

    cikarici = OzellikCikarici()

    print("\nMevcut temizleme metodlari:")
    print("  1. drop   - NaN iceren satirlari cikar")
    print("  2. mean   - NaN'lari sutun ortalamasi ile doldur")
    print("  3. median - NaN'lari sutun medyani ile doldur")
    print("  4. zero   - NaN'lari 0 ile doldur")

    metod = input("\nMetod secin (drop/mean/median/zero, Enter=drop): ").strip().lower()
    if metod not in ['drop', 'mean', 'median', 'zero', '']:
        print("[HATA] Gecersiz metod.")
        return
    if not metod:
        metod = 'drop'

    df = cikarici.nan_temizle(metod=metod)
    if not df.empty:
        print("\n[BASARILI] NaN temizleme tamamlandi.")
    else:
        print("\n[HATA] NaN temizleme basarisiz.")


def scaling_uygula():
    """Leakage-free split ve scaling uygula."""
    print("\n[4] VERI BOLME + OLCEKLENDIRME")
    print("-" * 60)
    print("\nScaler once egitim setine fit edilir, sonra dogrulama ve teste uygulanir.")
    print(f"Mevcut metod: {SCALING_METODU}")
    print("\nMetodlar:")
    print("  1. minmax")
    print("  2. robust")
    print("  3. standard")
    print("  4. maxabs")

    metod = input("\nMetod secin (minmax/robust/standard/maxabs, Enter=varsayilan): ").strip().lower()
    if metod not in ['minmax', 'robust', 'standard', 'maxabs', '']:
        print("[HATA] Gecersiz metod.")
        return
    if not metod:
        metod = SCALING_METODU

    splitler = veri_setini_bol_ve_olceklendir(metod=metod)
    if splitler and all(not df.empty for df in splitler):
        print("\n[BASARILI] Veri bolme ve olceklendirme tamamlandi.")
    else:
        print("\n[HATA] Veri bolme ve olceklendirme basarisiz.")


def istatistik_goster():
    """Istatistik raporu goster."""
    print("\n[5] ISTATISTIK RAPORU")
    print("-" * 60)
    cikarici = OzellikCikarici()
    cikarici.istatistik_raporu()


def veri_bol():
    """Ham CSV uzerinden veri setini bol."""
    print("\n[6] VERI SETI BOLME")
    print("-" * 60)
    print(f"\nOranlar: Egitim={EGITIM_ORANI}, Dogrulama={DOGRULAMA_ORANI}, Test={TEST_ORANI}")
    veri_boluntule()


def tum_islemleri_yap():
    """Tum islemleri otomatik ve guvenli sirada yap."""
    print("\n[7] TUM ISLEMLER OTOMATIK")
    print("-" * 60)
    print("\nSu islemler sirayla yapilacak:")
    print("  1. Goruntu on isleme")
    print("  2. Ozellik cikarma")
    print("  3. Veri bolme + egitim setine gore olceklendirme")
    print("  4. Istatistik raporu")

    onay = input("\nDevam etmek istiyor musunuz? (e/h): ").strip().lower()
    if onay != 'e':
        print("Islem iptal edildi.")
        return

    print("\n\n" + "=" * 60)
    print("ADIM 1/4: GORUNTU ON ISLEME")
    print("=" * 60)
    isleyici = GorselIsleyici()
    isleyici.tum_gorselleri_isle(CIKTI_KLASORU)

    print("\n\n" + "=" * 60)
    print("ADIM 2/4: OZELLIK CIKARMA")
    print("=" * 60)
    cikarici = OzellikCikarici()
    df = cikarici.csv_olustur(CIKTI_KLASORU)
    if df.empty:
        print("\n[HATA] Ozellik cikarimi basarisiz. Islem durduruluyor.")
        return

    print("\n\n" + "=" * 60)
    print("ADIM 3/4: VERI BOLME + OLCEKLENDIRME")
    print("=" * 60)
    veri_setini_bol_ve_olceklendir()

    print("\n\n" + "=" * 60)
    print("ADIM 4/4: ISTATISTIK RAPORU")
    print("=" * 60)
    cikarici.istatistik_raporu()

    print("\n\n" + "=" * 60)
    print("[BASARILI] Tum islemler tamamlandi.")
    print("=" * 60)


def main():
    """Ana program."""
    while True:
        try:
            ana_menu()
            secim = input("\nSeciminiz: ").strip()

            if secim == '0':
                print("\nCikiliyor...")
                break
            if secim == '1':
                goruntu_on_isleme()
            elif secim == '2':
                ozellik_cikar()
            elif secim == '3':
                nan_temizle()
            elif secim == '4':
                scaling_uygula()
            elif secim == '5':
                istatistik_goster()
            elif secim == '6':
                veri_bol()
            elif secim == '7':
                tum_islemleri_yap()
            else:
                print("\n[HATA] Gecersiz secim. Lutfen 0-7 arasi bir sayi girin.")

            input("\nDevam etmek icin Enter'a basin...")

        except KeyboardInterrupt:
            print("\n\nProgram kullanici tarafindan durduruldu.")
            break
        except Exception as e:
            print(f"\n[HATA] Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
            input("\nDevam etmek icin Enter'a basin...")


if __name__ == "__main__":
    main()
