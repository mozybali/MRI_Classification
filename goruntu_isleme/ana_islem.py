#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MRI goruntu isleme ana menu ve CLI giris noktasi."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from goruntu_isleme.ayarlar import (
        CIKTI_KLASORU,
        DOGRULAMA_ORANI,
        EGITIM_ORANI,
        ON_ISLEME_VARSAYILAN_GIRIS_KLASORU,
        SCALING_METODU,
        SINIF_KLASORLERI,
        TEST_ORANI,
    )
    from goruntu_isleme.goruntu_isleyici import GorselIsleyici
    from goruntu_isleme.ozellik_cikarici import (
        OzellikCikarici,
        veri_boluntule,
        veri_setini_bol_ve_olceklendir,
    )
else:
    from .ayarlar import (
        CIKTI_KLASORU,
        DOGRULAMA_ORANI,
        EGITIM_ORANI,
        ON_ISLEME_VARSAYILAN_GIRIS_KLASORU,
        SCALING_METODU,
        SINIF_KLASORLERI,
        TEST_ORANI,
    )
    from .goruntu_isleyici import GorselIsleyici
    from .ozellik_cikarici import (
        OzellikCikarici,
        veri_boluntule,
        veri_setini_bol_ve_olceklendir,
    )

OPTIONAL_3D_MODULES = (
    "goruntu_isleme.uc_boyutlu_isleme",
    "goruntu_isleme.three_d_isleme",
    "uc_boyutlu_isleme",
    "three_d_isleme",
)


def _load_optional_3d_module():
    """Opsiyonel 3D modulunu varsa yukle."""
    for module_name in OPTIONAL_3D_MODULES:
        if importlib.util.find_spec(module_name) is None:
            continue
        return importlib.import_module(module_name)
    return None


def parse_args(argv=None):
    """Komut satiri argumanlarini ayrisirt."""
    parser = argparse.ArgumentParser(
        description="MRI goruntu isleme modulu",
    )
    parser.add_argument(
        "--action",
        choices=["menu", "preprocess", "extract", "clean-nan", "scale", "report", "split", "all"],
        default="menu",
        help="Calistirilacak islem. Varsayilan interaktif menu.",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="On isleme modu. 3D ancak opsiyonel modul varsa kullanilabilir.",
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Girdi klasoru")
    parser.add_argument("--output-dir", type=Path, default=None, help="Cikti klasoru")
    parser.add_argument("--csv-path", type=Path, default=None, help="CSV dosya yolu")
    parser.add_argument(
        "--method",
        choices=["drop", "mean", "median", "zero", "minmax", "robust", "standard", "maxabs"],
        default=None,
        help="Temizleme veya scaling metodu",
    )
    parser.add_argument("--volume-path", type=Path, default=None, help="3D hacim dosyasi veya DICOM klasoru")
    parser.add_argument("--class-name", type=str, default=None, help="3D islem icin sinif adi")
    parser.add_argument("--model-path", type=Path, default=None, help="3D model dosyasi")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Toplu islemlerde etkilesimli onay isteme.",
    )
    return parser.parse_args(argv)


def ana_menu():
    """Ana menuyu goster."""
    print("\n" + "=" * 60)
    print("MRI GORUNTU ISLEME SISTEMI")
    print("=" * 60)
    print("\n1. Goruntuleri on isle")
    if _load_optional_3d_module() is not None:
        print("   3D destek modulu algilandi")
    print("2. Ozellik cikar ve CSV olustur")
    print("3. CSV'deki NaN degerleri temizle")
    print("4. Veri setini bol + egitim setine gore olceklendir")
    print("5. Istatistik raporu goster")
    print("6. Veri setini bol (ham CSV)")
    print("7. TUM ISLEMLERI OTOMATIK YAP")
    print("0. Cikis")
    print("\n" + "=" * 60)


def _resolve_3d_module_or_warn():
    """3D modul varsa dondur, yoksa acik mesaj yaz."""
    module = _load_optional_3d_module()
    if module is None:
        print("[HATA] 3D islem modulu bu repoda kurulu degil.")
        print("       3D secenegi modul eklenmeden kullanilamaz.")
    return module


def _run_3d_inference(
    islem3d,
    hacim_yolu: Path | None,
    sinif_adi: str,
    model_yolu: Path | None = None,
    cikti_klasoru: Path | None = None,
):
    """Opsiyonel 3D akisini calistir."""
    if hacim_yolu is None:
        print("[HATA] 3D islem icin hacim yolu gerekli.")
        return None
    if sinif_adi not in SINIF_KLASORLERI:
        print(f"[HATA] Gecersiz sinif: {sinif_adi}")
        return None

    default_model_path = getattr(islem3d, "DEFAULT_MODEL_PATH", None)
    if model_yolu is None:
        model_yolu = default_model_path
    if cikti_klasoru is None:
        cikti_klasoru = CIKTI_KLASORU / "3d"

    if model_yolu is None:
        print("[HATA] 3D modulunde varsayilan model yolu tanimli degil.")
        return None

    try:
        kayit = islem3d.calistir_3d_infer(
            Path(hacim_yolu),
            model_path=Path(model_yolu),
            output_dir=Path(cikti_klasoru),
            sinif_adi=sinif_adi,
        )
        print(f"\n3D islem tamamlandi. Maske: {kayit}")
        return kayit
    except Exception as exc:
        print(f"[HATA] 3D islem basarisiz: {exc}")
        return None


def goruntu_on_isleme(
    giris_klasoru: Path | None = None,
    cikti_klasoru: Path | None = None,
    mode: str = "2d",
    volume_path: Path | None = None,
    class_name: str | None = None,
    model_path: Path | None = None,
):
    """Ham goruntuleri on isle."""
    print("\n[1] GORUNTU ON ISLEME")
    print("-" * 60)

    if mode == "3d":
        islem3d = _resolve_3d_module_or_warn()
        if islem3d is None:
            return None
        sinif_adi = class_name or SINIF_KLASORLERI[0]
        return _run_3d_inference(
            islem3d=islem3d,
            hacim_yolu=Path(volume_path) if volume_path else None,
            sinif_adi=sinif_adi,
            model_yolu=Path(model_path) if model_path else None,
            cikti_klasoru=Path(cikti_klasoru) if cikti_klasoru else None,
        )

    isleyici = GorselIsleyici()
    giris_klasoru = Path(giris_klasoru) if giris_klasoru else ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
    cikti_klasoru = Path(cikti_klasoru) if cikti_klasoru else CIKTI_KLASORU

    istatistikler = isleyici.tum_gorselleri_isle(cikti_klasoru, giris_klasoru=giris_klasoru)

    if istatistikler:
        print("\n[BASARILI] Goruntu isleme tamamlandi.")
    else:
        print("\n[HATA] Goruntu isleme basarisiz.")
    return istatistikler


def ozellik_cikar(giris_klasoru: Path | None = None, cikti_csv: Path | None = None):
    """Islenmis goruntulerden ozellik cikar."""
    print("\n[2] OZELLIK CIKARMA VE CSV OLUSTURMA")
    print("-" * 60)

    cikarici = OzellikCikarici()
    giris_klasoru = Path(giris_klasoru) if giris_klasoru else CIKTI_KLASORU

    df = cikarici.csv_olustur(giris_klasoru, cikti_csv=cikti_csv)
    if not df.empty:
        print("\n[BASARILI] Ozellik cikarimi tamamlandi.")
    else:
        print("\n[HATA] Ozellik cikarimi basarisiz.")
    return df


def nan_temizle(csv_dosyasi: Path | None = None, metod: str = "drop"):
    """CSV'deki NaN degerleri temizle."""
    print("\n[3] NaN DEGERLERI TEMIZLEME")
    print("-" * 60)

    cikarici = OzellikCikarici()
    df = cikarici.nan_temizle(csv_dosyasi=csv_dosyasi, metod=metod)
    if not df.empty:
        print("\n[BASARILI] NaN temizleme tamamlandi.")
    else:
        print("\n[HATA] NaN temizleme basarisiz.")
    return df


def scaling_uygula(
    csv_dosyasi: Path | None = None,
    cikti_klasoru: Path | None = None,
    metod: str = SCALING_METODU,
):
    """Leakage-free split ve scaling uygula."""
    print("\n[4] VERI BOLME + OLCEKLENDIRME")
    print("-" * 60)
    print("\nScaler once egitim setine fit edilir, sonra dogrulama ve teste uygulanir.")
    print(f"Mevcut metod: {metod}")

    splitler = veri_setini_bol_ve_olceklendir(
        csv_dosyasi=csv_dosyasi,
        cikti_klasoru=cikti_klasoru,
        metod=metod,
    )
    if splitler and all(not df.empty for df in splitler):
        print("\n[BASARILI] Veri bolme ve olceklendirme tamamlandi.")
    else:
        print("\n[HATA] Veri bolme ve olceklendirme basarisiz.")
    return splitler


def istatistik_goster(csv_dosyasi: Path | None = None):
    """Istatistik raporu goster."""
    print("\n[5] ISTATISTIK RAPORU")
    print("-" * 60)
    cikarici = OzellikCikarici()
    cikarici.istatistik_raporu(csv_dosyasi=csv_dosyasi)


def veri_bol(csv_dosyasi: Path | None = None, cikti_klasoru: Path | None = None):
    """Ham CSV uzerinden veri setini bol."""
    print("\n[6] VERI SETI BOLME")
    print("-" * 60)
    print(f"\nOranlar: Egitim={EGITIM_ORANI}, Dogrulama={DOGRULAMA_ORANI}, Test={TEST_ORANI}")
    return veri_boluntule(csv_dosyasi=csv_dosyasi, cikti_klasoru=cikti_klasoru)


def tum_islemleri_yap(
    giris_klasoru: Path | None = None,
    cikti_klasoru: Path | None = None,
    metod: str = SCALING_METODU,
    skip_confirmation: bool = False,
):
    """Tum islemleri otomatik ve guvenli sirada yap."""
    print("\n[7] TUM ISLEMLER OTOMATIK")
    print("-" * 60)
    print("\nSu islemler sirayla yapilacak:")
    print("  1. Goruntu on isleme")
    print("  2. Ozellik cikarma")
    print("  3. Veri bolme + egitim setine gore olceklendirme")
    print("  4. Istatistik raporu")

    if not skip_confirmation:
        onay = input("\nDevam etmek istiyor musunuz? (e/h): ").strip().lower()
    else:
        onay = "e"
    if onay != "e":
        print("Islem iptal edildi.")
        return None

    giris_klasoru = Path(giris_klasoru) if giris_klasoru else ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
    cikti_klasoru = Path(cikti_klasoru) if cikti_klasoru else CIKTI_KLASORU

    print("\n\n" + "=" * 60)
    print("ADIM 1/4: GORUNTU ON ISLEME")
    print("=" * 60)
    isleyici = GorselIsleyici()
    isleyici.tum_gorselleri_isle(cikti_klasoru, giris_klasoru=giris_klasoru)

    print("\n\n" + "=" * 60)
    print("ADIM 2/4: OZELLIK CIKARMA")
    print("=" * 60)
    cikarici = OzellikCikarici()
    df = cikarici.csv_olustur(cikti_klasoru)
    if df.empty:
        print("\n[HATA] Ozellik cikarimi basarisiz. Islem durduruluyor.")
        return None

    print("\n\n" + "=" * 60)
    print("ADIM 3/4: VERI BOLME + OLCEKLENDIRME")
    print("=" * 60)
    sonuc = veri_setini_bol_ve_olceklendir(cikti_klasoru=cikti_klasoru, metod=metod)
    if sonuc is None:
        print("\n[HATA] Veri bolme ve olceklendirme basarisiz. Islem durduruluyor.")
        return None

    print("\n\n" + "=" * 60)
    print("ADIM 4/4: ISTATISTIK RAPORU")
    print("=" * 60)
    cikarici.istatistik_raporu(csv_dosyasi=cikti_klasoru / "goruntu_ozellikleri.csv")

    print("\n\n" + "=" * 60)
    print("[BASARILI] Tum islemler tamamlandi.")
    print("=" * 60)
    return sonuc


def _interactive_preprocess():
    """Interaktif on isleme akisi."""
    three_d_module = _load_optional_3d_module()
    mode_prompt = "2d/3d" if three_d_module is not None else "2d"
    mode = input(f"\nIsleme modu ({mode_prompt}, Enter=2d): ").strip().lower() or "2d"
    if mode == "3d":
        if three_d_module is None:
            _resolve_3d_module_or_warn()
            return None

        hacim_yolu = input("3D hacim dosyasi (nii/nii.gz/npy) veya DICOM klasoru: ").strip()
        if not hacim_yolu:
            print("[HATA] Hacim yolu gerekli.")
            return None

        sinif_adi = input(f"Sinif adi ({'/'.join(SINIF_KLASORLERI)}): ").strip() or SINIF_KLASORLERI[0]
        default_model_path = getattr(three_d_module, "DEFAULT_MODEL_PATH", "")
        model_yolu_girdi = input(f"3D model yolu (varsayilan: {default_model_path}): ").strip()
        model_yolu = Path(model_yolu_girdi) if model_yolu_girdi else None

        cikti3d_girdi = input(f"3D cikti klasoru (varsayilan: {CIKTI_KLASORU / '3d'}): ").strip()
        cikti3d = Path(cikti3d_girdi) if cikti3d_girdi else CIKTI_KLASORU / "3d"
        return goruntu_on_isleme(
            cikti_klasoru=cikti3d,
            mode="3d",
            volume_path=Path(hacim_yolu),
            class_name=sinif_adi,
            model_path=model_yolu,
        )

    giris = input(f"\nGirdi klasoru (varsayilan: {ON_ISLEME_VARSAYILAN_GIRIS_KLASORU}): ").strip()
    cikis = input(f"Cikti klasoru (varsayilan: {CIKTI_KLASORU}): ").strip()
    return goruntu_on_isleme(
        giris_klasoru=Path(giris) if giris else None,
        cikti_klasoru=Path(cikis) if cikis else None,
        mode="2d",
    )


def _interactive_feature_extraction():
    """Interaktif ozellik cikarma akisi."""
    giris = input(f"\nIslenmis goruntuler klasoru (varsayilan: {CIKTI_KLASORU}): ").strip()
    return ozellik_cikar(giris_klasoru=Path(giris) if giris else None)


def _interactive_nan_clean():
    """Interaktif NaN temizleme akisi."""
    print("\nMevcut temizleme metodlari:")
    print("  1. drop   - NaN iceren satirlari cikar")
    print("  2. mean   - NaN'lari sutun ortalamasi ile doldur")
    print("  3. median - NaN'lari sutun medyani ile doldur")
    print("  4. zero   - NaN'lari 0 ile doldur")

    metod = input("\nMetod secin (drop/mean/median/zero, Enter=drop): ").strip().lower() or "drop"
    if metod not in ["drop", "mean", "median", "zero"]:
        print("[HATA] Gecersiz metod.")
        return None
    return nan_temizle(metod=metod)


def _interactive_scaling():
    """Interaktif scaling akisi."""
    print(f"Mevcut metod: {SCALING_METODU}")
    print("\nMetodlar:")
    print("  1. minmax")
    print("  2. robust")
    print("  3. standard")
    print("  4. maxabs")

    metod = input("\nMetod secin (minmax/robust/standard/maxabs, Enter=varsayilan): ").strip().lower() or SCALING_METODU
    if metod not in ["minmax", "robust", "standard", "maxabs"]:
        print("[HATA] Gecersiz metod.")
        return None
    return scaling_uygula(metod=metod)


def run_action(args):
    """CLI action'i calistir."""
    if args.action == "preprocess":
        return goruntu_on_isleme(
            giris_klasoru=args.input_dir,
            cikti_klasoru=args.output_dir,
            mode=args.mode,
            volume_path=args.volume_path,
            class_name=args.class_name,
            model_path=args.model_path,
        )
    if args.action == "extract":
        return ozellik_cikar(giris_klasoru=args.input_dir, cikti_csv=args.csv_path)
    if args.action == "clean-nan":
        return nan_temizle(csv_dosyasi=args.csv_path, metod=args.method or "drop")
    if args.action == "scale":
        return scaling_uygula(
            csv_dosyasi=args.csv_path,
            cikti_klasoru=args.output_dir,
            metod=args.method or SCALING_METODU,
        )
    if args.action == "report":
        return istatistik_goster(csv_dosyasi=args.csv_path)
    if args.action == "split":
        return veri_bol(csv_dosyasi=args.csv_path, cikti_klasoru=args.output_dir)
    if args.action == "all":
        return tum_islemleri_yap(
            giris_klasoru=args.input_dir,
            cikti_klasoru=args.output_dir,
            metod=args.method or SCALING_METODU,
            skip_confirmation=args.yes,
        )
    return None


def main(argv=None):
    """Ana program."""
    args = parse_args(argv)
    if args.action != "menu":
        sonuc = run_action(args)
        return 0 if sonuc is not None else 1

    while True:
        try:
            ana_menu()
            secim = input("\nSeciminiz: ").strip()

            if secim == "0":
                print("\nCikiliyor...")
                break
            if secim == "1":
                _interactive_preprocess()
            elif secim == "2":
                _interactive_feature_extraction()
            elif secim == "3":
                _interactive_nan_clean()
            elif secim == "4":
                _interactive_scaling()
            elif secim == "5":
                istatistik_goster()
            elif secim == "6":
                veri_bol()
            elif secim == "7":
                tum_islemleri_yap()
            else:
                print("\n[HATA] Gecersiz secim. Lutfen 0-7 arasi bir sayi girin.")

            input("\nDevam etmek icin Enter'a basin...")

        except KeyboardInterrupt:
            print("\n\nProgram kullanici tarafindan durduruldu.")
            break
        except Exception as exc:
            print(f"\n[HATA] Beklenmeyen hata: {exc}")
            import traceback

            traceback.print_exc()
            input("\nDevam etmek icin Enter'a basin...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
