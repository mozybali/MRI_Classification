#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MRI goruntu isleme ana menu ve CLI giris noktasi."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from goruntu_isleme.ayarlar import (
        CIKTI_KLASORU,
        ON_ISLEME_VARSAYILAN_GIRIS_KLASORU,
    )
    from goruntu_isleme.goruntu_isleyici import GorselIsleyici
else:
    from .ayarlar import (
        CIKTI_KLASORU,
        ON_ISLEME_VARSAYILAN_GIRIS_KLASORU,
    )
    from .goruntu_isleyici import GorselIsleyici


def parse_args(argv=None):
    """Komut satiri argumanlarini ayrisirt."""
    parser = argparse.ArgumentParser(
        description="MRI goruntu isleme modulu",
    )
    parser.add_argument(
        "--action",
        choices=["menu", "preprocess"],
        default="menu",
        help="Calistirilacak islem. Varsayilan interaktif menu.",
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Girdi klasoru")
    parser.add_argument("--output-dir", type=Path, default=None, help="Cikti klasoru")
    return parser.parse_args(argv)


def ana_menu():
    """Ana menuyu goster."""
    print("\n" + "=" * 60)
    print("MRI GORUNTU ISLEME SISTEMI")
    print("=" * 60)
    print("\n1. Goruntuleri on isle")
    print("0. Cikis")
    print("\n" + "=" * 60)


def goruntu_on_isleme(
    giris_klasoru: Path | None = None,
    cikti_klasoru: Path | None = None,
):
    """Ham goruntuleri on isle."""
    print("\n[1] GORUNTU ON ISLEME")
    print("-" * 60)

    isleyici = GorselIsleyici()
    giris_klasoru = Path(giris_klasoru) if giris_klasoru else ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
    cikti_klasoru = Path(cikti_klasoru) if cikti_klasoru else CIKTI_KLASORU

    splitli_islem = getattr(isleyici, "tum_gorselleri_isle_ve_bol", None)
    if callable(splitli_islem):
        istatistikler = splitli_islem(cikti_klasoru, giris_klasoru=giris_klasoru)
    else:
        istatistikler = isleyici.tum_gorselleri_isle(cikti_klasoru, giris_klasoru=giris_klasoru)

    if istatistikler:
        print("\n[BASARILI] Goruntu isleme tamamlandi.")
    else:
        print("\n[HATA] Goruntu isleme basarisiz.")
    return istatistikler


def _interactive_preprocess():
    """Interaktif on isleme akisi."""
    giris = input(f"\nGirdi klasoru (varsayilan: {ON_ISLEME_VARSAYILAN_GIRIS_KLASORU}): ").strip()
    cikis = input(f"Cikti klasoru (varsayilan: {CIKTI_KLASORU}): ").strip()
    return goruntu_on_isleme(
        giris_klasoru=Path(giris) if giris else None,
        cikti_klasoru=Path(cikis) if cikis else None,
    )


def run_action(args):
    """CLI action'i calistir."""
    if args.action == "preprocess":
        return goruntu_on_isleme(
            giris_klasoru=args.input_dir,
            cikti_klasoru=args.output_dir,
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
            else:
                print("\n[HATA] Gecersiz secim. Lutfen 0 veya 1 girin.")

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
