#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""EDA analizi calistirma scripti."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERI_KLASORU = PROJECT_ROOT / "Veri_Seti" / "OriginalDataset"
DEFAULT_CIKTI_KLASORU = Path(__file__).resolve().parent / "eda_ciktilar"
EDAAnaliz = None

if __package__ in {None, ""}:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _guvenli_print(*args, sep: str = " ", end: str = "\n") -> None:
    # Ağır bağımlılıklar yüklenememişse plain print'e düş.
    try:
        if __package__ in {None, ""}:
            from eda_analiz.eda_araclar import _guvenli_print as _impl
        else:
            from .eda_araclar import _guvenli_print as _impl
    except Exception:
        print(*args, sep=sep, end=end)
        return
    _impl(*args, sep=sep, end=end)


def _load_eda_analiz_sinifi():
    """Ağır EDA bağımlılıklarını sadece analiz çalışırken yükle."""
    global EDAAnaliz
    if EDAAnaliz is not None:
        return EDAAnaliz

    if __package__ in {None, ""}:
        from eda_analiz.eda_araclar import EDAAnaliz as analiz_sinifi
    else:
        from .eda_araclar import EDAAnaliz as analiz_sinifi

    EDAAnaliz = analiz_sinifi
    return analiz_sinifi


def parse_args(argv=None):
    """CLI argumanlarini ayrisirt."""
    parser = argparse.ArgumentParser(
        description="MRI veri seti icin EDA analizi calistir",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Veri klasoru. Bos birakilirsa varsayilan dizin kullanilir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cikti klasoru. Bos birakilirsa varsayilan dizin kullanilir.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Eksik argumanlari soru-cevap ile tamamla.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Istatistik hesaplamada kullanilacak cekirdek sayisi. Varsayilan: otomatik.",
    )
    return parser.parse_args(argv)


def _resolve_paths(args) -> tuple[Path, Path]:
    """Varsayilan ve interaktif seceneklere gore yollari belirle."""
    veri_klasoru = args.data_dir
    cikti_klasoru = args.output_dir
    should_prompt = args.interactive

    if should_prompt:
        veri_girdi = input(
            f"Veri seti klasoru (Enter=varsayilan: {DEFAULT_VERI_KLASORU}): "
        ).strip()
        if veri_girdi:
            veri_klasoru = Path(veri_girdi).expanduser()

        cikti_girdi = input(
            f"Cikti klasoru (Enter=varsayilan: {DEFAULT_CIKTI_KLASORU}): "
        ).strip()
        if cikti_girdi:
            cikti_klasoru = Path(cikti_girdi).expanduser()

    if veri_klasoru is None:
        veri_klasoru = DEFAULT_VERI_KLASORU
    if cikti_klasoru is None:
        cikti_klasoru = DEFAULT_CIKTI_KLASORU

    return Path(veri_klasoru), Path(cikti_klasoru)


def main(argv=None):
    """Ana program."""
    args = parse_args(argv)
    veri_klasoru, cikti_klasoru = _resolve_paths(args)

    print("\nMRI Veri Seti EDA Analizi Baslatiliyor...\n")

    try:
        analiz_sinifi = _load_eda_analiz_sinifi()
        analizci = analiz_sinifi(
            veri_klasoru=veri_klasoru,
            cikti_klasoru=cikti_klasoru,
            n_jobs=args.jobs,
        )

        df = analizci.tam_analiz_yap()

        csv_yolu = Path(cikti_klasoru) / "veri_seti_istatistikler.csv"
        df.to_csv(csv_yolu, index=False, encoding="utf-8")
        _guvenli_print(f"\n[OK] Veri seti CSV kaydedildi: {csv_yolu}")
    except Exception as exc:
        _guvenli_print(f"\n[HATA] Analiz sirasinda hata olustu: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
