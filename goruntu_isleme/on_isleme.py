"""Tekil MRI goruntusu on isleme: alt mixin'leri birlesik bir mixin'e toplar.

Esas implementasyon su yan dosyalara bolunmustur:
- kalite_analizi.py   : ortak yardimcilar ve PipelineSonucu kontrati
- temizlik_normalize.py: kenar artefakt temizleme, yogunluk/CLAHE/z-score
- boyutlandirma.py    : resize + padding stratejileri
- gurultu.py          : median/gaussian/bilateral gurultu giderme
- skull_strip.py      : foreground/Otsu maske yardimcilari
- bias_alignment.py   : center-of-mass tabanli basit hizalama
- pipeline.py         : pipeline orkestrasyonu ve normalizasyon stratejisi

Bu modul yalnizca:
- `PipelineSonucu` ve `GorselOnIslemeMixin` icin geriye donuk giris noktasi
- `cv2` modul duzeyinde tutar (giris modulunun kose tasi sozlesmesi icin).
"""

import cv2  # noqa: F401

try:
    from .kalite_analizi import PipelineSonucu, GorselKaliteAnaliziMixin
    from .temizlik_normalize import GorselTemizlikNormalizeMixin
    from .boyutlandirma import GorselBoyutlandirmaMixin
    from .gurultu import GorselGurultuMixin
    from .skull_strip import GorselSkullStripMixin
    from .bias_alignment import GorselBiasAlignmentMixin
    from .pipeline import GorselPipelineMixin
except ImportError:
    from kalite_analizi import PipelineSonucu, GorselKaliteAnaliziMixin
    from temizlik_normalize import GorselTemizlikNormalizeMixin
    from boyutlandirma import GorselBoyutlandirmaMixin
    from gurultu import GorselGurultuMixin
    from skull_strip import GorselSkullStripMixin
    from bias_alignment import GorselBiasAlignmentMixin
    from pipeline import GorselPipelineMixin


class GorselOnIslemeMixin(
    GorselPipelineMixin,
    GorselBiasAlignmentMixin,
    GorselSkullStripMixin,
    GorselGurultuMixin,
    GorselBoyutlandirmaMixin,
    GorselTemizlikNormalizeMixin,
    GorselKaliteAnaliziMixin,
):
    """Alt-mixin'leri tek noktada birlestiren on isleme mixin'i."""


__all__ = ["PipelineSonucu", "GorselOnIslemeMixin"]
