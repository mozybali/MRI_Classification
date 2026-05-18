"""
goruntu_isleyici.py
-------------------
MRI goruntulerini isleme modulunun geriye donuk uyumlu giris noktasi.
"""

import sys
import types

try:
    from .ayarlar import *
    from . import temel as _temel_mod
    from . import veri as _veri_mod
    from . import kalite_io as _kalite_io_mod
    from . import on_isleme as _on_isleme_mod
    from . import kalite_analizi as _kalite_analizi_mod
    from . import temizlik_normalize as _temizlik_normalize_mod
    from . import boyutlandirma as _boyutlandirma_mod
    from . import gurultu as _gurultu_mod
    from . import skull_strip as _skull_strip_mod
    from . import bias_alignment as _bias_alignment_mod
    from . import pipeline as _pipeline_mod
    from . import toplu_islem as _toplu_islem_mod
    from .temel import GorselIsleyiciTemel
    from .veri import GorselVeriMixin
    from .kalite_io import GorselKaliteIOMixin
    from .on_isleme import GorselOnIslemeMixin
    from .toplu_islem import GorselTopluIslemMixin, _islem_worker_init, _islem_wrapper
except ImportError:
    from ayarlar import *
    import temel as _temel_mod
    import veri as _veri_mod
    import kalite_io as _kalite_io_mod
    import on_isleme as _on_isleme_mod
    import kalite_analizi as _kalite_analizi_mod
    import temizlik_normalize as _temizlik_normalize_mod
    import boyutlandirma as _boyutlandirma_mod
    import gurultu as _gurultu_mod
    import skull_strip as _skull_strip_mod
    import bias_alignment as _bias_alignment_mod
    import pipeline as _pipeline_mod
    import toplu_islem as _toplu_islem_mod
    from temel import GorselIsleyiciTemel
    from veri import GorselVeriMixin
    from kalite_io import GorselKaliteIOMixin
    from on_isleme import GorselOnIslemeMixin
    from toplu_islem import GorselTopluIslemMixin, _islem_worker_init, _islem_wrapper

# Eski tek-dosya modulunde dogrudan gorunen bagimlilik adlari korunur.
cv2 = getattr(_on_isleme_mod, "cv2", None)
CV2_AVAILABLE = True
Pool = _toplu_islem_mod.Pool
tqdm = _toplu_islem_mod.tqdm

_PATCH_TARGET_MODULES = (
    _temel_mod,
    _veri_mod,
    _kalite_io_mod,
    _on_isleme_mod,
    _kalite_analizi_mod,
    _temizlik_normalize_mod,
    _boyutlandirma_mod,
    _gurultu_mod,
    _skull_strip_mod,
    _bias_alignment_mod,
    _pipeline_mod,
    _toplu_islem_mod,
)

class GorselIsleyici(
    GorselTopluIslemMixin,
    GorselOnIslemeMixin,
    GorselKaliteIOMixin,
    GorselVeriMixin,
    GorselIsleyiciTemel,
):
    """MRI goruntu isleme sinifi."""


class _UyumluModul(types.ModuleType):
    """Testlerdeki module-level monkeypatch'leri parcalara da yay."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _PATCH_TARGET_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)

sys.modules[__name__].__class__ = _UyumluModul

__all__ = ["GorselIsleyici"]
