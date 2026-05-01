"""
goruntu_isleyici.py
-------------------
MRI goruntulerini isleme ve ozellik cikarma modulunun geriye donuk
uyumlu giris noktasi.
"""

import sys
import types

try:
    from .ayarlar import *
    from . import temel as _temel_mod
    from . import veri as _veri_mod
    from . import kalite_io as _kalite_io_mod
    from . import on_isleme as _on_isleme_mod
    from . import artirma as _artirma_mod
    from . import toplu_islem as _toplu_islem_mod
    from .temel import GorselIsleyiciTemel
    from .veri import GorselVeriMixin
    from .kalite_io import GorselKaliteIOMixin
    from .on_isleme import GorselOnIslemeMixin
    from .artirma import GorselArtirmaMixin
    from .toplu_islem import GorselTopluIslemMixin, _islem_worker_init, _islem_wrapper
except ImportError:
    from ayarlar import *
    import temel as _temel_mod
    import veri as _veri_mod
    import kalite_io as _kalite_io_mod
    import on_isleme as _on_isleme_mod
    import artirma as _artirma_mod
    import toplu_islem as _toplu_islem_mod
    from temel import GorselIsleyiciTemel
    from veri import GorselVeriMixin
    from kalite_io import GorselKaliteIOMixin
    from on_isleme import GorselOnIslemeMixin
    from artirma import GorselArtirmaMixin
    from toplu_islem import GorselTopluIslemMixin, _islem_worker_init, _islem_wrapper

# Eski tek-dosya modulunde dogrudan gorunen bagimlilik bayraklari korunur.
cv2 = getattr(_on_isleme_mod, "cv2", None)
CV2_AVAILABLE = _on_isleme_mod.CV2_AVAILABLE
exposure = getattr(_on_isleme_mod, "exposure", None)
SKIMAGE_AVAILABLE = _on_isleme_mod.SKIMAGE_AVAILABLE
sitk = getattr(_on_isleme_mod, "sitk", None)
SITK_AVAILABLE = _on_isleme_mod.SITK_AVAILABLE
Pool = _toplu_islem_mod.Pool
tqdm = _toplu_islem_mod.tqdm

_PATCH_TARGET_MODULES = (
    _temel_mod,
    _veri_mod,
    _kalite_io_mod,
    _on_isleme_mod,
    _artirma_mod,
    _toplu_islem_mod,
)

class GorselIsleyici(
    GorselTopluIslemMixin,
    GorselArtirmaMixin,
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
