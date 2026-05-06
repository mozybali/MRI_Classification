"""GorselIsleyici durum yonetimi ve ortak yardimcilari."""

import random
import re
from itertools import count
from multiprocessing import cpu_count, current_process
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselIsleyiciTemel:
    _isleyici_sayaci = count()

    def __init__(self):
        """İşleyiciyi başlat."""
        self.temel_tohum = RASTGELE_TOHUM
        self._random, self._np_random = self._rng_olustur(self.temel_tohum)
        self.template_image = None  # Registration için şablon görüntü
        self.son_egim_analizi = {}
        self.kalite_istatistikleri = {
            "toplam": 0,
            "basarili": 0,
            "kalite_hatasi": 0,
            "kaydetme_hatasi": 0,
            "kenar_artefakt_tespit": 0,
            "kenar_artefakt_temizlendi": 0,
            "egim_tespit": 0,
            "egim_duzeltildi": 0,
            "egim_gorsel_kontrol_adayi": 0,
            "egim_kalite_red": 0,
        }
        self.n_jobs = max(1, cpu_count() - 1)  # Bir çekirdek sisteme bırak

    @staticmethod
    def _worker_kimligi() -> int:
        """Worker bazli sabit bir kimlik dondur."""
        kimlik = getattr(current_process(), "_identity", ())
        return int(kimlik[0]) if kimlik else 0

    @classmethod
    def _benzersiz_tohum_uret(cls, temel_tohum: int) -> int:
        """Her instance icin ayri ama tekrar edilebilir bir tohum uret."""
        worker_kimligi = cls._worker_kimligi()
        instance_idx = next(cls._isleyici_sayaci)
        return int(temel_tohum + worker_kimligi * 10000 + instance_idx)

    @classmethod
    def _rng_olustur(cls, temel_tohum: int):
        """Instance icin ayrik Python ve NumPy RNG nesneleri olustur."""
        seed = cls._benzersiz_tohum_uret(temel_tohum)
        return random.Random(seed), np.random.default_rng(seed)

    @staticmethod
    def tohum_ayarla(tohum: int = RASTGELE_TOHUM):
        """Rastgelelik tohumu ayarla."""
        random.seed(tohum)
        np.random.seed(tohum)

    @staticmethod
    def _cikti_dosya_koku(dosya_yolu: str) -> str:
        """Cikti dosya kokunu giris adini benzersiz koruyacak sekilde uret."""
        kaynak = Path(dosya_yolu)
        uzanti = kaynak.suffix.lower().lstrip(".")
        if not uzanti:
            return kaynak.stem
        return f"{kaynak.stem}_{uzanti}"

    @staticmethod
    def kaynak_id_belirle(dosya_yolu: str) -> str:
        """Ayni kaynaktan tureyen dosyalari leak-free split icin grupla."""
        stem = Path(str(dosya_yolu)).stem
        stem = re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"\s*\(\d+\)$", "", stem)
        return stem

    @staticmethod
    def _sinif_kapsamini_dogrula(
        istatistikler: Dict[str, int],
        beklenen_siniflar: List[str],
        split_adi: str,
    ):
        """Kalite kontrol sonrasi split'in beklenen tum siniflari korudugunu dogrula."""
        eksik = sorted(
            sinif for sinif in beklenen_siniflar if int(istatistikler.get(sinif, 0)) <= 0
        )
        if eksik:
            raise ValueError(
                f"{split_adi} split'inde sinif kapsami eksik. "
                f"Kalite kontrol veya on isleme sonrasi goruntu kalmayan siniflar: {eksik}"
            )
    
    @staticmethod
    def klasor_olustur(yol: Path):
        """Klasör yoksa oluştur."""
        yol.mkdir(parents=True, exist_ok=True)
