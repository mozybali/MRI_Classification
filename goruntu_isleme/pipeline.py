"""Tekil goruntu pipeline orkestrasyonu ve normalizasyon stratejisi."""

from typing import Optional, Tuple

import numpy as np

try:
    from .ayarlar import *
    from .kalite_analizi import PipelineSonucu
except ImportError:
    from ayarlar import *
    from kalite_analizi import PipelineSonucu


class GorselPipelineMixin:
    def _pipeline_sonu_kalite_kontrol(self, goruntu: np.ndarray) -> Tuple[bool, str]:
        """Pipeline sonunda sessizce bozulmus goruntuleri yakala.

        - Tamamen siyah / cok kucuk foreground / cok dusuk foreground std
          ureten ciktilar dataset'e girmemeli.
        """
        on_ok, on_mesaji = self._temel_goruntu_on_kontrol(goruntu)
        if not on_ok:
            return False, on_mesaji

        arr = self._uint8_goruntu(goruntu)
        if arr.ndim != 2 or arr.size == 0:
            return False, "Geçersiz görüntü boyutu"

        mask = arr > 0
        min_piksel = self._min_foreground_piksel_sayisi(arr.size)
        if int(mask.sum()) < min_piksel:
            return False, "Yetersiz foreground"

        foreground_std = float(arr[mask].std())
        if foreground_std < float(MIN_STD_INTENSITY):
            return False, f"Düşük foreground kontrast (std={foreground_std:.1f})"

        return True, ""

    def goruntu_isle_sonuc(self, dosya_yolu: str) -> PipelineSonucu:
        """
        Tek bir goruntuye tam on isleme pipeline uygula ve yapisal sonuc don.

        Pipeline stratejileri (NORMALIZASYON_STRATEJISI ayarından):
        - "minimal": Sadece percentile clipping + resize
        - "standard": percentile + CLAHE + resize (önerilen)
        - "aggressive": percentile + CLAHE + z-score + resize

        Pipeline sırası:
        1. Görüntü yükle (gri ton)
        2. Temel boş/bozuk görüntü ön kontrolü
        3. Kenar artefakt tespiti (raporlama)
        4. Kenar artefakt temizligi
        5. Strict kalite kontrol
        6. Gürültü giderme (bilateral)
        7. Center-of-mass hizalama
        8. Strateji bazlı normalizasyon (percentile + CLAHE)
        9. Boyutlandırma (192x192, en-boy korumalı padding)
        10. Pipeline sonu kalite kontrol

        Args:
            dosya_yolu: Görüntü dosyasının yolu

        Returns:
            processed_image, quality_rejected ve quality_reason alanlarini
            iceren sonuc sozlugu.
        """
        # 1. Görüntüyü yükle
        goruntu = self.goruntu_yukle(dosya_yolu)
        if goruntu is None:
            return self._pipeline_sonucu()

        # 2. Temel on kontrol
        on_kontrol_ok, on_kontrol_mesaji = self._temel_goruntu_on_kontrol(goruntu)
        if not on_kontrol_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {on_kontrol_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        # 3. Kenar artefakt tespiti (raporlama amacli; reddetmez)
        if KENAR_ARTEFAKT_KONTROL_AKTIF:
            analiz = self.kenar_artefakt_analiz(self._uint8_goruntu(goruntu))
            if bool(analiz.get("artefakt_var", False)):
                self.kalite_istatistikleri["kenar_artefakt_tespit"] = (
                    self.kalite_istatistikleri.get("kenar_artefakt_tespit", 0) + 1
                )

        # 4. Kenar artefakt temizligi - normalize ve CLAHE'den ONCE.
        kenar_temizlendi = False
        if KENAR_ARTEFAKT_TEMIZLEME_AKTIF:
            oncesi_u8 = self._uint8_goruntu(goruntu)
            temiz_goruntu = self.kenar_artefakt_temizle(oncesi_u8)
            if not np.array_equal(temiz_goruntu, oncesi_u8):
                kenar_temizlendi = True
            goruntu = temiz_goruntu

        # 5. Strict kalite kontrol
        kalite_ok, hata_mesaji = self.goruntu_kalite_kontrol(goruntu)
        if not kalite_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {hata_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        # Temizlik sonrasi tamamen bosalan goruntuleri ele
        on_kontrol_ok, on_kontrol_mesaji = self._temel_goruntu_on_kontrol(goruntu)
        if not on_kontrol_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {on_kontrol_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        if kenar_temizlendi:
            self.kalite_istatistikleri["kenar_artefakt_temizlendi"] = (
                self.kalite_istatistikleri.get("kenar_artefakt_temizlendi", 0) + 1
            )

        # 6. Gürültü giderme
        goruntu = self.gurultu_gider(goruntu, metod='auto')

        # 7. Center-of-mass hizalama
        goruntu = self.center_of_mass_alignment(goruntu)

        # 8. Strateji bazlı normalizasyon
        goruntu = self._apply_normalization_strategy(goruntu)

        # 9. Boyutlandırma
        goruntu = self.boyutlandir(goruntu)

        # 10. Pipeline sonu kalite kontrol
        pipeline_ok, pipeline_mesaji = self._pipeline_sonu_kalite_kontrol(goruntu)
        if not pipeline_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: pipeline sonu reddi - {pipeline_mesaji}")
            self.kalite_istatistikleri["pipeline_sonu_red"] = (
                self.kalite_istatistikleri.get("pipeline_sonu_red", 0) + 1
            )
            return self._pipeline_sonucu(
                processed_image=None,
                quality_rejected=True,
                quality_reason="pipeline_sonu_red",
            )

        return self._pipeline_sonucu(goruntu)

    def goruntu_isle(self, dosya_yolu: str) -> Optional[np.ndarray]:
        """
        Tek bir goruntuye tam on isleme pipeline uygula.

        Geriye donuk API: kalite reddi dahil normal ciktiya alinmayacak
        goruntuler icin None, aksi halde islenmis goruntu dondurur.
        """
        sonuc = self.goruntu_isle_sonuc(dosya_yolu)
        if bool(sonuc.get("quality_rejected", False)):
            return None
        return sonuc.get("processed_image")

    def _apply_normalization_strategy(self, goruntu: np.ndarray) -> np.ndarray:
        """Seçilen normalizasyon stratejisini uygula."""
        strategy = NORMALIZASYON_STRATEJISI

        if strategy == "minimal":
            goruntu = self.yogunluk_normalize(goruntu)
        elif strategy == "standard":
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)
        elif strategy == "aggressive":
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)
            goruntu = self.z_score_normalize(goruntu)
        else:
            print(f"[UYARI] Bilinmeyen strateji: {strategy}, 'standard' kullanılıyor")
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)

        return goruntu
