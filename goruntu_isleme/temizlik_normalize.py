"""Kenar artefakt temizleme ve yogunluk/CLAHE/z-score normalizasyonu."""

import cv2
import numpy as np

try:
    from .ayarlar import *
    from . import opencv_duzeltmeler as _cv_yardimci
except ImportError:
    from ayarlar import *
    import opencv_duzeltmeler as _cv_yardimci


class GorselTemizlikNormalizeMixin:
    # Z-score'u uint8'e geri eslerken kullanilan +/- aralik (sigma cinsinden).
    # +/- 2.5 sigma disindaki degerler kirpilarak [0, 255]'e dogrusal eslenir;
    # boylece "*50 + 128" gibi sihirli sabitler yerine acik bir kontrat olur.
    _Z_SCORE_RANGE_SIGMA = 2.5

    def kenar_artefakt_temizle(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Kenar seritlerinde yogunlasmis parlak artefaktlari konservatif
        bicimde temizle.

        - Sadece KENAR_ARTEFAKT_TEMIZLEME_AKTIF True ise calisir.
        - Yogunluk normalize ve CLAHE'den ONCE cagrilmalidir.
        - Iki adimli temizleme kullanilir:
            1. Tum bilesen silme: Bir parlak baglantili bilesenin piksellerinin
               >= KENAR_BILESEN_SERIT_PAY_ESIGI orani kenar seritlerinde
               kaliyorsa, bilesen tamamen silinir (saf kenar artefakti).
            2. Kismi serit silme: Bilesen merkezdeki parlak yapilara
               (ornegin korteks) baglandigi icin (1) kosulu saglanmaz ama
               yine de seritte anlamli payi varsa, bilesenin SADECE serit
               icindeki piksellerini sil;
               merkezdeki anatomik kismi koru. Bu adim ozellikle ust/alt
               kenar parlak bantlarinin saturasyonlu beyin dokusuna
               baglandigi gercek dunya MRI vakalarinda kritiktir.
        - Tek bir bilesen goruntunun %50'sinden buyukse anatomik kabul
          edilir ve hicbir adim uygulanmaz.
        - Silinen pikseller KENAR_TEMIZLEME_DEGERI ile doldurulur.
        - Cikti uint8 ve deterministik.

        Args:
            goruntu: Gri tonlamali 2D numpy.ndarray.

        Returns:
            Kenar artefaktlari temizlenmis uint8 goruntu (girdiyle ayni
            sekilde). Temizleme kapali ise girdinin uint8 kopyasini doner.
        """
        if goruntu is None:
            return goruntu

        arr = self._uint8_goruntu(goruntu)
        if not KENAR_ARTEFAKT_TEMIZLEME_AKTIF or arr.ndim != 2:
            return arr

        h, w = arr.shape[:2]
        kalinlik_y, kalinlik_x = self._kenar_serit_kalinliklari((h, w))
        parlak_esigi = int(KENAR_PARLAKLIK_ESIGI)

        parlak_u8 = _cv_yardimci.parlak_maske(arr, parlak_esigi)
        if not parlak_u8.any():
            return arr

        num_labels, labels, stats, _ = _cv_yardimci.baglantili_bilesenler(
            parlak_u8, connectivity=8
        )
        if num_labels <= 1:
            return arr

        # Kenar seritleri maskesi: ust + alt + sol + sag.
        serit_maskesi = _cv_yardimci.kenar_serit_maskesi(
            (h, w), kalinlik_y, kalinlik_x
        )

        # Her etiket icin serit icindeki piksel sayisi (label 0 = arka plan).
        serit_sayilari = np.bincount(
            labels[serit_maskesi].ravel(), minlength=num_labels
        )

        toplam_alan = float(h * w)
        # Kismi temizleme tetigi: serit icindeki parlak piksel sayisi
        # ayarlarla belirlenen mutlak/esitsel siniri gecerse tetiklenir.
        # Boylece kucuk gurultu bilesenlerinin serit-icindeki birkac
        # pikselini silmeyiz; sadece anlamli kenar bantlarini hedefleriz.
        kismi_min_oran = max(0.0, float(KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI))
        kismi_min_mutlak = max(0, int(KENAR_KISMI_TEMIZLEME_MIN_PIXEL))
        kismi_min_pixel = max(kismi_min_mutlak, int(kismi_min_oran * toplam_alan))

        cikti = arr.copy()
        temizleme_degeri = int(KENAR_TEMIZLEME_DEGERI)
        # Anatomik koruma esigi: bilesen alani / toplam alan bu degeri
        # asarsa bilesen anatomik kabul edilir ve dokunulmaz.
        bilesen_max_oran = self._oran_ayari_dogrula(
            "KENAR_ANATOMI_KORUMA_ORANI",
            KENAR_ANATOMI_KORUMA_ORANI,
            0.5,
            0.0,
            1.0,
            clamp=True,
        )
        serit_pay_esigi = float(KENAR_BILESEN_SERIT_PAY_ESIGI)

        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            alan = int(stats[label_idx, cv2.CC_STAT_AREA])

            if alan <= 0 or alan / toplam_alan > bilesen_max_oran:
                continue

            serit_pay_pixel = int(serit_sayilari[label_idx])
            serit_pay = float(serit_pay_pixel) / float(alan)
            kenara_degiyor = (
                x == 0 or y == 0 or x + bw >= w or y + bh >= h
            )

            if serit_pay >= serit_pay_esigi or (
                kenara_degiyor and serit_pay_pixel == alan
            ):
                # 1) Tam bilesen silme: bilesen tamamen/agirlikli olarak
                #    kenar seridinde kalir, merkez anatomiyi temsil etmez.
                cikti[labels == label_idx] = temizleme_degeri
            elif serit_pay_pixel >= kismi_min_pixel or kenara_degiyor:
                # 2) Kismi silme: bilesen anatomik yapiya bagli ama
                #    serit icinde anlamli pay'i var -> sadece serit-icindeki
                #    pikselleri sil, anatomik kismi koru.
                kismi_maske = (labels == label_idx) & serit_maskesi
                cikti[kismi_maske] = temizleme_degeri

        return cikti

    def yogunluk_normalize(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Görüntü yoğunluğunu normalize et.

        Bu fonksiyon, görüntüdeki aşırı karanlık ve aşırı aydınlık pikselleri
        belirli yüzdeliklere göre kırpar ve 0-255 aralığına ölçeklendirir.
        Böylece görüntü kontrastı iyileştirilir.

        Args:
            goruntu: Girdi görüntüsü (numpy array)

        Returns:
            Normalize edilmiş görüntü (uint8, 0-255 arası)
        """
        # Geçersiz girdi kontrolü
        if goruntu is None or goruntu.size == 0:
            raise ValueError("Geçersiz görüntü")

        arr = np.asarray(goruntu)
        arr_float = arr.astype(np.float32)

        # Percentile istatistiklerini arka plan yerine foreground/beyin
        # aday maskesi icinden hesapla. Boylece buyuk siyah arka planlar
        # kirpma esiklerini domine etmez.
        maske = self._foreground_maskesi(arr)
        min_piksel = self._min_foreground_piksel_sayisi(arr.size)
        arka_plan_maskesi = None

        if maske.shape == arr.shape and int(maske.sum()) >= min_piksel:
            degerler = arr_float[maske]
            arka_plan_maskesi = ~maske
        else:
            u8 = self._uint8_goruntu(arr)
            nonzero = u8 > 0
            nonzero_sayisi = int(nonzero.sum())
            if nonzero_sayisi == 0:
                return np.zeros_like(arr, dtype=np.uint8)
            if nonzero_sayisi < min_piksel:
                return np.zeros_like(arr, dtype=np.uint8)
            degerler = arr_float[nonzero]
            arka_plan_maskesi = ~nonzero

        if degerler.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)

        # Alt ve üst yüzdelik değerlerini al (örn: %1 ve %99)
        alt_yuzde, ust_yuzde = KIRPMA_YUZDELERI
        alt_deger = float(np.percentile(degerler, alt_yuzde))  # Alt eşik
        ust_deger = float(np.percentile(degerler, ust_yuzde))  # Üst eşik

        # Değerleri belirlenen aralığa kırp (outlier'ları temizle)
        goruntu_kirp = np.clip(arr_float, alt_deger, ust_deger)

        # Eğer tüm piksel değerleri aynıysa sıfır dön
        if ust_deger - alt_deger < 1e-6:
            return np.zeros_like(goruntu_kirp, dtype=np.uint8)

        sonuc = cv2.normalize(goruntu_kirp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if arka_plan_maskesi is not None and arka_plan_maskesi.shape == sonuc.shape:
            sonuc[arka_plan_maskesi] = 0
        return sonuc

    def histogram_esitle(self, goruntu: np.ndarray, adaptive: bool = False) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula.

        Bu işlem, görüntünün kontrastını adaptif olarak iyileştirir.
        Görüntüyü küçük bloklara böler ve her blokta histogram eşitleme yapar,
        böylece aşırı güçlendirmeyi ve gürültü artışını önler.

        Normal histogram eşitlemeden farkları:
        - Lokal adaptif işlem (her bölge ayrı işlenir)
        - Kontrast sınırlama (clip_limit) ile aşırı güçlendirme önlenir
        - Düşük kontrastlı bölgelerde daha agresif, yüksek kontrastlılarda yumuşak

        Background invariant: CLAHE oncesi sifir piksel olan bolgeler
        sonucta da sifir tutulur; boylece pipeline boyunca background=0
        sozlesmesi korunur.

        Args:
            goruntu: Girdi görüntüsü (numpy array)
            adaptive: Görüntünün kontrast seviyesine göre clip_limit otomatik ayarlansın mı?
                     True: Düşük kontrast -> yüksek clip (3.0), yüksek kontrast -> düşük clip (1.5)
                     False: Sabit clip_limit kullan (ayarlar.py'den)

        Returns:
            Kontrast iyileştirilmiş 2D uint8 görüntü
        """
        if goruntu is None:
            return None

        arr = np.asarray(goruntu)
        if arr.size == 0:
            return goruntu

        # Renkli girdiyi guvenli sekilde gri tonlamaya cevir.
        # shape[2]==1 OpenCV'de cvtColor ile hata verir; squeeze ile inilir.
        if arr.ndim == 3:
            kanal = arr.shape[2] if arr.shape[2:] else 0
            if kanal == 1:
                arr = arr[..., 0]
            elif kanal == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            elif kanal == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
            else:
                # Beklenmedik kanal sayisi; ilk kanali al ve uyarmadan devam et
                arr = arr[..., 0]

        if arr.dtype != np.uint8:
            arr = self._uint8_goruntu(arr)

        # Ayarlardan histogram eşitleme kapalıysa guvenli 2D uint8 don
        if not HISTOGRAM_ESITLEME_AKTIF:
            return arr

        # Adaptif CLAHE: Görüntünün kontrast seviyesine göre clip limit ayarla
        clip_limit = CLAHE_CLIP_LIMIT
        if adaptive:
            contrast = float(cv2.meanStdDev(arr)[1][0][0])
            # Düşük kontrastlı görüntülerde daha agresif CLAHE
            if contrast < 30:
                clip_limit = 3.0
            # Yüksek kontrastlı görüntülerde daha yumuşak CLAHE
            elif contrast > 60:
                clip_limit = 1.5

        # CLAHE arka plani 0'dan farkli bir degere cekebilir; maskeyi
        # uygulamadan once kaydet ve sonucta sifir olarak geri yaz.
        background_mask = arr <= 0

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        sonuc = clahe.apply(arr)
        sonuc[background_mask] = 0
        return sonuc

    def z_score_normalize(self, goruntu: np.ndarray) -> np.ndarray:
        """Robust z-score: foreground mean/std cikarip +/- 2.5 sigma'yi [0, 255]'e esle.

        Strateji bazli cagrilir: yalnizca NORMALIZASYON_STRATEJISI="aggressive"
        oldugunda _apply_normalization_strategy bu metodu kullanir.

        Mean/std arka plani dahil etmez; sifir piksel olan bolgeler sonucta
        sifir kalir. Boylece aggressive strateji background'i griye taşımaz.
        """
        if goruntu is None:
            return goruntu

        arr = np.asarray(goruntu).astype(np.float32)
        if arr.size == 0:
            return goruntu

        mask = arr > 0
        min_piksel = self._min_foreground_piksel_sayisi(arr.size)
        # Erken donuslerde de uint8 sozlesmesi korunur; aksi halde float/uint16
        # girdiler aggressive akisinda goruntuyu bozabilir.
        if int(mask.sum()) < min_piksel:
            return self._uint8_goruntu(goruntu)

        foreground = arr[mask]
        mean = float(foreground.mean())
        std = float(foreground.std())

        if std < 1e-6:
            return self._uint8_goruntu(goruntu)

        zscored = (arr - mean) / std
        clipped = np.clip(zscored, -self._Z_SCORE_RANGE_SIGMA, self._Z_SCORE_RANGE_SIGMA)
        rescaled = (clipped + self._Z_SCORE_RANGE_SIGMA) / (2.0 * self._Z_SCORE_RANGE_SIGMA) * 255.0
        sonuc = rescaled.astype(np.uint8)
        sonuc[~mask] = 0
        return sonuc
