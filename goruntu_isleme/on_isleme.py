"""Tekil MRI goruntusu on isleme adimlari."""

import warnings
from typing import Dict, Optional, Tuple, TypedDict

import cv2
import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    from .ayarlar import *
    from . import opencv_duzeltmeler as _cv_yardimci
except ImportError:
    from ayarlar import *
    import opencv_duzeltmeler as _cv_yardimci


class PipelineSonucu(TypedDict):
    """Tekil goruntu pipeline sonucu icin tip kontratlari.

    `goruntu_isle_sonuc` ve `_tek_goruntu_isle` arasindaki sozlesme bu
    sozlukle sabittir. Yeni alan eklemeden once tum okuyucular gunceller.
    """

    processed_image: Optional[np.ndarray]
    quality_rejected: bool
    quality_reason: str
    tilt_angle: Optional[float]
    tilt_reliable: bool
    tilt_analysis: Dict[str, object]


class GorselOnIslemeMixin:
    @staticmethod
    def _uint8_goruntu(goruntu: np.ndarray) -> np.ndarray:
        """OpenCV islemleri icin goruntuyu guvenli uint8 araligina al."""
        return _cv_yardimci.uint8_goruntu(goruntu)

    @staticmethod
    def _bool_maske_uint8(mask: np.ndarray) -> np.ndarray:
        """Bool maskeyi OpenCV'nin bekledigi 0/255 uint8 formuna cevir."""
        return _cv_yardimci.bool_maske_uint8(mask)

    @staticmethod
    def _gaussian_blur_cv(goruntu: np.ndarray, sigma: float) -> np.ndarray:
        """SciPy gaussian_filter yerine OpenCV GaussianBlur uygula."""
        return cv2.GaussianBlur(
            goruntu,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _oran_ayari_dogrula(
        ad: str,
        deger: object,
        varsayilan: float,
        alt: float = 0.0,
        ust: float = 1.0,
        *,
        clamp: bool = True,
    ) -> float:
        """Oran ayarlarini uyarili sekilde dogrula."""
        try:
            sayi = float(deger)
        except (TypeError, ValueError):
            warnings.warn(
                f"{ad} gecersiz; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)

        if not np.isfinite(sayi):
            warnings.warn(
                f"{ad} sonlu bir sayi olmali; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)

        if alt <= sayi <= ust:
            return sayi

        if clamp:
            kirpilmis = min(max(sayi, alt), ust)
            warnings.warn(
                f"{ad} [{alt}, {ust}] araliginda olmali; {kirpilmis} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(kirpilmis)

        warnings.warn(
            f"{ad} [{alt}, {ust}] araliginda olmali; {varsayilan} kullaniliyor.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float(varsayilan)

    @staticmethod
    def _temel_goruntu_on_kontrol(goruntu: np.ndarray) -> Tuple[bool, str]:
        """Kenar temizligi oncesi cok temel bozuk/bos goruntu kontrolu."""
        if goruntu is None or np.asarray(goruntu).size == 0:
            return False, "Boş görüntü"

        arr = np.asarray(goruntu)
        if arr.ndim != 2:
            return False, "Geçersiz görüntü boyutu"

        if not np.isfinite(arr.astype(np.float32, copy=False)).all():
            return False, "Geçersiz piksel değerleri"

        if float(np.nanmax(arr)) <= 0.0:
            return False, "Boş/siyah görüntü"

        return True, ""

    @staticmethod
    def _egim_sonucu(
        aci: float = 0.0,
        rmse: Optional[float] = None,
        x_span: float = 0.0,
        satir_sayisi: int = 0,
        guvenilir: bool = False,
        sebep: str = "",
    ) -> Dict[str, object]:
        """Egim analizi icin standart sonuc sozlugu uret.

        rmse=None ise PCA-rect uyumsuzlugu hesaplanamamistir; sonuca NaN
        yazilir (yanliligi azaltmak icin "0.0" yerine acikca tanimsiz).
        """
        aci = float(aci)
        rmse_yaz = float("nan") if rmse is None else float(rmse)
        return {
            "aci": aci,
            "mutlak_aci": abs(aci),
            "rmse": rmse_yaz,
            "x_span": float(x_span),
            "satir_sayisi": int(satir_sayisi),
            "guvenilir": bool(guvenilir),
            "sebep": str(sebep),
        }

    @staticmethod
    def _egim_cizgi_acisindan_tilt(theta_x_deg: float) -> float:
        """X eksenine gore cizgi acisini pipeline tilt isaretine cevir."""
        theta_line = ((float(theta_x_deg) + 90.0) % 180.0) - 90.0
        aci = -theta_line
        return ((aci + 45.0) % 90.0) - 45.0

    @staticmethod
    def _egim_aci_farki(a: float, b: float) -> float:
        """Iki tilt acisi arasindaki en kucuk farki derece olarak don."""
        fark = ((float(a) - float(b) + 45.0) % 90.0) - 45.0
        return abs(fark)

    @staticmethod
    def _egim_float_ayar(
        ad: str,
        deger: object,
        varsayilan: float,
        alt: float,
        ust: float,
    ) -> float:
        """Egim ayarlarini uyarili ve guvenli aralikla oku."""
        try:
            sayi = float(deger)
        except (TypeError, ValueError):
            warnings.warn(
                f"{ad} gecersiz; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)

        if not np.isfinite(sayi) or sayi < alt or sayi > ust:
            warnings.warn(
                f"{ad} [{alt}, {ust}] araliginda olmali; {varsayilan} kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return float(varsayilan)
        return sayi

    @staticmethod
    def _pipeline_sonucu(
        processed_image: Optional[np.ndarray] = None,
        *,
        quality_rejected: bool = False,
        quality_reason: str = "",
        tilt_angle: Optional[float] = None,
        tilt_reliable: bool = False,
        tilt_analysis: Optional[Dict[str, object]] = None,
    ) -> PipelineSonucu:
        """Tekil goruntu pipeline'i icin standart sonuc sozlugu uret."""
        return PipelineSonucu(
            processed_image=processed_image,
            quality_rejected=bool(quality_rejected),
            quality_reason=str(quality_reason),
            tilt_angle=tilt_angle,
            tilt_reliable=bool(tilt_reliable),
            tilt_analysis=dict(tilt_analysis or {}),
        )

    def _egim_kalite_reddi_degerlendir(
        self, analiz: Optional[Dict[str, object]]
    ) -> Tuple[bool, str, Optional[float]]:
        """Guvenilir ve esik ustu egimi kalite reddi olarak degerlendir."""
        if not EGIM_KALITE_KONTROL_AKTIF or not analiz:
            return False, "", None

        try:
            aci = float(analiz.get("aci", 0.0))
        except (TypeError, ValueError):
            return False, "", None

        if not np.isfinite(aci):
            return False, "", None

        if not bool(analiz.get("guvenilir", False)):
            return False, "", aci

        esik = self._egim_float_ayar(
            "EGIM_KALITE_RED_ESIGI",
            EGIM_KALITE_RED_ESIGI,
            10.0,
            0.0,
            45.0,
        )
        if abs(aci) > esik:
            return True, "excessive_tilt", aci

        return False, "", aci

    def _egim_reddedilen_duzeltme_sayaclarini_guncelle(
        self, analiz: Dict[str, object]
    ) -> None:
        """Kalite reddi nedeniyle dondurulmayan egimler icin eski sayaclari koru."""
        if not EGIM_DUZELTME_AKTIF or not bool(analiz.get("guvenilir", False)):
            return

        aci = float(analiz.get("aci", 0.0))
        mutlak_aci = abs(aci)
        if mutlak_aci < float(EGIM_MIN_ACI):
            self.son_egim_analizi["sebep"] = "aci_min_altinda"
            return

        self.kalite_istatistikleri["egim_tespit"] = (
            self.kalite_istatistikleri.get("egim_tespit", 0) + 1
        )
        if mutlak_aci > float(EGIM_MAKS_ACI):
            self.kalite_istatistikleri["egim_gorsel_kontrol_adayi"] = (
                self.kalite_istatistikleri.get("egim_gorsel_kontrol_adayi", 0) + 1
            )
        self.son_egim_analizi["sebep"] = "asiri_egim_kalite_reddi"

    def egim_acisi_hesapla(self, goruntu: np.ndarray) -> Dict[str, object]:
        """
        Beyin maskesinin asal eksenine gore goruntu egimini tahmin et.

        Yontem: Otsu ile beyin maskesi uretilir, morfolojik open/close ile
        gurultu temizlenir ve en buyuk gecerli baglantili bilesen secilir.
        Bu bilesende PCA acisi hesaplanir; ayni maskenin ana konturu icin
        minAreaRect acisi da cikarilir. PCA acisi daha hassas tahmin olarak
        doner, ancak yalnizca iki yontem uyumluysa guvenilir sayilir.

        Guvenilirlik kriterleri:
        - Minor/major eksen orani EGIM_MIN_EKSEN_ORANI'ndan kucuk olmali
          (maske neredeyse dairesel ise tahmin guvenilmez sayilir).
        - Ana foreground alani EGIM_MIN_FOREGROUND_ORANI altinda olmamali.
        - PCA ve minAreaRect acilari EGIM_RMSE_MAKS derece icinde uyusmali.
        - Major eksen uzanti boyu EGIM_MIN_X_SPAN'dan buyuk olmali.
        - Maskenin dikey uzanti boyu EGIM_MIN_SATIR_SAYISI'ndan buyuk olmali.

        Donen sozlukteki anlamli alanlar:
        - aci: Major eksenin dikey eksene gore tilt (derece, [-90, 90]).
        - rmse: PCA ve minAreaRect acilari arasindaki mutlak fark (derece).
        - x_span: Major eksen uzunlugu olcusu (2*sqrt(maks ozdeger)).
        - satir_sayisi: Maskenin dikey bbox boyu.
        - guvenilir: Tum kriterler saglandi mi.
        - sebep: Reddedilirse hangi kriter dustu.
        """
        if goruntu is None:
            return self._egim_sonucu(sebep="bos_goruntu")

        arr = self._uint8_goruntu(goruntu)
        if arr.ndim != 2 or arr.size == 0:
            return self._egim_sonucu(sebep="gecersiz_boyut")

        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return self._egim_sonucu(sebep="goruntu_cok_kucuk")

        try:
            _, maske_u8 = cv2.threshold(
                arr,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        except cv2.error:
            return self._egim_sonucu(sebep="otsu_basarisiz")

        if not maske_u8.any():
            return self._egim_sonucu(sebep="maske_bos")

        temel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_boyut = max(5, int(round(min(h, w) * 0.04)))
        if close_boyut % 2 == 0:
            close_boyut += 1
        close_boyut = min(close_boyut, 15)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_boyut, close_boyut)
        )
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_OPEN,
            temel_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_CLOSE,
            close_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        num_labels, labels, stats, _ = _cv_yardimci.baglantili_bilesenler(
            maske_u8, connectivity=8
        )
        if num_labels <= 1:
            return self._egim_sonucu(sebep="bilesen_yok")

        toplam_alan = float(h * w)
        min_alan = max(20, int(0.005 * toplam_alan))
        en_iyi_label = None
        en_iyi_alan = 0
        for label_idx in range(1, num_labels):
            alan = int(stats[label_idx, cv2.CC_STAT_AREA])
            if alan >= min_alan and alan > en_iyi_alan:
                en_iyi_label = label_idx
                en_iyi_alan = alan

        if en_iyi_label is None:
            return self._egim_sonucu(sebep="gecerli_bilesen_yok")

        bilesen_maske = cv2.inRange(labels, int(en_iyi_label), int(en_iyi_label))
        coords = cv2.findNonZero(bilesen_maske)
        if coords is None or coords.size == 0:
            return self._egim_sonucu(sebep="bilesen_bos")

        _, _, _, y_span = cv2.boundingRect(coords)
        if y_span < max(10, int(EGIM_MIN_SATIR_SAYISI)):
            return self._egim_sonucu(
                satir_sayisi=int(y_span),
                sebep="yetersiz_yukseklik",
            )

        foreground_orani = float(en_iyi_alan) / toplam_alan
        min_foreground_orani = self._egim_float_ayar(
            "EGIM_MIN_FOREGROUND_ORANI",
            EGIM_MIN_FOREGROUND_ORANI,
            0.02,
            0.0,
            0.5,
        )

        # PCA: piksel koordinatlari (x, y) sirasinda OpenCV PCACompute2'ye verilir.
        coords_xy = coords.reshape(-1, 2).astype(np.float64)

        try:
            _, eigvecs, eigvals = _cv_yardimci.pca_compute_2d(coords_xy)
        except cv2.error:
            return self._egim_sonucu(
                satir_sayisi=int(y_span),
                sebep="pca_basarisiz",
            )

        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return self._egim_sonucu(
                satir_sayisi=int(y_span),
                sebep="pca_gecersiz",
            )

        # cv2.PCACompute2 azalan sirada doner; ilk satir/eigenvalue major.
        eig_max = float(max(eigvals[0], 0.0))
        eig_min = float(max(eigvals[1], 0.0)) if eigvals.size > 1 else 0.0
        eig_min = min(eig_min, eig_max)
        if eig_max <= 0.0 or eigvecs.shape[0] < 1:
            return self._egim_sonucu(
                satir_sayisi=int(y_span),
                sebep="pca_gecersiz",
            )
        v_major = eigvecs[0]

        major_extent = 2.0 * float(np.sqrt(eig_max))
        eksen_orani = float(np.sqrt(eig_min / eig_max)) if eig_max > 0 else 1.0

        # Major eksenin x-ekseninden acisi. Eigenvektor isaret belirsizligi
        # nedeniyle sonuc bir DOGRU yonelimidir; yardimci fonksiyon bunu
        # pipeline'in tilt isaretine ve [-45, 45) araligina sarar.
        theta_x_deg = float(
            np.degrees(np.arctan2(float(v_major[1]), float(v_major[0])))
        )
        pca_aci = self._egim_cizgi_acisindan_tilt(theta_x_deg)

        contours, _ = cv2.findContours(
            bilesen_maske,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return self._egim_sonucu(
                aci=pca_aci,
                x_span=major_extent,
                satir_sayisi=int(y_span),
                sebep="kontur_yok",
            )

        ana_kontur = max(contours, key=cv2.contourArea)
        if cv2.contourArea(ana_kontur) <= 0.0:
            return self._egim_sonucu(
                aci=pca_aci,
                x_span=major_extent,
                satir_sayisi=int(y_span),
                sebep="kontur_zayif",
            )

        box = cv2.boxPoints(cv2.minAreaRect(ana_kontur))
        kenarlar = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            dx_box = float(p2[0] - p1[0])
            dy_box = float(p2[1] - p1[1])
            uzunluk = float(np.hypot(dx_box, dy_box))
            if uzunluk > 0:
                kenarlar.append((uzunluk, dx_box, dy_box))

        if not kenarlar:
            return self._egim_sonucu(
                aci=pca_aci,
                x_span=major_extent,
                satir_sayisi=int(y_span),
                sebep="kontur_ambiguous",
            )

        kenarlar.sort(key=lambda item: item[0], reverse=True)
        rect_major, rect_dx, rect_dy = kenarlar[0]
        rect_minor = kenarlar[-1][0]
        if rect_major <= 0.0:
            return self._egim_sonucu(
                aci=pca_aci,
                x_span=major_extent,
                satir_sayisi=int(y_span),
                sebep="kontur_ambiguous",
            )

        rect_theta = float(np.degrees(np.arctan2(rect_dy, rect_dx)))
        rect_aci = self._egim_cizgi_acisindan_tilt(rect_theta)
        aci_farki = self._egim_aci_farki(pca_aci, rect_aci)

        if not np.isfinite(pca_aci) or not np.isfinite(aci_farki):
            return self._egim_sonucu(
                aci=0.0,
                rmse=aci_farki if np.isfinite(aci_farki) else None,
                x_span=major_extent,
                satir_sayisi=int(y_span),
                sebep="sonuc_gecersiz",
            )

        # Guvenilirlik: izotropik maske ve zayif uzanti reddedilir.
        eksen_esigi = self._egim_float_ayar(
            "EGIM_MIN_EKSEN_ORANI",
            EGIM_MIN_EKSEN_ORANI,
            0.90,
            0.1,
            0.999,
        )
        aci_farki_esigi = self._egim_float_ayar(
            "EGIM_RMSE_MAKS",
            EGIM_RMSE_MAKS,
            4.5,
            0.0,
            45.0,
        )
        rect_eksen_orani = rect_minor / rect_major if rect_major > 0 else 1.0

        # Cross-check tolerance is angle-relaxed. minAreaRect axis-quantization
        # error grows roughly linearly with the rotation magnitude on small
        # eccentric masks, so at small tilts the rect can latch to 0/90 while
        # PCA correctly recovers a few degrees. Allow up to |pca_aci| extra
        # disagreement before declaring the estimate unreliable.
        aci_uyum_toleransi = max(aci_farki_esigi, abs(float(pca_aci)))

        sebep = "ok"
        guvenilir = True
        if foreground_orani < min_foreground_orani:
            guvenilir = False
            sebep = "foreground_zayif"
        elif eksen_orani >= eksen_esigi:
            guvenilir = False
            sebep = "maske_isotropik"
        elif rect_eksen_orani >= eksen_esigi:
            guvenilir = False
            sebep = "kontur_ambiguous"
        elif major_extent < float(EGIM_MIN_X_SPAN):
            guvenilir = False
            sebep = "x_span_dusuk"
        elif aci_farki > aci_uyum_toleransi:
            guvenilir = False
            sebep = "aci_uyumsuz"

        return self._egim_sonucu(
            aci=float(pca_aci),
            rmse=aci_farki,
            x_span=major_extent,
            satir_sayisi=int(y_span),
            guvenilir=guvenilir,
            sebep=sebep,
        )

    def egim_duzelt(
        self,
        goruntu: np.ndarray,
        analiz: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        """
        Guvenilir ve hafif egimli goruntuyu merkez cizgisine gore dondur.

        Ozellik varsayilan olarak kapalidir. Kapaliyken goruntu aynen doner
        ve sayac/preview gibi yan etkiler olusmaz.
        """
        if not EGIM_DUZELTME_AKTIF:
            return goruntu
        if goruntu is None:
            return goruntu

        arr = self._uint8_goruntu(goruntu)
        if analiz is None:
            analiz = self.egim_acisi_hesapla(arr)
        self.son_egim_analizi = dict(analiz)

        if not bool(analiz.get("guvenilir", False)):
            return arr

        aci = float(analiz.get("aci", 0.0))
        mutlak_aci = abs(aci)
        if mutlak_aci < float(EGIM_MIN_ACI):
            self.son_egim_analizi["sebep"] = "aci_min_altinda"
            return arr

        self.kalite_istatistikleri["egim_tespit"] = (
            self.kalite_istatistikleri.get("egim_tespit", 0) + 1
        )

        if mutlak_aci > float(EGIM_MAKS_ACI):
            self.kalite_istatistikleri["egim_gorsel_kontrol_adayi"] = (
                self.kalite_istatistikleri.get("egim_gorsel_kontrol_adayi", 0) + 1
            )
            self.son_egim_analizi["sebep"] = "gorsel_kontrol_gerekli"
            return arr

        h, w = arr.shape[:2]
        pad_orani = max(0.0, float(EGIM_ROTASYON_PADDING_ORANI))
        pad = int(round(max(h, w) * pad_orani))
        dolgu = int(np.clip(EGIM_DOLDURMA_DEGERI, 0, 255))
        duzeltilmis = _cv_yardimci.affine_dondur_padding(arr, -aci, pad, dolgu)

        self.kalite_istatistikleri["egim_duzeltildi"] = (
            self.kalite_istatistikleri.get("egim_duzeltildi", 0) + 1
        )
        self.son_egim_analizi["sebep"] = "duzeltildi"
        return np.clip(duzeltilmis, 0, 255).astype(np.uint8)

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
        
        # 0-1 aralığına normalize et
        norm = (goruntu_kirp - alt_deger) / (ust_deger - alt_deger)
        # 0-255 aralığına ölçeklendir ve uint8'e çevir
        sonuc = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
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
        
        Args:
            goruntu: Girdi görüntüsü (numpy array, uint8 türünde olmalı)
            adaptive: Görüntünün kontrast seviyesine göre clip_limit otomatik ayarlansın mı?
                     True: Düşük kontrast -> yüksek clip (3.0), yüksek kontrast -> düşük clip (1.5)
                     False: Sabit clip_limit kullan (ayarlar.py'den)
            
        Returns:
            Kontrast iyileştirilmiş görüntü (uint8)
        """
        # Ayarlardan histogram eşitleme kapalıysa direkt dön
        if not HISTOGRAM_ESITLEME_AKTIF:
            return goruntu
        
        # Adaptif CLAHE: Görüntünün kontrast seviyesine göre clip limit ayarla
        clip_limit = CLAHE_CLIP_LIMIT
        if adaptive:
            contrast = np.std(goruntu)
            # Düşük kontrastlı görüntülerde daha agresif CLAHE
            if contrast < 30:
                clip_limit = 3.0
            # Yüksek kontrastlı görüntülerde daha yumuşak CLAHE
            elif contrast > 60:
                clip_limit = 1.5
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(goruntu)
    
    def boyutlandir(self, goruntu: np.ndarray,
                    genislik: int = HEDEF_GENISLIK,
                    yukseklik: int = HEDEF_YUKSEKLIK) -> np.ndarray:
        """
        Görüntüyü hedef boyuta yeniden boyutlandır.

        Makine öğrenmesi modellerinde tüm görüntülerin aynı boyutta olması gerekir.
        Bu fonksiyon MRI görüntülerini standart boyuta (örn: 256x256) getirir.

        BOYUTLANDIRMA_MODU ayarina gore iki strateji desteklenir:
        - "stretch": Goruntu dogrudan hedef boyuta gerilir (eski davranis).
                     En-boy orani korunmaz, ancak basit ve hizlidir.
        - "pad"    : En-boy orani korunarak goruntu hedef cerceveye sigdirilir
                     ve bos kenarlar PADDING_DEGERI ile doldurulur. MRI 2D
                     dilimlerinde anatomik distorsiyonu engeller.

        İnterpolasyon: LINEAR (bilinear interpolation)

        Args:
            goruntu: Kaynak görüntü (numpy array)
            genislik: Hedef genişlik (pixel)
            yukseklik: Hedef yükseklik (pixel)

        Returns:
            (yukseklik, genislik) seklinde yeniden boyutlandirilmis goruntu.
            Girdi dtype'i korunur.
        """
        mod = BOYUTLANDIRMA_MODU
        if mod == "stretch":
            return self._boyutlandir_dogrudan(goruntu, genislik, yukseklik)
        if mod != "pad":
            print(f"[UYARI] Bilinmeyen BOYUTLANDIRMA_MODU: {mod}, 'pad' kullanılıyor")
        return self._boyutlandir_pad(goruntu, genislik, yukseklik)

    def _boyutlandir_dogrudan(self, goruntu: np.ndarray,
                              genislik: int,
                              yukseklik: int) -> np.ndarray:
        """En-boy orani gozetmeden hedef boyuta dogrudan resize."""
        return cv2.resize(goruntu, (genislik, yukseklik), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _padding_fallback_degeri(dtype) -> int:
        """PADDING_DEGERI'ni uint8 araliginda guvenli oku."""
        try:
            deger = float(PADDING_DEGERI)
        except (TypeError, ValueError):
            warnings.warn(
                "PADDING_DEGERI gecersiz; 0 kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0

        if not np.isfinite(deger):
            warnings.warn(
                "PADDING_DEGERI sonlu bir sayi olmali; 0 kullaniliyor.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0

        if np.issubdtype(np.dtype(dtype), np.integer):
            if deger < 0 or deger > 255:
                kirpilmis = int(np.clip(deger, 0, 255))
                warnings.warn(
                    f"PADDING_DEGERI [0, 255] araliginda olmali; {kirpilmis} kullaniliyor.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return kirpilmis
            return int(round(deger))

        return int(np.clip(round(deger), 0, 255))

    def _padding_degeri_hesapla(self, olceklenmis: np.ndarray) -> int:
        """
        Padding icin post-normalizasyon arka plan degeri tahmin et.

        Tahmin yalnizca kose yamalari goruntunun dusuk yogunluklu arka
        planini guvenilir temsil ediyorsa kullanilir; aksi halde
        PADDING_DEGERI fallback olarak kalir.
        """
        fallback = self._padding_fallback_degeri(olceklenmis.dtype)
        if not PADDING_OTOMATIK_ARKAPLAN:
            return fallback

        arr = self._uint8_goruntu(olceklenmis)
        if arr.ndim != 2 or arr.size == 0 or min(arr.shape[:2]) < 4:
            return fallback

        h, w = arr.shape[:2]
        patch = max(2, min(12, h // 8, w // 8))
        kose_degerleri = np.concatenate(
            [
                arr[:patch, :patch].ravel(),
                arr[:patch, w - patch:].ravel(),
                arr[h - patch:, :patch].ravel(),
                arr[h - patch:, w - patch:].ravel(),
            ]
        )
        if kose_degerleri.size == 0:
            return fallback

        p25 = float(np.percentile(arr, 25))
        p90 = float(np.percentile(arr, 90))
        kose_medyan = float(np.median(kose_degerleri))

        # Tek renkli veya koseleri anatomik doku gibi parlak gorunen
        # goruntulerde otomatik tahmin kullanmak non-anatomik padding
        # uretebilir; fallback daha guvenlidir.
        if p90 - p25 < 1.0:
            return fallback
        if kose_medyan <= p25 + 5.0:
            return int(np.clip(round(kose_medyan), 0, 255))
        return fallback

    def _boyutlandir_pad(self, goruntu: np.ndarray,
                         genislik: int,
                         yukseklik: int) -> np.ndarray:
        """En-boy oranini koruyarak hedef cerceveye sigdir ve kenarlari doldur."""
        h, w = goruntu.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Geçersiz görüntü boyutu")

        olcek = min(genislik / float(w), yukseklik / float(h))
        yeni_g = max(1, int(round(w * olcek)))
        yeni_y = max(1, int(round(h * olcek)))
        # Yuvarlama hedefi asabilir; kirpalim
        yeni_g = min(yeni_g, genislik)
        yeni_y = min(yeni_y, yukseklik)

        olceklenmis = self._boyutlandir_dogrudan(goruntu, yeni_g, yeni_y)

        padding_degeri = self._padding_degeri_hesapla(olceklenmis)
        ust = (yukseklik - yeni_y) // 2
        sol = (genislik - yeni_g) // 2
        alt = yukseklik - yeni_y - ust
        sag = genislik - yeni_g - sol
        return cv2.copyMakeBorder(
            olceklenmis,
            ust,
            alt,
            sol,
            sag,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_degeri,
        )
    
    def _bilateral_filtre_uygula(self, goruntu: np.ndarray) -> np.ndarray:
        """Kenar korumali bilateral filtre uygula."""
        filtered = cv2.bilateralFilter(goruntu, d=5, sigmaColor=35, sigmaSpace=35)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def gurultu_gider(self, goruntu: np.ndarray, metod: str = 'auto') -> np.ndarray:
        """
        Görüntüden gürültüyü temizle.

        Args:
            goruntu: Girdi görüntüsü
            metod: 'auto' (FILTRE_METODU ayarini kullanir), 'off', 'median',
                   'gaussian' veya 'bilateral'.

        Returns:
            Gürültüsü azaltılmış görüntü
        """
        if metod == 'auto':
            metod = FILTRE_METODU

        if metod == 'off':
            # auto+off durumunda median 3x3'e dusmek kortikal dokuyu siler;
            # bu yuzden goruntu aynen donulur.
            return goruntu
        if metod == 'median':
            filtered = cv2.medianBlur(self._uint8_goruntu(goruntu), 3)
            return filtered.astype(np.uint8)
        if metod == 'gaussian':
            filtered = self._gaussian_blur_cv(
                self._uint8_goruntu(goruntu),
                sigma=GAUSSIAN_BLUR_SIGMA,
            )
            return filtered.astype(np.uint8)
        if metod == 'bilateral':
            return self._bilateral_filtre_uygula(goruntu)
        return goruntu
    
    def skull_strip(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Skull stripping (kafatası çıkarma).
        
        MRI görüntülerinde kafatası ve arka plan beyin dokusu için gereksizdir.
        Bu fonksiyon beyin bölgesini maskeleyerek çıkarır.
        
        İki metod desteklenir:
        1. "simple": Basit Otsu thresholding
        2. "advanced": Gelişmiş morfolojik işlemlerle iyileştirilmiş
        
        Daha profesyonel yöntemler için FSL/BET veya HD-BET önerilir.
        
        Args:
            goruntu: Girdi MRI görüntüsü
            
        Returns:
            Sadece beyin dokusunu içeren görüntü
        """
        if not SKULL_STRIPPING_AKTIF:
            return goruntu
        
        if SKULL_STRIPPING_METHOD == "advanced":
            return self._advanced_skull_strip(goruntu)
        else:
            return self._simple_skull_strip(goruntu)

    @staticmethod
    def _kenar_maskesini_temizle(mask: np.ndarray) -> np.ndarray:
        """Maske kenarlarindaki artefaktlari ayarlanabilir pay ile temizle."""
        temiz = mask.astype(bool, copy=True)
        pay = max(0, int(MASKE_KENAR_PAYI))
        if pay == 0:
            return temiz

        h, w = temiz.shape
        if pay * 2 >= min(h, w):
            return np.zeros_like(temiz, dtype=bool)

        temiz[:pay, :] = False
        temiz[-pay:, :] = False
        temiz[:, :pay] = False
        temiz[:, -pay:] = False
        return temiz

    @staticmethod
    def _morfolojik_yapi(kernel_boyutu: Optional[int] = None) -> np.ndarray:
        """Ayarlardaki kernel boyutunu kullanarak eliptik OpenCV yapi elemani uret."""
        boyut = int(kernel_boyutu or MORFOLOJIK_KERNEL_BOYUTU)
        boyut = max(1, boyut)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boyut, boyut))

    def _maskeyi_duzenle(
        self,
        mask: np.ndarray,
        *,
        closing_scale: int = 2,
        dilation_scale: int = 0,
    ) -> np.ndarray:
        """Maske kenarlarini ve morfolojik temizligini ayarlara gore uygula."""
        duzenli = self._kenar_maskesini_temizle(mask)
        if not MORFOLOJIK_OPERASYONLAR_AKTIF:
            return duzenli

        temel = self._morfolojik_yapi()
        close_kernel = self._morfolojik_yapi(MORFOLOJIK_KERNEL_BOYUTU * max(1, closing_scale))
        maske_u8 = self._bool_maske_uint8(duzenli)
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_OPEN,
            temel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        maske_u8 = cv2.morphologyEx(
            maske_u8,
            cv2.MORPH_CLOSE,
            close_kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if dilation_scale > 0:
            dilate_kernel = self._morfolojik_yapi(MORFOLOJIK_KERNEL_BOYUTU * dilation_scale)
            maske_u8 = cv2.dilate(
                maske_u8,
                dilate_kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return maske_u8 > 0

    def _otsu_maskesi(self, goruntu: np.ndarray) -> np.ndarray:
        """OpenCV Otsu thresholding ile beyin aday maskesini uret."""
        _, mask = cv2.threshold(
            self._uint8_goruntu(goruntu),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask > 0

    @staticmethod
    def _min_foreground_piksel_sayisi(toplam_piksel: int, oran: float = 0.001) -> int:
        """Kucuk test goruntulerini cezalandirmadan foreground alt siniri hesapla."""
        toplam = max(1, int(toplam_piksel))
        return min(max(16, int(round(toplam * oran))), max(1, toplam // 4))

    def _foreground_maskesi(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Per-image foreground/beyin aday maskesi uret.

        Maske yalnizca normalizasyon ve bias tahmini gibi istatistik
        adimlarinda kullanilir; goruntuyu tek basina kirpmaz veya reddetmez.
        """
        if goruntu is None:
            return np.zeros((0, 0), dtype=bool)

        arr = self._uint8_goruntu(goruntu)
        if arr.ndim != 2 or arr.size == 0:
            return np.zeros(arr.shape[:2], dtype=bool)

        min_piksel = self._min_foreground_piksel_sayisi(arr.size)
        if int(arr.max()) <= int(arr.min()):
            return np.zeros(arr.shape, dtype=bool)

        try:
            maske = self._otsu_maskesi(arr)
        except cv2.error:
            maske = arr > 0

        if not maske.any():
            maske = arr > 0

        if min(arr.shape[:2]) >= 8 and maske.any():
            maske = self._maskeyi_duzenle(maske, closing_scale=2)
            maske = self._kucuk_bilesenleri_temizle(maske, min_size=min_piksel)
            if maske.any():
                maske = self._kucuk_delikleri_doldur(
                    maske,
                    area_threshold=min_piksel,
                )

        maske = maske.astype(bool) & (arr > 0)
        if int(maske.sum()) >= min_piksel:
            return maske

        nonzero = arr > 0
        if int(nonzero.sum()) >= min_piksel:
            return nonzero

        return np.zeros(arr.shape, dtype=bool)

    def _kucuk_bilesenleri_temizle(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """OpenCV connected components ile min_size altindaki nesneleri sil."""
        if min_size <= 1:
            return mask.astype(bool)

        maske_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(maske_u8, connectivity=8)
        temiz = np.zeros_like(maske_u8, dtype=bool)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
                temiz[labels == label_idx] = True
        return temiz

    def _kucuk_delikleri_doldur(self, mask: np.ndarray, area_threshold: int) -> np.ndarray:
        """OpenCV connected components ile icteki kucuk delikleri doldur."""
        if area_threshold <= 0:
            return mask.astype(bool)

        dolu = mask.astype(bool, copy=True)
        ters = (~dolu).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ters, connectivity=8)
        h, w = dolu.shape

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area > area_threshold:
                continue

            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            genislik = stats[label_idx, cv2.CC_STAT_WIDTH]
            yukseklik = stats[label_idx, cv2.CC_STAT_HEIGHT]
            kenara_degiyor = x == 0 or y == 0 or x + genislik >= w or y + yukseklik >= h
            if not kenara_degiyor:
                dolu[labels == label_idx] = True

        return dolu

    def _simple_skull_strip(self, goruntu: np.ndarray) -> np.ndarray:
        """Basit skull stripping (Otsu thresholding)."""
        maske = self._otsu_maskesi(goruntu)
        maske = self._maskeyi_duzenle(maske, closing_scale=2)

        return cv2.bitwise_and(
            self._uint8_goruntu(goruntu),
            self._uint8_goruntu(goruntu),
            mask=self._bool_maske_uint8(maske),
        )
    
    def _advanced_skull_strip(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Gelişmiş skull stripping - morfolojik işlemlerle iyileştirilmiş.
        
        Bu metod daha agresif morfolojik operasyonlar ve bağlantılı bileşen
        analizi kullanarak daha iyi bir beyin maskesi oluşturur.
        
        Args:
            goruntu: Girdi görüntüsü
            
        Returns:
            Skull-stripped görüntü
        """
        try:
            # 1. Otsu eşikleme ile başlangıç maskesi
            maske = self._otsu_maskesi(goruntu)
            maske = self._kenar_maskesini_temizle(maske)
            
            # 2. Küçük nesneleri temizle (min_size = toplam pikselin %0.5'i)
            min_size = int(goruntu.size * 0.005)
            maske = self._kucuk_bilesenleri_temizle(maske, min_size=min_size)
            
            # 3. Morfolojik gürültü temizleme
            maske = self._maskeyi_duzenle(maske, closing_scale=1)
            
            # 4. Küçük delikleri kapat
            maske = self._kucuk_delikleri_doldur(maske, area_threshold=min_size)
            
            # 5. En büyük bağlantılı bileşeni bul (beyin olmalı)
            maske_u8 = maske.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(maske_u8, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_region = int(areas.argmax()) + 1
                maske = labels == largest_region

            # 6. Kenarlari yumusat ve beyin dokusunu korumak icin hafif genislet
            maske = self._maskeyi_duzenle(maske, closing_scale=2, dilation_scale=1)
            
            # 7. Maskeyi uygula
            result = cv2.bitwise_and(
                self._uint8_goruntu(goruntu),
                self._uint8_goruntu(goruntu),
                mask=self._bool_maske_uint8(maske),
            )
            
            return result
            
        except Exception as e:
            print(f"[UYARI] Advanced skull stripping başarısız: {e}")
            return self._simple_skull_strip(goruntu)
    
    def bias_field_correction(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Bias field correction (MRI yoğunluk düzensizliği düzeltme).

        MRI cihazındaki manyetik alan düzensizlikleri yüzünden, aynı doku tipinde
        farklı yoğunluk değerleri görülebilir. Bu düzeltme, smooth varying intensity
        düzensizliklerini giderir.

        İki metod desteklenir:
        1. "n4itk": N4ITK algoritması (profesyonel, yavaş) - SimpleITK gerekli
        2. "simple": Basit Gaussian blur tabanlı (hızlı)

        Args:
            goruntu: Girdi MRI görüntüsü

        Returns:
            Düzeltilmiş görüntü
        """
        if not BIAS_FIELD_CORRECTION_AKTIF:
            return goruntu
        
        # N4ITK metodu (profesyonel)
        if BIAS_FIELD_METHOD == "n4itk" and SITK_AVAILABLE:
            try:
                return self._n4itk_bias_correction(goruntu)
            except Exception as e:
                print(f"[UYARI] N4ITK bias correction başarısız, basit metoda geçiliyor: {e}")
                return self._simple_bias_correction(goruntu)
        
        # Basit metod
        return self._simple_bias_correction(goruntu)
    
    def _n4itk_bias_correction(self, goruntu: np.ndarray) -> np.ndarray:
        """
        N4ITK algoritması ile profesyonel bias field correction.
        
        N4ITK (N4 Bias Field Correction), MRI görüntülerinde yoğunluk
        düzensizliklerini düzeltmek için altın standart algoritmadır.
        
        Args:
            goruntu: Girdi görüntüsü (numpy array)
            
        Returns:
            Düzeltilmiş görüntü
        """
        # NumPy array'i SimpleITK image'e çevir
        img_sitk = sitk.GetImageFromArray(goruntu.astype(np.float32))
        
        # Maske oluştur (Otsu thresholding ile)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        mask = otsu_filter.Execute(img_sitk)
        
        # N4 Bias Field Correction uygula
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])  # 4 seviye, her seviyede 50 iterasyon
        corrector.SetConvergenceThreshold(0.001)
        
        # Düzeltmeyi çalıştır
        corrected = corrector.Execute(img_sitk, mask)
        
        # Geri NumPy array'e çevir ve maskeli normalize et
        corrected_array = sitk.GetArrayFromImage(corrected)
        mask_array = sitk.GetArrayFromImage(mask) > 0
        mask_values = corrected_array[mask_array]
        if mask_values.size == 0:
            return self._uint8_goruntu(goruntu)
        
        # 0-255 aralığına yalnizca foreground istatistikleriyle normalize et.
        alt = float(np.percentile(mask_values, 0.5))
        ust = float(np.percentile(mask_values, 99.5))
        if ust - alt < 1e-6:
            return self._uint8_goruntu(goruntu)

        normalized = np.clip(corrected_array, alt, ust)
        normalized = (normalized - alt) / (ust - alt) * 255.0
        sonuc = self._uint8_goruntu(goruntu).copy()
        sonuc[mask_array] = np.clip(normalized[mask_array], 0, 255).astype(np.uint8)
        return sonuc
    
    def _simple_bias_correction(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Basit bias field correction (Gaussian blur tabanlı).
        
        Hızlı ama daha az etkili bir metod. N4ITK mevcut değilse kullanılır.
        
        Args:
            goruntu: Girdi görüntüsü
            
        Returns:
            Düzeltilmiş görüntü
        """
        try:
            arr_u8 = self._uint8_goruntu(goruntu)
            img_float = arr_u8.astype(np.float32)
            maske = self._foreground_maskesi(arr_u8)
            min_piksel = self._min_foreground_piksel_sayisi(arr_u8.size)
            if maske.shape != arr_u8.shape or int(maske.sum()) < min_piksel:
                return arr_u8

            # Düşük frekanslı bias field'ı maskeli normalized convolution
            # ile tahmin et. Siyah kenar/padding bolgeleri blur'u asagi
            # cekip bolme sirasinda yapay kenar parlaklasmasi uretmesin.
            maske_float = maske.astype(np.float32)
            blur_img = self._gaussian_blur_cv(img_float * maske_float, sigma=50)
            blur_mask = self._gaussian_blur_cv(maske_float, sigma=50)
            valid = blur_mask > 1e-3
            bias_field = np.zeros_like(img_float, dtype=np.float32)
            bias_field[valid] = blur_img[valid] / blur_mask[valid]

            bias_degerleri = bias_field[maske & valid]
            if bias_degerleri.size == 0:
                return arr_u8

            mean_bias = float(np.median(bias_degerleri))
            if mean_bias < 1e-6:
                return arr_u8

            bias_floor = max(1.0, float(np.percentile(bias_degerleri, 5)) * 0.5)
            safe_bias = np.maximum(bias_field, bias_floor)

            corrected = img_float.copy()
            corrected[maske] = img_float[maske] / (safe_bias[maske] / mean_bias + 1e-6)
            corrected[~maske] = img_float[~maske]
            return np.clip(corrected, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"[UYARI] Bias field correction başarısız: {e}")
            return goruntu
    
    def center_of_mass_alignment(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Görüntüyü kütle merkezine göre hizala.
        
        Farklı açılardan çekilmiş MRI görüntülerini merkeze hizalar.
        
        Üç metod desteklenir:
        1. "simple": Center-of-mass tabanlı basit kaydırma
        2. "affine": Affine transformation (ölçek, dönme, kaydırma)
        3. "rigid": Rigid transformation (sadece dönme ve kaydırma)
        
        Args:
            goruntu: Girdi görüntüsü
            
        Returns:
            Hizalanmış görüntü
        """
        if not REGISTRATION_AKTIF:
            return goruntu
        
        # SimpleITK mevcut ve gelişmiş metod seçiliyse
        if SITK_AVAILABLE and REGISTRATION_METHOD in ["affine", "rigid"]:
            return self._advanced_registration(goruntu, method=REGISTRATION_METHOD)
        else:
            # Basit center-of-mass alignment
            return self._simple_center_alignment(goruntu)
    
    def _simple_center_alignment(self, goruntu: np.ndarray) -> np.ndarray:
        """Basit center-of-mass tabanlı hizalama."""
        try:
            # Eşikleme ile beyin bölgesini bul
            threshold = np.percentile(goruntu, 50)
            binary = goruntu > threshold
            
            if not np.any(binary):
                return goruntu

            moments = cv2.moments(binary.astype(np.uint8), binaryImage=True)
            if abs(moments["m00"]) < 1e-6:
                return goruntu

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            h, w = goruntu.shape[:2]
            shift_x = (w / 2.0) - center_x
            shift_y = (h / 2.0) - center_y
            matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
            aligned = cv2.warpAffine(
                self._uint8_goruntu(goruntu),
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return aligned.astype(np.uint8)
            
        except Exception as e:
            print(f"[UYARI] Center of mass alignment başarısız: {e}")
            return goruntu
    
    def _advanced_registration(self, goruntu: np.ndarray, method: str = "affine") -> np.ndarray:
        """
        SimpleITK ile gelişmiş registration.
        
        Bu fonksiyon, görüntüleri bir şablon görüntüye hizalar.
        İlk görüntü şablon olarak kaydedilir, diğerleri buna hizalanır.
        
        Args:
            goruntu: Hizalanacak görüntü
            method: "affine" veya "rigid"
            
        Returns:
            Hizalanmış görüntü
        """
        try:
            # İlk görüntüde güvenli başlangıç: en azından merkez hizalama uygula.
            if self.template_image is None:
                self.template_image = self._simple_center_alignment(goruntu)
                return self.template_image
            
            # Moving ve fixed image oluştur
            fixed_image = sitk.GetImageFromArray(self.template_image.astype(np.float32))
            moving_image = sitk.GetImageFromArray(goruntu.astype(np.float32))
            
            # Registration metodunu ayarla
            registration_method = sitk.ImageRegistrationMethod()
            
            # Metrik: Mean Squares (benzerlik ölçüsü)
            registration_method.SetMetricAsMeanSquares()
            
            # Optimizer: Gradient Descent
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=1.0,
                minStep=0.001,
                numberOfIterations=200,
                gradientMagnitudeTolerance=1e-6
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Transform tipi seç
            if method == "rigid":
                # Rigid: Sadece dönme ve kaydırma
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image, moving_image,
                    sitk.Euler2DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
            else:  # affine
                # Affine: Dönme, kaydırma, ölçekleme, kesme
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image, moving_image,
                    sitk.AffineTransform(2),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
            
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # Interpolator: Linear
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Registration'ı çalıştır
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Transform'u uygula
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_image = resampler.Execute(moving_image)
            
            # Geri numpy array'e çevir
            result = sitk.GetArrayFromImage(registered_image)
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"[UYARI] Advanced registration başarısız: {e}")
            return self._simple_center_alignment(goruntu)
    
    # Z-score'u uint8'e geri eslerken kullanilan +/- aralik (sigma cinsinden).
    # +/- 2.5 sigma disindaki degerler kirpilarak [0, 255]'e dogrusal eslenir;
    # boylece "*50 + 128" gibi sihirli sabitler yerine acik bir kontrat olur.
    _Z_SCORE_RANGE_SIGMA = 2.5

    def z_score_normalize(self, goruntu: np.ndarray) -> np.ndarray:
        """Robust z-score: mean/std cikarip +/- 2.5 sigma'yi [0, 255]'e dogrusal esle.

        Strateji bazli cagrilir: yalnizca NORMALIZASYON_STRATEJISI="aggressive"
        oldugunda _apply_normalization_strategy bu metodu kullanir.
        """
        arr = goruntu.astype(np.float32)
        mean = float(arr.mean())
        std = float(arr.std())

        if std < 1e-6:
            return goruntu

        zscored = (arr - mean) / std
        clipped = np.clip(zscored, -self._Z_SCORE_RANGE_SIGMA, self._Z_SCORE_RANGE_SIGMA)
        rescaled = (clipped + self._Z_SCORE_RANGE_SIGMA) / (2.0 * self._Z_SCORE_RANGE_SIGMA) * 255.0
        return rescaled.astype(np.uint8)
    
    def goruntu_isle_sonuc(self, dosya_yolu: str) -> PipelineSonucu:
        """
        Tek bir goruntuye tam on isleme pipeline uygula ve yapisal sonuc don.

        Pipeline stratejileri (NORMALIZASYON_STRATEJISI ayarından):
        - "minimal": Sadece percentile clipping + resize
        - "standard": percentile + CLAHE + resize (önerilen)
        - "aggressive": percentile + CLAHE + z-score + resize

        Pipeline sırası:
        1. Görüntü yükle
        2. Temel boş/bozuk görüntü ön kontrolü
        3. Kenar artefakt tespiti (raporlama)
        4. Kenar artefakt temizligi (varsa)
        5. Kalite kontrol
        6. Opsiyonel egim duzeltme
        7. Gürültü giderme (erken aşama)
        8. Bias field correction (N4ITK veya simple)
        9. Skull stripping (advanced veya simple)
        10. Registration (affine/rigid/simple)
        11. Strateji bazlı normalizasyon
        12. Boyutlandırma

        Args:
            dosya_yolu: Görüntü dosyasının yolu

        Returns:
            processed_image, quality_rejected, quality_reason ve tilt_angle
            alanlarini iceren sonuc sozlugu.
        """
        self.son_egim_analizi = {}

        # 1. Görüntüyü yükle
        goruntu = self.goruntu_yukle(dosya_yolu)
        if goruntu is None:
            return self._pipeline_sonucu()

        # 2. Temel on kontrol: strict kalite kontrol kenar temizliginden
        #    sonra calisir, ancak acikca bos/bozuk goruntuler erken elenir.
        on_kontrol_ok, on_kontrol_mesaji = self._temel_goruntu_on_kontrol(goruntu)
        if not on_kontrol_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {on_kontrol_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        # 3. Kenar artefakt tespiti (raporlama amacli; varsayilan olarak
        #    goruntuyu reddetmez, yalnizca sayac arttirir).
        if KENAR_ARTEFAKT_KONTROL_AKTIF:
            analiz = self.kenar_artefakt_analiz(self._uint8_goruntu(goruntu))
            if bool(analiz.get("artefakt_var", False)):
                self.kalite_istatistikleri["kenar_artefakt_tespit"] = (
                    self.kalite_istatistikleri.get("kenar_artefakt_tespit", 0) + 1
                )

        # 4. Kenar artefakt temizligi - normalize ve CLAHE'den ONCE.
        #    Tespit ile bagimsiz calisir: KENAR_ARTEFAKT_KONTROL_AKTIF
        #    kapali olsa bile temizleme aktifse her goruntuye uygulanir.
        #    Fonksiyonun kendisi parlak piksel yoksa erken cikar.
        kenar_temizlendi = False
        if KENAR_ARTEFAKT_TEMIZLEME_AKTIF:
            oncesi_u8 = self._uint8_goruntu(goruntu)
            temiz_goruntu = self.kenar_artefakt_temizle(oncesi_u8)
            if not np.array_equal(temiz_goruntu, oncesi_u8):
                kenar_temizlendi = True
            goruntu = temiz_goruntu

        # 5. Strict kalite kontrol: kenar temizligiyle kurtarilabilecek
        #    goruntuler temizlendikten sonra degerlendirilir.
        kalite_ok, hata_mesaji = self.goruntu_kalite_kontrol(goruntu)
        if not kalite_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {hata_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        # Temizlik sonrasi tamamen bosalan sentetik/bozuk goruntuleri,
        # kalite kontrol kapali olsa bile gecirmeyelim.
        on_kontrol_ok, on_kontrol_mesaji = self._temel_goruntu_on_kontrol(goruntu)
        if not on_kontrol_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {on_kontrol_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return self._pipeline_sonucu()

        # Sayaci ancak goruntu strict QC'yi de gectikten sonra arttir.
        # Boylece "temizlendi" sayisi gercekten ciktiya alinacak goruntuleri
        # yansitir; QC'de elenen artefaktli goruntuler over-count edilmez.
        if kenar_temizlendi:
            self.kalite_istatistikleri["kenar_artefakt_temizlendi"] = (
                self.kalite_istatistikleri.get("kenar_artefakt_temizlendi", 0) + 1
            )

        # 6. Opsiyonel egim duzeltme - normalize/CLAHE, skull strip ve
        #    registration oncesinde orijinal anatomik geometri uzerinde calisir.
        egim_analizi = None
        kalite_reddi = False
        kalite_reddi_nedeni = ""
        egim_acisi = None
        egim_guvenilir = False
        if EGIM_DUZELTME_AKTIF or EGIM_KALITE_KONTROL_AKTIF:
            egim_analizi = self.egim_acisi_hesapla(self._uint8_goruntu(goruntu))
            self.son_egim_analizi = dict(egim_analizi)
            egim_guvenilir = bool(egim_analizi.get("guvenilir", False))
            kalite_reddi, kalite_reddi_nedeni, egim_acisi = (
                self._egim_kalite_reddi_degerlendir(egim_analizi)
            )

        if kalite_reddi:
            self.kalite_istatistikleri["egim_kalite_red"] = (
                self.kalite_istatistikleri.get("egim_kalite_red", 0) + 1
            )
            self._egim_reddedilen_duzeltme_sayaclarini_guncelle(egim_analizi)
        elif EGIM_DUZELTME_AKTIF:
            goruntu = self.egim_duzelt(goruntu, analiz=egim_analizi)
            egim_analizi = dict(self.son_egim_analizi)
            egim_acisi = float(egim_analizi.get("aci", 0.0)) if egim_analizi else None
            egim_guvenilir = bool(egim_analizi.get("guvenilir", False)) if egim_analizi else False

        # 7. Gürültü giderme
        goruntu = self.gurultu_gider(goruntu, metod='auto')

        # 8. Bias field correction
        goruntu = self.bias_field_correction(goruntu)

        # 9. Skull stripping
        goruntu = self.skull_strip(goruntu)

        # 10. Registration
        goruntu = self.center_of_mass_alignment(goruntu)

        # 11. Strateji bazlı normalizasyon
        goruntu = self._apply_normalization_strategy(goruntu)

        # 12. Boyutlandırma
        goruntu = self.boyutlandir(goruntu)

        return self._pipeline_sonucu(
            goruntu,
            quality_rejected=kalite_reddi,
            quality_reason=kalite_reddi_nedeni,
            tilt_angle=egim_acisi,
            tilt_reliable=egim_guvenilir,
            tilt_analysis=egim_analizi,
        )

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
        """
        Seçilen normalizasyon stratejisini uygula.
        
        Bu fonksiyon, over-processing'i önlemek için farklı seviyede
        normalizasyon stratejileri sunar.
        
        Args:
            goruntu: Girdi görüntüsü
            
        Returns:
            Normalize edilmiş görüntü
        """
        strategy = NORMALIZASYON_STRATEJISI
        
        if strategy == "minimal":
            # Minimal: Sadece percentile clipping
            goruntu = self.yogunluk_normalize(goruntu)

        elif strategy == "standard":
            # Standard: percentile + sabit CLAHE (train/test arasi deterministik)
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)

        elif strategy == "aggressive":
            # Aggressive: percentile + sabit CLAHE + z-score
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)
            goruntu = self.z_score_normalize(goruntu)

        else:
            # Varsayılan: standard
            print(f"[UYARI] Bilinmeyen strateji: {strategy}, 'standard' kullanılıyor")
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=False)
        
        return goruntu
