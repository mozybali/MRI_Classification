"""Tekil MRI goruntusu on isleme adimlari."""

from typing import Optional

import cv2
import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselOnIslemeMixin:
    @staticmethod
    def _uint8_goruntu(goruntu: np.ndarray) -> np.ndarray:
        """OpenCV islemleri icin goruntuyu guvenli uint8 araligina al."""
        arr = np.asarray(goruntu)
        if arr.dtype == np.uint8:
            return arr
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _bool_maske_uint8(mask: np.ndarray) -> np.ndarray:
        """Bool maskeyi OpenCV'nin bekledigi 0/255 uint8 formuna cevir."""
        return (mask.astype(bool) * 255).astype(np.uint8)

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
        
        # Alt ve üst yüzdelik değerlerini al (örn: %1 ve %99)
        alt_yuzde, ust_yuzde = KIRPMA_YUZDELERI
        alt_deger = np.percentile(goruntu, alt_yuzde)  # Alt eşik
        ust_deger = np.percentile(goruntu, ust_yuzde)  # Üst eşik
        
        # Değerleri belirlenen aralığa kırp (outlier'ları temizle)
        goruntu_kirp = np.clip(goruntu, alt_deger, ust_deger)
        
        # Eğer tüm piksel değerleri aynıysa sıfır dön
        if ust_deger - alt_deger < 1e-6:
            return np.zeros_like(goruntu_kirp, dtype=np.uint8)
        
        # 0-1 aralığına normalize et
        norm = (goruntu_kirp - alt_deger) / (ust_deger - alt_deger)
        # 0-255 aralığına ölçeklendir ve uint8'e çevir
        return (norm * 255.0).astype(np.uint8)

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

        canvas_sekli = (yukseklik, genislik) + goruntu.shape[2:]
        canvas = np.full(canvas_sekli, PADDING_DEGERI, dtype=goruntu.dtype)
        y0 = (yukseklik - yeni_y) // 2
        x0 = (genislik - yeni_g) // 2
        canvas[y0:y0 + yeni_y, x0:x0 + yeni_g] = olceklenmis
        return canvas
    
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
        
        # Geri NumPy array'e çevir ve normalize et
        corrected_array = sitk.GetArrayFromImage(corrected)
        
        # 0-255 aralığına normalize et
        corrected_array = np.clip(corrected_array, 0, np.percentile(corrected_array, 99.5))
        corrected_array = ((corrected_array - corrected_array.min()) / 
                          (corrected_array.max() - corrected_array.min() + 1e-8) * 255.0)
        
        return corrected_array.astype(np.uint8)
    
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
            # Görüntüyü float'a çevir
            img_float = goruntu.astype(np.float32)
            
            # Düşük frekanslı bias field'ı tahmin etmek için Gaussian blur
            # Bias field, yavaş değişen bir alandır
            bias_field = self._gaussian_blur_cv(img_float, sigma=50)
            
            # Ortalamayı bul (sıfıra bölme önlemi)
            mean_bias = np.mean(bias_field)
            if mean_bias < 1e-6:
                return goruntu
            
            # Bias field'ı kaldır (orijinal / bias)
            # Skull-strip sonrası sıfır bölgeler inf üretebilir; epsilon ile koruma
            corrected = img_float / (bias_field / mean_bias + 1e-6)
            
            # 0-255 aralığına normalize et
            corrected = np.clip(corrected, 0, 255)
            return corrected.astype(np.uint8)
            
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
    
    def goruntu_isle(self, dosya_yolu: str) -> Optional[np.ndarray]:
        """
        Tek bir görüntüye tam ön işleme pipeline uygula.

        Pipeline stratejileri (NORMALIZASYON_STRATEJISI ayarından):
        - "minimal": Sadece percentile clipping + resize
        - "standard": percentile + CLAHE + resize (önerilen)
        - "aggressive": percentile + CLAHE + z-score + resize

        Pipeline sırası:
        1. Görüntü yükle
        2. Kalite kontrol
        3. Gürültü giderme (erken aşama)
        4. Bias field correction (N4ITK veya simple)
        5. Skull stripping (advanced veya simple)
        6. Registration (affine/rigid/simple)
        7. Strateji bazlı normalizasyon
        8. Boyutlandırma

        Args:
            dosya_yolu: Görüntü dosyasının yolu

        Returns:
            İşlenmiş görüntü veya None (kalite kontrolden geçmezse)
        """
        # 1. Görüntüyü yükle
        goruntu = self.goruntu_yukle(dosya_yolu)
        if goruntu is None:
            return None
        
        # 2. Kalite kontrol
        kalite_ok, hata_mesaji = self.goruntu_kalite_kontrol(goruntu)
        if not kalite_ok:
            print(f"[KALITE HATASI] {dosya_yolu}: {hata_mesaji}")
            self.kalite_istatistikleri["kalite_hatasi"] += 1
            return None

        # 3. Gürültü giderme
        goruntu = self.gurultu_gider(goruntu, metod='auto')

        # 4. Bias field correction
        goruntu = self.bias_field_correction(goruntu)

        # 5. Skull stripping
        goruntu = self.skull_strip(goruntu)

        # 6. Registration
        goruntu = self.center_of_mass_alignment(goruntu)

        # 7. Strateji bazlı normalizasyon
        goruntu = self._apply_normalization_strategy(goruntu)

        # 8. Boyutlandırma
        goruntu = self.boyutlandir(goruntu)
        
        return goruntu
    
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
