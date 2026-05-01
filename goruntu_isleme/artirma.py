"""Goruntu veri artirma islemleri."""

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselArtirmaMixin:
    @staticmethod
    def yatay_ayna(goruntu: np.ndarray) -> np.ndarray:
        """Yatay ayna (flip)."""
        if CV2_AVAILABLE:
            return cv2.flip(goruntu, 1)
        return np.fliplr(goruntu)
    
    @staticmethod
    def dikey_ayna(goruntu: np.ndarray) -> np.ndarray:
        """Dikey ayna (flip)."""
        if CV2_AVAILABLE:
            return cv2.flip(goruntu, 0)
        return np.flipud(goruntu)
    
    def rastgele_dondur(self, goruntu: np.ndarray) -> np.ndarray:
        """Küçük açılı rotasyon uygula."""
        if not ROTASYON_AKTIF:
            return goruntu
        aci = self._random.uniform(-ROTASYON_MAKS_ACI, ROTASYON_MAKS_ACI)
        if abs(aci) < 1e-3:
            return goruntu
        if CV2_AVAILABLE:
            h, w = goruntu.shape[:2]
            merkez = ((w - 1) / 2.0, (h - 1) / 2.0)
            matrix = cv2.getRotationMatrix2D(merkez, aci, 1.0)
            donmus = cv2.warpAffine(
                goruntu,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return np.clip(donmus, 0, 255).astype(np.uint8)
        donmus = ndimage.rotate(goruntu, angle=aci, reshape=False, order=1, mode='nearest')
        return np.clip(donmus, 0, 255).astype(np.uint8)
    
    def parlaklik_kontrast_degistir(self, goruntu: np.ndarray) -> np.ndarray:
        """Parlaklık ve kontrast rastgele değiştir."""
        b = self._random.uniform(*PARLAKLIK_ARALIK)
        c = self._random.uniform(*KONTRAST_ARALIK)
        
        degismis = goruntu.astype(np.float32) * c + b
        return np.clip(degismis, 0, 255).astype(np.uint8)
    
    def elastic_deformation(self, goruntu: np.ndarray, alpha: float = ELASTIC_ALPHA, 
                           sigma: float = ELASTIC_SIGMA) -> np.ndarray:
        """
        Elastik deformasyon uygula.
        
        Beyin dokusunun doğal varyasyonlarını simüle eder. Medical imaging
        için önemli bir augmentation tekniğidir.
        
        Args:
            goruntu: Girdi görüntüsü
            alpha: Deformasyon şiddeti (yüksek = daha fazla bozulma)
            sigma: Deformasyon yumuşaklığı (yüksek = daha yumuşak)
            
        Returns:
            Deforme edilmiş görüntü
        """
        if not ELASTIC_DEFORMATION_AKTIF:
            return goruntu
        
        h, w = goruntu.shape[:2]
        
        # Rastgele displacement field oluştur
        if CV2_AVAILABLE:
            dx = cv2.GaussianBlur(
                (self._np_random.random((h, w)) * 2 - 1).astype(np.float32),
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_CONSTANT,
            ) * alpha
            dy = cv2.GaussianBlur(
                (self._np_random.random((h, w)) * 2 - 1).astype(np.float32),
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_CONSTANT,
            ) * alpha

            x, y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
            )
            distorted = cv2.remap(
                goruntu,
                (x + dx).astype(np.float32),
                (y + dy).astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return np.clip(distorted, 0, 255).astype(np.uint8)

        shape = goruntu.shape
        dx = ndimage.gaussian_filter(
            (self._np_random.random(shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        dy = ndimage.gaussian_filter(
            (self._np_random.random(shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        
        # Mesh grid oluştur
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Görüntüyü deforme et
        distorted = ndimage.map_coordinates(goruntu, indices, order=1, mode='reflect')
        return distorted.reshape(shape).astype(np.uint8)
    
    def random_crop_resize(self, goruntu: np.ndarray, crop_ratio: float = RANDOM_CROP_RATIO) -> np.ndarray:
        """
        Rastgele kırp ve orijinal boyuta geri getir.
        
        Args:
            goruntu: Girdi görüntüsü
            crop_ratio: Kırpma oranı (0.9 = %90'ını al)
            
        Returns:
            Kırpılmış ve yeniden boyutlandırılmış görüntü
        """
        if not RANDOM_CROP_AKTIF:
            return goruntu
        
        h, w = goruntu.shape
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        
        # Rastgele başlangıç noktası seç
        top = self._random.randint(0, h - new_h)
        left = self._random.randint(0, w - new_w)
        
        # Kırp
        cropped = goruntu[top:top+new_h, left:left+new_w]
        
        # Orijinal boyuta geri getir
        if CV2_AVAILABLE:
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray(cropped)
            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            resized = np.array(pil_img)
        
        return resized
    
    def gaussian_noise(self, goruntu: np.ndarray, mean: float = GAUSSIAN_NOISE_MEAN,
                      sigma: float = GAUSSIAN_NOISE_SIGMA) -> np.ndarray:
        """
        Gaussian gürültü ekle.
        
        MRI cihazındaki termal gürültüyü simüle eder.
        
        Args:
            goruntu: Girdi görüntüsü
            mean: Gürültü ortalaması
            sigma: Gürültü standart sapması
            
        Returns:
            Gürültülü görüntü
        """
        if not GAUSSIAN_NOISE_AKTIF:
            return goruntu
        
        noise = self._np_random.normal(mean, sigma, goruntu.shape)
        noisy = goruntu.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def intensity_shift(self, goruntu: np.ndarray, limit: float = INTENSITY_SHIFT_LIMIT) -> np.ndarray:
        """
        Yoğunluk kayması uygula.
        
        Farklı MRI cihazlarındaki kalibrasyon farklılıklarını simüle eder.
        
        Args:
            goruntu: Girdi görüntüsü
            limit: Yoğunluk kayması limiti (0.1 = %10)
            
        Returns:
            Yoğunluğu kaymış görüntü
        """
        if not INTENSITY_SHIFT_AKTIF:
            return goruntu
        
        shift_factor = self._random.uniform(1 - limit, 1 + limit)
        shifted = goruntu.astype(np.float32) * shift_factor
        return np.clip(shifted, 0, 255).astype(np.uint8)
    
    def veri_artir(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Veri artırma (augmentation) işlemleri uygula.
        
        Veri artırma, mevcut görüntülerden yeni varyasyonlar oluşturarak
        veri setini genişletir. Bu, modelin daha iyi genelleme yapmasını sağlar.
        
        Uygulanan işlemler:
        - Basit: Aynalama, döndürme, parlaklık/kontrast değişimi
        - Gelişmiş: Elastik deformasyon, rastgele kırpma, gaussian gürültü, yoğunluk kayması
        
        Args:
            goruntu: Girdi görüntüsü
            
        Returns:
            Artırılmış görüntü
        """
        # Veri artırma kapalıysa direkt dön
        if not VERI_ARTIRMA_AKTIF:
            return goruntu
        
        g = goruntu.copy()  # Orijinali korumak için kopyala
        
        # BASIT AUGMENTATION
        # Beyin MR'larında ayna dönüşümleri anatomik yanlılık üretebilir.
        if YATAY_AYNA_AKTIF and self._random.random() < YATAY_AYNA_OLASILIK:
            g = self.yatay_ayna(g)

        # Küçük açılı rotasyon
        g = self.rastgele_dondur(g)
        
        # Parlaklık ve kontrast değişimi
        g = self.parlaklik_kontrast_degistir(g)
        
        # GELİŞMİŞ MEDİKAL AUGMENTATION
        # %40 ihtimalle elastik deformasyon
        if self._random.random() < 0.4:
            g = self.elastic_deformation(g)
        
        # %30 ihtimalle rastgele kırp ve yeniden boyutlandır
        if self._random.random() < 0.3:
            g = self.random_crop_resize(g)
        
        # %25 ihtimalle gaussian gürültü ekle
        if self._random.random() < 0.25:
            g = self.gaussian_noise(g)
        
        # %30 ihtimalle yoğunluk kayması
        if self._random.random() < 0.3:
            g = self.intensity_shift(g)
        
        return g
    
