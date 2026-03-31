"""
goruntu_isleyici.py
-------------------
MRI görüntülerini işleme ve özellik çıkarma modülü.
Tüm ön işleme, normalizasyon ve veri artırma işlevlerini içerir.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import random
import re
import math
from scipy import ndimage
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
from itertools import count
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[UYARI] OpenCV yüklü değil. Bazı özellikler çalışmayabilir.")

try:
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[UYARI] scikit-image yüklü değil. Histogram eşitleme devre dışı.")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("[UYARI] SimpleITK yüklü değil. N4ITK bias correction ve gelişmiş registration kullanılamayacak.")

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


# Multiprocessing için global fonksiyon (pickle edilebilir olmalı)
_worker_isleyici = None
_isleyici_sayaci = count()

def _islem_worker_init():
    """Worker başına tek GorselIsleyici instance oluştur."""
    global _worker_isleyici
    _worker_isleyici = GorselIsleyici()

def _islem_wrapper(args):
    """Tek bir görüntüyü işlemek için wrapper fonksiyon."""
    global _worker_isleyici
    if _worker_isleyici is None:
        _worker_isleyici = GorselIsleyici()
    dosya_info, cikti_klasoru, artirma_carpanlari = args
    return _worker_isleyici._tek_goruntu_isle(dosya_info, cikti_klasoru, artirma_carpanlari)


class GorselIsleyici:
    """MRI görüntü işleme sınıfı."""
    
    def __init__(self):
        """İşleyiciyi başlat."""
        self.temel_tohum = RASTGELE_TOHUM
        self._random, self._np_random = self._rng_olustur(self.temel_tohum)
        self.template_image = None  # Registration için şablon görüntü
        self.kalite_istatistikleri = {
            "toplam": 0,
            "basarili": 0,
            "kalite_hatasi": 0
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
        instance_idx = next(_isleyici_sayaci)
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

    @staticmethod
    def _sinif_klasorleri_var_mi(klasor_yolu: Path) -> bool:
        """Verilen klasörde en az bir bilinen sınıf klasörü var mı?"""
        return any((klasor_yolu / sinif).exists() for sinif in SINIF_KLASORLERI)

    @classmethod
    def _split_klasorleri_var_mi(cls, klasor_yolu: Path) -> bool:
        """Girdi kokunde trainval/test alt klasorleri var mi?"""
        return all(
            cls._sinif_klasorleri_var_mi(klasor_yolu / split_adi)
            for split_adi in ("trainval", "test")
        )

    def _giris_klasoru_cozumle(self, klasor_yolu: Path) -> Path:
        """
        Girdi klasörünü veri yapısına göre otomatik çöz.

        Desteklenen yapılar:
        1) Veri_Seti/OriginalDataset/<SinifAdi>/
        2) Özel bir klasörde doğrudan <SinifAdi>/
        """
        klasor_yolu = Path(klasor_yolu)

        # Klasör doğrudan sınıf klasörlerini içeriyorsa olduğu gibi kullan.
        if self._sinif_klasorleri_var_mi(klasor_yolu):
            return klasor_yolu

        # Kök klasör verildiğinde önce klasör altındaki bilinen alt yapıları dene.
        adaylar = [
            klasor_yolu / "OriginalDataset",
        ]
        for aday in adaylar:
            if aday.exists() and self._sinif_klasorleri_var_mi(aday):
                print(f"[BILGI] Girdi klasoru otomatik cozuldu: {aday}")
                return aday

        return klasor_yolu
    
    def gorselleri_listele(
        self, klasor_yolu: Path = ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
    ) -> List[Dict]:
        """
        Veri setindeki tüm görüntüleri listele.
        
        Bu fonksiyon, belirtilen klasör altındaki tüm sınıf klasörlerini tarar
        ve her görüntü için yol, sınıf adı ve etiket bilgilerini toplar.
        
        Returns:
            List[Dict]: [{"yol": dosya_yolu, "sinif": sınıf_adı, "etiket": etiket}, ...]
        """
        dosyalar = []  # Tüm görüntü bilgilerini saklayacak liste
        klasor_yolu = self._giris_klasoru_cozumle(klasor_yolu)
        
        # Her sınıf klasörünü sırayla tara
        for sinif_adi in SINIF_KLASORLERI:
            sinif_klasoru = klasor_yolu / sinif_adi
            
            # Klasör yoksa uyar ve devam et
            if not sinif_klasoru.exists():
                print(f"[UYARI] Klasör bulunamadı: {sinif_klasoru}")
                continue
            
            # Klasördeki tüm dosyaları tara
            for dosya in sorted(sinif_klasoru.iterdir(), key=lambda p: (p.name.lower(), p.name)):
                # Sadece görüntü dosyalarını işle (.jpg, .png, vb.)
                if dosya.suffix.lower() in GORUNTU_UZANTILARI:
                    kaynak_id = self.kaynak_id_belirle(dosya.name)
                    dosyalar.append({
                        "yol": str(dosya),
                        "sinif": sinif_adi,
                        "etiket": SINIF_ETIKETI[sinif_adi],
                        "kaynak_id": kaynak_id,
                        "kaynak_grup": f"{sinif_adi}::{kaynak_id}",
                    })
        
        return dosyalar

    def veri_dosyalarini_bol(self, dosyalar: List[Dict], test_orani: float = TEST_ORANI) -> Tuple[List[Dict], List[Dict]]:
        """Ham/original goruntuleri leak-free trainval ve test olarak bol."""
        from sklearn.model_selection import train_test_split

        if test_orani <= 0 or test_orani >= 1.0:
            raise ValueError("test_orani 0 ile 1 arasinda olmali.")

        grup_kayitlari: Dict[str, int] = {}
        for dosya_info in dosyalar:
            kaynak_grup = dosya_info["kaynak_grup"]
            etiket = int(dosya_info["etiket"])
            mevcut = grup_kayitlari.get(kaynak_grup)
            if mevcut is not None and mevcut != etiket:
                raise ValueError(
                    f"Tutarsiz etiket bulundu: {kaynak_grup} hem {mevcut} hem {etiket} ile eslendi."
                )
            grup_kayitlari[kaynak_grup] = etiket

        grup_anahtarlari = sorted(grup_kayitlari)
        grup_etiketleri = [grup_kayitlari[grup] for grup in grup_anahtarlari]
        benzersiz_etiketler = sorted(set(grup_etiketleri))
        sinif_sayisi = len(benzersiz_etiketler)

        if len(grup_anahtarlari) < sinif_sayisi * 2:
            raise ValueError(
                "Trainval/test bolmesi icin yeterli kaynak grup yok. "
                f"Bulunan grup: {len(grup_anahtarlari)}, gereken minimum: {sinif_sayisi * 2}"
            )

        sinif_grup_sayilari: Dict[int, int] = {}
        for etiket in grup_etiketleri:
            sinif_grup_sayilari[etiket] = sinif_grup_sayilari.get(etiket, 0) + 1
        yetersiz = sorted(etiket for etiket, sayi in sinif_grup_sayilari.items() if sayi < 2)
        if yetersiz:
            raise ValueError(
                "Harici test split'i olusturmak icin her sinifta en az 2 farkli kaynak grup gerekli. "
                f"Eksik etiketler: {yetersiz}"
            )

        toplam_grup = len(grup_anahtarlari)
        test_grup_sayisi = max(math.ceil(toplam_grup * test_orani), sinif_sayisi)
        trainval_grup_sayisi = toplam_grup - test_grup_sayisi
        if test_grup_sayisi >= toplam_grup or trainval_grup_sayisi < sinif_sayisi:
            raise ValueError(
                "Trainval/test bolmesi icin yeterli kaynak grup yok. "
                f"Toplam grup: {toplam_grup}, trainval grup: {trainval_grup_sayisi}, test grup: {test_grup_sayisi}"
            )

        trainval_gruplari, test_gruplari = train_test_split(
            grup_anahtarlari,
            test_size=test_grup_sayisi,
            stratify=grup_etiketleri,
            random_state=RASTGELE_TOHUM,
        )

        trainval_gruplari = set(trainval_gruplari)
        test_gruplari = set(test_gruplari)
        trainval_dosyalar = [d for d in dosyalar if d["kaynak_grup"] in trainval_gruplari]
        test_dosyalar = [d for d in dosyalar if d["kaynak_grup"] in test_gruplari]
        return trainval_dosyalar, test_dosyalar
    
    def goruntu_kalite_kontrol(self, goruntu: np.ndarray) -> Tuple[bool, str]:
        """
        Görüntü kalitesini kontrol et.
        
        Bu fonksiyon, bozuk, düşük kaliteli veya hatalı görüntüleri tespit eder.
        Model eğitiminde kullanılmaması gereken görüntüleri filtreler.
        
        Kontroller:
        1. Çok karanlık görüntü (ortalama yoğunluk < MIN_MEAN_INTENSITY)
        2. Çok aydınlık görüntü (ortalama yoğunluk > MAX_MEAN_INTENSITY)
        3. Düşük kontrast (std < MIN_STD_INTENSITY) - düz/tek renkli görüntü
        4. Çok fazla siyah piksel (> MAX_BLACK_RATIO) - boş/hatalı tarama
        
        Args:
            goruntu: Kontrol edilecek görüntü
            
        Returns:
            Tuple[bool, str]: (kalite_ok, hata_mesaji)
        """
        if not KALITE_KONTROL_AKTIF:
            return True, ""
        
        if goruntu is None or goruntu.size == 0:
            return False, "Boş görüntü"
        
        # Temel istatistikler
        mean_intensity = np.mean(goruntu)
        std_intensity = np.std(goruntu)
        
        # 1. Çok karanlık kontrol
        if mean_intensity < MIN_MEAN_INTENSITY:
            return False, f"Çok karanlık (mean={mean_intensity:.1f})"
        
        # 2. Çok aydınlık kontrol
        if mean_intensity > MAX_MEAN_INTENSITY:
            return False, f"Çok aydınlık (mean={mean_intensity:.1f})"
        
        # 3. Düşük kontrast kontrol (düz görüntü)
        if std_intensity < MIN_STD_INTENSITY:
            return False, f"Düşük kontrast (std={std_intensity:.1f})"
        
        # 4. Siyah piksel oranı kontrol (boş görüntü)
        black_pixels = np.sum(goruntu < 10)
        black_ratio = black_pixels / goruntu.size
        if black_ratio > MAX_BLACK_RATIO:
            return False, f"Çok fazla siyah piksel ({black_ratio*100:.1f}%)"
        
        # Tüm kontroller başarılı
        return True, ""
    
    def goruntu_yukle(self, dosya_yolu: str) -> Optional[np.ndarray]:
        """
        Görüntü dosyasını yükle ve gri tonlamaya çevir.
        
        Args:
            dosya_yolu: Görüntü dosyasının yolu
            
        Returns:
            np.ndarray veya None
        """
        try:
            goruntu = Image.open(dosya_yolu)
            if goruntu.mode != 'L':
                goruntu = goruntu.convert('L')
            return np.array(goruntu)
        except Exception as e:
            print(f"[HATA] Görüntü yüklenemedi {dosya_yolu}: {e}")
            return None
    
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
    
    def histogram_esitle(self, goruntu: np.ndarray, adaptive: bool = True) -> np.ndarray:
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
        
        # OpenCV varsa onu kullan (daha hızlı)
        if CV2_AVAILABLE:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(goruntu)
        # Değilse scikit-image kullan
        elif SKIMAGE_AVAILABLE:
            result = exposure.equalize_adapthist(goruntu, clip_limit=clip_limit / 100.0)
            # equalize_adapthist float64 [0,1] döner; uint8'e normalize et
            return (result * 255.0).clip(0, 255).astype(np.uint8)
        # Hiçbiri yoksa orijinal görüntüyü dön
        else:
            return goruntu
    
    def boyutlandir(self, goruntu: np.ndarray, 
                    genislik: int = HEDEF_GENISLIK, 
                    yukseklik: int = HEDEF_YUKSEKLIK) -> np.ndarray:
        """
        Görüntüyü hedef boyuta yeniden boyutlandır.
        
        Makine öğrenmesi modellerinde tüm görüntülerin aynı boyutta olması gerekir.
        Bu fonksiyon MRI görüntülerini standart boyuta (örn: 256x256) getirir.
        
        İnterpolasyon: LINEAR (bilinear interpolation)
        - Hızlı ve kaliteli
        - Piksel değerlerini yumuşak geçişlerle hesaplar
        
        Args:
            goruntu: Kaynak görüntü (numpy array)
            genislik: Hedef genişlik (pixel)
            yukseklik: Hedef yükseklik (pixel)
            
        Returns:
            Yeniden boyutlandırılmış görüntü
        """
        if CV2_AVAILABLE:
            # OpenCV ile hızlı yeniden boyutlandırma
            return cv2.resize(goruntu, (genislik, yukseklik), interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray(goruntu)
            pil_img = pil_img.resize((genislik, yukseklik), Image.LANCZOS)
            return np.array(pil_img)
    
    def _bilateral_filtre_uygula(self, goruntu: np.ndarray) -> np.ndarray:
        """OpenCV mevcutsa kenar korumali bilateral filtre uygula."""
        if not CV2_AVAILABLE:
            return goruntu

        filtered = cv2.bilateralFilter(goruntu, d=5, sigmaColor=35, sigmaSpace=35)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def gurultu_gider(self, goruntu: np.ndarray, metod: str = 'auto') -> np.ndarray:
        """
        Görüntüden gürültüyü temizle.

        MRI görüntülerinde sıkça salt-and-pepper ve Gaussian gürültü görülür.
        Bu gürültüler model performansını düşürür, temizlenmesi gerekir.

        Args:
            goruntu: Girdi görüntüsü
            metod: 'auto', 'median', 'gaussian' veya 'bilateral'

        Returns:
            Gürültüsü azaltılmış görüntü
        """
        if metod == 'auto':
            if GELISMIS_FILTRE_AKTIF:
                if BILATERAL_FILTRE_AKTIF and CV2_AVAILABLE:
                    return self._bilateral_filtre_uygula(goruntu)
                if GAUSSIAN_BLUR_AKTIF:
                    return self.gurultu_gider(goruntu, metod='gaussian')
            return self.gurultu_gider(goruntu, metod='median')

        if metod == 'median':
            # Median filtre: Salt-and-pepper gürültüsü için ideal
            filtered = ndimage.median_filter(goruntu, size=3)
            return np.clip(filtered, 0, 255).astype(np.uint8)
        elif metod == 'gaussian' and GAUSSIAN_BLUR_AKTIF:
            # Gaussian filtre: Genel gürültü azaltma
            filtered = ndimage.gaussian_filter(goruntu, sigma=GAUSSIAN_BLUR_SIGMA)
            return np.clip(filtered, 0, 255).astype(np.uint8)
        elif metod == 'bilateral' and BILATERAL_FILTRE_AKTIF and CV2_AVAILABLE:
            return self._bilateral_filtre_uygula(goruntu)
        else:
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
        """Ayarlardaki kernel boyutunu kullanarak kare yapı elemani üret."""
        boyut = int(kernel_boyutu or MORFOLOJIK_KERNEL_BOYUTU)
        boyut = max(1, boyut)
        return np.ones((boyut, boyut), dtype=bool)

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
        duzenli = ndimage.binary_opening(duzenli, structure=temel)
        duzenli = ndimage.binary_closing(duzenli, structure=close_kernel)

        if dilation_scale > 0:
            dilate_kernel = self._morfolojik_yapi(MORFOLOJIK_KERNEL_BOYUTU * dilation_scale)
            duzenli = ndimage.binary_dilation(duzenli, structure=dilate_kernel)

        return duzenli.astype(bool)
    
    def _simple_skull_strip(self, goruntu: np.ndarray) -> np.ndarray:
        """Basit skull stripping (Otsu thresholding)."""
        try:
            from skimage.filters import threshold_otsu
            
            # Eşik değeri bul
            esik = threshold_otsu(goruntu)
            
            # Binary maske oluştur
            maske = goruntu > esik
            maske = self._maskeyi_duzenle(maske, closing_scale=2)
            
            # Maskeyi uygula
            return (goruntu * maske).astype(np.uint8)
            
        except ImportError:
            # scikit-image yoksa basit eşikleme kullan
            esik = np.percentile(goruntu, 30)
            maske = goruntu > esik
            maske = self._maskeyi_duzenle(maske, closing_scale=2)
            return (goruntu * maske).astype(np.uint8)
    
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
            from skimage.filters import threshold_otsu
            from skimage.morphology import (
                remove_small_objects, remove_small_holes
            )
            from skimage.measure import label
            
            # 1. Otsu eşikleme ile başlangıç maskesi
            esik = threshold_otsu(goruntu)
            maske = goruntu > esik
            maske = self._kenar_maskesini_temizle(maske)
            
            # 2. Küçük nesneleri temizle (min_size = toplam pikselin %0.5'i)
            min_size = int(goruntu.size * 0.005)
            maske = remove_small_objects(maske, min_size=min_size)
            
            # 3. Morfolojik gürültü temizleme
            maske = self._maskeyi_duzenle(maske, closing_scale=1)
            
            # 4. Küçük delikleri kapat
            maske = remove_small_holes(maske, area_threshold=min_size)
            
            # 5. En büyük bağlantılı bileşeni bul (beyin olmalı)
            labeled_mask = label(maske)
            if labeled_mask.max() > 0:
                # Her bileşenin boyutunu hesapla
                regions = np.bincount(labeled_mask.ravel())
                # Arka plan (0) hariç en büyük bölgeyi bul
                largest_region = regions[1:].argmax() + 1
                maske = labeled_mask == largest_region

            # 6. Kenarlari yumusat ve beyin dokusunu korumak icin hafif genislet
            maske = self._maskeyi_duzenle(maske, closing_scale=2, dilation_scale=1)
            
            # 7. Maskeyi uygula
            result = (goruntu * maske).astype(np.uint8)
            
            return result
            
        except ImportError as e:
            print(f"[UYARI] Advanced skull stripping için gerekli kütüphane yok: {e}")
            return self._simple_skull_strip(goruntu)
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
        2. "simple": Basit Gaussian blur tabanlı (hızlı) - varsayılan fallback

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
        
        # Basit metod (fallback)
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
            bias_field = ndimage.gaussian_filter(img_float, sigma=50)
            
            # Ortalamayı bul (sıfıra bölme önlemi)
            mean_bias = np.mean(bias_field)
            if mean_bias < 1e-6:
                return goruntu
            
            # Bias field'ı kaldır (orijinal / bias)
            corrected = img_float / (bias_field / mean_bias)
            
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
            
            # Kütle merkezini hesapla
            center_of_mass = ndimage.center_of_mass(binary)
            
            # Görüntü merkezini hesapla
            img_center = np.array(goruntu.shape) / 2.0
            
            # Gerekli kaydırma miktarını hesapla
            shift = img_center - np.array(center_of_mass)
            
            # Görüntüyü kaydır
            aligned = ndimage.shift(goruntu, shift, mode='constant', cval=0)
            
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
            # İlk görüntüde güvenli fallback: en azından merkez hizalama uygula.
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
    
    def z_score_normalize(self, goruntu: np.ndarray) -> np.ndarray:
        """Z-score normalizasyonu uygula (mean=0, std=1)."""
        if not Z_SCORE_NORMALIZASYON_AKTIF:
            return goruntu
        
        mean = np.mean(goruntu)
        std = np.std(goruntu)
        
        if std < 1e-6:
            return goruntu
        
        return ((goruntu - mean) / std * 50 + 128).clip(0, 255).astype(np.uint8)
    
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
            # Standard: percentile + CLAHE (önerilen)
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=True)
            
        elif strategy == "aggressive":
            # Aggressive: percentile + CLAHE + z-score
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=True)
            goruntu = self.z_score_normalize(goruntu)
            
        else:
            # Varsayılan: standard
            print(f"[UYARI] Bilinmeyen strateji: {strategy}, 'standard' kullanılıyor")
            goruntu = self.yogunluk_normalize(goruntu)
            goruntu = self.histogram_esitle(goruntu, adaptive=True)
        
        return goruntu
    
    def goruntu_kaydet(self, goruntu: np.ndarray, cikti_yolu: str):
        """İşlenmiş görüntüyü kaydet."""
        try:
            pil_img = Image.fromarray(goruntu)
            pil_img.save(cikti_yolu)
        except Exception as e:
            print(f"[HATA] Görüntü kaydedilemedi {cikti_yolu}: {e}")
    
    # ==================== VERİ ARTIRMA FONKSİYONLARI ====================
    
    @staticmethod
    def yatay_ayna(goruntu: np.ndarray) -> np.ndarray:
        """Yatay ayna (flip)."""
        return np.fliplr(goruntu)
    
    @staticmethod
    def dikey_ayna(goruntu: np.ndarray) -> np.ndarray:
        """Dikey ayna (flip)."""
        return np.flipud(goruntu)
    
    def rastgele_dondur(self, goruntu: np.ndarray) -> np.ndarray:
        """Küçük açılı rotasyon uygula."""
        if not ROTASYON_AKTIF:
            return goruntu
        aci = self._random.uniform(-ROTASYON_MAKS_ACI, ROTASYON_MAKS_ACI)
        if abs(aci) < 1e-3:
            return goruntu
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
        
        shape = goruntu.shape
        
        # Rastgele displacement field oluştur
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
    
    # ==================== TOPLU İŞLEM FONKSİYONLARI ====================
    
    def _tek_goruntu_isle(self, dosya_info: Dict, cikti_klasoru: Path,
                          artirma_carpanlari: Dict[str, int]) -> Optional[Dict]:
        """
        Tek bir görüntüyü işle (paralel işlem için).

        Args:
            dosya_info: Dosya bilgileri sözlüğü
            cikti_klasoru: Çıktı klasörü
            artirma_carpanlari: Sınıf bazlı augmentation çarpanları

        Returns:
            İstatistikler sözlüğü veya None
        """
        try:
            # Çıktı klasörü oluştur
            sinif_cikti = cikti_klasoru / dosya_info["sinif"]
            self.klasor_olustur(sinif_cikti)

            # Görüntüyü işle (kalite kontrol içinde yapılır)
            goruntu = self.goruntu_isle(dosya_info["yol"])

            sonuc = {
                'basarili': 0,
                'basarisiz': 0,
                'kalite_hatasi': 0,
                'istatistikler': {sinif: 0 for sinif in SINIF_KLASORLERI}
            }

            if goruntu is not None:
                # Orijinal görüntüyü kaydet
                dosya_adi = self._cikti_dosya_koku(dosya_info["yol"])
                cikti_yolu = sinif_cikti / f"{dosya_adi}.png"
                self.goruntu_kaydet(goruntu, str(cikti_yolu))

                sonuc['basarili'] = 1
                sonuc['istatistikler'][dosya_info["sinif"]] = 1

                # Sınıf bazlı veri artırma
                if VERI_ARTIRMA_AKTIF:
                    sinif = dosya_info["sinif"]
                    carpan = artirma_carpanlari.get(sinif, ARTIRMA_CARPANI)

                    for i in range(carpan):
                        artirmis_goruntu = self.veri_artir(goruntu)
                        artirmis_yol = sinif_cikti / f"{dosya_adi}_aug{i+1}.png"
                        self.goruntu_kaydet(artirmis_goruntu, str(artirmis_yol))
                        sonuc['istatistikler'][dosya_info["sinif"]] += 1
            else:
                sonuc['basarisiz'] = 1
                sonuc['kalite_hatasi'] = self.kalite_istatistikleri.get('kalite_hatasi', 0)
                # Worker'daki sayacı sıfırla (sonraki görüntü için)
                self.kalite_istatistikleri['kalite_hatasi'] = 0

            return sonuc

        except Exception as e:
            print(f"[HATA] Goruntu islenemedi {dosya_info.get('yol', '?')}: {type(e).__name__}: {e}")
            return {
                'basarili': 0,
                'basarisiz': 1,
                'kalite_hatasi': 0,
                'istatistikler': {sinif: 0 for sinif in SINIF_KLASORLERI}
            }
    
    def sinif_bazli_artirma_carpani_hesapla(self, dosyalar: List[Dict]) -> Dict[str, int]:
        """
        Sınıf dengesizliğine göre augmentation çarpanını hesapla.
        
        Az örnekli sınıfları daha fazla artırarak veri dengesizliğini azaltır.
        
        Args:
            dosyalar: Tüm dosya bilgileri listesi
            
        Returns:
            Her sınıf için augmentation çarpanı sözlüğü
        """
        if not SINIF_BAZLI_ARTIRMA_AKTIF:
            # Tüm sınıflar için aynı çarpan
            return {sinif: ARTIRMA_CARPANI for sinif in SINIF_KLASORLERI}

        if SINIF_BAZLI_CARPANLAR:
            return {
                sinif: max(0, int(SINIF_BAZLI_CARPANLAR.get(sinif, ARTIRMA_CARPANI)))
                for sinif in SINIF_KLASORLERI
            }
        
        # Her sınıftaki örnek sayısını hesapla
        sinif_sayilari = {}
        for sinif in SINIF_KLASORLERI:
            sayi = sum(1 for d in dosyalar if d["sinif"] == sinif)
            sinif_sayilari[sinif] = sayi
        
        # En çok örnekli sınıfı bul
        max_sayi = max(sinif_sayilari.values())
        
        # Her sınıf için çarpan hesapla
        artirma_carpanlari = {}
        for sinif, sayi in sinif_sayilari.items():
            if sayi == 0:
                artirma_carpanlari[sinif] = 0
            else:
                # Az örnekli sınıflar daha fazla artırılır
                carpan = int(max_sayi / sayi)
                # Maksimum 5x sınırı koy (aşırı artırmayı önle)
                artirma_carpanlari[sinif] = min(carpan, 5)
        
        print("\n[BILGI] Sinif bazli augmentation carpanlari:")
        for sinif, carpan in artirma_carpanlari.items():
            print(f"   {sinif}: {carpan}x (mevcut: {sinif_sayilari[sinif]} ornek)")
        
        return artirma_carpanlari
    
    def tum_gorselleri_isle(
        self,
        cikti_klasoru: Path = CIKTI_KLASORU,
        giris_klasoru: Path = None,
        dosyalar: Optional[List[Dict]] = None,
        artirma_carpanlari: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """
        Tüm MRI görüntülerini toplu olarak işle ve kaydet.

        Her görüntüye goruntu_isle() pipeline'ı uygulanır, ardından
        sınıf bazlı augmentation ile veri artırma yapılır.

        Args:
            cikti_klasoru: İşlenmiş görüntülerin kaydedileceği klasör
            giris_klasoru: Ham görüntülerin bulunduğu klasör
            dosyalar: Önceden listelenmiş dosya bilgileri (None ise otomatik taranır)
            artirma_carpanlari: Sınıf bazlı augmentation çarpanları

        Returns:
            Dict: Sınıf bazlı istatistikler
        """
        self.klasor_olustur(cikti_klasoru)
        if dosyalar is None:
            if giris_klasoru is None:
                giris_klasoru = ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
            dosyalar = self.gorselleri_listele(giris_klasoru)
        else:
            dosyalar = list(dosyalar)
        
        if not dosyalar:
            print("[HATA] Hiç görüntü bulunamadı!")
            return {}
        
        print(f"\n{len(dosyalar)} görüntü bulundu. İşleniyor...\n")
        
        # Kalite istatistiklerini sıfırla
        self.kalite_istatistikleri = {
            "toplam": len(dosyalar),
            "basarili": 0,
            "kalite_hatasi": 0
        }
        
        if artirma_carpanlari is None:
            artirma_carpanlari = self.sinif_bazli_artirma_carpani_hesapla(dosyalar)
        
        basarili = 0
        basarisiz = 0
        kalite_hatasi_toplam = 0
        istatistikler = {sinif: 0 for sinif in SINIF_KLASORLERI}

        # Her görüntü için argümanları hazırla
        islem_args = [(dosya_info, cikti_klasoru, artirma_carpanlari) for dosya_info in dosyalar]

        # REGISTRATION_AKTIF + affine/rigid modunda template tutarlılığı
        # gerektiğinden sequential çalıştır; aksi halde paralel.
        paralel_kullan = True
        if REGISTRATION_AKTIF and SITK_AVAILABLE and REGISTRATION_METHOD in ("affine", "rigid"):
            paralel_kullan = False
            print("[BILGI] Affine/rigid registration aktif - sequential modda calisiyor (template tutarliligi icin)")

        sonuclar = []
        if paralel_kullan and self.n_jobs > 1:
            # Paralel işleme ile hızlandırma
            print(f"[BILGI] Paralel isleme aktif: {self.n_jobs} cekirdek kullaniliyor")
            try:
                with Pool(processes=self.n_jobs, initializer=_islem_worker_init) as pool:
                    sonuclar = list(tqdm(
                        pool.imap(_islem_wrapper, islem_args),
                        total=len(dosyalar),
                        desc="Goruntuler isleniyor (paralel)"
                    ))
            except Exception as e:
                print(
                    "[UYARI] Paralel isleme baslatilamadi; sequential moda geciliyor: "
                    f"{type(e).__name__}: {e}"
                )

        if not sonuclar:
            # Sequential işleme
            for args in tqdm(islem_args, desc="Goruntuler isleniyor"):
                sonuclar.append(self._tek_goruntu_isle(*args))

        # Sonuçları topla
        for sonuc in sonuclar:
            if sonuc is not None:
                basarili += sonuc['basarili']
                basarisiz += sonuc['basarisiz']
                kalite_hatasi_toplam += sonuc.get('kalite_hatasi', 0)
                for sinif, sayi in sonuc['istatistikler'].items():
                    istatistikler[sinif] += sayi

        self.kalite_istatistikleri['basarili'] = basarili
        self.kalite_istatistikleri['kalite_hatasi'] = kalite_hatasi_toplam

        # Sonuçları yazdır
        print(f"\n{'='*60}")
        print(f"Basarili: {basarili}")
        print(f"Basarisiz: {basarisiz}")
        print(f"Kalite hatasi: {kalite_hatasi_toplam}")
        print(f"\nSinif bazli istatistikler (augmentation sonrasi):")
        for sinif, sayi in istatistikler.items():
            print(f"   {sinif}: {sayi} goruntu")
        print(f"{'='*60}\n")
        
        return istatistikler

    def tum_gorselleri_isle_ve_bol(
        self,
        cikti_klasoru: Path = CIKTI_KLASORU,
        giris_klasoru: Path = None,
    ) -> Dict[str, Dict]:
        """Goruntuleri leak-free trainval/test yapisina ayirip isle."""
        self.klasor_olustur(cikti_klasoru)
        giris_klasoru = Path(giris_klasoru) if giris_klasoru else ON_ISLEME_VARSAYILAN_GIRIS_KLASORU
        giris_klasoru = self._giris_klasoru_cozumle(giris_klasoru)

        if self._split_klasorleri_var_mi(giris_klasoru):
            print("[BILGI] Girdi klasorunde trainval/test yapisi algilandi; mevcut split korunacak.")
            trainval_dosyalar = self.gorselleri_listele(giris_klasoru / "trainval")
            test_dosyalar = self.gorselleri_listele(giris_klasoru / "test")
        else:
            tum_dosyalar = self.gorselleri_listele(giris_klasoru)
            if not tum_dosyalar:
                print("[HATA] Hic goruntu bulunamadi!")
                return {}
            trainval_dosyalar, test_dosyalar = self.veri_dosyalarini_bol(tum_dosyalar)

        sifir_aug = {sinif: 0 for sinif in SINIF_KLASORLERI}
        print(
            "\n[BILGI] Islenmis goruntuler split bazinda kaydedilecek: "
            f"trainval={len(trainval_dosyalar)} goruntu, test={len(test_dosyalar)} goruntu"
        )
        beklenen_trainval_siniflari = sorted({dosya["sinif"] for dosya in trainval_dosyalar})
        beklenen_test_siniflari = sorted({dosya["sinif"] for dosya in test_dosyalar})

        trainval_istatistik = self.tum_gorselleri_isle(
            cikti_klasoru=Path(cikti_klasoru) / "trainval",
            dosyalar=trainval_dosyalar,
        )
        test_istatistik = self.tum_gorselleri_isle(
            cikti_klasoru=Path(cikti_klasoru) / "test",
            dosyalar=test_dosyalar,
            artirma_carpanlari=sifir_aug,
        )
        self._sinif_kapsamini_dogrula(trainval_istatistik, beklenen_trainval_siniflari, "TrainVal")
        self._sinif_kapsamini_dogrula(test_istatistik, beklenen_test_siniflari, "Test")

        return {
            "trainval": trainval_istatistik,
            "test": test_istatistik,
        }
