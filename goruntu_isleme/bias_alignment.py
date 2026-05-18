"""Center-of-mass tabanli basit registration/hizalama."""

import cv2
import numpy as np

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselBiasAlignmentMixin:
    def center_of_mass_alignment(self, goruntu: np.ndarray) -> np.ndarray:
        """
        Görüntüyü kütle merkezine göre hizala.

        Farklı kesit/merkez kaymalarına sahip 2D MRI dilimlerini hedef
        çerçevenin merkezine taşır. Yalnızca öteleme yapar; ölçekleme veya
        dönme uygulamaz, böylece anatomik distorsiyon üretmez.
        """
        if not REGISTRATION_AKTIF:
            return goruntu

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
