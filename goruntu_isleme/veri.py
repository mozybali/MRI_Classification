"""Girdi klasoru kesfi ve leak-free veri bolme islemleri."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *


class GorselVeriMixin:
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
