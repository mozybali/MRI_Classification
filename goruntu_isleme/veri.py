"""Girdi klasoru kesfi ve leak-free veri bolme islemleri."""

import math
import random
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
        """Ham/original goruntuleri leak-free trainval ve test olarak bol.

        Algoritma sklearn stratify yerine sinif basina kota tabanli calisir:
        her sinifin hem trainval hem test tarafinda en az 1 kaynak grubu
        bulunacak sekilde garanti uretir. Boylece dengesiz veri setlerinde
        azinlik siniflarinin test kapsamasindan dusmesi engellenir.
        """
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

        if not grup_kayitlari:
            raise ValueError("Bolme icin goruntu/kaynak grup bulunamadi.")

        sinif_kovalari: Dict[int, List[str]] = {}
        for grup in sorted(grup_kayitlari):
            sinif_kovalari.setdefault(grup_kayitlari[grup], []).append(grup)

        yetersiz = sorted(etiket for etiket, gruplar in sinif_kovalari.items() if len(gruplar) < 2)
        if yetersiz:
            raise ValueError(
                "Harici test split'i olusturmak icin her sinifta en az 2 farkli kaynak grup gerekli. "
                f"Eksik etiketler: {yetersiz}"
            )

        rng = random.Random(RASTGELE_TOHUM)
        for gruplar in sinif_kovalari.values():
            rng.shuffle(gruplar)

        sinif_sayilari = {etiket: len(gruplar) for etiket, gruplar in sinif_kovalari.items()}
        toplam_grup = sum(sinif_sayilari.values())
        hedef_test = max(math.ceil(toplam_grup * test_orani), len(sinif_kovalari))

        # Largest-remainder (Hamilton) yontemi:
        # taban[c] = floor(n_c * orani), [1, n_c-1]'e sikistir; kalan slotlari
        # en buyuk artik/eksik kesirsel paya gore sirayla dagit.
        idealler = {etiket: n * test_orani for etiket, n in sinif_sayilari.items()}
        kotalar: Dict[int, int] = {}
        for etiket, n in sinif_sayilari.items():
            kotalar[etiket] = max(1, min(n - 1, math.floor(idealler[etiket])))

        kalan = hedef_test - sum(kotalar.values())
        if kalan > 0:
            # Buyuk pozitif artik onceligi alir; tie-break: daha kalabalik sinif, sonra etiket.
            adaylar = sorted(
                sinif_kovalari,
                key=lambda e: (
                    -(idealler[e] - math.floor(idealler[e])),
                    -sinif_sayilari[e],
                    e,
                ),
            )
            i = 0
            while kalan > 0 and adaylar:
                etiket = adaylar[i % len(adaylar)]
                if kotalar[etiket] < sinif_sayilari[etiket] - 1:
                    kotalar[etiket] += 1
                    kalan -= 1
                    i += 1
                else:
                    adaylar.remove(etiket)
        elif kalan < 0:
            # Kucuk artik onceligi alir; tie-break: daha az kalabalik sinif, sonra etiket.
            adaylar = sorted(
                sinif_kovalari,
                key=lambda e: (
                    idealler[e] - math.floor(idealler[e]),
                    sinif_sayilari[e],
                    e,
                ),
            )
            i = 0
            while kalan < 0 and adaylar:
                etiket = adaylar[i % len(adaylar)]
                if kotalar[etiket] > 1:
                    kotalar[etiket] -= 1
                    kalan += 1
                    i += 1
                else:
                    adaylar.remove(etiket)

        trainval_gruplari: set = set()
        test_gruplari: set = set()
        for etiket, gruplar in sinif_kovalari.items():
            kota = kotalar[etiket]
            test_gruplari.update(gruplar[:kota])
            trainval_gruplari.update(gruplar[kota:])

        # Invariant'lari acikca dogrula: leak-free + sinif kapsamasi her iki tarafta.
        if trainval_gruplari & test_gruplari:
            raise RuntimeError("Veri sizintisi: trainval ve test kaynak gruplari kesisiyor.")
        beklenen_etiketler = set(sinif_kovalari)
        trainval_etiketler = {grup_kayitlari[g] for g in trainval_gruplari}
        test_etiketler = {grup_kayitlari[g] for g in test_gruplari}
        if trainval_etiketler != beklenen_etiketler or test_etiketler != beklenen_etiketler:
            raise RuntimeError(
                "Sinif kapsamasi ihlali: trainval="
                f"{sorted(trainval_etiketler)}, test={sorted(test_etiketler)}, "
                f"beklenen={sorted(beklenen_etiketler)}"
            )

        trainval_dosyalar = [d for d in dosyalar if d["kaynak_grup"] in trainval_gruplari]
        test_dosyalar = [d for d in dosyalar if d["kaynak_grup"] in test_gruplari]
        return trainval_dosyalar, test_dosyalar
