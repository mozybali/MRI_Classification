"""Toplu on isleme, paralel calisma ve split cikti uretimi."""

from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    from .ayarlar import *
except ImportError:
    from ayarlar import *

_worker_isleyici = None

def _gorsel_isleyici_sinifi():
    try:
        from .goruntu_isleyici import GorselIsleyici
    except ImportError:
        from goruntu_isleyici import GorselIsleyici
    return GorselIsleyici

def _islem_worker_init():
    """Worker basina tek GorselIsleyici instance olustur."""
    global _worker_isleyici
    _worker_isleyici = _gorsel_isleyici_sinifi()()

def _islem_wrapper(args):
    """Tek bir goruntuyu islemek icin wrapper fonksiyon."""
    global _worker_isleyici
    if _worker_isleyici is None:
        _worker_isleyici = _gorsel_isleyici_sinifi()()
    dosya_info, cikti_klasoru, artirma_carpanlari = args
    return _worker_isleyici._tek_goruntu_isle(dosya_info, cikti_klasoru, artirma_carpanlari)


class GorselTopluIslemMixin:
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

            # Kalite sayacını izle: işlem öncesi değeri kaydet
            kalite_oncesi = self.kalite_istatistikleri.get('kalite_hatasi', 0)

            # Görüntüyü işle (kalite kontrol içinde yapılır)
            goruntu = self.goruntu_isle(dosya_info["yol"])

            kalite_artis = self.kalite_istatistikleri.get('kalite_hatasi', 0) - kalite_oncesi

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
                sonuc['kalite_hatasi'] = kalite_artis

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
        efektif_isci = min(self.n_jobs, max(1, len(islem_args)))
        if paralel_kullan and efektif_isci > 1:
            # Paralel işleme ile hızlandırma
            print(f"[BILGI] Paralel isleme aktif: {efektif_isci} cekirdek kullaniliyor")
            try:
                with Pool(processes=efektif_isci, initializer=_islem_worker_init) as pool:
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

        # Template korunur: trainval ve test ayni anchor'a hizalanmali ki
        # affine/rigid registration train/test arasi sistematik kaymaya yol acmasin.

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
