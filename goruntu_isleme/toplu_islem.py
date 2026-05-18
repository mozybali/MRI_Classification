"""Toplu on isleme, paralel calisma ve split cikti uretimi."""

from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

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
    dosya_info, cikti_klasoru = args
    return _worker_isleyici._tek_goruntu_isle(dosya_info, cikti_klasoru)


class GorselTopluIslemMixin:
    @staticmethod
    def _goruntu_isle_metodu_ozel_mi(instance) -> bool:
        """Testlerdeki instance-level monkeypatch'leri geriye donuk destekle."""
        metod = getattr(instance, "goruntu_isle", None)
        sinif_metodu = getattr(type(instance), "goruntu_isle", None)
        return getattr(metod, "__func__", None) is not sinif_metodu

    def _goruntu_isle_sonucunu_al(self, dosya_yolu: Path):
        """Yapisal pipeline sonucunu al; eski goruntu_isle monkeypatch'lerini destekle."""
        if self._goruntu_isle_metodu_ozel_mi(self):
            goruntu = self.goruntu_isle(dosya_yolu)
            return self._pipeline_sonucu(processed_image=goruntu)
        return self.goruntu_isle_sonuc(dosya_yolu)

    def _tek_goruntu_isle(self, dosya_info: Dict, cikti_klasoru: Path) -> Optional[Dict]:
        """Tek bir görüntüyü işle (paralel işlem için)."""
        try:
            sinif_cikti = cikti_klasoru / dosya_info["sinif"]
            self.klasor_olustur(sinif_cikti)

            # Kalite sayacını izle: işlem öncesi değeri kaydet
            kalite_oncesi = self.kalite_istatistikleri.get('kalite_hatasi', 0)
            kenar_tespit_oncesi = self.kalite_istatistikleri.get('kenar_artefakt_tespit', 0)
            kenar_temizleme_oncesi = self.kalite_istatistikleri.get('kenar_artefakt_temizlendi', 0)
            pipeline_sonu_red_oncesi = self.kalite_istatistikleri.get('pipeline_sonu_red', 0)

            # Görüntüyü işle (kalite kontrol içinde yapılır)
            islem_sonucu = self._goruntu_isle_sonucunu_al(dosya_info["yol"])
            goruntu = islem_sonucu.get("processed_image")
            kalite_reddedildi = bool(islem_sonucu.get("quality_rejected", False))
            kalite_reddi_nedeni = str(islem_sonucu.get("quality_reason") or "")

            kalite_artis = self.kalite_istatistikleri.get('kalite_hatasi', 0) - kalite_oncesi
            kenar_tespit_artis = (
                self.kalite_istatistikleri.get('kenar_artefakt_tespit', 0) - kenar_tespit_oncesi
            )
            kenar_temizleme_artis = (
                self.kalite_istatistikleri.get('kenar_artefakt_temizlendi', 0) - kenar_temizleme_oncesi
            )
            pipeline_sonu_red_artis = (
                self.kalite_istatistikleri.get('pipeline_sonu_red', 0) - pipeline_sonu_red_oncesi
            )

            sonuc = {
                'basarili': 0,
                'basarisiz': 0,
                'kalite_hatasi': 0,
                'kaydetme_hatasi': 0,
                'kenar_artefakt_tespit': kenar_tespit_artis,
                'kenar_artefakt_temizlendi': kenar_temizleme_artis,
                'pipeline_sonu_red': pipeline_sonu_red_artis,
                'istatistikler': {sinif: 0 for sinif in SINIF_KLASORLERI}
            }

            dosya_adi = self._cikti_dosya_koku(dosya_info["yol"])
            normal_yol = sinif_cikti / f"{dosya_adi}.png"

            def normal_cikti_temizle() -> None:
                try:
                    if normal_yol.exists():
                        normal_yol.unlink()
                except OSError as exc:
                    print(f"[UYARI] Eski normal cikti temizlenemedi {normal_yol}: {exc}")

            # Pipeline sonu reddi: normal cikti yok.
            if kalite_reddedildi and kalite_reddi_nedeni == "pipeline_sonu_red":
                sonuc['basarisiz'] = 1
                sonuc['pipeline_sonu_red'] = max(sonuc.get('pipeline_sonu_red', 0), 1)
                normal_cikti_temizle()
                return sonuc

            if goruntu is not None:
                cikti_yolu = normal_yol
                if not self.goruntu_kaydet(goruntu, str(cikti_yolu)):
                    sonuc['basarisiz'] = 1
                    sonuc['kaydetme_hatasi'] = 1
                    return sonuc

                sonuc['basarili'] = 1
                sonuc['istatistikler'][dosya_info["sinif"]] = 1
            else:
                sonuc['basarisiz'] = 1
                sonuc['kalite_hatasi'] = kalite_artis
                normal_cikti_temizle()

            return sonuc

        except (ValueError, TypeError):
            # Config/programlama hatalari per-image except tarafindan yutulup
            # "basarisiz: 1" gibi gorunmemeli; gercek hatayi cagrana ilet.
            raise
        except Exception as e:
            print(f"[HATA] Goruntu islenemedi {dosya_info.get('yol', '?')}: {type(e).__name__}: {e}")
            return {
                'basarili': 0,
                'basarisiz': 1,
                'kalite_hatasi': 0,
                'kaydetme_hatasi': 0,
                'kenar_artefakt_tespit': 0,
                'kenar_artefakt_temizlendi': 0,
                'pipeline_sonu_red': 0,
                'istatistikler': {sinif: 0 for sinif in SINIF_KLASORLERI}
            }

    def tum_gorselleri_isle(
        self,
        cikti_klasoru: Path = CIKTI_KLASORU,
        giris_klasoru: Path = None,
        dosyalar: Optional[List[Dict]] = None,
    ) -> Dict:
        """Tüm MRI görüntülerini toplu olarak işle ve kaydet."""
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
            "kalite_hatasi": 0,
            "kaydetme_hatasi": 0,
            "kenar_artefakt_tespit": 0,
            "kenar_artefakt_temizlendi": 0,
            "pipeline_sonu_red": 0,
        }

        basarili = 0
        basarisiz = 0
        kalite_hatasi_toplam = 0
        kaydetme_hatasi_toplam = 0
        kenar_tespit_toplam = 0
        kenar_temizleme_toplam = 0
        pipeline_sonu_red_toplam = 0
        istatistikler = {sinif: 0 for sinif in SINIF_KLASORLERI}

        islem_args = [(dosya_info, cikti_klasoru) for dosya_info in dosyalar]

        sonuclar = []
        efektif_isci = min(self.n_jobs, max(1, len(islem_args)))
        if efektif_isci > 1:
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
            for args in tqdm(islem_args, desc="Goruntuler isleniyor"):
                sonuclar.append(self._tek_goruntu_isle(*args))

        for sonuc in sonuclar:
            if sonuc is not None:
                basarili += sonuc['basarili']
                basarisiz += sonuc['basarisiz']
                kalite_hatasi_toplam += sonuc.get('kalite_hatasi', 0)
                kaydetme_hatasi_toplam += sonuc.get('kaydetme_hatasi', 0)
                kenar_tespit_toplam += sonuc.get('kenar_artefakt_tespit', 0)
                kenar_temizleme_toplam += sonuc.get('kenar_artefakt_temizlendi', 0)
                pipeline_sonu_red_toplam += sonuc.get('pipeline_sonu_red', 0)
                for sinif, sayi in sonuc['istatistikler'].items():
                    istatistikler[sinif] += sayi

        self.kalite_istatistikleri['basarili'] = basarili
        self.kalite_istatistikleri['kalite_hatasi'] = kalite_hatasi_toplam
        self.kalite_istatistikleri['kaydetme_hatasi'] = kaydetme_hatasi_toplam
        self.kalite_istatistikleri['kenar_artefakt_tespit'] = kenar_tespit_toplam
        self.kalite_istatistikleri['kenar_artefakt_temizlendi'] = kenar_temizleme_toplam
        self.kalite_istatistikleri['pipeline_sonu_red'] = pipeline_sonu_red_toplam

        # Sonuçları yazdır
        print(f"\n{'='*60}")
        print(f"Basarili: {basarili}")
        print(f"Basarisiz: {basarisiz}")
        print(f"Kalite hatasi: {kalite_hatasi_toplam}")
        if kaydetme_hatasi_toplam:
            print(f"Kaydetme hatasi: {kaydetme_hatasi_toplam}")
        if KENAR_ARTEFAKT_RAPORLA and (
            KENAR_ARTEFAKT_KONTROL_AKTIF or KENAR_ARTEFAKT_TEMIZLEME_AKTIF
        ):
            print(f"Kenar artefakt tespit edilen: {kenar_tespit_toplam}")
            print(f"Kenar artefakt temizlenen: {kenar_temizleme_toplam}")
        if pipeline_sonu_red_toplam:
            print(f"Pipeline sonu reddi: {pipeline_sonu_red_toplam}")
        print(f"\nSinif bazli istatistikler:")
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
        )
        self._sinif_kapsamini_dogrula(trainval_istatistik, beklenen_trainval_siniflari, "TrainVal")
        self._sinif_kapsamini_dogrula(test_istatistik, beklenen_test_siniflari, "Test")

        return {
            "trainval": trainval_istatistik,
            "test": test_istatistik,
        }
