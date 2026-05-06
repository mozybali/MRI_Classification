"""Toplu on isleme, paralel calisma ve split cikti uretimi."""

import csv
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
    if len(args) == 4:
        dosya_info, cikti_klasoru, artirma_carpanlari, split_adi = args
    else:
        dosya_info, cikti_klasoru, artirma_carpanlari = args
        split_adi = None
    return _worker_isleyici._tek_goruntu_isle(
        dosya_info, cikti_klasoru, artirma_carpanlari, split_adi=split_adi
    )


class GorselTopluIslemMixin:
    @staticmethod
    def _split_adi_cozumle(cikti_klasoru: Path, split_adi: Optional[str] = None) -> str:
        """Cikti klasorunden trainval/test split adini guvenli sekilde cozumle."""
        if split_adi:
            return str(split_adi)
        klasor_adi = Path(cikti_klasoru).name
        if klasor_adi in {"trainval", "test"}:
            return klasor_adi
        return ""

    @staticmethod
    def _egim_kalite_kok_dizini(cikti_klasoru: Path, split_adi: str) -> Path:
        """Egim kalite adaylari icin preprocessing kok dizinini bul.

        Iki yerlesim destekli:
        - split_adi 'trainval' veya 'test' ise ve cikti_klasoru bu adla
          bitiyorsa: aday klasoru cikti_klasoru'nun parent'ina kardes olarak
          kok / EGIM_KALITE_ADAYLARI_KLASOR_ADI yoluyla yazilir (onerilen).
        - Diger durumlarda (no-split custom cikti, tek-seferlik koşular):
          aday klasoru cikti_klasoru icine yuvalanir. `_egim_kalite_aday_sinif_dizini`
          bu durumda split alt klasoru olusturmaz, dogrudan sinif altina
          yazar; manifest CSV ayni nested kok altinda tutulur.
        Boylece split bilgisi olmadan da deterministik ve tek bir yerlesim
        garanti edilir.
        """
        cikti_klasoru = Path(cikti_klasoru)
        if split_adi in {"trainval", "test"} and cikti_klasoru.name == split_adi:
            kok = cikti_klasoru.parent
        else:
            kok = cikti_klasoru
        return kok / EGIM_KALITE_ADAYLARI_KLASOR_ADI

    def _egim_kalite_manifest_yolu(self, cikti_klasoru: Path, split_adi: str) -> Path:
        """Egim kalite manifest dosyasinin yolunu dondur."""
        return self._egim_kalite_kok_dizini(cikti_klasoru, split_adi) / EGIM_KALITE_MANIFEST_DOSYA_ADI

    def _egim_kalite_aday_sinif_dizini(
        self, cikti_klasoru: Path, split_adi: str, sinif: str
    ) -> Path:
        """Aday goruntu icin split/sinif yapisini koruyan dizini dondur."""
        aday_kok = self._egim_kalite_kok_dizini(cikti_klasoru, split_adi)
        if split_adi:
            return aday_kok / split_adi / sinif
        return aday_kok / sinif

    @staticmethod
    def _goruntu_isle_metodu_ozel_mi(instance) -> bool:
        """Testlerdeki instance-level monkeypatch'leri geriye donuk destekle."""
        metod = getattr(instance, "goruntu_isle", None)
        sinif_metodu = getattr(type(instance), "goruntu_isle", None)
        return getattr(metod, "__func__", None) is not sinif_metodu

    def _goruntu_isle_sonucunu_al(self, dosya_yolu: Path):
        """Yapisal pipeline sonucunu al; eski goruntu_isle monkeypatch'lerini destekle.

        Donus tipi `on_isleme.PipelineSonucu` (TypedDict). Burada yumusak
        annot. yerine import dongusu kacinmak icin tip yazilmadi; yeni alan
        eklerken `_pipeline_sonucu` icindeki kontratin tek dogru kaynak
        oldugunu unutmayin.
        """
        if self._goruntu_isle_metodu_ozel_mi(self):
            goruntu = self.goruntu_isle(dosya_yolu)
            return self._pipeline_sonucu(processed_image=goruntu)
        return self.goruntu_isle_sonuc(dosya_yolu)

    def _egim_kalite_manifest_yaz(self, manifest_yolu: Path, satirlar: List[Dict[str, object]]) -> None:
        """Reddedilen egim adaylarini manifest CSV'sine ekle."""
        if not satirlar:
            return

        self.klasor_olustur(manifest_yolu.parent)
        alanlar = [
            "original_path",
            "split",
            "class_name",
            "detected_angle",
            "reason",
            "candidate_path",
        ]
        yeni_dosya = not manifest_yolu.exists() or manifest_yolu.stat().st_size == 0
        with manifest_yolu.open("a", newline="", encoding="utf-8") as dosya:
            yazici = csv.DictWriter(dosya, fieldnames=alanlar)
            if yeni_dosya:
                yazici.writeheader()
            for satir in satirlar:
                yazici.writerow({alan: satir.get(alan, "") for alan in alanlar})

    def _tek_goruntu_isle(self, dosya_info: Dict, cikti_klasoru: Path,
                          artirma_carpanlari: Dict[str, int],
                          split_adi: Optional[str] = None) -> Optional[Dict]:
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
            kenar_tespit_oncesi = self.kalite_istatistikleri.get('kenar_artefakt_tespit', 0)
            kenar_temizleme_oncesi = self.kalite_istatistikleri.get('kenar_artefakt_temizlendi', 0)
            egim_tespit_oncesi = self.kalite_istatistikleri.get('egim_tespit', 0)
            egim_duzeltme_oncesi = self.kalite_istatistikleri.get('egim_duzeltildi', 0)
            egim_kontrol_oncesi = self.kalite_istatistikleri.get('egim_gorsel_kontrol_adayi', 0)
            egim_kalite_red_oncesi = self.kalite_istatistikleri.get('egim_kalite_red', 0)

            # Görüntüyü işle (kalite kontrol içinde yapılır)
            islem_sonucu = self._goruntu_isle_sonucunu_al(dosya_info["yol"])
            goruntu = islem_sonucu.get("processed_image")
            kalite_reddedildi = bool(islem_sonucu.get("quality_rejected", False))

            kalite_artis = self.kalite_istatistikleri.get('kalite_hatasi', 0) - kalite_oncesi
            kenar_tespit_artis = (
                self.kalite_istatistikleri.get('kenar_artefakt_tespit', 0) - kenar_tespit_oncesi
            )
            kenar_temizleme_artis = (
                self.kalite_istatistikleri.get('kenar_artefakt_temizlendi', 0) - kenar_temizleme_oncesi
            )
            egim_tespit_artis = (
                self.kalite_istatistikleri.get('egim_tespit', 0) - egim_tespit_oncesi
            )
            egim_duzeltme_artis = (
                self.kalite_istatistikleri.get('egim_duzeltildi', 0) - egim_duzeltme_oncesi
            )
            egim_kontrol_artis = (
                self.kalite_istatistikleri.get('egim_gorsel_kontrol_adayi', 0) - egim_kontrol_oncesi
            )
            egim_kalite_red_artis = (
                self.kalite_istatistikleri.get('egim_kalite_red', 0) - egim_kalite_red_oncesi
            )

            sonuc = {
                'basarili': 0,
                'basarisiz': 0,
                'kalite_hatasi': 0,
                'kaydetme_hatasi': 0,
                'kenar_artefakt_tespit': kenar_tespit_artis,
                'kenar_artefakt_temizlendi': kenar_temizleme_artis,
                'egim_tespit': egim_tespit_artis,
                'egim_duzeltildi': egim_duzeltme_artis,
                'egim_gorsel_kontrol_adayi': egim_kontrol_artis,
                'egim_kalite_red': egim_kalite_red_artis,
                'kalite_aday_manifest_satirlari': [],
                'istatistikler': {sinif: 0 for sinif in SINIF_KLASORLERI}
            }

            dosya_adi = self._cikti_dosya_koku(dosya_info["yol"])
            if goruntu is not None and kalite_reddedildi:
                sonuc['basarisiz'] = 1
                sonuc['egim_kalite_red'] = max(sonuc.get('egim_kalite_red', 0), 1)

                split = self._split_adi_cozumle(cikti_klasoru, split_adi)
                aday_yolu = ""
                if EGIM_KALITE_ADAYLARI_KAYDET:
                    aday_sinif_cikti = self._egim_kalite_aday_sinif_dizini(
                        cikti_klasoru, split, dosya_info["sinif"]
                    )
                    self.klasor_olustur(aday_sinif_cikti)
                    aday_yolu_obj = aday_sinif_cikti / f"{dosya_adi}.png"
                    aday_yolu = str(aday_yolu_obj)
                    if not self.goruntu_kaydet(goruntu, str(aday_yolu_obj)):
                        sonuc['kaydetme_hatasi'] = 1
                        return sonuc

                aci = islem_sonucu.get("tilt_angle")
                aci_yaz = "" if aci is None else f"{float(aci):.6f}"
                sonuc['kalite_aday_manifest_satirlari'].append({
                    "original_path": str(dosya_info["yol"]),
                    "split": split,
                    "class_name": dosya_info["sinif"],
                    "detected_angle": aci_yaz,
                    "reason": islem_sonucu.get("quality_reason") or "excessive_tilt",
                    "candidate_path": aday_yolu,
                })
            elif goruntu is not None:
                # Orijinal görüntüyü kaydet
                cikti_yolu = sinif_cikti / f"{dosya_adi}.png"
                if not self.goruntu_kaydet(goruntu, str(cikti_yolu)):
                    sonuc['basarisiz'] = 1
                    sonuc['kaydetme_hatasi'] = 1
                    return sonuc

                sonuc['basarili'] = 1
                sonuc['istatistikler'][dosya_info["sinif"]] = 1

                # Sınıf bazlı veri artırma
                if VERI_ARTIRMA_AKTIF:
                    sinif = dosya_info["sinif"]
                    carpan = artirma_carpanlari.get(sinif, ARTIRMA_CARPANI)

                    for i in range(carpan):
                        artirmis_goruntu = self.veri_artir(goruntu)
                        artirmis_yol = sinif_cikti / f"{dosya_adi}_aug{i+1}.png"
                        if self.goruntu_kaydet(artirmis_goruntu, str(artirmis_yol)):
                            sonuc['istatistikler'][dosya_info["sinif"]] += 1
                        else:
                            sonuc['kaydetme_hatasi'] += 1
            else:
                sonuc['basarisiz'] = 1
                sonuc['kalite_hatasi'] = kalite_artis

            return sonuc

        except (ValueError, TypeError):
            # Config/programlama hatalari (orn. cozumlenemeyen aday yolu,
            # gecersiz split adi) per-image except tarafindan yutulup
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
                'egim_tespit': 0,
                'egim_duzeltildi': 0,
                'egim_gorsel_kontrol_adayi': 0,
                'egim_kalite_red': 0,
                'kalite_aday_manifest_satirlari': [],
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
        split_adi: Optional[str] = None,
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
            "kalite_hatasi": 0,
            "kaydetme_hatasi": 0,
            "kenar_artefakt_tespit": 0,
            "kenar_artefakt_temizlendi": 0,
            "egim_tespit": 0,
            "egim_duzeltildi": 0,
            "egim_gorsel_kontrol_adayi": 0,
            "egim_kalite_red": 0,
        }
        
        if artirma_carpanlari is None:
            artirma_carpanlari = self.sinif_bazli_artirma_carpani_hesapla(dosyalar)
        
        basarili = 0
        basarisiz = 0
        kalite_hatasi_toplam = 0
        kaydetme_hatasi_toplam = 0
        kenar_tespit_toplam = 0
        kenar_temizleme_toplam = 0
        egim_tespit_toplam = 0
        egim_duzeltme_toplam = 0
        egim_kontrol_toplam = 0
        egim_kalite_red_toplam = 0
        kalite_manifest_satirlari = []
        istatistikler = {sinif: 0 for sinif in SINIF_KLASORLERI}

        # Her görüntü için argümanları hazırla
        split = self._split_adi_cozumle(cikti_klasoru, split_adi)
        # split bos kalsa bile ileride aday yazimi `_egim_kalite_kok_dizini`
        # tarafindan deterministik nested layout ile karsilanir; bu yolu
        # docstring'de belgelemek isteniyorsa README "Eğim Kalite Kontrolü"
        # bolumune bakin.
        islem_args = [
            (dosya_info, cikti_klasoru, artirma_carpanlari, split)
            for dosya_info in dosyalar
        ]

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
                kaydetme_hatasi_toplam += sonuc.get('kaydetme_hatasi', 0)
                kenar_tespit_toplam += sonuc.get('kenar_artefakt_tespit', 0)
                kenar_temizleme_toplam += sonuc.get('kenar_artefakt_temizlendi', 0)
                egim_tespit_toplam += sonuc.get('egim_tespit', 0)
                egim_duzeltme_toplam += sonuc.get('egim_duzeltildi', 0)
                egim_kontrol_toplam += sonuc.get('egim_gorsel_kontrol_adayi', 0)
                egim_kalite_red_toplam += sonuc.get('egim_kalite_red', 0)
                kalite_manifest_satirlari.extend(
                    sonuc.get('kalite_aday_manifest_satirlari', [])
                )
                for sinif, sayi in sonuc['istatistikler'].items():
                    istatistikler[sinif] += sayi

        self.kalite_istatistikleri['basarili'] = basarili
        self.kalite_istatistikleri['kalite_hatasi'] = kalite_hatasi_toplam
        self.kalite_istatistikleri['kaydetme_hatasi'] = kaydetme_hatasi_toplam
        self.kalite_istatistikleri['kenar_artefakt_tespit'] = kenar_tespit_toplam
        self.kalite_istatistikleri['kenar_artefakt_temizlendi'] = kenar_temizleme_toplam
        self.kalite_istatistikleri['egim_tespit'] = egim_tespit_toplam
        self.kalite_istatistikleri['egim_duzeltildi'] = egim_duzeltme_toplam
        self.kalite_istatistikleri['egim_gorsel_kontrol_adayi'] = egim_kontrol_toplam
        self.kalite_istatistikleri['egim_kalite_red'] = egim_kalite_red_toplam

        if kalite_manifest_satirlari:
            self._egim_kalite_manifest_yaz(
                self._egim_kalite_manifest_yolu(cikti_klasoru, split),
                kalite_manifest_satirlari,
            )

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
        if EGIM_DUZELTME_RAPORLA and EGIM_DUZELTME_AKTIF:
            print(f"Egim tespit edilen: {egim_tespit_toplam}")
            print(f"Egim duzeltilen: {egim_duzeltme_toplam}")
            print(f"Egim gorsel kontrol adayi: {egim_kontrol_toplam}")
        if EGIM_KALITE_KONTROL_AKTIF:
            print(f"Egim kalite reddi: {egim_kalite_red_toplam}")
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
            split_adi="trainval",
        )

        # Template korunur: trainval ve test ayni anchor'a hizalanmali ki
        # affine/rigid registration train/test arasi sistematik kaymaya yol acmasin.

        test_istatistik = self.tum_gorselleri_isle(
            cikti_klasoru=Path(cikti_klasoru) / "test",
            dosyalar=test_dosyalar,
            artirma_carpanlari=sifir_aug,
            split_adi="test",
        )
        self._sinif_kapsamini_dogrula(trainval_istatistik, beklenen_trainval_siniflari, "TrainVal")
        self._sinif_kapsamini_dogrula(test_istatistik, beklenen_test_siniflari, "Test")

        return {
            "trainval": trainval_istatistik,
            "test": test_istatistik,
        }
