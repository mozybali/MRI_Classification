"""
Görüntü İşleyici Modülü Testleri
Tests for goruntu_isleyici.py module.

Bu dosya GorselIsleyici sınıfının tüm fonksiyonlarını test eder.
Görüntü yükleme, normalizasyon, histogram eşitleme gibi temel işlemleri doğrular.
"""

import sys
import importlib
import csv
import pytest
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "goruntu_isleme"))

import goruntu_isleyici as gi
from goruntu_isleyici import GorselIsleyici


class TestGorselIsleyici:
    """GorselIsleyici sınıfı için test suite."""

    def test_init(self):
        """GorselIsleyici başlatma testi."""
        isleyici = GorselIsleyici()
        assert isleyici is not None
        assert isleyici.kalite_istatistikleri['toplam'] == 0
        assert isleyici.kalite_istatistikleri['basarili'] == 0
        assert isleyici.kalite_istatistikleri['kalite_hatasi'] == 0
        assert isleyici.kalite_istatistikleri['kaydetme_hatasi'] == 0
        assert isleyici.kalite_istatistikleri['pipeline_sonu_red'] == 0

    def test_tohum_ayarla(self):
        """Aynı tohum ile aynı rastgele değer üretilmeli."""
        GorselIsleyici.tohum_ayarla(42)
        val1 = np.random.random()

        GorselIsleyici.tohum_ayarla(42)
        val2 = np.random.random()

        assert val1 == val2, "Aynı tohum aynı rastgele değerleri üretmeli"

    def test_klasor_olustur(self, tmp_path):
        """Yeni bir klasörün başarıyla oluşturulduğunu kontrol et."""
        test_klasor = tmp_path / "test_folder"
        GorselIsleyici.klasor_olustur(test_klasor)

        assert test_klasor.exists()
        assert test_klasor.is_dir()

    def test_cikti_dosya_koku_uzantiyi_koruyarak_benzersizlesir(self):
        """Ayni stem'e sahip farkli kaynaklar ayri cikti kokleri uretmeli."""
        assert GorselIsleyici._cikti_dosya_koku("26 (1).jpg") == "26 (1)_jpg"
        assert GorselIsleyici._cikti_dosya_koku("26 (1).png") == "26 (1)_png"

    def test_goruntu_yukle_valid(self, test_image_path):
        """Geçerli görüntü dosyası başarıyla yüklenmeli."""
        isleyici = GorselIsleyici()
        img = isleyici.goruntu_yukle(test_image_path)

        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2
        assert img.dtype == np.uint8

    def test_goruntu_yukle_invalid(self):
        """Var olmayan dosya None dönmeli."""
        isleyici = GorselIsleyici()
        img = isleyici.goruntu_yukle(Path("nonexistent.jpg"))

        assert img is None

    def test_yogunluk_normalize(self):
        """Normalize sonucu uint8 ve [0, 255] aralığında olmalı."""
        isleyici = GorselIsleyici()

        test_img = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        normalized = isleyici.yogunluk_normalize(test_img)

        assert normalized.min() >= 0
        assert normalized.max() <= 255
        assert normalized.dtype == np.uint8

    def test_yogunluk_normalize_percentilleri_foreground_maskesinde_hesaplar(self):
        """Buyuk siyah arka plan percentile esiklerini domine etmemeli."""
        isleyici = GorselIsleyici()
        img = np.zeros((160, 160), dtype=np.uint8)
        yy, xx = np.ogrid[:160, :160]
        beyin = ((yy - 80) ** 2) / (52 ** 2) + ((xx - 80) ** 2) / (42 ** 2) <= 1
        x_grid = np.broadcast_to(xx, img.shape)
        doku_degeri = 80 + ((x_grid - 38) * 50 / 84)
        img[beyin] = np.clip(doku_degeri[beyin], 80, 140).astype(np.uint8)
        parlak_doku = beyin & (yy > 70) & (yy < 90) & (xx > 78) & (xx < 92)
        img[parlak_doku] = 230

        # Eski tum-goruntu percentile hesabi, parlak dokuyu ust esikte
        # kirpacak kadar arka plan agirlikli olurdu.
        assert float(np.percentile(img, 99)) < 230

        normalized = isleyici.yogunluk_normalize(img)

        assert normalized.dtype == np.uint8
        assert int(normalized[~beyin].max()) == 0
        assert float(normalized[parlak_doku].mean()) > 220.0
        assert float(normalized[parlak_doku].mean()) > float(normalized[beyin & ~parlak_doku].mean()) + 80.0

    def test_histogram_esitle_dtype_uint8(self):
        """histogram_esitle her zaman uint8 dönmeli."""
        isleyici = GorselIsleyici()

        test_img = np.random.randint(100, 150, (256, 256), dtype=np.uint8)
        result = isleyici.histogram_esitle(test_img, adaptive=True)

        assert result.shape == test_img.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_histogram_esitle_contrast_artmali(self):
        """CLAHE düşük kontrastlı görüntünün kontrastını artırmalı."""
        isleyici = GorselIsleyici()

        test_img = np.random.randint(100, 150, (256, 256), dtype=np.uint8)
        result = isleyici.histogram_esitle(test_img, adaptive=True)

        assert result.std() >= test_img.std() * 0.8

    def test_boyutlandir(self):
        """512x512 görüntü 256x256'ya küçültülebilmeli."""
        isleyici = GorselIsleyici()

        test_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)

        assert resized.shape == (256, 256)
        assert resized.dtype == np.uint8

    def test_boyutlandir_pad_en_boy_oranini_korur(self, monkeypatch):
        """'pad' modu en-boy oranini korumali ve kenarlari PADDING_DEGERI ile doldurmali."""
        monkeypatch.setattr(gi, "BOYUTLANDIRMA_MODU", "pad")
        monkeypatch.setattr(gi, "PADDING_DEGERI", 0)

        isleyici = GorselIsleyici()
        # Tipik MRI dilim boyutu (yukseklik x genislik)
        test_img = np.full((208, 176), 200, dtype=np.uint8)
        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)

        assert resized.shape == (256, 256)
        assert resized.dtype == np.uint8

        # En-boy orani 176/208 ~= 0.846 oldugundan dikey eksen 256'ya
        # ulasmali, yatay eksen daha kucuk kalip soldan/sagdan padding almalidir.
        beklenen_genislik = int(round(176 * (256 / 208)))
        kenar = (256 - beklenen_genislik) // 2

        # Sol ve sag kenarlarda padding (PADDING_DEGERI=0) bulunmali
        assert kenar > 0
        assert (resized[:, :kenar] == 0).all()
        assert (resized[:, 256 - kenar:] == 0).all()
        # Merkez sutunda kaynak goruntu (sabit 200) yer almali
        assert (resized[:, 128] > 0).all()
        # Dikey eksen tamamen kaplanmali (ust ve alt kenarda padding olmamali)
        assert (resized[0, 128] > 0)
        assert (resized[-1, 128] > 0)

    def test_boyutlandir_pad_padding_degeri_kullanilir(self, monkeypatch):
        """PADDING_DEGERI ayari kenar piksellerine yansimalidir."""
        monkeypatch.setattr(gi, "BOYUTLANDIRMA_MODU", "pad")
        monkeypatch.setattr(gi, "PADDING_DEGERI", 42)

        isleyici = GorselIsleyici()
        test_img = np.full((100, 200), 200, dtype=np.uint8)
        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)

        assert resized.shape == (256, 256)
        # Genislik > yukseklik oldugundan ust ve alt kenarlarda padding olmali
        assert (resized[0, :] == 42).all()
        assert (resized[-1, :] == 42).all()

    def test_boyutlandir_pad_otomatik_arka_plan_sinir_bandi_azaltir(self, monkeypatch):
        """Post-CLAHE benzeri arka plan degeri padding'e yansimali."""
        monkeypatch.setattr(gi, "BOYUTLANDIRMA_MODU", "pad")
        monkeypatch.setattr(gi, "PADDING_OTOMATIK_ARKAPLAN", True)
        monkeypatch.setattr(gi, "PADDING_DEGERI", 0)

        isleyici = GorselIsleyici()
        test_img = np.full((120, 60), 7, dtype=np.uint8)
        test_img[30:90, 15:45] = 120

        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)

        x0 = (256 - int(round(60 * (256 / 120)))) // 2
        padding_medyan = float(np.median(resized[:, :x0]))
        komsu_arka_plan = np.concatenate(
            [
                resized[:25, x0:x0 + 4].ravel(),
                resized[-25:, x0:x0 + 4].ravel(),
            ]
        )
        komsu_medyan = float(np.median(komsu_arka_plan))

        assert abs(padding_medyan - komsu_medyan) <= 2.0
        assert padding_medyan == pytest.approx(7.0, abs=1.0)

    def test_boyutlandir_stretch_eski_davranisi_korur(self, monkeypatch):
        """'stretch' modu en-boy oranini gozetmeden dogrudan resize yapmali."""
        monkeypatch.setattr(gi, "BOYUTLANDIRMA_MODU", "stretch")

        isleyici = GorselIsleyici()
        test_img = np.full((208, 176), 200, dtype=np.uint8)
        resized = isleyici.boyutlandir(test_img, genislik=256, yukseklik=256)

        assert resized.shape == (256, 256)
        assert resized.dtype == np.uint8
        # Stretch modunda padding olmamali; sabit dolu goruntu her yerde
        # 200 civarinda kalmali (interpolasyon sapmasi minimal)
        assert resized.min() > 100
        # Kenarlarda PADDING_DEGERI=0 birikmesi olmamali
        assert (resized[:, 0] > 100).all()
        assert (resized[0, :] > 100).all()

    def test_gurultu_gider_auto_filtreler_kapaliyken_no_op(self, monkeypatch):
        """FILTRE_METODU='off' iken auto mod goruntuyu degistirmeden dondurmeli.

        Eski davranis: filtreler kapaliyken median 3x3'e dusuluyordu; bu
        kortikal dokuyu sessizce siliyordu. Yeni sozlesme: 'off' modu
        goruntuyu aynen dondurur.
        """
        monkeypatch.setattr(gi, "FILTRE_METODU", "off")

        isleyici = GorselIsleyici()
        test_img = np.full((9, 9), 128, dtype=np.uint8)
        test_img[4, 4] = 255

        result = isleyici.gurultu_gider(test_img, metod='auto')

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, test_img)

    def test_goruntu_isle_gurultu_giderme_ayar_gudumlu_calisir(self, monkeypatch):
        """goruntu_isle median'i zorlamamali; gurultu giderme auto modda cagrilmali."""
        isleyici = GorselIsleyici()
        test_img = np.random.randint(50, 180, (64, 64), dtype=np.uint8)
        cagrilan_metodlar = []

        monkeypatch.setattr(isleyici, "goruntu_yukle", lambda _: test_img)
        monkeypatch.setattr(isleyici, "goruntu_kalite_kontrol", lambda _: (True, ""))
        monkeypatch.setattr(isleyici, "center_of_mass_alignment", lambda img: img)
        monkeypatch.setattr(isleyici, "_apply_normalization_strategy", lambda img: img)
        monkeypatch.setattr(isleyici, "boyutlandir", lambda img: img)

        def spy(img, metod='auto'):
            cagrilan_metodlar.append(metod)
            return img

        monkeypatch.setattr(isleyici, "gurultu_gider", spy)

        result = isleyici.goruntu_isle("dummy.png")

        assert result is not None
        assert cagrilan_metodlar == ["auto"]

    def test_kenar_maskesi_ayari_maske_sinirlarini_temizler(self, monkeypatch):
        """MASKE_KENAR_PAYI kenardaki parlak artefaktlari maske disina itmeli."""
        monkeypatch.setattr(gi, "MASKE_KENAR_PAYI", 2)

        mask = np.ones((8, 8), dtype=bool)
        cleaned = GorselIsleyici._kenar_maskesini_temizle(mask)

        assert not cleaned[:2, :].any()
        assert not cleaned[-2:, :].any()
        assert not cleaned[:, :2].any()
        assert not cleaned[:, -2:].any()
        assert cleaned[2:6, 2:6].all()

    def test_maske_duzenleme_morfolojik_ayari_tekil_gurultuyu_temizler(self, monkeypatch):
        """Morfolojik ayar aktifken izole tek piksel maskeden silinmeli."""
        monkeypatch.setattr(gi, "MASKE_KENAR_PAYI", 0)
        monkeypatch.setattr(gi, "MORFOLOJIK_OPERASYONLAR_AKTIF", True)
        monkeypatch.setattr(gi, "MORFOLOJIK_KERNEL_BOYUTU", 3)

        isleyici = GorselIsleyici()
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True

        cleaned = isleyici._maskeyi_duzenle(mask, closing_scale=1)

        assert not cleaned.any()

    def test_goruntu_kalite_kontrol_valid(self):
        """Yeterli parlaklık ve kontrasta sahip görüntü kalite kontrolünden geçmeli."""
        isleyici = GorselIsleyici()

        valid_img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(valid_img)

        assert is_valid is True
        assert message == ""

    def test_goruntu_kalite_kontrol_too_dark(self):
        """Çok karanlık görüntü kalite kontrolünden geçmemeli."""
        isleyici = GorselIsleyici()

        dark_img = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(dark_img)

        assert is_valid is False

    def test_goruntu_kalite_kontrol_low_contrast(self):
        """Sıfır standart sapma görüntü kalite kontrolünden geçmemeli."""
        isleyici = GorselIsleyici()

        low_contrast_img = np.full((256, 256), 128, dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(low_contrast_img)

        assert is_valid is False

    def test_gorselleri_listele(self, test_dataset_structure):
        """Veri seti klasöründeki tüm görüntüler doğru şekilde listelenmeli."""
        isleyici = GorselIsleyici()
        dosyalar = isleyici.gorselleri_listele(test_dataset_structure)

        assert len(dosyalar) == 12  # 4 sınıf * 3 görüntü
        assert all('yol' in d for d in dosyalar)
        assert all('sinif' in d for d in dosyalar)
        assert all('etiket' in d for d in dosyalar)
        assert all('kaynak_grup' in d for d in dosyalar)

    def test_gorselleri_listele_dosya_sirasini_sabitleyerek_deterministik_kalir(self, tmp_path, monkeypatch):
        isleyici = GorselIsleyici()
        dataset = tmp_path / "dataset"
        sinif_klasoru = dataset / "NonDemented"
        sinif_klasoru.mkdir(parents=True)
        Image.fromarray(np.full((8, 8), 100, dtype=np.uint8), mode="L").save(sinif_klasoru / "b.jpg")
        Image.fromarray(np.full((8, 8), 120, dtype=np.uint8), mode="L").save(sinif_klasoru / "a.jpg")

        orijinal_iterdir = Path.iterdir

        def ters_iterdir(path_obj):
            oge_listesi = list(orijinal_iterdir(path_obj))
            if path_obj == sinif_klasoru:
                return iter(sorted(oge_listesi, key=lambda p: p.name, reverse=True))
            return iter(oge_listesi)

        monkeypatch.setattr(Path, "iterdir", ters_iterdir)

        dosyalar = isleyici.gorselleri_listele(dataset)
        adlar = [Path(d["yol"]).name for d in dosyalar if d["sinif"] == "NonDemented"]

        assert adlar == ["a.jpg", "b.jpg"]

    def test_veri_dosyalarini_bol_kaynak_gruplarini_ayirir(self, tmp_path):
        isleyici = GorselIsleyici()
        dataset = tmp_path / "dataset"

        for class_name in gi.SINIF_KLASORLERI:
            class_dir = dataset / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                Image.fromarray(
                    np.random.randint(80, 180, (32, 32), dtype=np.uint8), mode='L'
                ).save(class_dir / f"{class_name}_{idx}.jpg")

        dosyalar = isleyici.gorselleri_listele(dataset)
        trainval_dosyalar, test_dosyalar = isleyici.veri_dosyalarini_bol(dosyalar, test_orani=0.5)

        trainval_gruplar = {d["kaynak_grup"] for d in trainval_dosyalar}
        test_gruplar = {d["kaynak_grup"] for d in test_dosyalar}

        assert trainval_gruplar.isdisjoint(test_gruplar)
        assert {d["sinif"] for d in trainval_dosyalar} == set(gi.SINIF_KLASORLERI)
        assert {d["sinif"] for d in test_dosyalar} == set(gi.SINIF_KLASORLERI)

    def test_veri_dosyalarini_bol_dengesiz_dagilimda_tum_siniflari_korur(self):
        """Cogunluk sinifi azinliklari ezse bile her sinif iki tarafta da bulunmali."""
        isleyici = GorselIsleyici()
        sinif_grup_sayilari = {
            "NonDemented": 100,
            "VeryMildDemented": 2,
            "MildDemented": 2,
            "ModerateDemented": 2,
        }

        dosyalar = []
        for sinif_adi, grup_sayisi in sinif_grup_sayilari.items():
            for idx in range(grup_sayisi):
                kaynak_id = f"{sinif_adi.lower()}_{idx:03d}"
                dosyalar.append({
                    "yol": f"/sanal/{sinif_adi}/{kaynak_id}.jpg",
                    "sinif": sinif_adi,
                    "etiket": gi.SINIF_ETIKETI[sinif_adi],
                    "kaynak_id": kaynak_id,
                    "kaynak_grup": f"{sinif_adi}::{kaynak_id}",
                })

        trainval, test = isleyici.veri_dosyalarini_bol(dosyalar, test_orani=0.15)

        trainval_gruplar = {d["kaynak_grup"] for d in trainval}
        test_gruplar = {d["kaynak_grup"] for d in test}

        assert trainval_gruplar.isdisjoint(test_gruplar)
        assert {d["sinif"] for d in trainval} == set(sinif_grup_sayilari)
        assert {d["sinif"] for d in test} == set(sinif_grup_sayilari)

        # Her azinlik sinifindan test tarafinda en az 1, trainval tarafinda en az 1 grup olmali.
        for sinif_adi in sinif_grup_sayilari:
            assert any(d["sinif"] == sinif_adi for d in test), f"{sinif_adi} test tarafinda yok"
            assert any(d["sinif"] == sinif_adi for d in trainval), f"{sinif_adi} trainval tarafinda yok"

    def test_veri_dosyalarini_bol_esit_dagilimda_kalan_slotlari_yayilir(self):
        """Esit sinif buyuklugunde kalan kota tek sinifa yiglmamali (largest-remainder)."""
        isleyici = GorselIsleyici()
        siniflar = list(gi.SINIF_KLASORLERI)
        grup_basina = 10

        dosyalar = []
        for sinif_adi in siniflar:
            for idx in range(grup_basina):
                kaynak_id = f"{sinif_adi.lower()}_{idx:03d}"
                dosyalar.append({
                    "yol": f"/sanal/{sinif_adi}/{kaynak_id}.jpg",
                    "sinif": sinif_adi,
                    "etiket": gi.SINIF_ETIKETI[sinif_adi],
                    "kaynak_id": kaynak_id,
                    "kaynak_grup": f"{sinif_adi}::{kaynak_id}",
                })

        _, test = isleyici.veri_dosyalarini_bol(dosyalar, test_orani=0.25)

        sinif_test_sayilari = {sinif_adi: 0 for sinif_adi in siniflar}
        for d in test:
            sinif_test_sayilari[d["sinif"]] += 1

        # Toplam test grup sayisi ceil(40 * 0.25) = 10 olmali.
        assert sum(sinif_test_sayilari.values()) == 10
        # Esit ideal (2.5) durumunda dagilim {3,3,2,2} olmali; multiset karsilastirmasi.
        assert sorted(sinif_test_sayilari.values()) == [2, 2, 3, 3]

    def test_veri_dosyalarini_bol_bos_girdide_hata_verir(self):
        """Bos dosya listesi sessiz dis-akista bug yutmamali, acik hata vermeli."""
        isleyici = GorselIsleyici()
        with pytest.raises(ValueError, match="goruntu/kaynak grup bulunamadi"):
            isleyici.veri_dosyalarini_bol([], test_orani=0.15)

    def test_giris_klasoru_gercekten_kullaniliyor(self, test_dataset_structure, tmp_path):
        """tum_gorselleri_isle giris_klasoru parametresini kullanmalı."""
        isleyici = GorselIsleyici()
        cikti = tmp_path / "cikti"
        cikti.mkdir()

        # test_dataset_structure fixture'ından görüntüleri işle
        istatistikler = isleyici.tum_gorselleri_isle(cikti, giris_klasoru=test_dataset_structure)

        # İstatistikler boş olmamalı - en az bir sınıf klasörü işlenmeli
        assert isinstance(istatistikler, dict)
        assert len(istatistikler) > 0
        # cikti klasörü altında dosya oluşturulmuş olmalı
        islenmis = list(cikti.rglob("*.png"))
        assert len(islenmis) > 0

    def test_giris_klasoru_bos_oldugundan_default_kullanilir(self, tmp_path, monkeypatch):
        """giris_klasoru=None ise varsayilan original veri yolu kullanilmali."""
        # Bu test sadece giris_klasoru=None fallback davranisini dogrular;
        # sentetik gurultu goruntusunun kalite/egim filtrelerine takilmamasi
        # icin ilgili kontroller kapatilir.
        monkeypatch.setattr(gi, "KALITE_KONTROL_AKTIF", False)

        isleyici = GorselIsleyici()
        varsayilan_dataset = tmp_path / "varsayilan_dataset" / "OriginalDataset"
        sinif = "NonDemented"
        sinif_klasoru = varsayilan_dataset / sinif
        sinif_klasoru.mkdir(parents=True)

        img = Image.fromarray(
            np.random.randint(80, 180, (64, 64), dtype=np.uint8), mode='L'
        )
        img.save(sinif_klasoru / "sample.jpg")

        monkeypatch.setattr(gi, "ON_ISLEME_VARSAYILAN_GIRIS_KLASORU", varsayilan_dataset)

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        istatistikler = isleyici.tum_gorselleri_isle(cikti, giris_klasoru=None)
        assert istatistikler[sinif] >= 1
        assert len(list(cikti.rglob("*.png"))) >= 1

    def test_kalite_hatasi_paralel_akista_dogru_toplanir(self, tmp_path, monkeypatch):
        """Kalite hatası sayısı paralel/sequential işlemde doğru toplanmalı."""
        monkeypatch.setattr(gi, "KALITE_KONTROL_AKTIF", True)

        isleyici = GorselIsleyici()

        # Çok karanlık görüntülerle bir veri seti oluştur
        dataset = tmp_path / "dark_dataset"
        sinif = "NonDemented"
        sinif_klasoru = dataset / sinif
        sinif_klasoru.mkdir(parents=True)

        for i in range(3):
            img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode='L')
            img.save(sinif_klasoru / f"dark_{i}.jpg")

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        istatistikler = isleyici.tum_gorselleri_isle(cikti, giris_klasoru=dataset)

        assert isleyici.kalite_istatistikleri['toplam'] == 3
        assert isleyici.kalite_istatistikleri['basarili'] == 0
        assert isleyici.kalite_istatistikleri['kalite_hatasi'] == 3
        assert sum(istatistikler.values()) == 0

    def test_goruntu_kaydet_encode_hatasinda_false_doner(self, tmp_path, monkeypatch):
        isleyici = GorselIsleyici()

        def fail_imencode(*_args, **_kwargs):
            return False, None

        monkeypatch.setattr(gi._kalite_io_mod.cv2, "imencode", fail_imencode)

        hedef = tmp_path / "out.png"
        sonuc = isleyici.goruntu_kaydet(np.zeros((16, 16), dtype=np.uint8), hedef)

        assert sonuc is False
        assert not hedef.exists()

    def test_tek_goruntu_kaydetme_hatasi_false_success_uretmez(self, tmp_path, monkeypatch):
        isleyici = GorselIsleyici()
        img = np.full((32, 32), 120, dtype=np.uint8)

        monkeypatch.setattr(isleyici, "goruntu_isle", lambda _yol: img)
        monkeypatch.setattr(isleyici, "goruntu_kaydet", lambda *_args: False)

        sonuc = isleyici._tek_goruntu_isle(
            {"sinif": "NonDemented", "yol": tmp_path / "sample.jpg"},
            tmp_path / "cikti",
        )

        assert sonuc["basarili"] == 0
        assert sonuc["basarisiz"] == 1
        assert sonuc["kaydetme_hatasi"] == 1
        assert sonuc["istatistikler"]["NonDemented"] == 0

    def test_tum_gorselleri_isle_ve_bol_sinif_kapsami_dusunce_hata_verir(self, tmp_path, monkeypatch):
        isleyici = GorselIsleyici()
        isleyici.n_jobs = 1

        dataset = tmp_path / "dataset"
        for class_name in gi.SINIF_KLASORLERI:
            class_dir = dataset / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                Image.fromarray(
                    np.random.randint(80, 180, (32, 32), dtype=np.uint8), mode='L'
                ).save(class_dir / f"{class_name}_{idx}.jpg")

        def sahte_goruntu_isle(yol):
            if "ModerateDemented" in str(yol):
                return None
            return np.tile(np.arange(32, dtype=np.uint8), (32, 1))

        monkeypatch.setattr(isleyici, "goruntu_isle", sahte_goruntu_isle)

        with pytest.raises(ValueError, match="sinif kapsami eksik"):
            isleyici.tum_gorselleri_isle_ve_bol(tmp_path / "cikti", giris_klasoru=dataset)


class TestKenarArtefaktTespitVeTemizleme:
    """Kenar artefakt tespiti ve konservatif temizleme testleri."""

    @staticmethod
    def _temiz_mri_benzeri_goruntu(seed: int = 0) -> np.ndarray:
        """Kenarlari karanlik, merkezi orta parlaklikta sentetik 'temiz' MRI."""
        rng = np.random.default_rng(seed)
        img = rng.integers(20, 60, size=(128, 128), dtype=np.uint16).astype(np.uint8)
        # Merkezde brighter blob (beyin dokusu benzeri)
        cy, cx = 64, 64
        yy, xx = np.ogrid[:128, :128]
        merkez = (yy - cy) ** 2 + (xx - cx) ** 2 <= 30 ** 2
        img[merkez] = 130
        # Kenar seritleri yaklasik karanlik kalmali
        img[:8, :] = np.minimum(img[:8, :], 30)
        img[-8:, :] = np.minimum(img[-8:, :], 30)
        img[:, :8] = np.minimum(img[:, :8], 30)
        img[:, -8:] = np.minimum(img[:, -8:], 30)
        return img

    def test_top_serit_artefakti_tespit_edilir(self):
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=1)
        # Ust seritte parlak band ekle
        img[:10, 20:80] = 250

        analiz = isleyici.kenar_artefakt_analiz(img)

        assert analiz["artefakt_var"] is True
        assert "top" in analiz["suspicious_yonler"]

    def test_bottom_serit_artefakti_tespit_edilir(self):
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=2)
        img[-12:, 30:90] = 250

        analiz = isleyici.kenar_artefakt_analiz(img)

        assert analiz["artefakt_var"] is True
        assert "bottom" in analiz["suspicious_yonler"]

    def test_left_serit_artefakti_tespit_edilir(self):
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=3)
        img[20:90, :12] = 250

        analiz = isleyici.kenar_artefakt_analiz(img)

        assert analiz["artefakt_var"] is True
        assert "left" in analiz["suspicious_yonler"]

    def test_right_serit_artefakti_tespit_edilir(self):
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=4)
        img[20:90, -12:] = 250

        analiz = isleyici.kenar_artefakt_analiz(img)

        assert analiz["artefakt_var"] is True
        assert "right" in analiz["suspicious_yonler"]

    def test_temiz_goruntu_yanlislikla_isaretlenmez(self):
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=5)

        analiz = isleyici.kenar_artefakt_analiz(img)

        assert analiz["artefakt_var"] is False
        assert analiz["suspicious_yonler"] == []

    def test_kenar_temizligi_strict_kalite_kontrol_oncesi_goruntuyu_kurtarir(self, monkeypatch, tmp_path):
        """Parlak ust/alt bantlar kalite kontrolden once temizlenmeli."""
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_SERIT_ORANI", 0.25)
        monkeypatch.setattr(gi, "KENAR_PARLAKLIK_ESIGI", 250)

        isleyici = GorselIsleyici()
        merkez = np.tile(np.linspace(235, 244, 128, dtype=np.uint8), (128, 1))
        img = merkez.copy()
        img[:30, :] = 255
        img[-30:, :] = 255
        assert isleyici.goruntu_kalite_kontrol(img)[0] is False

        path = tmp_path / "saturated_edges.png"
        Image.fromarray(img, mode="L").save(path)

        sonuc = isleyici.goruntu_isle(str(path))

        assert sonuc is not None
        assert sonuc.dtype == np.uint8
        assert isleyici.kalite_istatistikleri["kalite_hatasi"] == 0
        assert isleyici.kalite_istatistikleri["kenar_artefakt_temizlendi"] == 1

    def test_temizleme_kenara_degen_parlak_bolgeyi_baski_alir(self, monkeypatch):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=6)
        img[:10, 20:80] = 250

        oncesi_parlak = int((img[:15, :] >= 220).sum())
        assert oncesi_parlak > 0

        temiz = isleyici.kenar_artefakt_temizle(img)

        sonra_parlak = int((temiz[:15, :] >= 220).sum())
        assert sonra_parlak < oncesi_parlak
        # Esik altina cekilmeli
        toplam = temiz[:15, :].size
        assert sonra_parlak / toplam < gi.KENAR_PARLAK_PIXEL_ORANI_ESIGI

    def test_temizleme_merkezdeki_parlak_bolgeyi_korur(self, monkeypatch):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=7)
        # Merkezde, kenarlara degmeyen parlak nokta
        img[58:68, 58:68] = 250

        temiz = isleyici.kenar_artefakt_temizle(img)

        # Merkez yapisi korunmali
        assert (temiz[58:68, 58:68] == 250).all()

    def test_kenar_anatomi_koruma_orani_bilesen_buyuklugunu_ayarlar(self, monkeypatch):
        """Esik altindaki kenar bileseni temizlenir, ustundeki korunur."""
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_ANATOMI_KORUMA_ORANI", 0.5)
        monkeypatch.setattr(gi, "KENAR_SERIT_ORANI", 0.12)
        # Test, KENAR_ANATOMI_KORUMA_ORANI'nin bilesen alani uzerindeki
        # gating davranisini olcer; bu nedenle serit pay esigini deterministik
        # bir degere sabitleriz, ayarlar.py'deki default tweaklerinden bagimsiz olsun.
        monkeypatch.setattr(gi, "KENAR_BILESEN_SERIT_PAY_ESIGI", 0.6)
        isleyici = GorselIsleyici()

        kucuk = np.full((100, 100), 25, dtype=np.uint8)
        kucuk[:20, :] = 250
        kucuk_temiz = isleyici.kenar_artefakt_temizle(kucuk)
        assert int((kucuk_temiz[:20, :] >= 220).sum()) == 0

        buyuk = np.full((100, 100), 25, dtype=np.uint8)
        buyuk[:70, :] = 250
        buyuk_temiz = isleyici.kenar_artefakt_temizle(buyuk)
        np.testing.assert_array_equal(buyuk_temiz, buyuk)

    def test_temizleme_seride_uzanan_band_temizlenir(self, monkeypatch):
        """Bbox seridi asip merkeze tasan parlak band, cogunlugu seritteyse silinmeli.

        Gercek veri setinde gozlenen senaryo: parlak ust band serit
        kalinligini biraz asar ve goruntu sinirina degmeyebilir. Eski
        'tamamen seritte' kosulu bu durumu kacirip artefakti birakiyordu.
        """
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        # Senaryo deterministik olsun diye serit kalinligi ve serit pay
        # esigini test'e sabitle; ayarlar.py default tweaklerinden bagimsiz.
        monkeypatch.setattr(gi, "KENAR_SERIT_ORANI", 0.12)
        monkeypatch.setattr(gi, "KENAR_BILESEN_SERIT_PAY_ESIGI", 0.6)
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=11)
        # 128x128 icin serit kalinligi ~15. Band rows 2..21 = 20 rows;
        # 13 satiri (2..14) seritte, 7 satiri (15..21) merkezde.
        # strip_pay ~13/20=0.65 >= 0.6 -> temizlenmeli.
        # y=2 oldugundan goruntu sinirina degmiyor; eski mantik kacirirdi.
        img[2:22, 20:80] = 250

        oncesi_parlak = int((img[2:22, 20:80] >= 220).sum())
        assert oncesi_parlak > 0

        temiz = isleyici.kenar_artefakt_temizle(img)

        # Band tamamen baski altina alinmali (>=220 piksel kalmamali).
        assert int((temiz[2:22, 20:80] >= 220).sum()) == 0
        # Goruntu degisti -> uzanan band yakalandi.
        assert not np.array_equal(temiz, img)

    def test_temizleme_merkeze_bagli_bandi_kismi_temizler(self, monkeypatch):
        """Parlak band, parlak korteks/orta dokuya bagli olsa bile serit
        icindeki pay temizlenmeli; merkez doku korunmalidir.

        Bu, gercek veri setinde sik gorulen senaryodur: ust kenar bandi
        beyin korteksindeki saturasyonlu pikseller araciligiyla beyne
        baglidir; tek bir buyuk baglantili bilesen olusur. Bu durumda
        bilesenin sadece serit icindeki pikselleri silinmeli.
        """
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        isleyici = GorselIsleyici()

        # 128x128 sentetik MRI: merkezde parlak (saturasyonlu) blob,
        # ust seritte parlak band, ikisi parlak ince bir koprode birlesir.
        img = np.full((128, 128), 25, dtype=np.uint8)
        # Merkezdeki parlak blob (anatomik) - kenarlardan uzak
        cy, cx = 70, 64
        yy, xx = np.ogrid[:128, :128]
        merkez = (yy - cy) ** 2 + (xx - cx) ** 2 <= 28 ** 2
        img[merkez] = 250
        # Ust kenar bandi (artefakt) - rows 0..6, cols 30..100
        img[0:7, 30:100] = 250
        # Bandı merkeze baglayan dik parlak kopru (rows 7..43, col 64-65)
        img[7:43, 63:66] = 250

        # Beklenti: tek baglantili bilesen olusur (band+kopru+merkez).
        merkez_oncesi = int((img[merkez] >= 220).sum())
        ust_oncesi = int((img[:8, :] >= 220).sum())
        assert merkez_oncesi > 0 and ust_oncesi > 0

        temiz = isleyici.kenar_artefakt_temizle(img)

        # Ust serit (kalinlik = round(128*0.12) = 15) parlaklik orani
        # esigin altina dusurulmeli.
        ust_serit = temiz[:15, :]
        ust_oran = float((ust_serit >= 220).sum()) / float(ust_serit.size)
        assert ust_oran < gi.KENAR_PARLAK_PIXEL_ORANI_ESIGI

        # Merkez parlak yapi korunmali (cogunlugu hala parlak).
        merkez_sonrasi = int((temiz[merkez] >= 220).sum())
        assert merkez_sonrasi >= int(0.8 * merkez_oncesi)

    def test_kismi_temizleme_min_pixel_ayarlari_davranisi_degistirir(self, monkeypatch):
        """Kismi temizlikteki oran/mutlak piksel esikleri ayarlardan gelmeli."""
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_BILESEN_SERIT_PAY_ESIGI", 0.9)
        isleyici = GorselIsleyici()

        img = np.full((128, 128), 25, dtype=np.uint8)
        # Ust seride kismen giren ama tamamen kenar artefakti sayilmayacak
        # kadar merkeze uzayan parlak bilesen: serit payi ~0.325 < 0.9.
        img[2:42, 30:50] = 250
        serit_once = int((img[:15, :] >= 220).sum())
        assert serit_once > 0

        monkeypatch.setattr(gi, "KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI", 0.01)
        monkeypatch.setattr(gi, "KENAR_KISMI_TEMIZLEME_MIN_PIXEL", 50)
        temiz_dusuk_esik = isleyici.kenar_artefakt_temizle(img)
        assert int((temiz_dusuk_esik[:15, :] >= 220).sum()) < serit_once

        monkeypatch.setattr(gi, "KENAR_KISMI_TEMIZLEME_MIN_PIXEL_ORANI", 1.0)
        monkeypatch.setattr(gi, "KENAR_KISMI_TEMIZLEME_MIN_PIXEL", 10_000)
        temiz_yuksek_esik = isleyici.kenar_artefakt_temizle(img)
        np.testing.assert_array_equal(temiz_yuksek_esik, img)

    def test_temizleme_tespit_kapaliyken_de_calisir(self, monkeypatch):
        """KONTROL_AKTIF=False, TEMIZLEME_AKTIF=True kombinasyonu temizlemeli.

        Ayar dokumantasyonu tespit ve temizlemenin bagimsiz oldugunu
        soyler; pipeline da bu sozlesmeye uymali.
        """
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", False)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)

        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=12)
        img[:12, 20:80] = 250

        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = Path(tmp) / "edge.png"
            Image.fromarray(img, mode="L").save(path)

            sonuc = isleyici.goruntu_isle(str(path))

        assert sonuc is not None
        # Tespit kapali oldugundan tespit sayaci sifir kalmali.
        assert isleyici.kalite_istatistikleri.get("kenar_artefakt_tespit", 0) == 0
        # Temizleme yine de calismali.
        assert isleyici.kalite_istatistikleri.get("kenar_artefakt_temizlendi", 0) == 1

    def test_temizleme_kapaliyken_goruntu_degismez(self, monkeypatch):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", False)
        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=8)
        img[:10, 20:80] = 250

        temiz = isleyici.kenar_artefakt_temizle(img)

        np.testing.assert_array_equal(temiz, img)

    def test_goruntu_isle_kenar_artefakt_temizleme_ile_256x256_uint8(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)

        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=9)
        img[:12, 20:80] = 250
        path = tmp_path / "edge.png"
        Image.fromarray(img, mode="L").save(path)

        sonuc = isleyici.goruntu_isle(str(path))

        assert sonuc is not None
        assert sonuc.shape == (gi.HEDEF_YUKSEKLIK, gi.HEDEF_GENISLIK)
        assert sonuc.dtype == np.uint8

    def test_goruntu_isle_temizleme_kapaliyken_uyumluluk_korunur(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", False)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", False)

        isleyici = GorselIsleyici()
        img = self._temiz_mri_benzeri_goruntu(seed=10)
        path = tmp_path / "clean.png"
        Image.fromarray(img, mode="L").save(path)

        sonuc = isleyici.goruntu_isle(str(path))

        assert sonuc is not None
        assert sonuc.shape == (gi.HEDEF_YUKSEKLIK, gi.HEDEF_GENISLIK)
        assert sonuc.dtype == np.uint8
        # Yeni sayaclar varsa bile sifir kalmali
        assert isleyici.kalite_istatistikleri.get("kenar_artefakt_tespit", 0) == 0
        assert isleyici.kalite_istatistikleri.get("kenar_artefakt_temizlendi", 0) == 0

    def test_toplu_isleme_kenar_artefakt_sayaclarini_toplar(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", True)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", True)

        isleyici = GorselIsleyici()
        isleyici.n_jobs = 1

        dataset = tmp_path / "dataset"
        sinif_klasoru = dataset / "NonDemented"
        sinif_klasoru.mkdir(parents=True)

        # Iki goruntu, ikisinde de ust kenar artefakti
        for i in range(2):
            img = self._temiz_mri_benzeri_goruntu(seed=20 + i)
            img[:12, 20:80] = 250
            Image.fromarray(img, mode="L").save(sinif_klasoru / f"art_{i}.png")

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        isleyici.tum_gorselleri_isle(cikti, giris_klasoru=dataset)

        assert isleyici.kalite_istatistikleri["kenar_artefakt_tespit"] >= 2
        assert isleyici.kalite_istatistikleri["kenar_artefakt_temizlendi"] >= 2


class TestGorselIsleyiciEdgeCases:
    """Sınır durumları ve hata yönetimi testleri."""

    def test_empty_image(self):
        """Tüm pikselleri sıfır olan görüntü kalite kontrolünden geçmemeli."""
        isleyici = GorselIsleyici()

        empty_img = np.zeros((256, 256), dtype=np.uint8)
        is_valid, message = isleyici.goruntu_kalite_kontrol(empty_img)

        assert is_valid is False

    def test_very_small_image(self):
        """10x10 görüntü 256x256'ya büyütülebilmeli."""
        isleyici = GorselIsleyici()

        small_img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        resized = isleyici.boyutlandir(small_img, 256, 256)

        assert resized.shape == (256, 256)

    def test_very_large_image(self):
        """2048x2048 görüntü 256x256'ya küçültülebilmeli."""
        isleyici = GorselIsleyici()

        large_img = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
        resized = isleyici.boyutlandir(large_img, 256, 256)

        assert resized.shape == (256, 256)

    def test_yogunluk_normalize_constant_image(self):
        """Sabit değerli görüntü normalizasyonda NaN üretmemeli."""
        isleyici = GorselIsleyici()

        constant_img = np.full((256, 256), 128, dtype=np.uint8)
        normalized = isleyici.yogunluk_normalize(constant_img)

        assert normalized.shape == constant_img.shape
        assert not np.isnan(normalized).any()
        assert normalized.dtype == np.uint8

    def test_yogunluk_normalize_all_zero_image_background_kalir(self):
        """Tam siyah goruntu maske yokken sifir ve uint8 kalmali."""
        isleyici = GorselIsleyici()
        img = np.zeros((64, 64), dtype=np.uint8)

        normalized = isleyici.yogunluk_normalize(img)

        np.testing.assert_array_equal(normalized, img)
        assert normalized.dtype == np.uint8

    def test_yogunluk_normalize_tiny_foreground_reddedilir(self):
        """Cok kucuk foreground percentile hesabi icin guvenilir sayilmamali."""
        isleyici = GorselIsleyici()
        img = np.zeros((64, 64), dtype=np.uint8)
        img[30:32, 30:32] = 180

        normalized = isleyici.yogunluk_normalize(img)

        assert int(normalized.max()) == 0
        assert normalized.dtype == np.uint8

    def test_yogunluk_normalize_low_contrast_foreground_sifir_doner(self):
        """Dusuk kontrastli foreground normalize edilirken NaN veya yapay sinyal uretmemeli."""
        isleyici = GorselIsleyici()
        img = np.zeros((96, 96), dtype=np.uint8)
        yy, xx = np.ogrid[:96, :96]
        beyin = (yy - 48) ** 2 + (xx - 48) ** 2 <= 24 ** 2
        img[beyin] = 120

        normalized = isleyici.yogunluk_normalize(img)

        assert int(normalized.max()) == 0
        assert normalized.dtype == np.uint8


class TestGorselIsleyiciRegressions:
    """Regresyon testleri."""

    def test_ozel_giris_klasoru_global_veri_setine_dusmez(self, tmp_path, monkeypatch):
        """Ozel giris klasoru cozulurken repo varsayilanina sessizce dusulmemeli."""
        isleyici = GorselIsleyici()

        ozel_giris = tmp_path / "ozel_giris"
        ozel_giris.mkdir()

        varsayilan_orig = tmp_path / "varsayilan" / "OriginalDataset" / "NonDemented"
        varsayilan_orig.mkdir(parents=True)
        Image.fromarray(
            np.random.randint(80, 180, (64, 64), dtype=np.uint8), mode='L'
        ).save(varsayilan_orig / "sample.jpg")

        assert isleyici._giris_klasoru_cozumle(ozel_giris) == ozel_giris

    def test_instance_rngleri_bagimsizdir(self):
        """Her isleyici instance'i ayri RNG kullanmali."""
        isleyici_a = GorselIsleyici()
        isleyici_b = GorselIsleyici()

        assert isleyici_a._random.random() != isleyici_b._random.random()

    def test_ayni_stem_farkli_uzantilar_birbirini_ezmez(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gi, "MIN_STD_INTENSITY", 5)
        isleyici = GorselIsleyici()
        isleyici.n_jobs = 1

        dataset = tmp_path / "dataset"
        sinif_klasoru = dataset / "NonDemented"
        sinif_klasoru.mkdir(parents=True)

        arr_a = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) + 80
        arr_b = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) + 120
        Image.fromarray(arr_a, mode="L").save(sinif_klasoru / "sample.jpg")
        Image.fromarray(arr_b, mode="L").save(sinif_klasoru / "sample.png")

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        istatistikler = isleyici.tum_gorselleri_isle(cikti, giris_klasoru=dataset)
        kaydedilenler = sorted(p.name for p in (cikti / "NonDemented").glob("*.png"))

        assert istatistikler["NonDemented"] == 2
        assert kaydedilenler == ["sample_jpg.png", "sample_png.png"]

    def test_paralel_hata_olursa_sequential_fallback_kullanilir(self, test_dataset_structure, tmp_path, monkeypatch):
        class FailingPool:
            def __init__(self, *args, **kwargs):
                raise PermissionError("pool blocked")

        monkeypatch.setattr(gi, "Pool", FailingPool)
        monkeypatch.setattr(gi, "tqdm", lambda iterable, **kwargs: iterable)

        isleyici = GorselIsleyici()
        isleyici.n_jobs = 2

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        istatistikler = isleyici.tum_gorselleri_isle(cikti, giris_klasoru=test_dataset_structure)

        assert sum(istatistikler.values()) > 0
        assert len(list(cikti.rglob("*.png"))) > 0

    def test_modul_paket_olarak_import_edilebilir(self):
        """goruntu_isleme.goruntu_isleyici paket import'u ile yuklenebilmeli."""
        modul = importlib.import_module("goruntu_isleme.goruntu_isleyici")
        assert hasattr(modul, "GorselIsleyici")

    def test_log_mesajlari_ascii_kalir(self, test_dataset_structure, tmp_path, monkeypatch, capsys):
        class DummyPool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                for item in iterable:
                    yield func(item)

        monkeypatch.setattr(gi, "Pool", DummyPool)
        monkeypatch.setattr(gi, "tqdm", lambda iterable, **kwargs: iterable)

        isleyici = GorselIsleyici()
        isleyici.n_jobs = 2

        cikti = tmp_path / "cikti"
        cikti.mkdir()

        isleyici.tum_gorselleri_isle(cikti, giris_klasoru=test_dataset_structure)
        output = capsys.readouterr().out

        assert "⚡" not in output
        assert "📊" not in output


class TestBackgroundInvariant:
    """Pipeline boyunca background=0 invariant'i ve final QC testleri."""

    @staticmethod
    def _beyin_benzeri_uint8(seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.zeros((128, 128), dtype=np.uint8)
        yy, xx = np.ogrid[:128, :128]
        beyin = (yy - 64) ** 2 + (xx - 64) ** 2 <= 40 ** 2
        doku = rng.integers(80, 180, size=img.shape, dtype=np.uint8)
        img[beyin] = doku[beyin]
        return img

    def test_histogram_esitle_none_girdi(self):
        isleyici = GorselIsleyici()
        assert isleyici.histogram_esitle(None) is None

    def test_histogram_esitle_bos_array(self):
        isleyici = GorselIsleyici()
        bos = np.zeros((0, 0), dtype=np.uint8)
        sonuc = isleyici.histogram_esitle(bos)
        assert isinstance(sonuc, np.ndarray)
        assert sonuc.size == 0

    def test_histogram_esitle_background_sifir_kalir(self):
        isleyici = GorselIsleyici()
        img = self._beyin_benzeri_uint8(seed=1)
        background = img == 0
        assert background.any()

        sonuc = isleyici.histogram_esitle(img, adaptive=False)

        assert sonuc.dtype == np.uint8
        assert sonuc.ndim == 2
        # CLAHE arka plani 0'dan kaydirabilir; helper'in geri sifirlamasi gerek
        assert int(sonuc[background].max()) == 0
        # Foreground'da hala sinyal olmali
        assert int(sonuc[~background].max()) > 0

    def test_z_score_normalize_background_sifir_kalir(self):
        isleyici = GorselIsleyici()
        img = self._beyin_benzeri_uint8(seed=2).astype(np.uint8)
        background = img == 0
        assert background.any()

        sonuc = isleyici.z_score_normalize(img)

        assert sonuc.dtype == np.uint8
        assert sonuc.shape == img.shape
        assert int(sonuc[background].max()) == 0

    def test_aggressive_strateji_background_griye_tasinmaz(self, monkeypatch):
        monkeypatch.setattr(gi, "NORMALIZASYON_STRATEJISI", "aggressive")
        isleyici = GorselIsleyici()
        img = self._beyin_benzeri_uint8(seed=3)
        background = img == 0
        assert background.any()

        sonuc = isleyici._apply_normalization_strategy(img)

        assert sonuc.dtype == np.uint8
        # Aggressive (yogunluk_normalize + CLAHE + z-score) sonucunda
        # background hala sifir kalmalidir.
        assert int(sonuc[background].max()) == 0

    def test_pipeline_sonu_kalite_kontrol_helper_siyah_reddeder(self):
        isleyici = GorselIsleyici()
        siyah = np.zeros((64, 64), dtype=np.uint8)
        ok, sebep = isleyici._pipeline_sonu_kalite_kontrol(siyah)
        assert ok is False
        assert sebep

    def test_pipeline_sonu_kalite_kontrol_helper_dusuk_kontrast_reddeder(self):
        isleyici = GorselIsleyici()
        img = np.zeros((96, 96), dtype=np.uint8)
        # Genis ama tek tonlu foreground; std cok dusuk
        img[20:80, 20:80] = 100
        ok, sebep = isleyici._pipeline_sonu_kalite_kontrol(img)
        assert ok is False
        assert sebep

    def test_pipeline_sonu_kalite_kontrol_helper_saglikli_kabul_eder(self):
        isleyici = GorselIsleyici()
        img = self._beyin_benzeri_uint8(seed=4)
        ok, _ = isleyici._pipeline_sonu_kalite_kontrol(img)
        assert ok is True

    @staticmethod
    def _pipeline_no_op_monkeypatch(monkeypatch, isleyici, kaynak):
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_KONTROL_AKTIF", False)
        monkeypatch.setattr(gi, "KENAR_ARTEFAKT_TEMIZLEME_AKTIF", False)
        monkeypatch.setattr(isleyici, "goruntu_yukle", lambda _: kaynak)
        monkeypatch.setattr(isleyici, "goruntu_kalite_kontrol", lambda _: (True, ""))
        monkeypatch.setattr(isleyici, "gurultu_gider", lambda img, metod='auto': img)
        monkeypatch.setattr(isleyici, "center_of_mass_alignment", lambda img: img)
        monkeypatch.setattr(isleyici, "_apply_normalization_strategy", lambda img: img)

    def test_final_qc_siyah_pipeline_ciktisini_reddeder(self, monkeypatch):
        isleyici = GorselIsleyici()
        kaynak = self._beyin_benzeri_uint8(seed=5)
        self._pipeline_no_op_monkeypatch(monkeypatch, isleyici, kaynak)
        # boyutlandir sonrasi siyah cikti simulasyonu
        siyah = np.zeros((64, 64), dtype=np.uint8)
        monkeypatch.setattr(isleyici, "boyutlandir", lambda img: siyah)

        sonuc = isleyici.goruntu_isle_sonuc("dummy.png")

        assert sonuc["processed_image"] is None
        assert sonuc["quality_rejected"] is True
        assert sonuc["quality_reason"] == "pipeline_sonu_red"
        assert isleyici.kalite_istatistikleri["pipeline_sonu_red"] == 1

    def test_final_qc_dusuk_kontrastli_pipeline_ciktisini_reddeder(self, monkeypatch):
        isleyici = GorselIsleyici()
        kaynak = self._beyin_benzeri_uint8(seed=6)
        self._pipeline_no_op_monkeypatch(monkeypatch, isleyici, kaynak)
        # Foreground dolu ama std neredeyse sifir
        dusuk = np.full((64, 64), 100, dtype=np.uint8)
        monkeypatch.setattr(isleyici, "boyutlandir", lambda img: dusuk)

        sonuc = isleyici.goruntu_isle_sonuc("dummy.png")

        assert sonuc["processed_image"] is None
        assert sonuc["quality_rejected"] is True
        assert sonuc["quality_reason"] == "pipeline_sonu_red"
        assert isleyici.kalite_istatistikleri["pipeline_sonu_red"] == 1

    def test_final_qc_saglikli_ciktiyi_gecirir(self, monkeypatch):
        isleyici = GorselIsleyici()
        kaynak = self._beyin_benzeri_uint8(seed=7)
        self._pipeline_no_op_monkeypatch(monkeypatch, isleyici, kaynak)
        monkeypatch.setattr(isleyici, "boyutlandir", lambda img: img)

        sonuc = isleyici.goruntu_isle_sonuc("dummy.png")

        assert sonuc["processed_image"] is not None
        assert sonuc["quality_rejected"] is False
        assert isleyici.kalite_istatistikleri.get("pipeline_sonu_red", 0) == 0

    def test_histogram_esitle_tek_kanalli_3d_girdi(self):
        # shape[2]==1 OpenCV cvtColor'da hata verir; helper squeeze etmeli.
        isleyici = GorselIsleyici()
        img = self._beyin_benzeri_uint8(seed=11)[..., None]
        sonuc = isleyici.histogram_esitle(img, adaptive=False)
        assert sonuc.dtype == np.uint8
        assert sonuc.ndim == 2

    def test_z_score_normalize_uint16_girdi_uint8_doner(self):
        # std cok dusuk ve foreground yetersiz oldugu erken donus akislarinda
        # bile sozlesme uint8 olmali.
        isleyici = GorselIsleyici()
        sabit_uint16 = np.full((64, 64), 1000, dtype=np.uint16)
        sonuc = isleyici.z_score_normalize(sabit_uint16)
        assert sonuc.dtype == np.uint8

