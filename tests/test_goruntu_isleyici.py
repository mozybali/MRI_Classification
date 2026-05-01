"""
Görüntü İşleyici Modülü Testleri
Tests for goruntu_isleyici.py module.

Bu dosya GorselIsleyici sınıfının tüm fonksiyonlarını test eder.
Görüntü yükleme, normalizasyon, histogram eşitleme gibi temel işlemleri doğrular.
"""

import sys
import importlib
import pytest
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

    def test_histogram_esitle_dtype_uint8(self):
        """histogram_esitle her zaman uint8 dönmeli (skimage fallback dahil)."""
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
        monkeypatch.setattr(isleyici, "bias_field_correction", lambda img: img)
        monkeypatch.setattr(isleyici, "skull_strip", lambda img: img)
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

    def test_tum_gorselleri_isle_ve_bol_test_splitinde_augmentation_kapatir(self, tmp_path, monkeypatch):
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

        monkeypatch.setattr(
            isleyici,
            "goruntu_isle",
            lambda _yol: np.tile(np.arange(32, dtype=np.uint8), (32, 1)),
        )

        cikti = tmp_path / "cikti"
        sonuc = isleyici.tum_gorselleri_isle_ve_bol(cikti, giris_klasoru=dataset)

        assert set(sonuc) == {"trainval", "test"}
        assert any((cikti / "trainval").rglob("*.png"))
        assert any((cikti / "test").rglob("*.png"))
        assert all("_aug" not in path.name for path in (cikti / "test").rglob("*.png"))

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

    def test_instance_rngleri_bagimsizdir(self, monkeypatch):
        """Her isleyici instance'i augmentation icin ayri RNG kullanmali."""
        monkeypatch.setattr(gi, "GAUSSIAN_NOISE_AKTIF", True)

        isleyici_a = GorselIsleyici()
        isleyici_b = GorselIsleyici()
        img = np.full((32, 32), 128, dtype=np.uint8)

        assert isleyici_a._random.random() != isleyici_b._random.random()
        assert not np.array_equal(isleyici_a.gaussian_noise(img), isleyici_b.gaussian_noise(img))

    def test_ayni_stem_farkli_uzantilar_birbirini_ezmez(self, tmp_path):
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
