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
        """giris_klasoru=None ise varsayılan VERI_SETI_KLASORU gerçekten kullanılmalı."""
        isleyici = GorselIsleyici()
        varsayilan_dataset = tmp_path / "varsayilan_dataset"
        sinif = "NonDemented"
        sinif_klasoru = varsayilan_dataset / sinif
        sinif_klasoru.mkdir(parents=True)

        img = Image.fromarray(
            np.random.randint(80, 180, (64, 64), dtype=np.uint8), mode='L'
        )
        img.save(sinif_klasoru / "sample.jpg")

        monkeypatch.setattr(gi, "VERI_SETI_KLASORU", varsayilan_dataset)

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

        varsayilan_aug = tmp_path / "varsayilan" / "AugmentedAlzheimerDataset" / "NonDemented"
        varsayilan_aug.mkdir(parents=True)
        Image.fromarray(
            np.random.randint(80, 180, (64, 64), dtype=np.uint8), mode='L'
        ).save(varsayilan_aug / "sample.jpg")

        monkeypatch.setattr(gi, "AUGMENTED_VERI_SETI_KLASORU", varsayilan_aug.parent)
        monkeypatch.setattr(gi, "ORIGINAL_VERI_SETI_KLASORU", tmp_path / "varsayilan" / "OriginalDataset")

        assert isleyici._giris_klasoru_cozumle(ozel_giris) == ozel_giris

    def test_instance_rngleri_bagimsizdir(self, monkeypatch):
        """Her isleyici instance'i augmentation icin ayri RNG kullanmali."""
        monkeypatch.setattr(gi, "GAUSSIAN_NOISE_AKTIF", True)

        isleyici_a = GorselIsleyici()
        isleyici_b = GorselIsleyici()
        img = np.full((32, 32), 128, dtype=np.uint8)

        assert isleyici_a._random.random() != isleyici_b._random.random()
        assert not np.array_equal(isleyici_a.gaussian_noise(img), isleyici_b.gaussian_noise(img))

    def test_modul_paket_olarak_import_edilebilir(self):
        """goruntu_isleme.goruntu_isleyici paket import'u ile yuklenebilmeli."""
        modul = importlib.import_module("goruntu_isleme.goruntu_isleyici")
        assert hasattr(modul, "GorselIsleyici")
