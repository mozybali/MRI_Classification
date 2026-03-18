"""
Tests for eda_araclar.py module.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "eda_analiz"))

import eda_araclar
from eda_araclar import EDAAnaLiz


SINIFLAR = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


def _ornek_goruntu_olustur(hedef: Path, adet: int = 3):
    hedef.mkdir(parents=True, exist_ok=True)
    for i in range(adet):
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(hedef / f"ornek_{i}.jpg")


def _dataset_yapisi_olustur(kok: Path):
    for sinif in SINIFLAR:
        _ornek_goruntu_olustur(kok / sinif, adet=3)


class TestEDAAnaLiz:
    def test_guvenli_print_unicode_encode_hatasinda_dusmez(self, monkeypatch):
        class FakeBuffer:
            def __init__(self):
                self.data = b""

            def write(self, chunk):
                self.data += chunk

            def flush(self):
                return None

        class FakeStdout:
            encoding = "ascii"

            def __init__(self):
                self.buffer = FakeBuffer()

            def write(self, metin):
                raise UnicodeEncodeError("ascii", metin, 0, 1, "bad char")

            def flush(self):
                return None

        sahte_stdout = FakeStdout()
        monkeypatch.setattr(eda_araclar.sys, "stdout", sahte_stdout)

        eda_araclar._guvenli_print("⚡ deneme")

        assert b"? deneme" in sahte_stdout.buffer.data

    def test_init_custom_paths(self, tmp_path):
        veri_klasoru = tmp_path / "veri"
        cikti_klasoru = tmp_path / "cikti"
        _dataset_yapisi_olustur(veri_klasoru)

        eda = EDAAnaLiz(veri_klasoru=veri_klasoru, cikti_klasoru=cikti_klasoru)

        assert eda.veri_klasoru == veri_klasoru.resolve()
        assert eda.cikti_klasoru.exists()

    def test_veri_yukle_class_dirs(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")

        df = eda.veri_yukle()

        assert len(df) == 12
        assert {"id", "filepath", "label", "label_name"}.issubset(df.columns)
        assert set(df["label_name"].unique()) == set(SINIFLAR)

    def test_veri_yukle_root_auto_resolve_augmented(self, tmp_path):
        veri_koku = tmp_path / "Veri_Seti"
        augmented = veri_koku / "AugmentedAlzheimerDataset"
        _dataset_yapisi_olustur(augmented)

        eda = EDAAnaLiz(veri_klasoru=veri_koku, cikti_klasoru=tmp_path / "out")
        df = eda.veri_yukle()

        assert len(df) == 12
        assert eda.veri_klasoru == augmented.resolve()

    def test_veri_yukle_invalid_path_raises(self, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=tmp_path / "yok", cikti_klasoru=tmp_path / "out")

        with pytest.raises(FileNotFoundError):
            eda.veri_yukle()

    def test_goruntu_istatistikleri_hesapla(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1
        df = eda.veri_yukle()

        enriched = eda.goruntu_istatistikleri_hesapla(df)

        beklenen = {"genislik", "yukseklik", "en_boy_orani", "int_ort", "int_std", "int_p99"}
        assert beklenen.issubset(enriched.columns)
        assert len(enriched) == len(df)
        assert enriched["int_ort"].notna().all()

    def test_goruntu_istatistikleri_hata_raporlar(self, test_dataset_structure, tmp_path, capsys):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1
        df = eda.veri_yukle()

        bozuk = Path(df.loc[0, "filepath"])
        bozuk.unlink()

        enriched = eda.goruntu_istatistikleri_hesapla(df)
        captured = capsys.readouterr()

        assert len(enriched) == len(df)
        assert enriched["int_ort"].isna().sum() >= 1
        assert "[UYARI]" in captured.out

    def test_grafikler_olusturulur(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1
        df = eda.goruntu_istatistikleri_hesapla(eda.veri_yukle())

        eda.sinif_dagilimi_ciz(df)
        eda.boyut_analizi_ciz(df)
        eda.yogunluk_analizi_ciz(df)
        eda.korelasyon_analizi_ciz(df)
        eda.pca_analizi_ciz(df, n_ornekler=12)

        assert (eda.cikti_klasoru / "1_sinif_dagilimi.png").exists()
        assert (eda.cikti_klasoru / "2_boyut_analizi.png").exists()
        assert (eda.cikti_klasoru / "3_yogunluk_analizi.png").exists()
        assert (eda.cikti_klasoru / "4_korelasyon_matrisi.png").exists()
        assert (eda.cikti_klasoru / "5_pca_analizi.png").exists()

    def test_korelasyon_analizi_all_nan_ise_atlanir(self, tmp_path, capsys):
        veri_klasoru = tmp_path / "veri"
        _dataset_yapisi_olustur(veri_klasoru)
        eda = EDAAnaLiz(veri_klasoru=veri_klasoru, cikti_klasoru=tmp_path / "out")

        df = pd.DataFrame(
            {
                "genislik": [256],
                "yukseklik": [256],
                "en_boy_orani": [1.0],
                "int_ort": [100.0],
                "int_std": [0.0],
                "int_min": [100.0],
                "int_max": [100.0],
                "int_p1": [100.0],
                "int_p99": [100.0],
            }
        )

        eda.korelasyon_analizi_ciz(df)
        captured = capsys.readouterr()

        assert "Korelasyon analizi atlandı" in captured.out
        assert not (eda.cikti_klasoru / "4_korelasyon_matrisi.png").exists()

    def test_tam_analiz_yap(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1

        df = eda.tam_analiz_yap()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert (eda.cikti_klasoru / "0_ozet_istatistikler.txt").exists()
