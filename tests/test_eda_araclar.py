"""
Tests for eda_araclar.py module.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")

from eda_analiz import eda_araclar
from eda_analiz.eda_araclar import EDAAnaLiz


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
    def test_eda_araclar_bom_ile_baslamaz(self):
        assert Path(eda_araclar.__file__).read_bytes().startswith(b"#!/")

    def test_import_headless_backend_ile_savefig_yapar(self, tmp_path):
        cikti = tmp_path / "backend_probe.png"
        kod = (
            "import sys; "
            "from eda_analiz import eda_araclar; "
            "fig, ax = eda_araclar.plt.subplots(); "
            "ax.plot([1, 2]); "
            "fig.savefig(sys.argv[1]); "
            "print(eda_araclar.plt.get_backend().lower())"
        )

        sonuc = subprocess.run(
            [sys.executable, "-c", kod, str(cikti)],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

        assert "agg" in sonuc.stdout
        assert cikti.exists()

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

    def test_veri_yukle_dosyalari_deterministik_sirada_toplar(self, tmp_path):
        veri_klasoru = tmp_path / "veri"
        for sinif in SINIFLAR:
            (veri_klasoru / sinif).mkdir(parents=True, exist_ok=True)

        arr = np.full((16, 16), 100, dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(veri_klasoru / "NonDemented" / "b_ornek.png")
        Image.fromarray(arr, mode="L").save(veri_klasoru / "NonDemented" / "a_ornek.png")

        eda = EDAAnaLiz(veri_klasoru=veri_klasoru, cikti_klasoru=tmp_path / "out")
        df = eda.veri_yukle()

        non_demented = (
            df[df["label_name"] == "NonDemented"]["filepath"].map(lambda yol: Path(yol).name).tolist()
        )
        assert non_demented == ["a_ornek.png", "b_ornek.png"]

    def test_veri_yukle_root_auto_resolve_original(self, tmp_path):
        veri_koku = tmp_path / "Veri_Seti"
        original = veri_koku / "OriginalDataset"
        _dataset_yapisi_olustur(original)

        eda = EDAAnaLiz(veri_klasoru=veri_koku, cikti_klasoru=tmp_path / "out")
        df = eda.veri_yukle()

        assert len(df) == 12
        assert eda.veri_klasoru == original.resolve()

    def test_veri_yukle_bos_ozel_klasorde_varsayilana_dusmez(self, tmp_path):
        veri_klasoru = tmp_path / "bos_veri"
        veri_klasoru.mkdir()
        eda = EDAAnaLiz(veri_klasoru=veri_klasoru, cikti_klasoru=tmp_path / "out")

        with pytest.raises(FileNotFoundError):
            eda.veri_yukle()

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

    def test_goruntu_istatistikleri_paralel_hata_olursa_tek_cekirdege_duser(
        self, test_dataset_structure, tmp_path, monkeypatch, capsys
    ):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out", n_jobs=2)
        df = eda.veri_yukle()

        class BrokenPool:
            def __init__(self, *args, **kwargs):
                raise PermissionError("blocked")

        monkeypatch.setattr(eda_araclar, "Pool", BrokenPool)

        enriched = eda.goruntu_istatistikleri_hesapla(df)
        captured = capsys.readouterr()

        assert len(enriched) == len(df)
        assert eda.n_jobs == 1
        assert "tek cekirdege dusuluyor" in captured.out

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

    def test_yogunluk_analizi_girdi_dataframeini_degistirmez(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1
        df = eda.goruntu_istatistikleri_hesapla(eda.veri_yukle())
        onceki_kolonlar = list(df.columns)

        eda.yogunluk_analizi_ciz(df)

        assert list(df.columns) == onceki_kolonlar

    def test_pca_analizi_olceklenmis_veriyi_kullanir(self, tmp_path, monkeypatch):
        veri_klasoru = tmp_path / "veri"
        _dataset_yapisi_olustur(veri_klasoru)
        eda = EDAAnaLiz(veri_klasoru=veri_klasoru, cikti_klasoru=tmp_path / "out")
        gozlem = {}

        class FakeScaler:
            def fit_transform(self, X):
                gozlem["scaler_input"] = X.copy()
                donusmus = X + 7
                gozlem["scaled_output"] = donusmus.copy()
                return donusmus

        class FakePCA:
            def __init__(self, n_components, random_state):
                self.explained_variance_ratio_ = np.array([0.6, 0.4])

            def fit_transform(self, X):
                gozlem["pca_input"] = X.copy()
                return np.column_stack([np.arange(len(X)), np.arange(len(X))])

        monkeypatch.setattr(eda_araclar, "StandardScaler", FakeScaler)
        monkeypatch.setattr(eda_araclar, "PCA", FakePCA)

        df = pd.DataFrame(
            {
                "label_name": ["NonDemented", "MildDemented", "ModerateDemented"],
                "genislik": [10.0, 20.0, 30.0],
                "yukseklik": [11.0, 21.0, 31.0],
                "en_boy_orani": [1.0, 1.1, 1.2],
                "int_ort": [50.0, 60.0, 70.0],
                "int_std": [5.0, 6.0, 7.0],
                "int_min": [1.0, 2.0, 3.0],
                "int_max": [100.0, 110.0, 120.0],
                "int_p1": [2.0, 3.0, 4.0],
                "int_p99": [98.0, 108.0, 118.0],
            }
        )

        eda.pca_analizi_ciz(df, n_ornekler=3)

        np.testing.assert_array_equal(gozlem["pca_input"], gozlem["scaled_output"])

    def test_tam_analiz_yap(self, test_dataset_structure, tmp_path):
        eda = EDAAnaLiz(veri_klasoru=test_dataset_structure, cikti_klasoru=tmp_path / "out")
        eda.n_jobs = 1

        df = eda.tam_analiz_yap()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert (eda.cikti_klasoru / "0_ozet_istatistikler.txt").exists()
