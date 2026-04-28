import subprocess
import sys
from pathlib import Path

import pandas as pd

from eda_analiz import eda_calistir
from goruntu_isleme import ana_islem


def test_run_action_preprocess_parametreleri_iletir(monkeypatch, tmp_path):
    captured = {}

    def fake_preprocess(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(ana_islem, "goruntu_on_isleme", fake_preprocess)

    args = ana_islem.parse_args(
        [
            "--action",
            "preprocess",
            "--input-dir",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    result = ana_islem.run_action(args)

    assert result == {"ok": True}
    assert captured["giris_klasoru"] == tmp_path / "in"
    assert captured["cikti_klasoru"] == tmp_path / "out"


def test_main_non_menu_sonucuna_gore_cikis_kodu_verir(monkeypatch):
    monkeypatch.setattr(ana_islem, "run_action", lambda args: {"ok": True})
    assert ana_islem.main(["--action", "preprocess"]) == 0

    monkeypatch.setattr(ana_islem, "run_action", lambda args: None)
    assert ana_islem.main(["--action", "preprocess"]) == 1


def test_eda_resolve_paths_defaults_noninteractive(monkeypatch, tmp_path):
    monkeypatch.setattr(eda_calistir.sys.stdin, "isatty", lambda: False)

    args = eda_calistir.parse_args([])
    data_dir, output_dir = eda_calistir._resolve_paths(args)

    assert isinstance(data_dir, Path)
    assert isinstance(output_dir, Path)


def test_eda_calistir_importu_eda_araclarini_lazy_yukler():
    kod = (
        "import sys; "
        "from eda_analiz import eda_calistir; "
        "print('eda_analiz.eda_araclar' in sys.modules)"
    )

    sonuc = subprocess.run(
        [sys.executable, "-c", kod],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert sonuc.stdout.strip() == "False"


def test_eda_parse_args_jobs_degerini_cozer():
    args = eda_calistir.parse_args(["--jobs", "3"])

    assert args.jobs == 3


def test_eda_main_analizi_calistirip_csv_yazar(monkeypatch, tmp_path):
    captured = {}

    class DummyAnaliz:
        def __init__(self, veri_klasoru, cikti_klasoru, n_jobs=None):
            self.veri_klasoru = veri_klasoru
            self.cikti_klasoru = cikti_klasoru
            captured["n_jobs"] = n_jobs
            Path(cikti_klasoru).mkdir(parents=True, exist_ok=True)

        def tam_analiz_yap(self):
            return pd.DataFrame({"label": [0], "int_ort": [123.0]})

    monkeypatch.setattr(eda_calistir, "EDAAnaliz", DummyAnaliz)

    result = eda_calistir.main(
        [
            "--data-dir",
            str(tmp_path / "veri"),
            "--output-dir",
            str(tmp_path / "cikti"),
            "--jobs",
            "2",
        ]
    )

    assert result == 0
    assert captured["n_jobs"] == 2
    assert (tmp_path / "cikti" / "veri_seti_istatistikler.csv").exists()


def test_eda_main_hata_durumunda_bir_doner(monkeypatch, tmp_path):
    class FailingAnaliz:
        def __init__(self, veri_klasoru, cikti_klasoru, n_jobs=None):
            pass

        def tam_analiz_yap(self):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(eda_calistir, "EDAAnaliz", FailingAnaliz)

    result = eda_calistir.main(
        [
            "--data-dir",
            str(tmp_path / "veri"),
            "--output-dir",
            str(tmp_path / "cikti"),
        ]
    )

    assert result == 1
