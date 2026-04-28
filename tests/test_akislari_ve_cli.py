import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eda_analiz import eda_calistir
from goruntu_isleme import ana_islem
from goruntu_isleme.ayarlar import SCALER_DOSYA_ADI
from goruntu_isleme.ozellik_cikarici import OzellikCikarici, veri_setini_bol_ve_olceklendir


def _grouped_features_df(prefix: str = "aug", per_class: int = 4) -> pd.DataFrame:
    rows = []
    classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    for class_index, class_name in enumerate(classes):
        for source_idx in range(per_class):
            rows.append(
                {
                    "dosya_adi": f"{prefix}_{class_name}_{source_idx}.png",
                    "feature1": class_index * 10 + source_idx,
                    "feature2": class_index * 100 + source_idx,
                    "sinif": class_name,
                    "etiket": class_index,
                    "tam_yol": f"/tmp/{prefix}_{class_name}_{source_idx}.png",
                }
            )
    return pd.DataFrame(rows)


def test_tek_goruntu_ozellikleri_beklenen_alanlari_uretir(tmp_path):
    image_path = tmp_path / "gradient.png"
    arr = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    Image.fromarray(arr, mode="L").save(image_path)

    sonuc = OzellikCikarici().tek_goruntu_ozellikleri(str(image_path))

    assert sonuc is not None
    assert sonuc["dosya_adi"] == "gradient.png"
    assert sonuc["genislik"] == 16
    assert sonuc["yukseklik"] == 16
    assert sonuc["piksel_sayisi"] == 256
    assert "entropi" in sonuc
    assert "otsu_esik" in sonuc


def test_kaynak_kolonlarini_hazirla_eksik_kolonlari_tamamlar():
    df = pd.DataFrame(
        {
            "dosya_adi": ["26.png", "26_aug1.png"],
            "sinif": ["NonDemented", "NonDemented"],
        }
    )

    hazir = OzellikCikarici.kaynak_kolonlarini_hazirla(df)

    assert hazir["kaynak_id"].tolist() == ["26", "26"]
    assert hazir["kaynak_grup"].tolist() == ["NonDemented::26", "NonDemented::26"]
    assert hazir["augmentasyon_mu"].tolist() == [False, True]


def test_nan_temizle_mean_sayisal_nanlari_doldurur_ve_kaydeder(tmp_path):
    csv_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "dosya_adi": ["a.png", "a_aug1.png", "b.png"],
            "feature1": [1.0, np.nan, 4.0],
            "feature2": [np.nan, 5.0, 8.0],
            "sinif": ["NonDemented", "NonDemented", "MildDemented"],
            "etiket": [0, 0, 2],
        }
    ).to_csv(csv_path, index=False)

    temiz = OzellikCikarici().nan_temizle(csv_dosyasi=csv_path, metod="mean")
    dosyadan = pd.read_csv(csv_path)

    assert temiz["feature1"].isna().sum() == 0
    assert temiz["feature2"].isna().sum() == 0
    assert {"kaynak_id", "kaynak_grup", "augmentasyon_mu"}.issubset(temiz.columns)
    pd.testing.assert_frame_equal(dosyadan, temiz, check_dtype=False)


def test_veri_setini_bol_ve_olceklendir_scaler_ve_csvleri_kaydeder(tmp_path):
    trainval_csv = tmp_path / "grouped_augmented.csv"
    test_csv = tmp_path / "grouped_original.csv"
    trainval_df = _grouped_features_df(prefix="aug", per_class=4)
    test_df_raw = _grouped_features_df(prefix="orig", per_class=2)
    trainval_df.loc[0, "feature1"] = np.nan
    trainval_df.loc[1, "feature2"] = np.nan
    test_df_raw.loc[0, "feature2"] = np.nan
    trainval_df.to_csv(trainval_csv, index=False)
    test_df_raw.to_csv(test_csv, index=False)

    train_df, val_df, test_df = veri_setini_bol_ve_olceklendir(
        csv_dosyasi=trainval_csv,
        cikti_klasoru=tmp_path,
        metod="minmax",
        test_csv_dosyasi=test_csv,
    )

    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty
    assert train_df[["feature1", "feature2"]].isna().sum().sum() == 0
    assert val_df[["feature1", "feature2"]].isna().sum().sum() == 0
    assert test_df[["feature1", "feature2"]].isna().sum().sum() == 0
    assert (tmp_path / "egitim_scaled.csv").exists()
    assert (tmp_path / "dogrulama_scaled.csv").exists()
    assert (tmp_path / "test_scaled.csv").exists()
    assert not (tmp_path / "goruntu_ozellikleri_scaled.csv").exists()
    assert (tmp_path / SCALER_DOSYA_ADI).exists()

    with open(tmp_path / SCALER_DOSYA_ADI, "rb") as file:
        scaler_info = pickle.load(file)

    assert scaler_info["method"] == "minmax"
    assert scaler_info["columns"] == ["feature1", "feature2"]


def test_parse_args_extract_action_yollarini_cozer(tmp_path):
    args = ana_islem.parse_args(
        [
            "--action",
            "extract",
            "--input-dir",
            str(tmp_path / "girdi"),
            "--csv-path",
            str(tmp_path / "out.csv"),
            "--test-csv-path",
            str(tmp_path / "test.csv"),
        ]
    )

    assert args.action == "extract"
    assert args.input_dir == tmp_path / "girdi"
    assert args.csv_path == tmp_path / "out.csv"
    assert args.test_csv_path == tmp_path / "test.csv"


def test_ozellik_cikar_splitli_ciktida_trainval_ve_test_csvlerini_uretir(monkeypatch, tmp_path):
    root = tmp_path / "cikti"
    for split_name in ("trainval", "test"):
        for class_name in ("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"):
            (root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    calls = []

    class DummyCikarici:
        def csv_olustur(self, giris_klasoru, cikti_csv=None):
            calls.append((giris_klasoru, cikti_csv))
            return pd.DataFrame({"feature1": [1], "sinif": ["NonDemented"], "etiket": [0]})

    monkeypatch.setattr(ana_islem, "OzellikCikarici", DummyCikarici)

    df = ana_islem.ozellik_cikar(giris_klasoru=root)

    assert not df.empty
    assert calls == [
        (root / "trainval", root / "goruntu_ozellikleri.csv"),
        (root / "test", root / "test_goruntu_ozellikleri.csv"),
    ]


def test_scaling_uygula_test_csvsini_otomatik_algilar(monkeypatch, tmp_path):
    captured = {}
    trainval_csv = tmp_path / "goruntu_ozellikleri.csv"
    test_csv = tmp_path / "test_goruntu_ozellikleri.csv"
    trainval_csv.write_text("feature1,sinif,etiket\n1,NonDemented,0\n", encoding="utf-8")
    test_csv.write_text("feature1,sinif,etiket\n2,NonDemented,0\n", encoding="utf-8")

    def fake_scale(**kwargs):
        captured.update(kwargs)
        return (
            pd.DataFrame({"x": [1]}),
            pd.DataFrame({"x": [2]}),
            pd.DataFrame({"x": [3]}),
        )

    monkeypatch.setattr(ana_islem, "veri_setini_bol_ve_olceklendir", fake_scale)

    ana_islem.scaling_uygula(csv_dosyasi=trainval_csv, cikti_klasoru=tmp_path, metod="robust")

    assert captured["csv_dosyasi"] == trainval_csv
    assert captured["test_csv_dosyasi"] == test_csv


def test_scaling_uygula_ozel_csv_adi_icin_eslesen_test_csvsini_algilar(monkeypatch, tmp_path):
    captured = {}
    trainval_csv = tmp_path / "custom_features.csv"
    test_csv = tmp_path / "test_custom_features.csv"
    trainval_csv.write_text("feature1,sinif,etiket\n1,NonDemented,0\n", encoding="utf-8")
    test_csv.write_text("feature1,sinif,etiket\n2,NonDemented,0\n", encoding="utf-8")

    def fake_scale(**kwargs):
        captured.update(kwargs)
        return (
            pd.DataFrame({"x": [1]}),
            pd.DataFrame({"x": [2]}),
            pd.DataFrame({"x": [3]}),
        )

    monkeypatch.setattr(ana_islem, "veri_setini_bol_ve_olceklendir", fake_scale)

    ana_islem.scaling_uygula(csv_dosyasi=trainval_csv, metod="robust")

    assert captured["csv_dosyasi"] == trainval_csv
    assert captured["test_csv_dosyasi"] == test_csv


def test_ozellik_cikar_test_csv_bos_donerken_hata_verir(monkeypatch, tmp_path):
    root = tmp_path / "cikti"
    for split_name in ("trainval", "test"):
        for class_name in ("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"):
            (root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    class DummyCikarici:
        def csv_olustur(self, giris_klasoru, cikti_csv=None):
            if Path(giris_klasoru).name == "test":
                return pd.DataFrame()
            return pd.DataFrame({"feature1": [1], "sinif": ["NonDemented"], "etiket": [0]})

    monkeypatch.setattr(ana_islem, "OzellikCikarici", DummyCikarici)

    sonuc = ana_islem.ozellik_cikar(giris_klasoru=root)

    assert sonuc is None


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
    assert ana_islem.main(["--action", "report"]) == 0

    monkeypatch.setattr(ana_islem, "run_action", lambda args: None)
    assert ana_islem.main(["--action", "report"]) == 1


def test_tum_islemleri_yap_skip_confirmation_ile_sirayi_calistirir(monkeypatch, tmp_path):
    called = {"preprocess": 0, "csv": 0, "report": 0}

    class DummyIsleyici:
        def tum_gorselleri_isle(self, cikti_klasoru, giris_klasoru=None):
            called["preprocess"] += 1
            return {"NonDemented": 1}

    class DummyCikarici:
        def csv_olustur(self, cikti_klasoru, cikti_csv=None):
            called["csv"] += 1
            return pd.DataFrame({"feature1": [1], "sinif": ["NonDemented"], "etiket": [0]})

        def istatistik_raporu(self, csv_dosyasi=None):
            called["report"] += 1

    monkeypatch.setattr(ana_islem, "GorselIsleyici", DummyIsleyici)
    monkeypatch.setattr(ana_islem, "OzellikCikarici", DummyCikarici)
    monkeypatch.setattr(
        ana_islem,
        "veri_setini_bol_ve_olceklendir",
        lambda csv_dosyasi=None, cikti_klasoru=None, metod=None: (
            pd.DataFrame({"x": [1]}),
            pd.DataFrame({"x": [2]}),
            pd.DataFrame({"x": [3]}),
        ),
    )

    sonuc = ana_islem.tum_islemleri_yap(
        giris_klasoru=tmp_path / "girdi",
        cikti_klasoru=tmp_path / "cikti",
        skip_confirmation=True,
    )

    assert sonuc is not None
    assert called == {"preprocess": 1, "csv": 1, "report": 1}


def test_tum_islemleri_yap_ozel_cikti_csv_yolunu_sabitleyerek_ilerler(monkeypatch, tmp_path):
    captured = {}

    class DummyIsleyici:
        def tum_gorselleri_isle(self, cikti_klasoru, giris_klasoru=None):
            captured["preprocess_output"] = cikti_klasoru
            return {"NonDemented": 1}

    class DummyCikarici:
        def csv_olustur(self, cikti_klasoru, cikti_csv=None):
            captured["feature_dir"] = cikti_klasoru
            captured["feature_csv"] = cikti_csv
            return pd.DataFrame({"feature1": [1], "sinif": ["NonDemented"], "etiket": [0]})

        def istatistik_raporu(self, csv_dosyasi=None):
            captured["report_csv"] = csv_dosyasi

    def fake_split(csv_dosyasi=None, cikti_klasoru=None, metod=None):
        captured["split_csv"] = csv_dosyasi
        captured["split_output"] = cikti_klasoru
        return pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}), pd.DataFrame({"x": [3]})

    monkeypatch.setattr(ana_islem, "GorselIsleyici", DummyIsleyici)
    monkeypatch.setattr(ana_islem, "OzellikCikarici", DummyCikarici)
    monkeypatch.setattr(ana_islem, "veri_setini_bol_ve_olceklendir", fake_split)

    cikti = tmp_path / "ozel_cikti"
    ana_islem.tum_islemleri_yap(
        giris_klasoru=tmp_path / "girdi",
        cikti_klasoru=cikti,
        skip_confirmation=True,
    )

    beklenen_csv = cikti / "goruntu_ozellikleri.csv"
    assert captured["feature_csv"] == beklenen_csv
    assert captured["split_csv"] == beklenen_csv
    assert captured["report_csv"] == beklenen_csv


def test_parse_args_gecersiz_scaling_methodunu_erken_reddeder():
    with pytest.raises(SystemExit):
        ana_islem.parse_args(["--action", "scale", "--method", "mean"])


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
            raise RuntimeError("boom")

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
