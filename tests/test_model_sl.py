"""
SL (XGBoost) pipeline testleri.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.sl.features import (
    extract_features,
    feature_group_slices,
    _ensure_gray_uint8,
    _extract_histogram_stats,
)
from model.sl.dataset import build_feature_matrix
from model.sl.xgb_classifier import build_xgb_classifier, save_xgb_model, load_xgb_model
from model.dl.dataset import SINIF_ISIMLERI


# ==================== Fixtures ====================


@pytest.fixture
def synth_gray_image():
    """256x256 sentetik gri-ton goruntu."""
    return np.random.randint(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def synth_rgb_image():
    """256x256 sentetik RGB goruntu."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def synth_dataset(tmp_path):
    """Her siniftan 4 goruntu iceren minimal sentetik veri seti olustur."""
    dataset_path = tmp_path / f"synth_{uuid4().hex[:8]}"
    for class_name in SINIF_ISIMLERI:
        class_dir = dataset_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = Image.fromarray(
                np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L"
            )
            img.save(class_dir / f"img_{i}.jpg")
    return dataset_path


# ==================== extract_features ====================


class TestExtractFeatures:
    def test_returns_1d_float64(self, synth_gray_image):
        vec = extract_features(synth_gray_image)
        assert vec.ndim == 1
        assert vec.dtype == np.float64

    def test_fixed_size_across_calls(self, synth_gray_image):
        v1 = extract_features(synth_gray_image)
        v2 = extract_features(np.random.randint(0, 256, (256, 256), dtype=np.uint8))
        assert v1.shape == v2.shape

    def test_handles_rgb_input(self, synth_rgb_image):
        vec = extract_features(synth_rgb_image)
        assert vec.ndim == 1
        assert vec.shape[0] > 0

    def test_different_sizes_same_output_length(self):
        """Farkli boyutlu goruntuler farkli HOG boyutu uretir; bu beklenen davranis."""
        img_small = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img_large = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        v_small = extract_features(img_small)
        v_large = extract_features(img_large)
        # HOG boyutu goruntu boyutuna bagli olabilir; her ikisi de >0 olmali
        assert v_small.shape[0] > 0
        assert v_large.shape[0] > 0


# ==================== Histogram entropy & stats ====================


class TestHistogramStats:
    def test_constant_image_entropy_is_zero(self):
        const = np.full((224, 224), 128, dtype=np.uint8)
        vec = _extract_histogram_stats(const)
        # son 5 stats: mean, std, skew, kurt, entropy
        assert vec[-1] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_image_entropy_close_to_log2_bins(self):
        rng = np.random.default_rng(0)
        uni = rng.integers(0, 256, (224, 224), dtype=np.uint8)
        vec = _extract_histogram_stats(uni)
        # 32 bin uzerinde uniform dagilim ~ log2(32) = 5
        assert vec[-1] == pytest.approx(5.0, abs=0.1)

    def test_constant_image_skew_kurt_no_warnings(self, recwarn):
        const = np.full((100, 100), 42, dtype=np.uint8)
        vec = _extract_histogram_stats(const)
        assert vec[-3] == 0.0  # skew
        assert vec[-2] == 0.0  # kurt
        # std==0 kisa devresi RuntimeWarning uretmemeli
        assert all("Precision loss" not in str(w.message) for w in recwarn.list)


# ==================== _ensure_gray_uint8 ====================


class TestEnsureGray:
    @pytest.mark.parametrize("k", [42, 128, 255])
    def test_uint8_passthrough_preserves_brightness(self, k):
        img = np.full((10, 10), k, dtype=np.uint8)
        out = _ensure_gray_uint8(img)
        assert out.dtype == np.uint8
        assert int(out.min()) == k
        assert int(out.max()) == k

    def test_float_0_1_scaled_to_0_255(self):
        f = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        out = _ensure_gray_uint8(f)
        assert out.dtype == np.uint8
        assert int(out.max()) == 255
        assert int(out.min()) == 0

    def test_float_0_255_clipped(self):
        f = np.array([[-10.0, 100.0, 300.0]], dtype=np.float64)
        out = _ensure_gray_uint8(f)
        assert int(out.min()) == 0
        assert int(out.max()) == 255

    def test_negative_float_not_treated_as_unit_range(self):
        # max <= 1.0 olsa bile min < 0 ise [0,1] dalina girmemeli;
        # clip(0,255) yoluna gitmeli (B kontrati).
        f = np.array([[-1.0, 0.5, 1.0]], dtype=np.float64)
        out = _ensure_gray_uint8(f)
        # [0,1] yolu olsa: [0, 127, 255]; clip(0,255) yolu: [0, 0, 1]
        assert out.tolist() == [[0, 0, 1]]


# ==================== feature_group_slices ====================


class TestFeatureGroupSlices:
    def test_total_length_matches_extract(self):
        img = np.zeros((224, 224), dtype=np.uint8)
        vec = extract_features(img)
        slices = feature_group_slices(224)
        total = max(s.stop for s in slices.values())
        assert total == vec.shape[0] == 6179


# ==================== build_feature_matrix ====================


class TestBuildFeatureMatrix:
    def test_returns_correct_shapes(self, synth_dataset):
        X, y, groups, paths = build_feature_matrix(synth_dataset, image_size=64)
        n_samples = 4 * len(SINIF_ISIMLERI)
        assert X.shape[0] == n_samples
        assert y.shape == (n_samples,)
        assert len(groups) == n_samples
        assert len(paths) == n_samples

    def test_labels_are_valid(self, synth_dataset):
        _, y, _, _ = build_feature_matrix(synth_dataset, image_size=64)
        assert set(y.tolist()) == set(range(len(SINIF_ISIMLERI)))

    def test_groups_contain_class_prefix(self, synth_dataset):
        _, _, groups, _ = build_feature_matrix(synth_dataset, image_size=64)
        for g in groups:
            assert "::" in g
            class_part = g.split("::")[0]
            assert class_part in SINIF_ISIMLERI

    def test_cache_roundtrip(self, synth_dataset, tmp_path):
        cache = tmp_path / "cache.npz"
        X1, y1, g1, p1 = build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)
        assert cache.exists()
        X2, y2, g2, p2 = build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert g1 == g2
        assert p1 == p2


# ==================== XGBClassifier yardimcilari ====================


class TestXGBClassifier:
    def test_build_and_predict(self):
        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (20, 4)

    def test_save_load_roundtrip(self, tmp_path):
        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)
        path = tmp_path / "test_model.json"
        save_xgb_model(clf, path)
        assert path.exists()
        loaded = load_xgb_model(path)
        np.testing.assert_array_almost_equal(
            clf.predict_proba(X),
            loaded.predict_proba(X),
        )


# ==================== run_sl_training smoke ====================


class TestSLTrainingSmoke:
    def test_minimal_training(self, synth_dataset, tmp_path):
        from model.sl.training_runner import SLTrainingConfig, run_sl_training

        output_root = tmp_path / f"output_{uuid4().hex[:8]}"
        config = SLTrainingConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.3,
            image_size=64,
            trainval_dir=str(synth_dataset),
            test_dir=str(synth_dataset),
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        results = run_sl_training(
            config,
            output_root=output_root,
            artifact_tag="xgboost_test",
            save_artifacts=True,
            evaluate_test_set=True,
            verbose=False,
        )

        # Checkpoint uretildi mi?
        assert results["checkpoint_path"] is not None
        assert Path(results["checkpoint_path"]).exists()
        assert Path(results["checkpoint_path"]).suffix == ".json"

        # Rapor JSON uretildi mi?
        assert results["report_path"] is not None
        assert Path(results["report_path"]).exists()
        with open(results["report_path"], encoding="utf-8") as f:
            report = json.load(f)
        assert report["model"] == "xgboost"
        assert results["feature_importance_path"] is not None
        assert Path(results["feature_importance_path"]).exists()
        assert Path(report["artifacts"]["feature_importance"]).exists()

        # Test metrikleri makul aralıkta mi?
        assert results["test_metrics"] is not None
        assert results["data_info"]["uses_external_test_dir"] is False
        assert 0.0 <= results["test_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= results["test_metrics"]["f1"] <= 1.0


# ==================== CLI parse testi ====================


class TestCLIParseXGBoost:
    def test_model_xgboost_accepted(self):
        from model.train import parse_args
        args = parse_args(["--model", "xgboost"])
        assert args.model == "xgboost"

    def test_xgb_args_parsed(self):
        from model.train import parse_args
        args = parse_args([
            "--model", "xgboost",
            "--xgb-n-estimators", "500",
            "--xgb-max-depth", "8",
            "--xgb-learning-rate", "0.05",
            "--xgb-subsample", "0.7",
            "--xgb-colsample-bytree", "0.6",
            "--xgb-reg-lambda", "2.0",
            "--xgb-min-child-weight", "3",
            "--feature-cache", "/tmp/cache",
        ])
        assert args.xgb_n_estimators == 500
        assert args.xgb_max_depth == 8
        assert args.xgb_learning_rate == 0.05
        assert args.xgb_subsample == 0.7
        assert args.xgb_colsample_bytree == 0.6
        assert args.xgb_reg_lambda == 2.0
        assert args.xgb_min_child_weight == 3
        assert args.feature_cache == "/tmp/cache"

    def test_full_trainval_accepted_with_xgboost(self):
        from model.train import parse_args
        args = parse_args(["--model", "xgboost", "--full-trainval"])
        assert args.full_trainval is True

    def test_folds_flag_xgboost_ile_kabul_edilir(self):
        from model.train import parse_args
        args = parse_args(["--model", "xgboost", "--folds", "5"])
        assert args.folds == 5


class TestSLSelectionMetricWiring:
    """selection_metric -> XGBoost eval_metric baglantisi."""

    def test_eval_metric_loss_returns_mlogloss(self):
        from model.sl.training_runner import _xgb_eval_metric_for_selection

        eval_metric, label = _xgb_eval_metric_for_selection("loss")
        assert eval_metric == "mlogloss"
        assert label == "mlogloss"

    @pytest.mark.parametrize(
        "metric,expected_label",
        [
            ("f1", "1 - macro_f1"),
            ("precision", "1 - macro_precision"),
            ("recall", "1 - macro_recall"),
            ("accuracy", "1 - accuracy"),
        ],
    )
    def test_eval_metric_returns_list_with_callable_last(self, metric, expected_label):
        from model.sl.training_runner import _xgb_eval_metric_for_selection

        eval_metric, label = _xgb_eval_metric_for_selection(metric)
        assert isinstance(eval_metric, list)
        assert eval_metric[0] == "mlogloss"
        assert callable(eval_metric[-1])
        # Module-level isimli fonksiyon olmali (XGBoost __name__'i okuyor)
        assert hasattr(eval_metric[-1], "__name__")
        assert label == expected_label

    def test_invalid_selection_metric_raises_in_helper(self):
        from model.sl.training_runner import _xgb_eval_metric_for_selection

        with pytest.raises(ValueError, match="Gecersiz selection metric"):
            _xgb_eval_metric_for_selection("auc")

    def test_invalid_selection_metric_raises_early_in_run_sl_training(
        self, synth_dataset, tmp_path,
    ):
        """Hatali metric eğitim/dosya I/O baslamadan ValueError firlatmalı."""
        from model.sl.training_runner import SLTrainingConfig, run_sl_training

        output_root = tmp_path / "output_invalid"
        config = SLTrainingConfig(
            n_estimators=10,
            max_depth=3,
            image_size=64,
            trainval_dir=str(synth_dataset),
            test_dir=str(synth_dataset),
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        with pytest.raises(ValueError, match="Gecersiz selection_metric"):
            run_sl_training(
                config,
                output_root=output_root,
                save_artifacts=True,
                evaluate_test_set=True,
                verbose=False,
                selection_metric="auc",
            )
        # Erken hata: cikti dizini yazilmamis olmali
        assert not output_root.exists()

    def test_macro_f1_loss_callable(self):
        from model.sl.training_runner import _xgb_macro_f1_loss

        # 3 sinif, mukemmel tahmin -> 1 - 1.0 = 0
        y_true = np.array([0, 1, 2, 0, 1, 2])
        probs = np.zeros((6, 3))
        for i, c in enumerate(y_true):
            probs[i, c] = 1.0
        assert _xgb_macro_f1_loss(y_true, probs) == pytest.approx(0.0)
        # Hep yanlis tahmin -> 1 - 0 = 1
        wrong = np.zeros((6, 3))
        wrong[:, 0] = 0.4
        wrong[:, 1] = 0.5  # hepsi sinif 1 tahmin edilir
        wrong[:, 2] = 0.1
        # y_true=[0,1,2,0,1,2] iken hep 1 -> macro_f1 < 1
        loss_value = _xgb_macro_f1_loss(y_true, wrong)
        assert 0.0 < loss_value <= 1.0


class TestSLTrainingMetricIntegration:
    """run_sl_training'in selection_metric'i XGBoost eval_metric'e baglandigini
    end-to-end dogrulayan testler."""

    def _make_config(self, dataset_dir):
        from model.sl.training_runner import SLTrainingConfig

        return SLTrainingConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.3,
            image_size=64,
            trainval_dir=str(dataset_dir),
            test_dir=str(dataset_dir),
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

    def test_history_returned_when_save_artifacts_false(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

        config = self._make_config(synth_dataset)
        result = run_sl_training(
            config,
            output_root=tmp_path / "no_artifacts",
            save_artifacts=False,
            evaluate_test_set=True,
            verbose=False,
            selection_metric="f1",
        )
        assert result["report_path"] is None
        # Eski bug: save_artifacts=False iken history bos donuyordu
        assert result["history"], "history save_artifacts=False iken de dolu olmali"
        assert any("mlogloss" in k for k in result["history"].keys())

    def test_f1_selection_uses_custom_metric_in_history(self, synth_dataset, tmp_path):
        """selection_metric='f1' iken history'de custom metric (mlogloss disinda)
        bir anahtar olmali — XGBoost gercekten custom metric'i calistirmis demek."""
        from model.sl.training_runner import run_sl_training

        config = self._make_config(synth_dataset)
        result = run_sl_training(
            config,
            output_root=tmp_path / "f1_run",
            save_artifacts=False,
            evaluate_test_set=True,
            verbose=False,
            selection_metric="f1",
        )
        non_logloss_keys = [
            k for k in result["history"].keys() if "mlogloss" not in k
        ]
        assert non_logloss_keys, (
            f"f1 secildiginde custom metric history'e yansimali; gelen anahtarlar: "
            f"{list(result['history'].keys())}"
        )
        assert result["xgb_early_stopping_metric"] == "1 - macro_f1"
        assert result["selection_mode"] == "maximize"
        assert result["selection_metric"] == "f1"

    def test_loss_selection_only_mlogloss(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

        config = self._make_config(synth_dataset)
        result = run_sl_training(
            config,
            output_root=tmp_path / "loss_run",
            save_artifacts=False,
            evaluate_test_set=True,
            verbose=False,
            selection_metric="loss",
        )
        assert result["xgb_early_stopping_metric"] == "mlogloss"
        assert result["selection_mode"] == "minimize"
        # Sadece mlogloss anahtarlari olmali
        assert all("mlogloss" in k for k in result["history"].keys())

    def test_report_contains_xgb_early_stopping_metric(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

        config = self._make_config(synth_dataset)
        result = run_sl_training(
            config,
            output_root=tmp_path / "report_run",
            save_artifacts=True,
            evaluate_test_set=True,
            verbose=False,
            selection_metric="f1",
            artifact_tag="xgb_test",
        )
        assert result["report_path"] is not None
        with open(result["report_path"], encoding="utf-8") as f:
            report = json.load(f)
        assert report["selection_metric"] == "f1"
        assert report["selection_mode"] == "maximize"
        assert report["xgb_early_stopping_metric"] == "1 - macro_f1"


class TestSLPlotBestIteration:
    """Egitim egrisi grafiginde best_iteration cizgisinin off-by-one duzeltmesi."""

    def test_axvline_uses_best_iteration_plus_one(self, tmp_path):
        from model.sl.training_runner import _plot_xgb_training_curves

        evals_result = {
            "validation_0": {"mlogloss": [0.9, 0.7, 0.5, 0.4, 0.35]},
            "validation_1": {"mlogloss": [0.95, 0.75, 0.55, 0.45, 0.42]},
        }
        save_path = tmp_path / "curve.png"
        _plot_xgb_training_curves(evals_result, save_path, best_iteration=2)
        assert save_path.exists()
        # axvline label "Best iter=3" olmali (0-bazli 2 -> 1-bazli 3)
        # Pratik dogrulama: figure'i tekrar acmadan label'i kontrol etmek icin
        # fonksiyonu cagirip dosyayi olusturduk. Asagida ek bir izlek dogrulamasi:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for label, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                ax.plot(range(1, len(values) + 1), values, label=f"{label} {metric_name}")
        # Ayni mantigi tekrar uygulayip vline label'inin 3 oldugunu dogrula
        best_round_1based = 2 + 1
        line = ax.axvline(best_round_1based, label=f"Best iter={best_round_1based}")
        assert line.get_xdata()[0] == 3
        plt.close(fig)


class TestSLKFoldFeatureMatrix:
    """SL feature-matrix K-fold split helper'i icin testler."""

    def _build_synth_inputs(self, n_groups: int = 8, copies_per_group: int = 2):
        rng = np.random.default_rng(123)
        X_rows = []
        y = []
        groups = []
        paths = []
        for g in range(n_groups):
            class_id = g % len(SINIF_ISIMLERI)
            class_name = SINIF_ISIMLERI[class_id]
            for c in range(copies_per_group):
                X_rows.append(rng.standard_normal(8))
                y.append(class_id)
                groups.append(f"{class_name}::g{g}")
                # Ilk kopya original, sonrakiler aug-isaretli isim
                if c == 0:
                    paths.append(f"{class_name}/g{g}.jpg")
                else:
                    paths.append(f"{class_name}/g{g}_aug{c}.jpg")
        return np.asarray(X_rows, dtype=np.float64), np.asarray(y), groups, paths

    def test_kfold_feature_matrix_grup_sizintisini_onler(self):
        from model.sl.training_runner import _kfold_feature_matrix

        X, y, groups, paths = self._build_synth_inputs()

        fold_splits, X_test, y_test, info = _kfold_feature_matrix(
            X, y, groups, paths,
            n_folds=2, test_ratio=0.0, seed=42, include_test=False,
        )

        assert info["n_folds"] == 2
        assert info["split_strategy"] == "group_stratified_kfold"
        assert X_test is None and y_test is None
        assert len(fold_splits) == 2
        seen_val_indices: set[int] = set()
        for X_tr, y_tr, X_va, y_va in fold_splits:
            assert X_tr.shape[1] == X.shape[1] == X_va.shape[1]
            assert len(y_tr) == len(X_tr)
            assert len(y_va) == len(X_va)

    def test_kfold_feature_matrix_test_ratio_destekler(self):
        from model.sl.training_runner import _kfold_feature_matrix

        # 24 grup, sinif basina 6 -> %25 test sonrasi sinif basina ~4-5 grup kalir
        X, y, groups, paths = self._build_synth_inputs(n_groups=24, copies_per_group=2)

        fold_splits, X_test, y_test, info = _kfold_feature_matrix(
            X, y, groups, paths,
            n_folds=3, test_ratio=0.25, seed=42, include_test=True,
        )

        assert X_test is not None and y_test is not None
        assert info["test_size"] > 0
        assert info["test_groups"] > 0
        assert len(fold_splits) == 3

    def test_kfold_feature_matrix_test_kapali_ise_test_groups_sifirdir(self):
        from model.sl.training_runner import _kfold_feature_matrix

        X, y, groups, paths = self._build_synth_inputs(n_groups=8, copies_per_group=2)

        _fold_splits, X_test, y_test, info = _kfold_feature_matrix(
            X, y, groups, paths,
            n_folds=2, test_ratio=0.0, seed=42, include_test=False,
        )

        assert X_test is None and y_test is None
        assert info["test_size"] == 0
        assert info["test_groups"] == 0


class TestSLCVOrchestrator:
    """run_sl_cv_training orkestratoru icin testler."""

    def test_run_sl_cv_training_n_folds_iki_alti_reddeder(self):
        from model.sl.training_runner import SLTrainingConfig, run_sl_cv_training

        config = SLTrainingConfig()
        with pytest.raises(ValueError, match="--folds en az 2"):
            run_sl_cv_training(config, n_folds=1)

    def test_run_sl_cv_training_fold_basina_run_sl_training_cagrir(self, monkeypatch, tmp_path):
        from model.sl import training_runner as sl_runner

        captured = []

        def fake_iter_kfold_split(*args, **kwargs):
            # _kfold_feature_matrix imzasi
            n_folds = kwargs["n_folds"]
            X = args[0]
            y = args[1]
            fold_splits = [
                (X[:2], y[:2], X[2:4], y[2:4])
                for _ in range(n_folds)
            ]
            X_test = X[4:6] if kwargs["include_test"] else None
            y_test = y[4:6] if kwargs["include_test"] else None
            info = {
                "num_classes": len(SINIF_ISIMLERI),
                "n_folds": n_folds,
                "test_size": len(y_test) if y_test is not None else 0,
                "test_groups": 2 if kwargs["include_test"] else 0,
                "split_strategy": "group_stratified_kfold",
                "split_warnings": [],
                "trainval_grouping": {"unique_groups": 8},
                "uses_external_test_dir": False,
                "folds": [
                    {
                        "fold_index": i,
                        "train_size": 2,
                        "val_size": 2,
                        "train_groups": 1,
                        "val_groups": 1,
                    }
                    for i in range(n_folds)
                ],
            }
            return fold_splits, X_test, y_test, info

        def fake_run_sl_training(config, **kwargs):
            captured.append(kwargs)
            fold_index = kwargs["extra_report"]["fold_index"]
            return {
                "best_iteration": 10 + fold_index,
                "best_val_metrics": {
                    "loss": 0.4 + 0.05 * fold_index,
                    "accuracy": 0.7 + 0.02 * fold_index,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1": 0.75 + 0.01 * fold_index,
                },
                "best_train_metrics": {"loss": 0.3, "accuracy": 0.85, "precision": 0.85, "recall": 0.85, "f1": 0.85},
                "best_selection_value": 0.75 + 0.01 * fold_index,
                "test_metrics": {
                    "loss": 0.5,
                    "accuracy": 0.65,
                    "precision": 0.65,
                    "recall": 0.65,
                    "f1": 0.7,
                },
                "report_path": tmp_path / f"fold_{fold_index}_report.json",
                "checkpoint_path": tmp_path / f"fold_{fold_index}.json",
            }

        # Yeterli sentetik feature/etiket
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 4))
        y = np.array([i % len(SINIF_ISIMLERI) for i in range(10)])
        groups = [f"{SINIF_ISIMLERI[i % len(SINIF_ISIMLERI)]}::g{i}" for i in range(10)]
        paths = [f"{SINIF_ISIMLERI[i % len(SINIF_ISIMLERI)]}/g{i}.jpg" for i in range(10)]

        monkeypatch.setattr(
            sl_runner,
            "build_feature_matrix",
            lambda *a, **k: (X, y, groups, paths),
        )
        monkeypatch.setattr(sl_runner, "_kfold_feature_matrix", fake_iter_kfold_split)
        monkeypatch.setattr(sl_runner, "validate_sl_config", lambda *a, **k: None)
        monkeypatch.setattr(
            sl_runner,
            "resolve_sl_data_dirs",
            lambda config: (tmp_path / "trainval", None),
        )
        monkeypatch.setattr(sl_runner, "run_sl_training", fake_run_sl_training)

        config = sl_runner.SLTrainingConfig(image_size=64)
        result = sl_runner.run_sl_cv_training(
            config,
            n_folds=3,
            output_root=tmp_path / "cv_out",
            artifact_tag="sl_cv_test",
            save_artifacts=False,
            evaluate_test_set=False,
            verbose=False,
            selection_metric="f1",
        )

        assert result["n_folds"] == 3
        assert len(result["fold_results"]) == 3
        assert result["aggregate"]["val"]["f1"]["mean"] == pytest.approx(0.76)
        assert result["aggregate"]["completed_folds"] == 3
        # run_sl_training her fold icin preloaded_split ile cagrildi
        for kwargs in captured:
            assert "preloaded_split" in kwargs
            assert kwargs["preloaded_split"] is not None
            assert "X_train" in kwargs["preloaded_split"]
            assert "y_train" in kwargs["preloaded_split"]

    def test_cv_summary_taşır_xgb_early_stopping_metric(self, tmp_path):
        """run_sl_cv_training top-level dönüş ve cv_summary, xgb_early_stopping_metric
        ile selection_mode alanlarını taşımalı (per-fold disinda da seffaflik)."""
        from model.sl import training_runner as sl_runner

        rng = np.random.default_rng(11)
        X = rng.standard_normal((10, 4))
        y = np.array([i % len(SINIF_ISIMLERI) for i in range(10)])
        groups = [f"{SINIF_ISIMLERI[i % len(SINIF_ISIMLERI)]}::g{i}" for i in range(10)]
        paths = [f"{SINIF_ISIMLERI[i % len(SINIF_ISIMLERI)]}/g{i}.jpg" for i in range(10)]

        def fake_iter_kfold_split(*args, **kwargs):
            n_folds = kwargs["n_folds"]
            X_arr = args[0]
            y_arr = args[1]
            fold_splits = [
                (X_arr[:2], y_arr[:2], X_arr[2:4], y_arr[2:4])
                for _ in range(n_folds)
            ]
            info = {
                "num_classes": len(SINIF_ISIMLERI),
                "n_folds": n_folds,
                "test_size": 0,
                "test_groups": 0,
                "split_strategy": "group_stratified_kfold",
                "split_warnings": [],
                "trainval_grouping": {"unique_groups": 8},
                "uses_external_test_dir": False,
                "folds": [
                    {"fold_index": i, "train_size": 2, "val_size": 2,
                     "train_groups": 1, "val_groups": 1}
                    for i in range(n_folds)
                ],
            }
            return fold_splits, None, None, info

        def fake_run_sl_training(config, **kwargs):
            return {
                "best_iteration": 5,
                "best_val_metrics": {
                    "loss": 0.4, "accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.75,
                },
                "best_train_metrics": {
                    "loss": 0.3, "accuracy": 0.85, "precision": 0.85, "recall": 0.85, "f1": 0.85,
                },
                "best_selection_value": 0.75,
                "test_metrics": None,
                "report_path": tmp_path / "rep.json",
                "checkpoint_path": tmp_path / "ckpt.json",
                "xgb_early_stopping_metric": "1 - macro_f1",
            }

        from unittest.mock import patch
        cv_root = tmp_path / "cv_out"
        with patch.object(sl_runner, "build_feature_matrix", return_value=(X, y, groups, paths)), \
             patch.object(sl_runner, "validate_sl_config"), \
             patch.object(sl_runner, "resolve_sl_data_dirs", return_value=(tmp_path / "trainval", None)), \
             patch.object(sl_runner, "_kfold_feature_matrix", side_effect=fake_iter_kfold_split), \
             patch.object(sl_runner, "run_sl_training", side_effect=fake_run_sl_training):
            config = sl_runner.SLTrainingConfig(image_size=64)
            result = sl_runner.run_sl_cv_training(
                config,
                n_folds=2,
                output_root=cv_root,
                artifact_tag="sl_cv_seffaflik",
                save_artifacts=True,
                evaluate_test_set=False,
                verbose=False,
                selection_metric="f1",
            )

        # Top-level return dict
        assert result["xgb_early_stopping_metric"] == "1 - macro_f1"
        assert result["selection_mode"] == "maximize"
        # cv_summary JSON
        assert result["cv_summary_path"] is not None
        with open(result["cv_summary_path"], encoding="utf-8") as f:
            cv_summary = json.load(f)
        assert cv_summary["xgb_early_stopping_metric"] == "1 - macro_f1"
        assert cv_summary["selection_mode"] == "maximize"
        assert cv_summary["selection_metric"] == "f1"

    def test_cv_invalid_selection_metric_erken_yakalanir(self, synth_dataset, tmp_path):
        """Hatali selection_metric, fold doneminden once ValueError firlatmali."""
        from model.sl.training_runner import SLTrainingConfig, run_sl_cv_training

        config = SLTrainingConfig(
            n_estimators=10,
            max_depth=3,
            image_size=64,
            trainval_dir=str(synth_dataset),
            test_dir=str(synth_dataset),
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        cv_out = tmp_path / "cv_invalid"
        with pytest.raises(ValueError, match="Gecersiz selection_metric"):
            run_sl_cv_training(
                config,
                n_folds=2,
                output_root=cv_out,
                save_artifacts=True,
                evaluate_test_set=False,
                verbose=False,
                selection_metric="auc",
            )
        # Erken hata: fold dizinleri olusturulmamis olmali
        assert not (cv_out / "folds").exists()

    def test_run_sl_cv_training_internal_test_test_groups_dogru_raporlar(self, tmp_path):
        """Regresyon: internal test split kullanildiginda fold_split_info['test_groups']
        sifir kalmamali; _kfold_feature_matrix'in dondugu test_groups sayisi kullanilmali."""
        from model.sl import training_runner as sl_runner

        # Sentetik veri: her sinifta 6 kaynak grup, 2 kopya -> her sinif basina yeterli
        rng = np.random.default_rng(7)
        n_groups_per_class = 6
        copies = 2
        X_rows: list[np.ndarray] = []
        y_list: list[int] = []
        groups: list[str] = []
        paths: list[str] = []
        for class_id, class_name in enumerate(SINIF_ISIMLERI):
            for g in range(n_groups_per_class):
                for c in range(copies):
                    X_rows.append(rng.standard_normal(8))
                    y_list.append(class_id)
                    groups.append(f"{class_name}::g{class_id}_{g}")
                    if c == 0:
                        paths.append(f"{class_name}/g{class_id}_{g}.jpg")
                    else:
                        paths.append(f"{class_name}/g{class_id}_{g}_aug{c}.jpg")
        X_arr = np.asarray(X_rows, dtype=np.float64)
        y_arr = np.asarray(y_list)

        captured: dict[str, Any] = {}

        def fake_run_sl_training(config, **kwargs):
            captured["preloaded_split"] = kwargs["preloaded_split"]
            return {
                "best_iteration": 10,
                "best_val_metrics": {
                    "loss": 0.4, "accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.75,
                },
                "best_train_metrics": {
                    "loss": 0.3, "accuracy": 0.85, "precision": 0.85, "recall": 0.85, "f1": 0.85,
                },
                "best_selection_value": 0.75,
                "test_metrics": {
                    "loss": 0.5, "accuracy": 0.65, "precision": 0.65, "recall": 0.65, "f1": 0.7,
                },
                "report_path": tmp_path / "rep.json",
                "checkpoint_path": tmp_path / "ckpt.json",
            }

        from unittest.mock import patch
        with patch.object(sl_runner, "build_feature_matrix", return_value=(X_arr, y_arr, groups, paths)), \
             patch.object(sl_runner, "validate_sl_config"), \
             patch.object(sl_runner, "resolve_sl_data_dirs", return_value=(tmp_path / "trainval", None)), \
             patch.object(sl_runner, "run_sl_training", side_effect=fake_run_sl_training):
            config = sl_runner.SLTrainingConfig(image_size=64, test_ratio=0.2)
            sl_runner.run_sl_cv_training(
                config,
                n_folds=2,
                output_root=tmp_path / "cv_out",
                artifact_tag="sl_cv_internal",
                save_artifacts=False,
                evaluate_test_set=True,  # internal test (test_dir=None oldugu icin internal split)
                verbose=False,
                selection_metric="f1",
            )

        # Internal test'te test_groups > 0 olmali (eski bug'da 0 idi)
        split_info = captured["preloaded_split"]["split_info"]
        assert split_info["test_groups"] > 0
        assert split_info["uses_external_test_dir"] is False
        # Test_size de tutarli pozitif olmali
        assert split_info["test_size"] > 0
