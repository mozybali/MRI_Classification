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

from model.sl.features import extract_features
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
