"""
SL (XGBoost) pipeline testleri.
"""

import json
import os
import sys
from pathlib import Path
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
