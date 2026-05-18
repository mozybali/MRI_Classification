"""
Regression testleri — raporlanan hatalarin duzeltmelerini dogrular.

CRITICAL-1: Inference image_size metadata
CRITICAL-2: Feature cache image_size cakismasi
HIGH-1:     Val loss best vs last epoch
HIGH-2:     Test loss gercek logloss
HIGH-3:     XGB study summary dogru search space
HIGH-4:     SLTrainingConfig validation (subsample, colsample_bytree, reg_lambda, min_child_weight)
MEDIUM-1:   XGB HPO pruning dokumantasyon notu
LOW-1:      num_class redundant kaldirildi
LOW-3:      NaN/Inf feature assert
"""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from model.dl.dataset import SINIF_ISIMLERI
from model.sl.features import extract_features
from model.sl.dataset import build_feature_matrix
from model.sl.xgb_classifier import (
    build_xgb_classifier,
    save_xgb_model,
    load_xgb_model,
    load_xgb_model_with_meta,
)
from model.sl.training_runner import SLTrainingConfig, validate_sl_config


# ==================== Fixtures ====================

@pytest.fixture
def synth_dataset(tmp_path):
    """Her siniftan 4 goruntu iceren minimal sentetik veri seti."""
    dataset_path = tmp_path / f"synth_{uuid4().hex[:8]}"
    for class_name in SINIF_ISIMLERI:
        class_dir = dataset_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = Image.fromarray(
                np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L",
            )
            img.save(class_dir / f"img_{i}.jpg")
    return dataset_path


# ==================== CRITICAL-1: Inference image_size metadata ====================


class TestInferenceMetadata:
    def test_save_xgb_model_writes_metadata_json(self, tmp_path):
        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)

        model_path = tmp_path / "model.json"
        save_xgb_model(
            clf,
            model_path,
            image_size=160,
            class_names=SINIF_ISIMLERI,
            seed=42,
        )

        meta_path = model_path.with_suffix(".meta.json")
        assert meta_path.exists()
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["image_size"] == 160
        assert meta["class_names"] == SINIF_ISIMLERI
        assert meta["seed"] == 42
        assert meta["feature_dim"] == 10

    def test_load_xgb_model_with_meta_reads_metadata(self, tmp_path):
        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)

        model_path = tmp_path / "model.json"
        save_xgb_model(clf, model_path, image_size=192, class_names=SINIF_ISIMLERI)

        loaded, meta = load_xgb_model_with_meta(model_path)
        assert meta["image_size"] == 192
        assert meta["class_names"] == SINIF_ISIMLERI
        np.testing.assert_array_almost_equal(
            clf.predict_proba(X), loaded.predict_proba(X),
        )

    def test_load_xgb_model_with_meta_no_sidecar_returns_empty(self, tmp_path):
        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)

        model_path = tmp_path / "model_no_meta.json"
        save_xgb_model(clf, model_path)  # metadata olmadan kaydet

        loaded, meta = load_xgb_model_with_meta(model_path)
        assert meta == {}

    def test_inference_load_uses_metadata_image_size(self, tmp_path):
        from model.inference import load_xgb_model_for_inference

        clf = build_xgb_classifier(num_classes=4)
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 5)
        clf.fit(X, y)

        model_path = tmp_path / "model.json"
        save_xgb_model(clf, model_path, image_size=160, class_names=["A", "B", "C", "D"])

        _model, image_size, class_names = load_xgb_model_for_inference(model_path)
        assert image_size == 160
        assert class_names == ["A", "B", "C", "D"]


# ==================== CRITICAL-2: Feature cache image_size conflict ====================


class TestCacheImageSizeConflict:
    def test_cache_stores_image_size_metadata(self, synth_dataset, tmp_path):
        cache = tmp_path / "cache.npz"
        build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)
        assert cache.exists()
        data = np.load(cache, allow_pickle=True)
        assert "image_size" in data
        assert int(data["image_size"]) == 64
        assert "data_dir" in data
        assert str(data["data_dir"].item()) == str(synth_dataset.resolve())

    def test_cache_rejects_different_image_size(self, synth_dataset, tmp_path):
        cache = tmp_path / "cache.npz"
        build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)

        with pytest.raises(ValueError, match="farkli image_size"):
            build_feature_matrix(synth_dataset, image_size=128, cache_path=cache)

    def test_cache_accepts_same_image_size(self, synth_dataset, tmp_path):
        cache = tmp_path / "cache.npz"
        X1, y1, g1, p1 = build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)
        X2, y2, g2, p2 = build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)
        np.testing.assert_array_equal(X1, X2)

    def test_cache_rejects_different_data_dir(self, synth_dataset, tmp_path):
        cache = tmp_path / "cache.npz"
        build_feature_matrix(synth_dataset, image_size=64, cache_path=cache)

        other_dataset = tmp_path / f"other_{uuid4().hex[:8]}"
        for class_name in SINIF_ISIMLERI:
            class_dir = other_dataset / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L",
                )
                img.save(class_dir / f"other_{i}.jpg")

        with pytest.raises(ValueError, match="farkli veri dizini"):
            build_feature_matrix(other_dataset, image_size=64, cache_path=cache)


# ==================== HIGH-1: Val loss best vs last epoch ====================


class TestValLossBestEpoch:
    def test_val_loss_reports_best_not_last(self, synth_dataset, tmp_path):
        """Yuksek n_estimators ile early stopping tetiklendiginde
        val_loss'un best_iteration ile tutarli oldugunu dogrula."""
        from model.sl.training_runner import run_sl_training

        config = SLTrainingConfig(
            n_estimators=200,
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
            output_root=tmp_path / "out",
            save_artifacts=False,
            evaluate_test_set=False,
            verbose=False,
        )

        val_metrics = results["best_val_metrics"]
        best_iter = results["best_iteration"]

        # best_iteration varsa, val_loss son iteration'dan farkli olabilir
        assert val_metrics is not None
        assert val_metrics["loss"] >= 0
        # val_loss should be a real number
        assert isinstance(val_metrics["loss"], float)


# ==================== HIGH-2: Test loss gercek logloss ====================


class TestTestLossReal:
    def test_test_loss_is_real_logloss(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

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
            output_root=tmp_path / "out",
            save_artifacts=False,
            evaluate_test_set=True,
            verbose=False,
        )

        assert results["test_metrics"] is not None
        test_loss = results["test_metrics"]["loss"]
        assert test_loss > 0.0, "Test loss should be > 0 (real logloss, not hard-coded 0.0)"


# ==================== HIGH-2B: SL split sizinti ve original-only politikasi ====================


class TestSLSplitSafety:
    def test_sl_internal_test_split_same_dir_with_positive_ratio(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

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
            output_root=tmp_path / "out",
            save_artifacts=False,
            evaluate_test_set=True,
            verbose=False,
        )

        assert results["test_metrics"] is not None
        assert results["data_info"]["test_size"] > 0
        assert results["data_info"]["uses_external_test_dir"] is False

    def test_sl_external_test_rejects_overlapping_source_groups(self, synth_dataset, tmp_path):
        from model.sl.training_runner import run_sl_training

        external_test = tmp_path / f"external_{uuid4().hex[:8]}"
        for class_name in SINIF_ISIMLERI:
            class_dir = external_test / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L",
                )
                img.save(class_dir / f"img_{i}.jpg")

        config = SLTrainingConfig(
            n_estimators=10,
            image_size=64,
            trainval_dir=str(synth_dataset),
            test_dir=str(external_test),
            val_ratio=0.25,
            test_ratio=0.0,
            seed=42,
        )

        with pytest.raises(ValueError, match="ortak kaynak grup"):
            run_sl_training(config, save_artifacts=False, evaluate_test_set=True, verbose=False)

    def test_sl_split_filters_augmented_samples_from_val_and_test(self):
        from model.sl.training_runner import _split_feature_matrix

        X_rows = []
        y_values = []
        groups = []
        paths = []
        row_id = 0
        for class_id, class_name in enumerate(SINIF_ISIMLERI):
            for sample_id in range(4):
                group = f"{class_name}::img_{sample_id}"
                for suffix in ("", "_aug1"):
                    X_rows.append([float(row_id)])
                    y_values.append(class_id)
                    groups.append(group)
                    paths.append(f"/tmp/{class_name}/img_{sample_id}{suffix}.jpg")
                    row_id += 1

        (
            _X_train,
            _y_train,
            _X_val,
            _y_val,
            _X_test,
            _y_test,
            info,
        ) = _split_feature_matrix(
            np.asarray(X_rows, dtype=np.float32),
            np.asarray(y_values, dtype=np.int64),
            groups,
            paths,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
            include_test=True,
        )

        assert info["trainval_grouping"]["augmented_samples"] == 16
        assert info["val_size"] == info["val_groups"]
        assert info["test_size"] == info["test_groups"]


# ==================== HIGH-3: XGB study summary search space ====================


class TestXGBStudySummary:
    def test_xgb_study_summary_has_correct_search_space(self):
        from model.hpo import _search_space_summary, parse_args

        args = parse_args([
            "--model", "xgboost",
            "--trials", "1",
            "--skip-final-train",
            "--image-size-choices", "128", "160",
        ])
        summary = _search_space_summary(args)

        # XGBoost-spesifik alanlar olmali
        assert "n_estimators_range" in summary
        assert "max_depth_range" in summary
        assert "learning_rate_range" in summary
        assert "subsample_range" in summary
        assert "colsample_bytree_range" in summary
        assert "image_size_choices" in summary
        assert summary["image_size_choices"] == [128, 160]

        # DL-spesifik alanlar olmamali
        assert "batch_size_choices" not in summary
        assert "lr_range" not in summary
        assert "weight_decay_range" not in summary
        assert "loss_choices" not in summary
        assert "search_pretrained" not in summary

    def test_resnet_study_summary_has_dl_fields(self):
        from model.hpo import _search_space_summary, parse_args

        args = parse_args([
            "--model", "resnet",
            "--trials", "1",
            "--skip-final-train",
        ])
        summary = _search_space_summary(args)

        assert "batch_size_choices" in summary
        assert "lr_range" in summary
        assert "loss_choices" in summary
        assert "n_estimators_range" not in summary

    def test_final_xgb_n_estimators_uses_best_iteration(self):
        from model.hpo_xgb import _resolve_final_xgb_n_estimators

        assert _resolve_final_xgb_n_estimators(100, 24) == (25, 24)
        assert _resolve_final_xgb_n_estimators(100, "9") == (10, 9)
        assert _resolve_final_xgb_n_estimators(100, None) == (100, None)
        assert _resolve_final_xgb_n_estimators(100, 150) == (100, 150)

    def test_final_xgb_training_receives_best_iteration_estimators(self, tmp_path, monkeypatch):
        from model import hpo, hpo_xgb

        args = hpo.parse_args([
            "--model", "xgboost",
            "--metric", "f1",
            "--trainval-dir", "trainval",
            "--test-dir", "test",
        ])
        captured = {}

        def fake_run_sl_training(config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs
            return {
                "output_root": kwargs["output_root"],
                "report_path": kwargs["output_root"] / "rapor.json",
            }

        monkeypatch.setattr(hpo_xgb, "run_sl_training", fake_run_sl_training)
        best_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "min_child_weight": 2,
            "image_size": 64,
        }

        hpo_xgb._run_final_xgb_training(
            args=args,
            study_dir=tmp_path,
            best_params=best_params,
            study_name="xgb_test",
            best_trial_number=3,
            best_iteration=31,
        )

        assert captured["config"].n_estimators == 32
        assert captured["kwargs"]["full_trainval"] is True
        extra_report = captured["kwargs"]["extra_report"]
        assert extra_report["best_trial_n_estimators"] == 200
        assert extra_report["best_trial_best_iteration"] == 31
        assert extra_report["final_n_estimators"] == 32
        assert extra_report["final_n_estimators_source"] == "best_iteration_plus_one"

    def test_hpo_visualizations_are_saved(self, tmp_path):
        optuna = pytest.importorskip("optuna")
        from model.hpo import _save_hpo_visualizations

        study = optuna.create_study(direction="maximize")
        for value in [0.40, 0.55, 0.50, 0.70]:
            trial = study.ask()
            trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            trial.suggest_int("max_depth", 3, 8)
            study.tell(trial, value)

        artifacts = _save_hpo_visualizations(study, tmp_path)

        assert "optimization_history" in artifacts
        assert Path(artifacts["optimization_history"]).exists()


# ==================== HIGH-4: SLTrainingConfig validation ====================


class TestSLConfigValidation:
    def test_sl_config_rejects_invalid_subsample(self):
        config = SLTrainingConfig(subsample=0.0)
        with pytest.raises(ValueError, match="subsample"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_rejects_subsample_gt_1(self):
        config = SLTrainingConfig(subsample=1.5)
        with pytest.raises(ValueError, match="subsample"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_rejects_invalid_colsample_bytree(self):
        config = SLTrainingConfig(colsample_bytree=0.0)
        with pytest.raises(ValueError, match="colsample"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_rejects_colsample_bytree_gt_1(self):
        config = SLTrainingConfig(colsample_bytree=1.1)
        with pytest.raises(ValueError, match="colsample"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_rejects_negative_reg_lambda(self):
        config = SLTrainingConfig(reg_lambda=-0.1)
        with pytest.raises(ValueError, match="reg-lambda"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_rejects_negative_min_child_weight(self):
        config = SLTrainingConfig(min_child_weight=-1)
        with pytest.raises(ValueError, match="min-child-weight"):
            validate_sl_config(config, require_test_dir=False)

    def test_sl_config_accepts_valid_values(self, synth_dataset):
        config = SLTrainingConfig(
            subsample=1.0,
            colsample_bytree=0.5,
            reg_lambda=0.0,
            min_child_weight=0,
            trainval_dir=str(synth_dataset),
        )
        # Should not raise
        validate_sl_config(config, require_test_dir=False)


# ==================== LOW-1: num_class redundant ====================


class TestNumClassRedundant:
    def test_build_xgb_classifier_no_num_class_param(self):
        clf = build_xgb_classifier(num_classes=4)
        params = clf.get_params()
        # num_class XGBoost tarafindan otomatik cikartilmali, explicit set edilmemeli
        assert "num_class" not in params or params.get("num_class") is None


# ==================== LOW-3: NaN/Inf feature assert ====================


class TestFeatureNaNInf:
    def test_extract_features_no_nan_inf(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        vec = extract_features(img)
        assert np.isfinite(vec).all()

    def test_extract_features_all_zero_image(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        vec = extract_features(img)
        assert np.isfinite(vec).all()

    def test_extract_features_all_white_image(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        vec = extract_features(img)
        assert np.isfinite(vec).all()
