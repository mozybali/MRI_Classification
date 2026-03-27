"""
Derin ogrenme model katmani icin temel testler.
Not: Dosya adi geriye donuk uyumluluk icin korunmustur.
"""

import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

if os.environ.get("MRI_RUN_TORCH_TESTS") != "1":
    pytest.skip(
        "Torch bagimli testler varsayilan olarak atlanir. Calistirmak icin "
        "MRI_RUN_TORCH_TESTS=1 ayarlayin.",
        allow_module_level=True,
    )

import torch
import torch.nn.functional as F

from model.dl.dataset import kaynak_id_belirle, _group_stratified_train_val_split
from model.dl.losses import FocalLoss, compute_class_weights
from model.dl.models.resnet_classifier import ResNetClassifier
from model.dl.utils import (
    load_checkpoint,
    plot_classification_summary,
    plot_confusion_matrix,
    plot_multiclass_roc_pr_curves,
    plot_prediction_confidence,
    plot_training_curves,
)
from model import hpo, training_runner
from model.train import build_model, parse_args as parse_train_args


def test_kaynak_id_belirle_parantez_ve_aug_duzgun_gruplar():
    assert kaynak_id_belirle("26 (19).jpg") == "26"
    assert kaynak_id_belirle("26_aug2.png") == "26"
    assert kaynak_id_belirle("26 (19)_aug2.png") == "26"


def test_group_split_kaynak_sizintisini_onler():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    groups = [
        "A::g1", "A::g1",  # ayni grup
        "B::g2", "B::g2",  # ayni grup
        "C::g3", "C::g3",  # ayni grup
        "D::g4", "D::g4",  # ayni grup
    ]

    train_idxs, val_idxs = _group_stratified_train_val_split(
        labels=labels,
        groups=groups,
        val_ratio=0.25,
        seed=42,
        num_classes=4,
    )

    train_set = set(train_idxs)
    val_set = set(val_idxs)

    assert train_set.isdisjoint(val_set)
    assert len(train_set) + len(val_set) == len(labels)


def test_group_split_her_sinifi_iki_splitte_de_temsil_eder():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    groups = [
        "A::g1", "A::g2",
        "B::g1", "B::g2",
        "C::g1", "C::g2",
        "D::g1", "D::g2",
    ]

    train_idxs, val_idxs = _group_stratified_train_val_split(
        labels=labels,
        groups=groups,
        val_ratio=0.5,
        seed=42,
        num_classes=4,
    )

    train_counts = np.bincount([labels[idx] for idx in train_idxs], minlength=4)
    val_counts = np.bincount([labels[idx] for idx in val_idxs], minlength=4)

    assert np.all(train_counts > 0)
    assert np.all(val_counts > 0)


def test_group_split_tek_kaynakli_sinifta_best_effort_bolme_yapar():
    labels = [0, 0, 1, 1]
    groups = [
        "A::g1", "A::g1",
        "B::g1", "B::g2",
    ]

    train_idxs, val_idxs = _group_stratified_train_val_split(
        labels=labels,
        groups=groups,
        val_ratio=0.5,
        seed=42,
        num_classes=2,
    )

    assert set(train_idxs).isdisjoint(val_idxs)
    assert len(train_idxs) + len(val_idxs) == len(labels)


def test_group_split_strict_modda_eksik_sinif_kapsamini_reddeder():
    labels = [0, 0, 1, 1]
    groups = [
        "A::g1", "A::g1",
        "B::g1", "B::g2",
    ]

    with pytest.raises(RuntimeError, match="sinif kapsami eksik kaldi"):
        _group_stratified_train_val_split(
            labels=labels,
            groups=groups,
            val_ratio=0.5,
            seed=42,
            num_classes=2,
            require_all_classes_in_each_split=True,
        )


def test_compute_class_weights_pozitif_deger_uretir():
    weights = compute_class_weights([0, 0, 1, 2, 2, 3], num_classes=4)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == 4
    assert torch.all(weights > 0).item()


def test_focal_loss_gamma_sifirken_weighted_ce_ile_eslesir():
    inputs = torch.tensor([[2.0, 0.5], [0.1, 1.3]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    alpha = torch.tensor([1.0, 3.0], dtype=torch.float32)

    focal = FocalLoss(alpha=alpha, gamma=0.0, reduction="mean")
    focal_loss = focal(inputs, targets)
    ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction="mean")

    assert torch.allclose(focal_loss, ce_loss)


def test_confusion_matrix_eksik_sinifta_da_cizer(tmp_path):
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    class_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    out_path = tmp_path / "cm.png"

    plot_confusion_matrix(labels, preds, class_names, out_path)

    assert out_path.exists()


def test_detayli_degerlendirme_grafikleri_olusturulur(tmp_path):
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    preds = np.array([0, 1, 2, 2, 0, 0, 2, 3])
    probs = np.array(
        [
            [0.92, 0.03, 0.03, 0.02],
            [0.10, 0.80, 0.07, 0.03],
            [0.04, 0.06, 0.84, 0.06],
            [0.08, 0.12, 0.60, 0.20],
            [0.88, 0.05, 0.04, 0.03],
            [0.52, 0.28, 0.10, 0.10],
            [0.05, 0.08, 0.79, 0.08],
            [0.06, 0.08, 0.12, 0.74],
        ],
        dtype=np.float32,
    )
    confidences = probs.max(axis=1)
    class_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

    plot_confusion_matrix(labels, preds, class_names, tmp_path / "cm_norm.png", normalize=True)
    plot_classification_summary(labels, preds, class_names, tmp_path / "class_summary.png")
    plot_prediction_confidence(confidences, labels, preds, tmp_path / "confidence.png")
    plot_multiclass_roc_pr_curves(labels, probs, class_names, tmp_path / "roc_pr.png")
    plot_training_curves(
        [1.0, 0.7, 0.4],
        [1.1, 0.8, 0.5],
        [0.5, 0.7, 0.9],
        [0.4, 0.65, 0.82],
        tmp_path / "training_dashboard.png",
        train_precisions=[0.45, 0.68, 0.91],
        val_precisions=[0.38, 0.62, 0.84],
        train_recalls=[0.44, 0.69, 0.90],
        val_recalls=[0.40, 0.63, 0.81],
        train_f1s=[0.44, 0.68, 0.90],
        val_f1s=[0.39, 0.62, 0.82],
        best_epoch=3,
    )

    assert (tmp_path / "cm_norm.png").exists()
    assert (tmp_path / "class_summary.png").exists()
    assert (tmp_path / "confidence.png").exists()
    assert (tmp_path / "roc_pr.png").exists()
    assert (tmp_path / "training_dashboard.png").exists()


def test_prediction_confidence_bos_veride_placeholder_cizer(tmp_path):
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    out_path = tmp_path / "confidence_placeholder.png"

    plot_prediction_confidence(np.array([], dtype=np.float32), labels, preds, out_path)

    assert out_path.exists()


def test_detayli_rapor_auc_ap_ortalamasinda_eksik_siniflari_atlar():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    probs = np.array(
        [
            [0.95, 0.05, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.05, 0.90, 0.05, 0.00],
            [0.10, 0.85, 0.05, 0.00],
        ],
        dtype=np.float32,
    )

    report = training_runner._build_detailed_eval_report(
        labels,
        preds,
        probs,
        ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"],
    )

    assert report["macro_auc_ovr"] == pytest.approx(1.0)
    assert report["macro_average_precision"] == pytest.approx(1.0)
    assert report["per_class"]["MildDemented"]["support"] == 0
    assert report["per_class"]["ModerateDemented"]["support"] == 0


def test_build_model_resnet_cpu_olusturur():
    device = torch.device("cpu")
    model = build_model("resnet", num_classes=4, device=device, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)


def test_resnet_pretrained_yuklenemezse_acik_hata_verir(monkeypatch):
    def fake_resnet18(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("model.dl.models.resnet_classifier.models.resnet18", fake_resnet18)

    try:
        ResNetClassifier(num_classes=4, pretrained=True)
    except RuntimeError as exc:
        assert "pretrained agirliklar istendi ama yuklenemedi" in str(exc)
    else:
        raise AssertionError("Beklenen RuntimeError olusmadi.")


def test_load_checkpoint_guvenli_modu_kullanir(monkeypatch, tmp_path):
    calls = {}

    def fake_load(path, map_location=None, weights_only=None):
        calls["path"] = path
        calls["map_location"] = map_location
        calls["weights_only"] = weights_only
        return {"model_state_dict": {"weight": torch.tensor([1.0])}}

    monkeypatch.setattr("model.dl.utils.torch.load", fake_load)

    checkpoint = load_checkpoint(tmp_path / "dummy.pt", map_location="cpu")

    assert checkpoint["model_state_dict"]["weight"].item() == 1.0
    assert calls["weights_only"] is True


def test_train_epochs_sifir_icin_anlamli_hata_verir():
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(project_root / "model" / "train.py"),
        "--epochs", "0",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "--epochs en az 1 olmali" in result.stdout


def test_inference_image_ve_batch_ayni_anda_verilemez(tmp_path):
    project_root = Path(__file__).resolve().parent.parent
    dummy_model = tmp_path / "dummy.pt"
    dummy_image = tmp_path / "img.jpg"
    dummy_batch = tmp_path / "batch"

    dummy_model.write_bytes(b"not-a-real-checkpoint")
    dummy_image.write_bytes(b"not-a-real-image")
    dummy_batch.mkdir()

    cmd = [
        sys.executable,
        str(project_root / "model" / "inference.py"),
        "--model-path", str(dummy_model),
        "--image", str(dummy_image),
        "--batch", str(dummy_batch),
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_train_parse_args_ek_hiperparametreleri_cozer():
    args = parse_train_args(
        [
            "--loss",
            "focal",
            "--focal-gamma",
            "2.7",
            "--weight-decay",
            "0.002",
            "--scheduler-factor",
            "0.4",
            "--scheduler-patience",
            "3",
        ]
    )

    assert args.loss == "focal"
    assert args.focal_gamma == pytest.approx(2.7)
    assert args.weight_decay == pytest.approx(0.002)
    assert args.scheduler_factor == pytest.approx(0.4)
    assert args.scheduler_patience == 3


def test_train_parse_args_islenmis_goruntu_flaglerini_cozer():
    args = parse_train_args(
        [
            "--use-processed-trainval",
            "--use-processed-test",
        ]
    )

    assert args.use_processed_trainval is True
    assert args.use_processed_test is True


def test_train_parse_args_full_trainval_flagini_cozer():
    args = parse_train_args(["--full-trainval"])

    assert args.full_trainval is True


def test_resolve_data_dirs_islenmis_trainval_kokunu_otomatik_kullanir(monkeypatch):
    processed_root = Path("tmp_test_artifacts") / f"processed_trainval_{uuid4().hex}"
    for class_name in training_runner.SINIF_ISIMLERI:
        (processed_root / class_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training_runner, "ISLENMIS_VERI_KLASORU", processed_root)
    monkeypatch.setattr(training_runner, "ISLENMIS_TRAINVAL_VERI_DIZINI", processed_root / "trainval")

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(
            use_processed_trainval=True,
            test_dir="custom/test",
        )
    )

    assert trainval_dir == processed_root
    assert test_dir == Path("custom/test")


def test_resolve_data_dirs_islenmis_test_varsayilanini_kullanir(monkeypatch):
    processed_test = Path("tmp_test_artifacts") / f"processed_test_{uuid4().hex}"
    processed_test.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(training_runner, "ISLENMIS_TEST_VERI_DIZINI", processed_test)

    _trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(use_processed_test=True)
    )

    assert test_dir == processed_test


def test_resolve_data_dirs_islenmis_trainval_secildiginde_islenmis_testi_de_kullanir(monkeypatch):
    processed_trainval = Path("tmp_test_artifacts") / f"processed_trainval_split_{uuid4().hex}"
    processed_test = Path("tmp_test_artifacts") / f"processed_test_split_{uuid4().hex}"
    for root in (processed_trainval, processed_test):
        for class_name in training_runner.SINIF_ISIMLERI:
            (root / class_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training_runner, "ISLENMIS_TRAINVAL_VERI_DIZINI", processed_trainval)
    monkeypatch.setattr(training_runner, "ISLENMIS_TEST_VERI_DIZINI", processed_test)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(use_processed_trainval=True)
    )

    assert trainval_dir == processed_trainval
    assert test_dir == processed_test


def test_resolve_data_dirs_trainval_root_verilince_split_alt_dizinlerini_cozer():
    processed_root = Path("tmp_test_artifacts") / f"processed_root_{uuid4().hex}"
    for split_name in ("trainval", "test"):
        for class_name in training_runner.SINIF_ISIMLERI:
            (processed_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(
            trainval_dir=processed_root,
            use_processed_trainval=True,
        )
    )

    assert trainval_dir == processed_root / "trainval"
    assert test_dir == processed_root / "test"


def test_resolve_data_dirs_test_root_verilince_test_alt_dizinini_cozer():
    processed_root = Path("tmp_test_artifacts") / f"processed_test_root_{uuid4().hex}"
    for split_name in ("trainval", "test"):
        for class_name in training_runner.SINIF_ISIMLERI:
            (processed_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(
            trainval_dir=processed_root / "trainval",
            test_dir=processed_root,
        )
    )

    assert trainval_dir == processed_root / "trainval"
    assert test_dir == processed_root / "test"


def test_resolve_data_dirs_varsayilanda_islenmis_splitleri_kullanir(monkeypatch):
    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", Path("goruntu_isleme/cikti/trainval"))
    monkeypatch.setattr(training_runner, "VARSAYILAN_VERI_DIZINI", Path("goruntu_isleme/cikti/trainval"))
    monkeypatch.setattr(training_runner, "ISLENMIS_TRAINVAL_VERI_DIZINI", Path("goruntu_isleme/cikti/trainval"))
    monkeypatch.setattr(training_runner, "ISLENMIS_TEST_VERI_DIZINI", Path("goruntu_isleme/cikti/test"))
    monkeypatch.setattr(training_runner, "TEST_VERI_DIZINI", Path("goruntu_isleme/cikti/test"))

    trainval_dir, test_dir = training_runner.resolve_data_dirs(training_runner.TrainingConfig())

    assert trainval_dir == Path("goruntu_isleme/cikti/trainval")
    assert test_dir == Path("goruntu_isleme/cikti/test")


def test_resolve_data_dirs_varsayilan_veri_dizinini_onceliklendirir(monkeypatch):
    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", Path("fallback/trainval"))
    monkeypatch.setattr(training_runner, "VARSAYILAN_VERI_DIZINI", Path("."))
    monkeypatch.setattr(training_runner, "TEST_VERI_DIZINI", Path("fallback/trainval"))

    trainval_dir, _test_dir = training_runner.resolve_data_dirs(training_runner.TrainingConfig())

    assert trainval_dir == Path(".")


def test_resolve_data_dirs_harici_test_ayarini_kullanir(monkeypatch):
    external_test = Path("tmp_test_artifacts") / f"external_test_{uuid4().hex}"
    for class_name in training_runner.SINIF_ISIMLERI:
        (external_test / class_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", Path("Veri_Seti/OriginalDataset"))
    monkeypatch.setattr(training_runner, "VARSAYILAN_VERI_DIZINI", Path("Veri_Seti/OriginalDataset"))
    monkeypatch.setattr(training_runner, "TEST_VERI_DIZINI", external_test)

    _trainval_dir, test_dir = training_runner.resolve_data_dirs(training_runner.TrainingConfig())

    assert test_dir == external_test


def test_build_output_dirs_varsayilan_alt_klasor_ayarlarini_kullanir(monkeypatch):
    root = Path("tmp_test_artifacts") / f"output_root_{uuid4().hex}"
    models = root / "weights"
    reports = root / "json_reports"
    visuals = root / "plots"

    monkeypatch.setattr(training_runner, "CIKTI_KLASORU", root)
    monkeypatch.setattr(training_runner, "MODELS_KLASORU", models)
    monkeypatch.setattr(training_runner, "RAPORLAR_KLASORU", reports)
    monkeypatch.setattr(training_runner, "GORSELLER_KLASORU", visuals)

    output_dirs = training_runner._build_output_dirs(root)

    assert output_dirs["models"] == models
    assert output_dirs["reports"] == reports
    assert output_dirs["visuals"] == visuals


def test_hpo_parse_args_bayes_search_parametrelerini_cozer():
    args = hpo.parse_args(
        [
            "--model",
            "unet",
            "--trials",
            "9",
            "--metric",
            "loss",
            "--batch-size-choices",
            "8",
            "16",
            "--image-size-choices",
            "128",
            "160",
            "--skip-final-train",
        ]
    )

    assert args.model == "unet"
    assert args.trials == 9
    assert args.metric == "loss"
    assert args.batch_size_choices == [8, 16]
    assert args.image_size_choices == [128, 160]
    assert args.skip_final_train is True


def test_hpo_main_optuna_yokken_acik_hata_verir(monkeypatch):
    monkeypatch.setattr(hpo, "optuna", None)

    result = hpo.main(["--trials", "1"])

    assert result == 1


def test_hpo_skip_final_train_icin_test_dizini_zorunlu_degildir(monkeypatch):
    calls = {}

    class FakeOptuna:
        pass

    def fake_validate_training_config(config, require_test_dir=True, full_trainval=False):
        calls["require_test_dir"] = require_test_dir
        calls["full_trainval"] = full_trainval

    monkeypatch.setattr(hpo, "optuna", FakeOptuna())
    monkeypatch.setattr(hpo, "validate_training_config", fake_validate_training_config)

    args = hpo.parse_args(["--trials", "1", "--skip-final-train"])
    hpo.validate_search_args(args)

    assert calls["require_test_dir"] is False
    assert calls["full_trainval"] is False


def test_run_training_hedef_metrige_gore_best_epoch_secer(monkeypatch):
    eval_metrics = iter(
        [
            {
                "loss": 0.30,
                "accuracy": 0.70,
                "precision": 0.70,
                "recall": 0.70,
                "f1": 0.70,
            },
            {
                "loss": 0.40,
                "accuracy": 0.75,
                "precision": 0.75,
                "recall": 0.75,
                "f1": 0.90,
            },
            {
                "loss": 0.20,
                "accuracy": 0.65,
                "precision": 0.65,
                "recall": 0.65,
                "f1": 0.60,
            },
            {
                "loss": 0.40,
                "accuracy": 0.75,
                "precision": 0.75,
                "recall": 0.75,
                "f1": 0.90,
            },
        ]
    )
    create_calls = {}

    def fake_create_dataloaders(**kwargs):
        create_calls.update(kwargs)
        info = {
            "num_classes": 4,
            "train_size": 8,
            "val_size": 4,
            "test_size": 0,
            "train_groups": 4,
            "val_groups": 2,
            "split_strategy": "group_stratified",
            "split_warnings": [],
            "train_labels": [0, 1, 2, 3],
            "trainval_grouping": {"grouping_reliable": True},
            "test_grouping": None,
            "test_labels": [],
            "test_grouping": None,
        }
        return object(), object(), None, info

    def fake_evaluate(_model, _loader, _criterion, _device):
        return dict(next(eval_metrics))

    class NeverStop:
        def __init__(self, patience):
            self.patience = patience

        def __call__(self, _value):
            return False

    monkeypatch.setattr(training_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))
    monkeypatch.setattr(training_runner, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(training_runner, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        training_runner,
        "compute_class_weights",
        lambda labels, num_classes: torch.ones(num_classes, dtype=torch.float32),
    )
    monkeypatch.setattr(
        training_runner,
        "train_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 0.5,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
    )
    monkeypatch.setattr(training_runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(training_runner, "EarlyStopping", NeverStop)

    trainval_dir = Path("tmp_test_artifacts") / f"train_{uuid4().hex}"
    trainval_dir.mkdir(parents=True, exist_ok=True)

    result = training_runner.run_training(
        training_runner.TrainingConfig(
            model="resnet",
            epochs=3,
            batch_size=2,
            trainval_dir=trainval_dir,
            test_dir=None,
            test_ratio=0.2,
        ),
        save_artifacts=False,
        evaluate_test_set=False,
        verbose=False,
        selection_metric="f1",
    )

    assert create_calls["include_test"] is False
    assert create_calls["test_dir"] is None
    assert create_calls["test_ratio"] == pytest.approx(0.2)
    assert result["best_epoch"] == 2
    assert result["best_val_metrics"]["f1"] == pytest.approx(0.90)
    assert result["best_val_loss"] == pytest.approx(0.40)
    assert result["lowest_val_loss"] == pytest.approx(0.20)
    assert result["selected_epoch_val_loss"] == pytest.approx(0.40)
    assert "train_f1" in result["history"]


def test_run_training_full_trainval_modunda_validation_atlamaz(monkeypatch):
    create_calls = {}
    eval_metrics = iter(
        [
            {
                "loss": 0.25,
                "accuracy": 0.80,
                "precision": 0.80,
                "recall": 0.80,
                "f1": 0.80,
            }
        ]
    )

    def fake_create_full_train_test_loaders(**kwargs):
        create_calls.update(kwargs)
        info = {
            "num_classes": 4,
            "train_size": 12,
            "val_size": 0,
            "test_size": 4,
            "train_groups": 8,
            "val_groups": 0,
            "split_strategy": "full_trainval_external_test",
            "split_warnings": [],
            "train_labels": [0, 1, 2, 3],
            "val_labels": [],
            "test_labels": [0, 1, 2, 3],
            "trainval_grouping": {"grouping_reliable": True},
            "test_grouping": {"grouping_reliable": True},
        }
        return object(), object(), info

    monkeypatch.setattr(training_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))
    monkeypatch.setattr(training_runner, "create_full_train_test_loaders", fake_create_full_train_test_loaders)
    monkeypatch.setattr(
        training_runner,
        "create_dataloaders",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("create_dataloaders cagrilmamali")),
    )
    monkeypatch.setattr(training_runner, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        training_runner,
        "compute_class_weights",
        lambda labels, num_classes: torch.ones(num_classes, dtype=torch.float32),
    )
    monkeypatch.setattr(
        training_runner,
        "train_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 0.4,
            "accuracy": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "f1": 0.6,
        },
    )
    monkeypatch.setattr(training_runner, "evaluate", lambda *_args, **_kwargs: dict(next(eval_metrics)))

    trainval_dir = Path("tmp_test_artifacts") / f"full_train_{uuid4().hex}"
    test_dir = Path("tmp_test_artifacts") / f"full_test_{uuid4().hex}"
    trainval_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    result = training_runner.run_training(
        training_runner.TrainingConfig(
            model="resnet",
            epochs=3,
            batch_size=2,
            trainval_dir=trainval_dir,
            test_dir=test_dir,
        ),
        save_artifacts=False,
        evaluate_test_set=True,
        full_trainval=True,
        verbose=False,
    )

    assert create_calls["trainval_dir"] == trainval_dir
    assert create_calls["test_dir"] == test_dir
    assert result["best_epoch"] == 3
    assert result["best_val_metrics"] is None
    assert result["best_train_metrics"]["f1"] == pytest.approx(0.6)
    assert result["selection_mode"] == "fixed_epoch_full_trainval"
    assert result["test_metrics"]["f1"] == pytest.approx(0.80)


def test_run_final_training_en_iyi_epoch_ve_full_trainval_kullanir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_training(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"output_root": tmp_path / "best_run", "report_path": tmp_path / "report.json"}

    monkeypatch.setattr(hpo, "run_training", fake_run_training)

    result = hpo._run_final_training(
        args=hpo.parse_args(["--model", "resnet", "--epochs", "12"]),
        study_dir=tmp_path,
        best_params={
            "batch_size": 16,
            "lr": 1e-4,
            "image_size": 224,
            "loss": "ce",
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 4,
        },
        study_name="demo_study",
        best_trial_number=7,
        best_epoch=5,
    )

    assert result["output_root"] == tmp_path / "best_run"
    assert captured["config"].epochs == 5
    assert captured["kwargs"]["full_trainval"] is True
    assert captured["kwargs"]["evaluate_test_set"] is True


def test_hpo_main_storage_varken_trials_hedef_toplam_olarak_yorumlanir(monkeypatch):
    optimize_calls = []

    class FakeTrialState:
        name = "COMPLETE"

    class FakeTrial:
        def __init__(self, number, value):
            self.number = number
            self.value = value
            self.params = {}
            self.user_attrs = {}
            self.state = FakeTrialState()

    class FakeStudy:
        def __init__(self):
            self.trials = [FakeTrial(0, 0.8), FakeTrial(1, 0.9)]
            self.best_trial = self.trials[1]
            self.best_value = self.best_trial.value

        def optimize(self, objective, n_trials, timeout, catch, gc_after_trial):
            optimize_calls.append(n_trials)

    fake_study = FakeStudy()
    fake_optuna = type(
        "FakeOptuna",
        (),
        {
            "samplers": type("Samplers", (), {"TPESampler": staticmethod(lambda **kwargs: object())}),
            "pruners": type("Pruners", (), {"MedianPruner": staticmethod(lambda **kwargs: object())}),
            "create_study": staticmethod(lambda **kwargs: fake_study),
        },
    )()

    monkeypatch.setattr(hpo, "optuna", fake_optuna)
    monkeypatch.setattr(hpo, "validate_search_args", lambda args: None)
    monkeypatch.setattr(hpo, "_save_study_artifacts", lambda **kwargs: None)

    output_dir = Path("tmp_test_artifacts") / f"hpo_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = hpo.main(
        [
            "--trials",
            "2",
            "--storage",
            "sqlite:///study.db",
            "--skip-final-train",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert optimize_calls == []
