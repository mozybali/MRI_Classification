"""
Derin ogrenme model katmani icin temel testler.
Not: Dosya adi geriye donuk uyumluluk icin korunmustur.
"""

import json
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

from model.dl.dataset import (
    kaynak_id_belirle,
    _group_stratified_train_val_split,
    _split_group_keys_kfold,
    compute_dataset_stats,
)
from model.dl.engine import evaluate, train_one_epoch
from model.dl.losses import FocalLoss, compute_class_weights
from model.dl.models.resnet_classifier import ResNetClassifier
from model.dl.utils import (
    configure_torch_runtime,
    load_checkpoint,
    plot_classification_summary,
    plot_confusion_matrix,
    plot_multiclass_roc_pr_curves,
    plot_prediction_confidence,
    plot_training_curves,
    set_seed,
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


def test_split_group_keys_kfold_grup_sizintisini_onler():
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3]
    groups = [
        "A::g1", "A::g1",
        "B::g1", "B::g1",
        "C::g1", "C::g1",
        "D::g1", "D::g1",
        "A::g2", "B::g2", "C::g2", "D::g2",
    ]

    fold_assignments, test_groups, strategy = _split_group_keys_kfold(
        labels=labels,
        groups=groups,
        n_folds=2,
        test_ratio=0.0,
        seed=42,
        use_group_split=True,
    )

    assert strategy == "group_stratified_kfold"
    assert test_groups == set()
    assert len(fold_assignments) == 2
    all_val_groups = set()
    for train_groups, val_groups in fold_assignments:
        assert train_groups.isdisjoint(val_groups)
        all_val_groups.update(val_groups)
    # Her kaynak grup tam olarak bir fold'un val'inde
    assert all_val_groups == set(groups)


def test_split_group_keys_kfold_test_ratio_onceden_grubu_ayirir():
    # Her sinifta 4 farkli kaynak grup -> test_ratio sonrasi her sinifta yeterli grup kalir
    labels = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    groups = [
        "A::g1", "A::g2", "A::g3", "A::g4",
        "B::g1", "B::g2", "B::g3", "B::g4",
        "C::g1", "C::g2", "C::g3", "C::g4",
        "D::g1", "D::g2", "D::g3", "D::g4",
    ]

    fold_assignments, test_groups, _strategy = _split_group_keys_kfold(
        labels=labels,
        groups=groups,
        n_folds=2,
        test_ratio=0.25,
        seed=42,
        use_group_split=True,
    )

    assert len(test_groups) > 0
    for train_groups, val_groups in fold_assignments:
        assert train_groups.isdisjoint(val_groups)
        # Test setine ayrilan gruplar fold'larda gorunmemeli
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)


def test_split_group_keys_kfold_yetersiz_grup_hata_verir():
    labels = [0, 0, 1, 1]
    groups = ["A::g1", "A::g1", "B::g1", "B::g1"]

    with pytest.raises(ValueError, match="yeterli kaynak grup yok"):
        _split_group_keys_kfold(
            labels=labels,
            groups=groups,
            n_folds=5,
            test_ratio=0.0,
            seed=42,
            use_group_split=True,
        )


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


def test_focal_loss_gamma_sifirken_ornek_bazli_ce_ile_eslesir():
    inputs = torch.tensor([[2.0, 0.5], [0.1, 1.3]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    alpha = torch.tensor([1.0, 3.0], dtype=torch.float32)

    focal = FocalLoss(alpha=alpha, gamma=0.0, reduction="mean")
    focal_loss = focal(inputs, targets)
    per_sample_ce = F.cross_entropy(inputs, targets, weight=alpha, reduction="none")
    expected = per_sample_ce.mean()

    assert torch.allclose(focal_loss, expected)


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


def test_detayli_rapor_ikili_2d_probs_calisir():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    probs = np.array(
        [
            [0.80, 0.20],
            [0.40, 0.60],
            [0.30, 0.70],
            [0.10, 0.90],
        ],
        dtype=np.float32,
    )

    report = training_runner._build_detailed_eval_report(
        labels, preds, probs, ["neg", "poz"]
    )

    assert report["macro_auc_ovr"] is not None
    assert report["macro_average_precision"] is not None
    assert report["per_class"]["neg"]["support"] == 2
    assert report["per_class"]["poz"]["support"] == 2


def test_detayli_rapor_ikili_1d_probs_otomatik_genisletir():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    probs_1d = np.array([0.20, 0.60, 0.70, 0.90], dtype=np.float32)

    report = training_runner._build_detailed_eval_report(
        labels, preds, probs_1d, ["neg", "poz"]
    )

    assert report["macro_auc_ovr"] is not None
    assert report["confidence"]["mean_confidence"] is not None


def test_detayli_rapor_probs_none_ise_auc_ap_bos():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    report = training_runner._build_detailed_eval_report(
        labels, preds, None, ["neg", "poz"]
    )

    assert report["macro_auc_ovr"] is None
    assert report["macro_average_precision"] is None
    assert report["confidence"]["mean_confidence"] is None


def test_detayli_rapor_label_aralik_disi_hata_verir():
    labels = np.array([0, 1, 2, 0])
    preds = np.array([0, 1, 1, 0])

    with pytest.raises(ValueError, match="araligi disinda"):
        training_runner._build_detailed_eval_report(
            labels, preds, None, ["neg", "poz"]
        )


def test_detayli_rapor_labels_preds_uzunlugu_uyumsuz_hata_verir():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1])

    with pytest.raises(ValueError, match="sekilleri uyusmuyor"):
        training_runner._build_detailed_eval_report(
            labels, preds, None, ["neg", "poz"]
        )


def test_detayli_rapor_probs_kolon_sayisi_uyumsuz_hata_verir():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])
    probs = np.array(
        [[0.6, 0.3, 0.1]] * 4,
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="probs sekli"):
        training_runner._build_detailed_eval_report(
            labels, preds, probs, ["neg", "poz"]
        )


def test_detayli_rapor_probs_nan_hata_verir():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])
    probs = np.array(
        [
            [0.6, 0.4],
            [np.nan, 0.5],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="NaN/Inf"):
        training_runner._build_detailed_eval_report(
            labels, preds, probs, ["neg", "poz"]
        )


def test_detayli_rapor_1d_probs_aralik_disi_hata_verir():
    labels = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])
    probs_1d = np.array([0.2, 1.5, 0.7, 0.3], dtype=np.float32)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        training_runner._build_detailed_eval_report(
            labels, preds, probs_1d, ["neg", "poz"]
        )


def test_detayli_rapor_1d_probs_cok_sinifli_hata_verir():
    labels = np.array([0, 1, 2, 0])
    preds = np.array([0, 1, 2, 0])
    probs_1d = np.array([0.2, 0.6, 0.7, 0.3], dtype=np.float32)

    with pytest.raises(ValueError, match="1D probs yalnizca ikili"):
        training_runner._build_detailed_eval_report(
            labels, preds, probs_1d, ["a", "b", "c"]
        )


def test_build_model_resnet_cpu_olusturur():
    device = torch.device("cpu")
    model = build_model("resnet", num_classes=4, device=device, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)


def test_resnet_pretrained_indirilemezse_acik_hata_verir(monkeypatch):
    from urllib.error import URLError

    def fake_resnet18(*args, **kwargs):
        raise URLError("download failed")

    monkeypatch.setattr("model.dl.models.resnet_classifier.models.resnet18", fake_resnet18)

    with pytest.raises(RuntimeError, match="indirilemedi") as exc_info:
        ResNetClassifier(num_classes=4, pretrained=True)
    assert isinstance(exc_info.value.__cause__, URLError)


def test_resnet_pretrained_oserror_da_sarilir(monkeypatch):
    def fake_resnet18(*args, **kwargs):
        raise OSError("cache write failed")

    monkeypatch.setattr("model.dl.models.resnet_classifier.models.resnet18", fake_resnet18)

    with pytest.raises(RuntimeError, match="indirilemedi"):
        ResNetClassifier(num_classes=4, pretrained=True)


def test_resnet_pretrained_api_hatasi_sarilmadan_yukselir(monkeypatch):
    def fake_resnet18(*args, **kwargs):
        raise TypeError("unexpected keyword argument 'weights'")

    monkeypatch.setattr("model.dl.models.resnet_classifier.models.resnet18", fake_resnet18)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ResNetClassifier(num_classes=4, pretrained=True)


def test_resnet_num_classes_gecersizse_hata_verir():
    with pytest.raises(ValueError, match="num_classes"):
        ResNetClassifier(num_classes=1, pretrained=False)
    with pytest.raises(ValueError, match="num_classes"):
        ResNetClassifier(num_classes=0, pretrained=False)


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


def test_train_parse_args_trainval_ve_test_dir_flaglerini_cozer():
    args = parse_train_args(
        [
            "--trainval-dir",
            "goruntu_isleme/cikti/trainval",
            "--test-dir",
            "goruntu_isleme/cikti/test",
        ]
    )

    assert args.trainval_dir == "goruntu_isleme/cikti/trainval"
    assert args.test_dir == "goruntu_isleme/cikti/test"


def test_train_parse_args_full_trainval_flagini_cozer():
    args = parse_train_args(["--full-trainval"])

    assert args.full_trainval is True


def test_resolve_data_dirs_explicit_trainval_sinif_dizinlerini_kullanir():
    processed_root = Path("tmp_test_artifacts") / f"processed_trainval_{uuid4().hex}"
    for class_name in training_runner.SINIF_ISIMLERI:
        (processed_root / class_name).mkdir(parents=True, exist_ok=True)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(
            trainval_dir=str(processed_root),
            test_dir="custom/test",
        )
    )

    assert trainval_dir == processed_root
    assert test_dir == Path("custom/test")


def test_resolve_data_dirs_explicit_test_dir_kullanir():
    processed_test = Path("tmp_test_artifacts") / f"processed_test_{uuid4().hex}"
    for class_name in training_runner.SINIF_ISIMLERI:
        (processed_test / class_name).mkdir(parents=True, exist_ok=True)

    _trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(test_dir=str(processed_test))
    )

    assert test_dir == processed_test


def test_resolve_data_dirs_explicit_trainval_ve_test_kullanir():
    processed_trainval = Path("tmp_test_artifacts") / f"processed_trainval_split_{uuid4().hex}"
    processed_test = Path("tmp_test_artifacts") / f"processed_test_split_{uuid4().hex}"
    for root in (processed_trainval, processed_test):
        for class_name in training_runner.SINIF_ISIMLERI:
            (root / class_name).mkdir(parents=True, exist_ok=True)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(
        training_runner.TrainingConfig(
            trainval_dir=str(processed_trainval),
            test_dir=str(processed_test),
        )
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
            trainval_dir=str(processed_root),
            test_dir=str(processed_root),
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


def test_resolve_data_dirs_varsayilanda_trainval_ve_test_dizinlerini_kullanir(monkeypatch):
    tv_path = Path("tmp_test_artifacts") / f"resolve_tv_{uuid4().hex}"
    te_path = Path("tmp_test_artifacts") / f"resolve_te_{uuid4().hex}"
    for root in (tv_path, te_path):
        for class_name in training_runner.SINIF_ISIMLERI:
            (root / class_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", tv_path)
    monkeypatch.setattr(training_runner, "TEST_VERI_DIZINI", te_path)

    trainval_dir, test_dir = training_runner.resolve_data_dirs(training_runner.TrainingConfig())

    assert trainval_dir == tv_path
    assert test_dir == te_path


def test_resolve_data_dirs_trainval_dizinini_varsayilan_olarak_kullanir(monkeypatch):
    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", Path("fallback/trainval"))
    monkeypatch.setattr(training_runner, "TEST_VERI_DIZINI", Path("fallback/trainval"))

    trainval_dir, _test_dir = training_runner.resolve_data_dirs(training_runner.TrainingConfig())

    assert trainval_dir == Path("fallback/trainval")


def test_resolve_data_dirs_harici_test_ayarini_kullanir(monkeypatch):
    external_test = Path("tmp_test_artifacts") / f"external_test_{uuid4().hex}"
    for class_name in training_runner.SINIF_ISIMLERI:
        (external_test / class_name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(training_runner, "TRAINVAL_VERI_DIZINI", Path("Veri_Seti/OriginalDataset"))
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
            "resnet",
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

    assert args.model == "resnet"
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

    def fake_evaluate(_model, _loader, _criterion, _device, **_kwargs):
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


def test_run_training_class_weightleri_original_train_etiketlerinden_hesaplar(monkeypatch):
    captured = {}

    def fake_create_dataloaders(**_kwargs):
        info = {
            "num_classes": 4,
            "train_size": 12,
            "val_size": 4,
            "test_size": 0,
            "train_groups": 4,
            "val_groups": 2,
            "split_strategy": "group_stratified",
            "split_warnings": [],
            "train_labels": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "train_original_labels": [0, 1, 2, 3],
            "trainval_grouping": {"grouping_reliable": True},
            "test_grouping": None,
            "test_labels": [],
        }
        return object(), object(), None, info

    def fake_compute_class_weights(labels, num_classes):
        captured["labels"] = list(labels)
        captured["num_classes"] = num_classes
        return torch.ones(num_classes, dtype=torch.float32)

    monkeypatch.setattr(training_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))
    monkeypatch.setattr(training_runner, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(training_runner, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(training_runner, "compute_class_weights", fake_compute_class_weights)
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
    monkeypatch.setattr(
        training_runner,
        "evaluate",
        lambda *_args, **_kwargs: {
            "loss": 0.4,
            "accuracy": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "f1": 0.6,
        },
    )

    trainval_dir = Path("tmp_test_artifacts") / f"train_weights_{uuid4().hex}"
    trainval_dir.mkdir(parents=True, exist_ok=True)

    training_runner.run_training(
        training_runner.TrainingConfig(
            model="resnet",
            epochs=1,
            batch_size=2,
            trainval_dir=trainval_dir,
            test_dir=None,
            test_ratio=0.2,
        ),
        save_artifacts=False,
        evaluate_test_set=False,
        verbose=False,
    )

    assert captured["labels"] == [0, 1, 2, 3]
    assert captured["num_classes"] == 4


def test_checkpoint_payload_full_trainval_validation_loss_metadata_bos():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    payload = training_runner._checkpoint_payload(
        config=training_runner.TrainingConfig(model="resnet"),
        epoch=3,
        selection_metric="f1",
        selection_value=0.876,
        selection_source="train",
        val_loss=None,
        model=model,
        optimizer=optimizer,
        num_classes=4,
    )

    assert payload["val_loss"] is None
    assert payload["selection_metric"] == "f1"
    assert payload["selection_value"] == pytest.approx(0.876)
    assert payload["selection_source"] == "train"


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


def test_hpo_main_xgboost_modunda_nop_pruner_kullanir(monkeypatch):
    calls = {"median": 0, "nop": 0, "pruner": None}

    class FakeTrialState:
        name = "COMPLETE"

    class FakeTrial:
        def __init__(self):
            self.number = 0
            self.value = 0.8
            self.params = {}
            self.user_attrs = {}
            self.state = FakeTrialState()

    class FakeStudy:
        def __init__(self):
            self.trials = [FakeTrial()]
            self.best_trial = self.trials[0]
            self.best_value = self.best_trial.value

        def optimize(self, *_args, **_kwargs):
            raise AssertionError("Yeni trial calistirilmamali")

    fake_study = FakeStudy()

    def fake_create_study(**kwargs):
        calls["pruner"] = kwargs["pruner"]
        return fake_study

    fake_optuna = type(
        "FakeOptuna",
        (),
        {
            "samplers": type("Samplers", (), {"TPESampler": staticmethod(lambda **kwargs: object())}),
            "pruners": type(
                "Pruners",
                (),
                {
                    "MedianPruner": staticmethod(
                        lambda **kwargs: calls.__setitem__("median", calls["median"] + 1) or "median"
                    ),
                    "NopPruner": staticmethod(
                        lambda: calls.__setitem__("nop", calls["nop"] + 1) or "nop"
                    ),
                },
            ),
            "create_study": staticmethod(fake_create_study),
        },
    )()

    monkeypatch.setattr(hpo, "optuna", fake_optuna)
    monkeypatch.setattr(hpo, "validate_search_args", lambda args: None)
    monkeypatch.setattr(hpo, "_save_study_artifacts", lambda **kwargs: None)

    output_dir = Path("tmp_test_artifacts") / f"hpo_xgb_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = hpo.main(
        [
            "--model",
            "xgboost",
            "--trials",
            "1",
            "--skip-final-train",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert calls["nop"] == 1
    assert calls["median"] == 0
    assert calls["pruner"] == "nop"


def test_train_parse_args_folds_flagini_cozer():
    args = parse_train_args(["--folds", "5"])
    assert args.folds == 5

    args_default = parse_train_args([])
    assert args_default.folds == 1


def test_hpo_parse_args_hpo_folds_flagini_cozer():
    args = hpo.parse_args(["--trials", "2", "--hpo-folds", "3"])
    assert args.hpo_folds == 3


def test_hpo_validate_args_hpo_folds_negatif_reddeder(monkeypatch):
    monkeypatch.setattr(hpo, "optuna", object())
    monkeypatch.setattr(hpo, "validate_training_config", lambda *a, **k: None)

    args = hpo.parse_args(["--trials", "1", "--hpo-folds", "0", "--skip-final-train"])
    with pytest.raises(ValueError, match="--hpo-folds en az 1"):
        hpo.validate_search_args(args)


def test_hpo_main_cv_modunda_nop_pruner_kullanir(monkeypatch):
    calls = {"median": 0, "nop": 0, "pruner": None}

    class FakeTrialState:
        name = "COMPLETE"

    class FakeTrial:
        def __init__(self):
            self.number = 0
            self.value = 0.8
            self.params = {}
            self.user_attrs = {}
            self.state = FakeTrialState()

    class FakeStudy:
        def __init__(self):
            self.trials = [FakeTrial()]
            self.best_trial = self.trials[0]
            self.best_value = self.best_trial.value

        def optimize(self, *_args, **_kwargs):
            raise AssertionError("Yeni trial calistirilmamali")

    fake_study = FakeStudy()

    def fake_create_study(**kwargs):
        calls["pruner"] = kwargs["pruner"]
        return fake_study

    fake_optuna = type(
        "FakeOptuna",
        (),
        {
            "samplers": type(
                "Samplers",
                (),
                {"TPESampler": staticmethod(lambda **kwargs: object())},
            ),
            "pruners": type(
                "Pruners",
                (),
                {
                    "MedianPruner": staticmethod(
                        lambda **kwargs: calls.__setitem__("median", calls["median"] + 1) or "median"
                    ),
                    "NopPruner": staticmethod(
                        lambda: calls.__setitem__("nop", calls["nop"] + 1) or "nop"
                    ),
                },
            ),
            "create_study": staticmethod(fake_create_study),
        },
    )()

    monkeypatch.setattr(hpo, "optuna", fake_optuna)
    monkeypatch.setattr(hpo, "validate_search_args", lambda args: None)
    monkeypatch.setattr(hpo, "_save_study_artifacts", lambda **kwargs: None)

    output_dir = Path("tmp_test_artifacts") / f"hpo_cv_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = hpo.main(
        [
            "--model",
            "resnet",
            "--trials",
            "1",
            "--hpo-folds",
            "3",
            "--skip-final-train",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert calls["nop"] == 1
    assert calls["median"] == 0
    assert calls["pruner"] == "nop"


def test_aggregate_cv_metrics_mean_ve_std_uretir():
    fold_results = [
        {
            "best_val_metrics": {
                "loss": 0.40, "accuracy": 0.80, "precision": 0.78, "recall": 0.81, "f1": 0.79,
            },
            "test_metrics": {
                "loss": 0.45, "accuracy": 0.75, "precision": 0.73, "recall": 0.76, "f1": 0.74,
            },
            "best_selection_value": 0.79,
        },
        {
            "best_val_metrics": {
                "loss": 0.50, "accuracy": 0.70, "precision": 0.69, "recall": 0.72, "f1": 0.71,
            },
            "test_metrics": {
                "loss": 0.55, "accuracy": 0.65, "precision": 0.63, "recall": 0.66, "f1": 0.64,
            },
            "best_selection_value": 0.71,
        },
    ]

    aggregate = training_runner._aggregate_cv_metrics(fold_results, selection_metric="f1")

    assert aggregate["completed_folds"] == 2
    assert aggregate["val"]["f1"]["mean"] == pytest.approx(0.75)
    assert aggregate["val"]["f1"]["std"] == pytest.approx(0.04)
    assert aggregate["test"]["accuracy"]["mean"] == pytest.approx(0.70)
    assert aggregate["selection"]["mean"] == pytest.approx(0.75)
    assert aggregate["selection"]["values"] == [0.79, 0.71]


def test_run_cv_training_n_folds_iki_alti_reddeder():
    config = training_runner.TrainingConfig()
    with pytest.raises(ValueError, match="--folds en az 2"):
        training_runner.run_cv_training(config, n_folds=1)


def test_run_cv_training_fold_basina_run_training_cagrir(monkeypatch, tmp_path):
    captured_calls = []

    def fake_iter_kfold(**kwargs):
        captured_calls.append({"kfold_kwargs": kwargs})
        for fold_idx in range(kwargs["n_folds"]):
            yield fold_idx, "train_loader", "val_loader", "test_loader", {
                "fold_index": fold_idx,
                "n_folds": kwargs["n_folds"],
            }

    def fake_run_training(config, **kwargs):
        captured_calls.append({"run_training_kwargs": kwargs})
        # train.py'nin run_training arabirimine uygun donus
        return {
            "best_epoch": 5,
            "best_val_metrics": {
                "loss": 0.4 + 0.05 * kwargs["extra_report"]["fold_index"],
                "accuracy": 0.7,
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.75,
            },
            "best_selection_value": 0.75,
            "test_metrics": {
                "loss": 0.5,
                "accuracy": 0.65,
                "precision": 0.65,
                "recall": 0.65,
                "f1": 0.7,
            },
            "report_path": tmp_path / f"fold_{kwargs['extra_report']['fold_index']}_report.json",
            "checkpoint_path": tmp_path / f"fold_{kwargs['extra_report']['fold_index']}.pt",
        }

    monkeypatch.setattr(training_runner, "iter_kfold_dataloaders", fake_iter_kfold)
    monkeypatch.setattr(training_runner, "validate_training_config", lambda *a, **k: None)
    monkeypatch.setattr(
        training_runner,
        "resolve_data_dirs",
        lambda config: (tmp_path / "trainval", tmp_path / "test"),
    )
    monkeypatch.setattr(training_runner, "run_training", fake_run_training)

    config = training_runner.TrainingConfig(model="resnet", epochs=3, batch_size=2)

    result = training_runner.run_cv_training(
        config,
        n_folds=3,
        output_root=tmp_path / "cv_out",
        artifact_tag="cv_test",
        save_artifacts=False,
        evaluate_test_set=True,
        verbose=False,
        selection_metric="f1",
    )

    assert result["n_folds"] == 3
    assert len(result["fold_results"]) == 3
    assert result["aggregate"]["val"]["f1"]["mean"] == pytest.approx(0.75)
    assert result["aggregate"]["completed_folds"] == 3
    # Run_training her fold icin fold_index iceren extra_report ile cagrildi
    run_training_calls = [c for c in captured_calls if "run_training_kwargs" in c]
    assert len(run_training_calls) == 3
    fold_indices = [c["run_training_kwargs"]["extra_report"]["fold_index"] for c in run_training_calls]
    assert fold_indices == [0, 1, 2]
    # preloaded_loaders enjekte edildigini dogrula
    for c in run_training_calls:
        assert c["run_training_kwargs"]["preloaded_loaders"] == (
            "train_loader", "val_loader", "test_loader",
            {"fold_index": fold_indices[run_training_calls.index(c)], "n_folds": 3},
        )


def test_train_main_folds_full_trainval_birlikte_reddeder():
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(project_root / "model" / "train.py"),
        "--folds", "5",
        "--full-trainval",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "--folds > 1 ile --full-trainval birlikte kullanilamaz" in result.stdout


# ============================================================================
# AMP / cuDNN runtime ayarlari (configure_torch_runtime, set_seed etkilesimi)
# ============================================================================


def test_configure_torch_runtime_deterministik_modu_etkinlestirir():
    configure_torch_runtime(deterministic=True, allow_tf32=False)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        assert torch.backends.cudnn.allow_tf32 is False
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        assert matmul_backend.allow_tf32 is False


def test_configure_torch_runtime_hpo_modunda_tf32_acik():
    try:
        configure_torch_runtime(deterministic=False, allow_tf32=True)

        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            assert torch.backends.cudnn.allow_tf32 is True
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            assert matmul_backend.allow_tf32 is True
    finally:
        # Diger testleri etkilememesi icin tam deterministik moda donus
        set_seed(42)


def test_set_seed_oncesi_acilan_tf32yi_kapatir():
    configure_torch_runtime(deterministic=False, allow_tf32=True)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        assert torch.backends.cudnn.allow_tf32 is True

    set_seed(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        assert torch.backends.cudnn.allow_tf32 is False


# ============================================================================
# compute_dataset_stats disk cache davranisi
# ============================================================================


def _create_dummy_image(path: Path, *, color: tuple[int, int, int] = (128, 64, 32)) -> None:
    from PIL import Image

    Image.new("RGB", (32, 32), color=color).save(path)


def test_compute_dataset_stats_cache_dir_none_iken_disk_kullanmaz(tmp_path):
    img_path = tmp_path / "im.png"
    _create_dummy_image(img_path)

    mean, std = compute_dataset_stats(
        [img_path], image_size=32, max_samples=None, cache_dir=None
    )

    assert len(mean) == 3 and len(std) == 3
    assert all(s > 0 for s in std)


def test_compute_dataset_stats_cache_diskten_okur(tmp_path):
    img_path = tmp_path / "im.png"
    _create_dummy_image(img_path)
    cache_dir = tmp_path / "stats_cache"

    mean1, std1 = compute_dataset_stats(
        [img_path], image_size=32, max_samples=None, cache_dir=cache_dir
    )

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1

    # Bayrak: image_size + path listesi degismedi -> ikinci cagri cache hit.
    mean2, std2 = compute_dataset_stats(
        [img_path], image_size=32, max_samples=None, cache_dir=cache_dir
    )

    assert mean1 == mean2
    assert std1 == std2


def test_compute_dataset_stats_image_size_degisince_yeni_cache_olusur(tmp_path):
    img_path = tmp_path / "im.png"
    _create_dummy_image(img_path)
    cache_dir = tmp_path / "stats_cache"

    compute_dataset_stats(
        [img_path], image_size=32, max_samples=None, cache_dir=cache_dir
    )
    compute_dataset_stats(
        [img_path], image_size=64, max_samples=None, cache_dir=cache_dir
    )

    cache_files = list(cache_dir.glob("*.json"))
    # Ayri image_size icin ayri cache anahtari uretilir
    assert len(cache_files) == 2


# ============================================================================
# Engine: bos veri yukleyici hata mesajlari
# ============================================================================


def test_train_one_epoch_bos_loaderda_aciklayici_hata_verir():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    with pytest.raises(RuntimeError, match="bos veri yukleyici"):
        train_one_epoch(model, [], criterion, optimizer, torch.device("cpu"))


def test_evaluate_bos_loaderda_aciklayici_hata_verir():
    model = torch.nn.Linear(4, 2)
    criterion = torch.nn.CrossEntropyLoss()

    with pytest.raises(RuntimeError, match="bos veri yukleyici"):
        evaluate(model, [], criterion, torch.device("cpu"))


def test_evaluate_cpu_loaderda_metrik_ve_prob_dondurur():
    model = torch.nn.Linear(4, 3)
    criterion = torch.nn.CrossEntropyLoss()
    inputs = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    loader = [(inputs[:3], targets[:3]), (inputs[3:], targets[3:])]

    result = evaluate(model, loader, criterion, torch.device("cpu"), use_amp=False)

    assert set(result.keys()) >= {"loss", "accuracy", "preds", "labels", "probs", "confidences"}
    assert result["probs"].shape == (6, 3)
    assert result["preds"].shape == (6,)
    assert result["labels"].shape == (6,)
    # CPU yolunda use_amp=True bile autocast'i no-op'a indiriyor; ek olarak
    # use_amp=True ile cagrildiginda da hata vermemeli.
    result_amp = evaluate(model, loader, criterion, torch.device("cpu"), use_amp=True)
    assert result_amp["probs"].shape == (6, 3)


# ============================================================================
# run_training: deterministic / use_amp parametre yolu
# ============================================================================


def test_run_training_deterministic_false_iken_configure_torch_runtimei_etkinlestirir(monkeypatch):
    captured = {}

    def fake_configure(*, deterministic, allow_tf32):
        captured["deterministic"] = deterministic
        captured["allow_tf32"] = allow_tf32

    monkeypatch.setattr(training_runner, "configure_torch_runtime", fake_configure)
    monkeypatch.setattr(training_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))
    monkeypatch.setattr(
        training_runner,
        "create_dataloaders",
        lambda **kwargs: (
            object(), object(), None,
            {
                "num_classes": 4, "train_size": 4, "val_size": 2, "test_size": 0,
                "train_groups": 2, "val_groups": 1,
                "split_strategy": "group_stratified", "split_warnings": [],
                "train_labels": [0, 1, 2, 3],
                "trainval_grouping": {"grouping_reliable": True},
                "test_grouping": None, "test_labels": [],
            },
        ),
    )
    monkeypatch.setattr(training_runner, "build_model", lambda *a, **k: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        training_runner,
        "compute_class_weights",
        lambda labels, num_classes: torch.ones(num_classes, dtype=torch.float32),
    )
    monkeypatch.setattr(
        training_runner,
        "train_one_epoch",
        lambda *a, **k: {"loss": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
    )
    monkeypatch.setattr(
        training_runner,
        "evaluate",
        lambda *a, **k: {"loss": 0.4, "accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6},
    )

    trainval_dir = Path("tmp_test_artifacts") / f"deterministic_{uuid4().hex}"
    trainval_dir.mkdir(parents=True, exist_ok=True)

    training_runner.run_training(
        training_runner.TrainingConfig(
            model="resnet", epochs=1, batch_size=2,
            trainval_dir=trainval_dir, test_dir=None, test_ratio=0.0,
        ),
        save_artifacts=False,
        evaluate_test_set=False,
        verbose=False,
        deterministic=False,
    )

    assert captured["deterministic"] is False
    assert captured["allow_tf32"] is True


# ============================================================================
# train.py: yeni XGB device/n-jobs flagleri
# ============================================================================


def test_train_parse_args_xgb_device_ve_n_jobs_flaglerini_cozer():
    args = parse_train_args(
        ["--xgb-device", "cpu", "--xgb-n-jobs", "4"]
    )

    assert args.xgb_device == "cpu"
    assert args.xgb_n_jobs == 4


def test_train_parse_args_xgb_device_varsayilan_auto():
    args = parse_train_args([])

    assert args.xgb_device == "auto"
    assert args.xgb_n_jobs is None


# ============================================================================
# HPO: yeni paralel/cache/plot bayraklari
# ============================================================================


def test_hpo_parse_args_paralel_ve_cache_flaglerini_cozer():
    args = hpo.parse_args(
        [
            "--trials", "1",
            "--n-jobs", "4",
            "--no-feature-cache",
            "--xgb-device", "cpu",
            "--xgb-n-jobs", "2",
            "--no-hpo-plots",
        ]
    )

    assert args.n_jobs == 4
    assert args.no_feature_cache is True
    assert args.xgb_device == "cpu"
    assert args.xgb_n_jobs == 2
    assert args.no_hpo_plots is True


def test_hpo_resolve_feature_cache_no_feature_cache_iken_none_doner():
    args = hpo.parse_args(["--trials", "1", "--no-feature-cache"])
    assert hpo._resolve_feature_cache(args) is None


def test_hpo_resolve_feature_cache_acik_dizini_kullanir():
    args = hpo.parse_args(
        ["--trials", "1", "--feature-cache", "custom/cache_dir"]
    )
    assert hpo._resolve_feature_cache(args) == "custom/cache_dir"


def test_hpo_validate_search_args_n_jobs_negatif_reddeder(monkeypatch):
    monkeypatch.setattr(hpo, "optuna", object())
    monkeypatch.setattr(hpo, "validate_training_config", lambda *a, **k: None)

    args = hpo.parse_args(["--trials", "1", "--n-jobs", "0", "--skip-final-train"])
    with pytest.raises(ValueError, match="--n-jobs en az 1"):
        hpo.validate_search_args(args)


def test_hpo_validate_search_args_xgb_n_jobs_negatif_reddeder(monkeypatch):
    monkeypatch.setattr(hpo, "optuna", object())
    monkeypatch.setattr(hpo, "validate_sl_config", lambda *a, **k: None)

    args = hpo.parse_args(
        [
            "--model", "xgboost",
            "--trials", "1",
            "--xgb-n-jobs", "0",
            "--skip-final-train",
        ]
    )
    with pytest.raises(ValueError, match="--xgb-n-jobs"):
        hpo.validate_search_args(args)


def test_hpo_save_study_artifacts_no_hpo_plots_iken_gorseli_atlar(monkeypatch, tmp_path):
    plot_calls = {"count": 0}

    def fake_save_visualizations(*_args, **_kwargs):
        plot_calls["count"] += 1
        return {"history": "history.png"}

    monkeypatch.setattr(hpo, "_save_hpo_visualizations", fake_save_visualizations)

    class FakeTrialState:
        name = "COMPLETE"

    class FakeTrial:
        def __init__(self):
            self.number = 0
            self.value = 0.5
            self.params = {"x": 1}
            self.user_attrs = {}
            self.state = FakeTrialState()
            self.datetime_start = None
            self.datetime_complete = None

    class FakeStudy:
        def __init__(self):
            self.trials = [FakeTrial()]
            self.best_trial = self.trials[0]
            self.best_value = self.best_trial.value
            self.study_name = "demo"

        def trials_dataframe(self):
            import pandas as pd

            return pd.DataFrame({"number": [0], "value": [0.5]})

    args = hpo.parse_args(["--trials", "1", "--no-hpo-plots", "--skip-final-train"])

    hpo._save_study_artifacts(
        study=FakeStudy(),
        args=args,
        study_dir=tmp_path,
        study_name="demo",
        final_run=None,
        existing_trial_count=0,
        trials_executed_this_run=1,
    )

    assert plot_calls["count"] == 0
    assert (tmp_path / "trial_history.csv").exists()


# ============================================================================
# SL config dogrulama: device ve n_jobs
# ============================================================================


def test_validate_sl_config_gecersiz_device_reddeder():
    from model.sl.training_runner import SLTrainingConfig, validate_sl_config

    cfg = SLTrainingConfig(device="gpu")
    with pytest.raises(ValueError, match="--xgb-device"):
        validate_sl_config(cfg, require_test_dir=False)


def test_validate_sl_config_gecersiz_n_jobs_reddeder():
    from model.sl.training_runner import SLTrainingConfig, validate_sl_config

    cfg = SLTrainingConfig(n_jobs=0)
    with pytest.raises(ValueError, match="--xgb-n-jobs"):
        validate_sl_config(cfg, require_test_dir=False)


# ============================================================================
# VRAM/RAM cleanup helper'lari ve HPO/CV entegrasyonu
# ============================================================================


def test_release_cuda_memory_cuda_yokken_hata_vermeden_calisir(monkeypatch):
    """CPU-only ortamda release_cuda_memory bir gc turu yapmali ve hata atmamali."""
    from model.dl import utils as dl_utils

    # Cuda mevcut olsa bile testi predictable tutmak icin kapatiyoruz.
    monkeypatch.setattr(dl_utils.torch.cuda, "is_available", lambda: False)

    gc_calls = {"count": 0}
    real_collect = dl_utils.gc.collect

    def counting_collect(*args, **kwargs):
        gc_calls["count"] += 1
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(dl_utils.gc, "collect", counting_collect)

    dl_utils.release_cuda_memory()

    assert gc_calls["count"] >= 1


def test_release_cuda_memory_cuda_varken_empty_cache_ve_ipc_collect_cagrir(monkeypatch):
    """CUDA mevcut hayalinde empty_cache ve ipc_collect'in her ikisi de cagirilmali."""
    from model.dl import utils as dl_utils

    calls = {"empty_cache": 0, "ipc_collect": 0, "gc": 0}

    monkeypatch.setattr(dl_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        dl_utils.torch.cuda,
        "empty_cache",
        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )
    monkeypatch.setattr(
        dl_utils.torch.cuda,
        "ipc_collect",
        lambda: calls.__setitem__("ipc_collect", calls["ipc_collect"] + 1),
    )
    real_collect = dl_utils.gc.collect

    def counting_collect(*args, **kwargs):
        calls["gc"] += 1
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(dl_utils.gc, "collect", counting_collect)

    dl_utils.release_cuda_memory()

    assert calls["empty_cache"] == 1
    assert calls["ipc_collect"] == 1
    assert calls["gc"] >= 1


def test_release_cuda_memory_empty_cache_hatasi_akisi_bloklamaz(monkeypatch):
    """empty_cache exception atsa bile ipc_collect cagirilmali, fonksiyon sessiz gecmeli."""
    from model.dl import utils as dl_utils

    calls = {"ipc_collect": 0}

    monkeypatch.setattr(dl_utils.torch.cuda, "is_available", lambda: True)

    def boom():
        raise RuntimeError("driver hiccup")

    monkeypatch.setattr(dl_utils.torch.cuda, "empty_cache", boom)
    monkeypatch.setattr(
        dl_utils.torch.cuda,
        "ipc_collect",
        lambda: calls.__setitem__("ipc_collect", calls["ipc_collect"] + 1),
    )

    # Exception gizlenmeli
    dl_utils.release_cuda_memory()

    assert calls["ipc_collect"] == 1


def test_clone_state_dict_to_cpu_tensorlari_cpuya_tasir():
    from model.dl.utils import clone_state_dict_to_cpu

    model = torch.nn.Linear(4, 3)
    src = model.state_dict()

    cloned = clone_state_dict_to_cpu(src)

    assert set(cloned.keys()) == set(src.keys())
    for key, tensor in cloned.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        # Ayni degerleri tasimali (allclose, deterministik karsilastirma)
        assert torch.allclose(tensor, src[key].detach().cpu())


def test_clone_state_dict_to_cpu_orijinali_etkilemez():
    from model.dl.utils import clone_state_dict_to_cpu

    model = torch.nn.Linear(4, 3)
    src = model.state_dict()
    cloned = clone_state_dict_to_cpu(src)

    # Klonu yerinde degistirmek orijinal storage'i etkilememeli
    first_key = next(iter(cloned))
    cloned[first_key].add_(1.0)

    assert not torch.allclose(cloned[first_key], src[first_key].detach().cpu())


def test_clone_state_dict_to_cpu_tensor_olmayan_degerleri_korur():
    from model.dl.utils import clone_state_dict_to_cpu

    src = {
        "weight": torch.randn(2, 3),
        "meta_int": 7,
        "meta_str": "abc",
    }

    cloned = clone_state_dict_to_cpu(src)

    assert cloned["meta_int"] == 7
    assert cloned["meta_str"] == "abc"
    assert cloned["weight"].device.type == "cpu"


def test_run_training_best_state_dict_cpu_uzerinde_tutulur(monkeypatch):
    """run_training trial-ici VRAM'i sismememesi icin best state_dict'i CPU'ya
    klonlamali."""
    captured = {}

    def fake_clone(state_dict):
        captured["clone_called"] = captured.get("clone_called", 0) + 1
        captured["device_types"] = sorted({v.device.type for v in state_dict.values()})
        return {k: v.detach().to("cpu", copy=True) for k, v in state_dict.items()}

    monkeypatch.setattr(training_runner, "clone_state_dict_to_cpu", fake_clone)
    monkeypatch.setattr(training_runner, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_runner, "configure_torch_runtime", lambda **kwargs: None)
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))

    def fake_create_dataloaders(**_kwargs):
        info = {
            "num_classes": 4,
            "train_size": 4,
            "val_size": 2,
            "test_size": 0,
            "train_groups": 2,
            "val_groups": 1,
            "split_strategy": "group_stratified",
            "split_warnings": [],
            "train_labels": [0, 1, 2, 3],
            "trainval_grouping": {"grouping_reliable": True},
            "test_grouping": None,
            "test_labels": [],
        }
        return object(), object(), None, info

    monkeypatch.setattr(training_runner, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(training_runner, "build_model", lambda *a, **k: torch.nn.Linear(4, 4))
    monkeypatch.setattr(
        training_runner,
        "compute_class_weights",
        lambda labels, num_classes: torch.ones(num_classes, dtype=torch.float32),
    )
    monkeypatch.setattr(
        training_runner,
        "train_one_epoch",
        lambda *a, **k: {"loss": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
    )
    monkeypatch.setattr(
        training_runner,
        "evaluate",
        lambda *a, **k: {"loss": 0.4, "accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6},
    )

    trainval_dir = Path("tmp_test_artifacts") / f"clone_state_{uuid4().hex}"
    trainval_dir.mkdir(parents=True, exist_ok=True)

    training_runner.run_training(
        training_runner.TrainingConfig(
            model="resnet", epochs=1, batch_size=2,
            trainval_dir=trainval_dir, test_dir=None, test_ratio=0.0,
        ),
        save_artifacts=False,
        evaluate_test_set=False,
        verbose=False,
    )

    assert captured.get("clone_called", 0) >= 1
    # Klonlanan tensor'lar CPU'da uretilen modelden gelmis olmali
    assert captured["device_types"] == ["cpu"]


def test_run_cv_training_fold_basina_release_cuda_memory_cagirir(monkeypatch, tmp_path):
    """Her fold tamamlandiktan sonra release_cuda_memory tetiklenmeli."""
    release_calls = {"count": 0}

    monkeypatch.setattr(
        training_runner,
        "release_cuda_memory",
        lambda: release_calls.__setitem__("count", release_calls["count"] + 1),
    )

    def fake_iter_kfold(**kwargs):
        for fold_idx in range(kwargs["n_folds"]):
            yield fold_idx, "train_loader", "val_loader", "test_loader", {
                "fold_index": fold_idx,
                "n_folds": kwargs["n_folds"],
            }

    def fake_run_training(config, **kwargs):
        return {
            "best_epoch": 1,
            "best_val_metrics": {
                "loss": 0.4, "accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7,
            },
            "best_selection_value": 0.7,
            "test_metrics": None,
            "report_path": None,
            "checkpoint_path": None,
        }

    monkeypatch.setattr(training_runner, "iter_kfold_dataloaders", fake_iter_kfold)
    monkeypatch.setattr(training_runner, "validate_training_config", lambda *a, **k: None)
    monkeypatch.setattr(
        training_runner,
        "resolve_data_dirs",
        lambda config: (tmp_path / "trainval", tmp_path / "test"),
    )
    monkeypatch.setattr(training_runner, "run_training", fake_run_training)

    config = training_runner.TrainingConfig(model="resnet", epochs=1, batch_size=2)

    training_runner.run_cv_training(
        config,
        n_folds=3,
        output_root=tmp_path / "cv_release",
        artifact_tag="cv_release",
        save_artifacts=False,
        evaluate_test_set=False,
        verbose=False,
        selection_metric="f1",
    )

    assert release_calls["count"] == 3


def _build_fake_trial():
    class FakeTrial:
        def __init__(self, number=0):
            self.number = number
            self.user_attrs: dict = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

        def report(self, *_args, **_kwargs):
            pass

        def should_prune(self):
            return False

    return FakeTrial()


def _hpo_args_for_dl():
    return hpo.parse_args(
        [
            "--model", "resnet",
            "--trials", "1",
            "--epochs", "1",
            "--batch-size-choices", "2",
            "--image-size-choices", "32",
            "--skip-final-train",
        ]
    )


def test_hpo_dl_objective_basari_yolunda_release_cuda_memory_cagrir(monkeypatch, tmp_path):
    """Trial basariyla bitse de cleanup tetiklenmeli (try/finally finally bloku)."""
    release_calls = {"count": 0}

    monkeypatch.setattr(
        hpo,
        "release_cuda_memory",
        lambda: release_calls.__setitem__("count", release_calls["count"] + 1),
    )

    fake_results = {
        "best_epoch": 2,
        "best_val_loss": 0.30,
        "lowest_val_loss": 0.25,
        "best_val_metrics": {
            "loss": 0.30, "accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.85,
        },
        "config": {"model": "resnet"},
    }
    monkeypatch.setattr(hpo, "run_training", lambda *a, **k: fake_results)

    def fake_sample(_trial, _args):
        return {
            "batch_size": 2,
            "image_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "loss": "ce",
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "dropout": 0.5,
            "hflip_p": 0.0,
            "rotation_degrees": 5,
            "color_jitter": 0.1,
            "pretrained": False,
        }

    monkeypatch.setattr(hpo, "_sample_params", fake_sample)

    args = _hpo_args_for_dl()
    objective = hpo._objective_factory(args, tmp_path)
    trial = _build_fake_trial()

    value = objective(trial)

    assert value == pytest.approx(0.85)  # default metric "f1"
    assert release_calls["count"] == 1
    summary = json.loads(
        (tmp_path / "trials" / "trial_000" / "trial_summary.json").read_text(encoding="utf-8")
    )
    assert summary["state"] == "COMPLETE"


def test_hpo_dl_objective_basarisiz_trialde_de_release_cuda_memory_cagrir(monkeypatch, tmp_path):
    """Trial hata ile bitse bile finally bloku cleanup'i garanti etmeli."""
    release_calls = {"count": 0}

    monkeypatch.setattr(
        hpo,
        "release_cuda_memory",
        lambda: release_calls.__setitem__("count", release_calls["count"] + 1),
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory simulasyonu")

    monkeypatch.setattr(hpo, "run_training", boom)

    def fake_sample(_trial, _args):
        return {
            "batch_size": 2,
            "image_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "loss": "ce",
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "dropout": 0.5,
            "hflip_p": 0.0,
            "rotation_degrees": 5,
            "color_jitter": 0.1,
            "pretrained": False,
        }

    monkeypatch.setattr(hpo, "_sample_params", fake_sample)

    args = _hpo_args_for_dl()
    objective = hpo._objective_factory(args, tmp_path)
    trial = _build_fake_trial()

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        objective(trial)

    assert release_calls["count"] == 1
    summary = json.loads(
        (tmp_path / "trials" / "trial_000" / "trial_summary.json").read_text(encoding="utf-8")
    )
    assert summary["state"] == "FAILED"


def test_hpo_dl_objective_cv_basari_yolunda_release_cuda_memory_cagrir(monkeypatch, tmp_path):
    """CV modunda da finally bloku cleanup'i garanti etmeli."""
    release_calls = {"count": 0}

    monkeypatch.setattr(
        hpo,
        "release_cuda_memory",
        lambda: release_calls.__setitem__("count", release_calls["count"] + 1),
    )

    fake_cv = {
        "aggregate": {
            "val": {
                "f1": {"mean": 0.78, "std": 0.02, "values": [0.76, 0.80]},
                "loss": {"mean": 0.40, "std": 0.05, "values": [0.45, 0.35]},
            },
            "test": None,
            "selection": {
                "metric": "f1", "mean": 0.78, "std": 0.02, "values": [0.76, 0.80],
            },
            "completed_folds": 2,
        },
        "fold_results": [{"best_epoch": 3}, {"best_epoch": 4}],
        "config": {"model": "resnet"},
    }
    monkeypatch.setattr(hpo, "run_cv_training", lambda *a, **k: fake_cv)

    def fake_sample(_trial, _args):
        return {
            "batch_size": 2,
            "image_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "loss": "ce",
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "dropout": 0.5,
            "hflip_p": 0.0,
            "rotation_degrees": 5,
            "color_jitter": 0.1,
            "pretrained": False,
        }

    monkeypatch.setattr(hpo, "_sample_params", fake_sample)

    args = hpo.parse_args(
        [
            "--model", "resnet",
            "--trials", "1",
            "--epochs", "1",
            "--hpo-folds", "2",
            "--batch-size-choices", "2",
            "--image-size-choices", "32",
            "--skip-final-train",
        ]
    )
    objective = hpo._objective_factory(args, tmp_path)
    trial = _build_fake_trial()

    value = objective(trial)

    assert value == pytest.approx(0.78)
    assert release_calls["count"] == 1
    assert trial.user_attrs["best_epoch"] == 4  # round((3+4)/2)


def test_hpo_dl_objective_pruned_trialde_de_release_cuda_memory_cagrir(monkeypatch, tmp_path):
    """Optuna TrialPruned akisinda da finally bloku cleanup'i tetiklemeli."""
    if hpo.optuna is None:
        pytest.skip("Optuna kurulu degil; pruned-path testi atlandi.")

    release_calls = {"count": 0}

    monkeypatch.setattr(
        hpo,
        "release_cuda_memory",
        lambda: release_calls.__setitem__("count", release_calls["count"] + 1),
    )

    def pruned(*_args, **_kwargs):
        raise hpo.optuna.TrialPruned("erken pruning")

    monkeypatch.setattr(hpo, "run_training", pruned)

    def fake_sample(_trial, _args):
        return {
            "batch_size": 2,
            "image_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "loss": "ce",
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "dropout": 0.5,
            "hflip_p": 0.0,
            "rotation_degrees": 5,
            "color_jitter": 0.1,
            "pretrained": False,
        }

    monkeypatch.setattr(hpo, "_sample_params", fake_sample)

    args = _hpo_args_for_dl()
    objective = hpo._objective_factory(args, tmp_path)
    trial = _build_fake_trial()

    with pytest.raises(hpo.optuna.TrialPruned):
        objective(trial)

    assert release_calls["count"] == 1
    summary = json.loads(
        (tmp_path / "trials" / "trial_000" / "trial_summary.json").read_text(encoding="utf-8")
    )
    assert summary["state"] == "PRUNED"


# ---------------------------------------------------------------------------
# NaN guard ve checkpoint bug fix testleri
# ---------------------------------------------------------------------------

def test_is_improved_nan_false_doner():
    """NaN/Inf candidate asla improvement sayilmamali."""
    assert training_runner._is_improved(float("nan"), None, "minimize") is False
    assert training_runner._is_improved(float("inf"), None, "minimize") is False
    assert training_runner._is_improved(float("nan"), 0.5, "minimize") is False
    assert training_runner._is_improved(float("inf"), 0.5, "maximize") is False


def test_is_improved_gecerli_degerler_dogru_calisir():
    """Finite degerler icin normal karsilastirma dogru calisir."""
    assert training_runner._is_improved(0.3, None, "minimize") is True
    assert training_runner._is_improved(0.3, 0.5, "minimize") is True   # 0.3 < 0.5 → iyilesme
    assert training_runner._is_improved(0.6, 0.5, "maximize") is True   # 0.6 > 0.5 → iyilesme
    assert training_runner._is_improved(0.6, 0.5, "minimize") is False  # 0.6 > 0.5 → kotulasma
    assert training_runner._is_improved(0.4, 0.5, "maximize") is False  # 0.4 < 0.5 → kotulasma


def test_is_improved_best_value_nan_iken_finite_candidate_true():
    """best_value NaN iken gecerli candidate improvement sayilmali."""
    assert training_runner._is_improved(0.5, float("nan"), "minimize") is True
    assert training_runner._is_improved(0.5, float("inf"), "minimize") is True


def test_nan_loss_checkpoint_kaydedilmez(monkeypatch, tmp_path):
    """NaN loss uretildiginde best checkpoint kaydedilmemeli."""
    import math

    saved_files = []

    def fake_save(obj, path):
        saved_files.append(path)

    monkeypatch.setattr(torch, "save", fake_save)

    call_count = [0]

    def fake_train_one_epoch(*args, **kwargs):
        call_count[0] += 1
        return {"loss": float("nan"), "accuracy": 0.25, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    def fake_evaluate(*args, **kwargs):
        return {"loss": float("nan"), "accuracy": 0.25, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "preds": np.array([0]), "labels": np.array([0]), "probs": np.zeros((1, 4)),
                "confidences": np.array([0.25])}

    monkeypatch.setattr(training_runner, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(training_runner, "evaluate", fake_evaluate)

    dummy_loader = [(torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long))]

    def fake_create_dataloaders(**kwargs):
        info = {
            "num_classes": 4, "train_size": 2, "val_size": 2, "test_size": 2,
            "train_groups": 2, "val_groups": 2, "split_strategy": "test",
            "split_warnings": [], "train_labels": np.array([0, 1]),
            "trainval_grouping": None, "test_grouping": None,
            "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5],
        }
        return dummy_loader, dummy_loader, dummy_loader, info

    monkeypatch.setattr(training_runner, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(training_runner, "validate_training_config", lambda *a, **k: None)
    monkeypatch.setattr(training_runner, "resolve_data_dirs", lambda c: (tmp_path, tmp_path))
    monkeypatch.setattr(training_runner, "get_device", lambda verbose=True: torch.device("cpu"))

    import model.dl.utils as _utils
    monkeypatch.setattr(_utils, "configure_torch_runtime", lambda **kw: None)

    config = training_runner.TrainingConfig(epochs=2, batch_size=2, lr=1e-4, patience=10)
    result = training_runner.run_training(
        config,
        output_root=tmp_path,
        save_artifacts=True,
        evaluate_test_set=True,
        verbose=False,
    )

    # NaN loss ile hicbir checkpoint kaydedilmemeli
    best_ckpt = result["checkpoint_path"]
    assert best_ckpt not in saved_files, (
        f"NaN loss ile checkpoint kaydedildi: {best_ckpt}"
    )


def test_train_one_epoch_nan_loss_runtime_error_firlatir():
    """train_one_epoch NaN loss urettiginde RuntimeError firlatmali."""
    import torch.nn as nn

    model = nn.Linear(4, 4)
    model.train()

    # NaN weight ile criterion NaN loss uretir
    nan_weight = torch.tensor([float("nan"), 1.0, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=nan_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    class NaNLoader:
        def __iter__(self):
            yield torch.randn(2, 4), torch.tensor([0, 1])

    from model.dl.engine import train_one_epoch

    with pytest.raises(RuntimeError, match="NaN/Inf loss"):
        train_one_epoch(model, NaNLoader(), criterion, optimizer, torch.device("cpu"))


def test_evaluate_nan_loss_runtime_error_firlatir():
    """evaluate NaN loss urettiginde RuntimeError firlatmali."""
    import torch.nn as nn

    model = nn.Linear(4, 4)
    nan_weight = torch.tensor([float("nan"), 1.0, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=nan_weight)

    class NaNLoader:
        def __iter__(self):
            yield torch.randn(2, 4), torch.tensor([0, 1])

    from model.dl.engine import evaluate

    with pytest.raises(RuntimeError, match="NaN/Inf loss"):
        evaluate(model, NaNLoader(), criterion, torch.device("cpu"))
