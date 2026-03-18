"""
Derin ogrenme model katmani icin temel testler.
Not: Dosya adi geriye donuk uyumluluk icin korunmustur.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model.dl.dataset import kaynak_id_belirle, _group_stratified_train_val_split
from model.dl.losses import FocalLoss, compute_class_weights
from model.dl.models.resnet_classifier import ResNetClassifier
from model.dl.utils import plot_confusion_matrix
from model.train import build_model


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


def test_group_split_yetersiz_kaynak_grubunda_hata_verir():
    labels = [0, 0, 1, 1]
    groups = [
        "A::g1", "A::g1",
        "B::g1", "B::g2",
    ]

    try:
        _group_stratified_train_val_split(
            labels=labels,
            groups=groups,
            val_ratio=0.5,
            seed=42,
            num_classes=2,
        )
    except ValueError as exc:
        assert "en az 2 farkli kaynak grup" in str(exc)
    else:
        raise AssertionError("Beklenen ValueError olusmadi.")


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
