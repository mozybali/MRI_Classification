from pathlib import Path
import os
import sys

import numpy as np
import pytest
from PIL import Image

if os.environ.get("MRI_RUN_TORCH_TESTS") != "1":
    pytest.skip(
        "Torch bagimli testler varsayilan olarak atlanir. Calistirmak icin "
        "MRI_RUN_TORCH_TESTS=1 ayarlayin.",
        allow_module_level=True,
    )

import torch
import torch.nn as nn

from model.dl.dataset import (
    MRIDataset,
    SINIF_ISIMLERI,
    create_dataloaders,
    get_transforms,
    kaynak_id_belirle,
    _validate_class_match,
)
from model.dl.engine import EarlyStopping, evaluate, train_one_epoch
from model.dl.models.unet_classifier import UNetClassifier
from model.inference import collect_batch_images, load_model, main as inference_main, predict_image


def _create_image(path: Path, value: int = 128) -> None:
    arr = np.full((32, 32), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _create_class_dataset(root: Path, per_class: int, prefix: str) -> None:
    for class_index, class_name in enumerate(SINIF_ISIMLERI):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_idx in range(per_class):
            value = 40 + (class_index * 30) + image_idx
            _create_image(class_dir / f"{prefix}_{class_index}_{image_idx}.jpg", value=value)


def _create_grouped_class_dataset(
    root: Path,
    groups_per_class: int,
    copies_per_group: int,
    prefix: str,
) -> None:
    for class_index, class_name in enumerate(SINIF_ISIMLERI):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for group_idx in range(groups_per_class):
            base_name = f"{prefix}_{class_index}_{group_idx}"
            for copy_idx in range(copies_per_group):
                value = 40 + (class_index * 30) + group_idx + copy_idx
                suffix = ".jpg" if copy_idx == 0 else f" ({copy_idx}).jpg"
                _create_image(class_dir / f"{base_name}{suffix}", value=value)


def test_train_one_epoch_returns_metrics_and_updates_weights():
    model = nn.Linear(2, 2)
    before = model.weight.detach().clone()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        (
            torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
    ]

    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert set(metrics) == {"loss", "accuracy", "precision", "recall", "f1"}
    assert metrics["loss"] >= 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert not torch.allclose(before, model.weight.detach())


def test_evaluate_returns_numpy_predictions_and_labels():
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    loader = [
        (
            torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
    ]

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))

    assert set(metrics) == {"loss", "accuracy", "precision", "recall", "f1", "preds", "labels"}
    assert metrics["preds"].shape == (2,)
    assert metrics["labels"].shape == (2,)


def test_early_stopping_patience_and_reset():
    stopper = EarlyStopping(patience=2, min_delta=0.01)

    assert stopper(1.00) is False
    assert stopper.counter == 0

    assert stopper(0.80) is False
    assert stopper.counter == 0

    assert stopper(0.805) is False
    assert stopper.counter == 1

    assert stopper(0.79) is False
    assert stopper.counter == 0

    assert stopper(0.795) is False
    assert stopper(0.796) is True
    assert stopper.should_stop is True


def test_mri_dataset_returns_rgb_image_without_transform(tmp_path):
    image_path = tmp_path / "sample.png"
    _create_image(image_path, value=100)
    dataset = MRIDataset([image_path], [3])

    image, label = dataset[0]

    assert image.mode == "RGB"
    assert image.size == (32, 32)
    assert label == 3


def test_get_transforms_train_and_eval_produce_expected_tensor_shape():
    image = Image.fromarray(np.full((18, 25), 150, dtype=np.uint8), mode="L").convert("RGB")

    train_tensor = get_transforms(image_size=64, is_train=True)(image)
    eval_tensor = get_transforms(image_size=64, is_train=False)(image)

    assert tuple(train_tensor.shape) == (3, 64, 64)
    assert tuple(eval_tensor.shape) == (3, 64, 64)


def test_validate_class_match_mismatch_raises(tmp_path):
    trainval_dir = tmp_path / "trainval"
    test_dir = tmp_path / "test"
    (trainval_dir / "NonDemented").mkdir(parents=True)
    (test_dir / "MildDemented").mkdir(parents=True)

    with pytest.raises(ValueError, match="Sinif isimleri eslesmiyor"):
        _validate_class_match(trainval_dir, test_dir)


def test_validate_class_match_beklenmeyen_klasoru_reddeder(tmp_path):
    trainval_dir = tmp_path / "trainval"
    test_dir = tmp_path / "test"
    for class_name in SINIF_ISIMLERI:
        (trainval_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
    (trainval_dir / "UnknownClass").mkdir()

    with pytest.raises(ValueError, match="beklenmeyen"):
        _validate_class_match(trainval_dir, test_dir)


def test_create_dataloaders_builds_leak_free_splits(tmp_path):
    trainval_dir = tmp_path / "trainval"
    _create_grouped_class_dataset(trainval_dir, groups_per_class=4, copies_per_group=2, prefix="train")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=None,
        batch_size=2,
        image_size=32,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
        num_workers=0,
    )

    assert info["train_size"] + info["val_size"] + info["test_size"] == 24
    assert info["test_size"] == 4
    assert info["split_strategy"] == "group_stratified"
    assert len(train_loader.dataset) == info["train_size"]
    assert len(val_loader.dataset) == info["val_size"]
    assert len(test_loader.dataset) == info["test_size"]

    train_sources = {kaynak_id_belirle(path.name) for path in train_loader.dataset.image_paths}
    val_sources = {kaynak_id_belirle(path.name) for path in val_loader.dataset.image_paths}
    test_sources = {kaynak_id_belirle(path.name) for path in test_loader.dataset.image_paths}
    assert train_sources.isdisjoint(val_sources)
    assert train_sources.isdisjoint(test_sources)
    assert val_sources.isdisjoint(test_sources)
    assert all("(" not in path.name for path in val_loader.dataset.image_paths)
    assert all("(" not in path.name for path in test_loader.dataset.image_paths)


def test_create_dataloaders_sets_seeded_generators(tmp_path):
    trainval_dir = tmp_path / "trainval"
    _create_grouped_class_dataset(trainval_dir, groups_per_class=4, copies_per_group=2, prefix="train")

    train_loader, val_loader, test_loader, _info = create_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=None,
        batch_size=2,
        image_size=32,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
        num_workers=0,
    )

    assert train_loader.worker_init_fn is not None
    assert val_loader.worker_init_fn is not None
    assert test_loader.worker_init_fn is not None
    assert train_loader.generator.initial_seed() == 123
    assert val_loader.generator.initial_seed() == 124
    assert test_loader.generator.initial_seed() == 125


def test_create_dataloaders_warns_when_grouping_cannot_be_inferred(tmp_path):
    trainval_dir = tmp_path / "trainval"
    _create_class_dataset(trainval_dir, per_class=4, prefix="train")

    _, _, _, info = create_dataloaders(
        trainval_dir=trainval_dir,
        test_dir=None,
        batch_size=2,
        image_size=32,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
        num_workers=0,
    )

    assert info["split_strategy"] == "stratified_without_groups"
    assert info["trainval_grouping"]["grouping_reliable"] is False
    assert info["split_warnings"] == []


def test_create_dataloaders_rejects_trainval_test_source_overlap(tmp_path):
    trainval_dir = tmp_path / "trainval"
    test_dir = tmp_path / "test"
    _create_class_dataset(trainval_dir, per_class=2, prefix="shared")
    _create_class_dataset(test_dir, per_class=1, prefix="shared")

    with pytest.raises(ValueError, match="ortak kaynak grup"):
        create_dataloaders(
            trainval_dir=trainval_dir,
            test_dir=test_dir,
            batch_size=2,
            image_size=32,
            val_ratio=0.5,
            seed=42,
            num_workers=0,
        )


def test_unet_classifier_forward_shape():
    model = UNetClassifier(num_classes=4)
    batch = torch.randn(2, 3, 64, 64)

    output = model(batch)

    assert output.shape == (2, 4)


def test_load_model_uses_checkpoint_metadata(monkeypatch, tmp_path):
    class DummyModel(nn.Module):
        def __init__(self, num_classes=4, pretrained=False):
            super().__init__()
            self.num_classes = num_classes
            self.pretrained = pretrained
            self.loaded_state = None
            self.eval_called = False

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    monkeypatch.setattr("model.inference.ResNetClassifier", DummyModel)

    checkpoint_path = tmp_path / "dummy_resnet.pt"
    torch.save(
        {
            "model_name": "resnet",
            "num_classes": 3,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "image_size": 96,
            "class_names": ["A", "B", "C"],
        },
        checkpoint_path,
    )

    model, image_size, class_names = load_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(model, DummyModel)
    assert model.num_classes == 3
    assert torch.equal(model.loaded_state["weight"], torch.tensor([1.0]))
    assert model.eval_called is True
    assert image_size == 96
    assert class_names == ["A", "B", "C"]


def test_predict_image_returns_ranked_probabilities(tmp_path):
    class FixedModel(nn.Module):
        def forward(self, tensor):
            batch_size = tensor.shape[0]
            logits = torch.tensor([[0.1, 0.2, 2.4]], dtype=torch.float32)
            return logits.repeat(batch_size, 1)

    image_path = tmp_path / "input.png"
    Image.fromarray(np.full((20, 20), 180, dtype=np.uint8), mode="L").save(image_path)

    result = predict_image(
        model=FixedModel(),
        image_path=image_path,
        image_size=32,
        class_names=["A", "B", "C"],
        device=torch.device("cpu"),
    )

    assert result["tahmin_sinif"] == 2
    assert result["tahmin_adi"] == "C"
    assert abs(sum(result["olasiliklar"].values()) - 1.0) < 1e-6


def test_collect_batch_images_uppercase_uzantilari_da_toplar(tmp_path):
    image_upper = tmp_path / "sample.JPG"
    image_lower = tmp_path / "sample2.png"
    text_file = tmp_path / "notes.txt"
    _create_image(image_upper, value=110)
    _create_image(image_lower, value=120)
    text_file.write_text("ignore", encoding="utf-8")

    images = collect_batch_images(tmp_path)

    assert images == sorted([image_upper, image_lower])


def test_inference_main_gecersiz_checkpoint_icin_temiz_hata_verir(capsys, monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "bad_model.pt"
    checkpoint_path.write_bytes(b"not-a-checkpoint")

    monkeypatch.setattr("model.inference.get_device", lambda: torch.device("cpu"))

    result = inference_main(
        ["--model-path", str(checkpoint_path), "--batch", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "[HATA]" in captured.out
    assert "Checkpoint guvenli modda yuklenemedi" in captured.out
