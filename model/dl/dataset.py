#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py
----------
MRI goruntuleri icin PyTorch Dataset ve DataLoader olusturma.
Sinif klasorlerinden (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
goruntuleri okur, kaynak-grup sizintisini engelleyerek train/val/test olarak boler.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SINIF_ISIMLERI = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SINIF_ETIKETI = {name: idx for idx, name in enumerate(SINIF_ISIMLERI)}
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png"}


class MRIDataset(Dataset):
    """MRI goruntu siniflandirma veri seti."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Egitim veya degerlendirme icin goruntu donusumleri."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def kaynak_id_belirle(dosya_adi: str) -> str:
    """Augmentasyon kopyalarini ayni kaynaga baglamak icin kaynak ID cikar."""
    stem = Path(str(dosya_adi)).stem
    stem = re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return stem


def collect_images(data_dir: Path) -> Tuple[List[Path], List[int], List[str]]:
    """Sinif klasorlerinden goruntu yollari, etiketleri ve kaynak grup anahtarlarini topla."""
    image_paths: List[Path] = []
    labels: List[int] = []
    groups: List[str] = []

    for class_name, label in SINIF_ETIKETI.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  [UYARI] Sinif klasoru bulunamadi: {class_dir}")
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in GORUNTU_UZANTILARI:
                image_paths.append(img_file)
                labels.append(label)
                # Sinif + kaynak id birlikte tutulur; ayni kaynagin turevleri ayni split'te kalir.
                groups.append(f"{class_name}::{kaynak_id_belirle(img_file.name)}")

    return image_paths, labels, groups


def _validate_class_match(trainval_dir: Path, test_dir: Path) -> None:
    """İki dizindeki sınıf klasörlerinin eşleştiğini doğrula."""
    tv_classes = {d.name for d in trainval_dir.iterdir() if d.is_dir()} & set(SINIF_ISIMLERI)
    te_classes = {d.name for d in test_dir.iterdir() if d.is_dir()} & set(SINIF_ISIMLERI)
    if not tv_classes:
        raise FileNotFoundError(
            f"Trainval dizininde bilinen sinif klasoru yok: {trainval_dir}"
        )
    if not te_classes:
        raise FileNotFoundError(
            f"Test dizininde bilinen sinif klasoru yok: {test_dir}"
        )
    if tv_classes != te_classes:
        raise ValueError(
            f"Sinif isimleri eslesmiyor!\n"
            f"  Trainval ({trainval_dir}): {sorted(tv_classes)}\n"
            f"  Test     ({test_dir}):     {sorted(te_classes)}"
        )


def _validate_dataset_separation(
    trainval_groups: List[str],
    test_groups: List[str],
    trainval_dir: Path,
    test_dir: Path,
) -> None:
    """Train/val ve test veri kaynaklarinin ayrik oldugunu dogrula."""
    overlap = sorted(set(trainval_groups) & set(test_groups))
    if not overlap:
        return

    sample = ", ".join(overlap[:5])
    raise ValueError(
        "Train/validation icin augmented, test icin original veri bekleniyor; "
        "iki dizin arasinda ortak kaynak grup bulundu.\n"
        f"  TrainVal: {trainval_dir}\n"
        f"  Test    : {test_dir}\n"
        f"  Ortak grup sayisi: {len(overlap)}\n"
        f"  Ornekler: {sample}"
    )


def _group_stratified_train_val_split(
    labels: List[int],
    groups: List[str],
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> Tuple[List[int], List[int]]:
    """
    Grup sizintisini engelleyerek (ayni kaynak ayni split'te) yaklasik
    stratified train/val bolme yap. Greedy atama.
    """
    train_ratio = 1.0 - val_ratio
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio 0 ile 1 arasinda olmali.")

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, grp in enumerate(groups):
        group_to_indices[grp].append(idx)

    if len(group_to_indices) < 2:
        raise ValueError(
            f"Leak-free bolme icin en az 2 kaynak grup gerekli (bulunan: {len(group_to_indices)})."
        )

    class_to_groups: Dict[int, set[str]] = defaultdict(set)
    for label, grp in zip(labels, groups):
        class_to_groups[label].add(grp)

    insufficient_classes = [
        SINIF_ISIMLERI[class_id]
        for class_id in range(num_classes)
        if len(class_to_groups[class_id]) < 2
    ]
    if insufficient_classes:
        joined = ", ".join(insufficient_classes)
        raise ValueError(
            "Leak-free train/val bolme icin her sinifta en az 2 farkli kaynak grup gerekli. "
            f"Eksik siniflar: {joined}"
        )

    group_items = []
    for grp, idxs in group_to_indices.items():
        cls_counts = np.zeros(num_classes, dtype=np.int64)
        for idx in idxs:
            cls_counts[labels[idx]] += 1
        group_items.append((grp, idxs, cls_counts))

    rng = np.random.default_rng(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: (len(item[1]), int(item[2].max())), reverse=True)

    target_ratios = np.array([train_ratio, val_ratio], dtype=np.float64)
    total_samples = float(len(labels))
    target_samples = target_ratios * total_samples
    total_class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    target_class_counts = target_ratios[:, None] * total_class_counts[None, :]

    split_groups: List[list] = [[], []]
    split_sample_counts = np.zeros(2, dtype=np.float64)
    split_class_counts = np.zeros((2, num_classes), dtype=np.float64)

    for grp, idxs, cls_counts in group_items:
        best_split = None
        best_cost = None
        for split_id in rng.permutation(2):
            new_sample_counts = split_sample_counts.copy()
            new_class_counts = split_class_counts.copy()
            new_sample_counts[split_id] += len(idxs)
            new_class_counts[split_id] += cls_counts

            size_error = np.mean(
                ((new_sample_counts - target_samples) / (target_samples + 1e-6)) ** 2
            )
            class_error = np.mean(
                ((new_class_counts - target_class_counts) / (target_class_counts + 1e-6)) ** 2
            )
            overfill = np.maximum(0.0, new_sample_counts - target_samples * 1.25)
            overfill_penalty = float(np.sum(overfill) / (total_samples + 1e-6))
            cost = (3.0 * class_error) + size_error + overfill_penalty

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = int(split_id)

        split_groups[best_split].append((grp, idxs, cls_counts))
        split_sample_counts[best_split] += len(idxs)
        split_class_counts[best_split] += cls_counts

    for split_id in range(2):
        if split_groups[split_id]:
            continue
        donor = 1 - split_id
        if len(split_groups[donor]) < 2:
            raise RuntimeError("Splitler dengelenemedi: bos split olustu.")
        donor_group_idx = int(np.argmin([len(item[1]) for item in split_groups[donor]]))
        moved = split_groups[donor].pop(donor_group_idx)
        split_groups[split_id].append(moved)

    def _move_smallest_group_with_class(source_id: int, target_id: int, class_id: int) -> bool:
        candidate_idxs = [
            idx for idx, item in enumerate(split_groups[source_id])
            if item[2][class_id] > 0
        ]
        if len(candidate_idxs) < 2:
            return False

        donor_group_idx = min(candidate_idxs, key=lambda idx: len(split_groups[source_id][idx][1]))
        moved = split_groups[source_id].pop(donor_group_idx)
        split_groups[target_id].append(moved)
        return True

    for class_id in range(num_classes):
        split_class_totals = [
            int(sum(item[2][class_id] for item in split_groups[split_id]))
            for split_id in range(2)
        ]
        if split_class_totals[0] > 0 and split_class_totals[1] > 0:
            continue

        missing_split = 0 if split_class_totals[0] == 0 else 1
        donor_split = 1 - missing_split
        moved = _move_smallest_group_with_class(donor_split, missing_split, class_id)
        if not moved:
            raise RuntimeError(
                "Leak-free split sonrasi her sinif train ve validation icinde temsil edilemedi. "
                f"Sorunlu sinif: {SINIF_ISIMLERI[class_id]}"
            )

    split_indices = []
    split_group_keys = []
    for items in split_groups:
        idxs = []
        keys = set()
        for grp, group_idxs, _ in items:
            idxs.extend(group_idxs)
            keys.add(grp)
        split_indices.append(idxs)
        split_group_keys.append(keys)

    if split_group_keys[0] & split_group_keys[1]:
        raise RuntimeError("Train/Val arasinda kaynak grup sizintisi tespit edildi.")

    for split_name, idxs in (("train", split_indices[0]), ("validation", split_indices[1])):
        class_counts = np.bincount([labels[idx] for idx in idxs], minlength=num_classes)
        missing_classes = [
            SINIF_ISIMLERI[class_id]
            for class_id, count in enumerate(class_counts)
            if count == 0
        ]
        if missing_classes:
            joined = ", ".join(missing_classes)
            raise RuntimeError(
                f"{split_name} split'inde sinif kapsami eksik kaldi: {joined}"
            )

    return split_indices[0], split_indices[1]


def create_dataloaders(
    trainval_dir: Path,
    test_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Train, validation ve test DataLoader'lari olustur.

    - Train/Val goruntulerini ``trainval_dir`` (augmented) icerisinden toplar.
    - Test goruntuleri ``test_dir`` (original) icerisinden alinir; split yapilmaz.
    """
    _validate_class_match(trainval_dir, test_dir)

    # --- Augmented: train + val ---
    tv_paths, tv_labels, tv_groups = collect_images(trainval_dir)
    if len(tv_paths) == 0:
        raise FileNotFoundError(f"Trainval verisi bulunamadi: {trainval_dir}")

    paths_test, labels_test, test_groups = collect_images(test_dir)
    if len(paths_test) == 0:
        raise FileNotFoundError(f"Test verisi bulunamadi: {test_dir}")

    _validate_dataset_separation(tv_groups, test_groups, trainval_dir, test_dir)

    print(f"  [TrainVal] Kaynak: {trainval_dir}")
    print(f"  [TrainVal] Toplam goruntu: {len(tv_paths)}")
    for name, lbl in SINIF_ETIKETI.items():
        print(f"    {name}: {tv_labels.count(lbl)}")
    print(f"  [TrainVal] Kaynak grup sayisi: {len(set(tv_groups))}")

    train_idxs, val_idxs = _group_stratified_train_val_split(
        labels=tv_labels,
        groups=tv_groups,
        val_ratio=val_ratio,
        seed=seed,
        num_classes=len(SINIF_ISIMLERI),
    )

    paths_train = [tv_paths[i] for i in train_idxs]
    labels_train = [tv_labels[i] for i in train_idxs]
    paths_val = [tv_paths[i] for i in val_idxs]
    labels_val = [tv_labels[i] for i in val_idxs]

    print(f"  [Test] Kaynak: {test_dir}")
    print(f"  [Test] Toplam goruntu: {len(paths_test)}")
    for name, lbl in SINIF_ETIKETI.items():
        print(f"    {name}: {labels_test.count(lbl)}")

    # --- Dataset ve DataLoader ---
    train_ds = MRIDataset(paths_train, labels_train, get_transforms(image_size, is_train=True))
    val_ds = MRIDataset(paths_val, labels_val, get_transforms(image_size, is_train=False))
    test_ds = MRIDataset(paths_test, labels_test, get_transforms(image_size, is_train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    info = {
        "num_classes": len(SINIF_ISIMLERI),
        "class_names": SINIF_ISIMLERI,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_labels": labels_train,
        "val_labels": labels_val,
        "test_labels": labels_test,
        "train_groups": len({tv_groups[i] for i in train_idxs}),
        "val_groups": len({tv_groups[i] for i in val_idxs}),
        "trainval_dir": str(trainval_dir),
        "test_dir": str(test_dir),
        "val_ratio": val_ratio,
    }

    return train_loader, val_loader, test_loader, info
