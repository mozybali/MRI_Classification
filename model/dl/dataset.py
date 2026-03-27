#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py
----------
MRI goruntuleri icin PyTorch Dataset ve DataLoader olusturma.
Sinif klasorlerinden (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
goruntuleri okur; dosya adindan kaynak grup cikarilabiliyorsa leak-free bolme uygular,
aksi halde uyari ile stratified fallback kullanir.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import re
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SINIF_ISIMLERI = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SINIF_ETIKETI = {name: idx for idx, name in enumerate(SINIF_ISIMLERI)}
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png"}


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker'larinda tekrar uretilebilir RNG durumu kur."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def _augmentasyon_kopyasi_mi(dosya_adi: str) -> bool:
    """Dosya adinin augment/turev kopya olup olmadigini belirle."""
    stem = Path(str(dosya_adi)).stem
    return stem != kaynak_id_belirle(dosya_adi)


def collect_images(data_dir: Path) -> Tuple[List[Path], List[int], List[str], List[bool]]:
    """Sinif klasorlerinden goruntu yollari, etiketleri ve kaynak grup anahtarlarini topla."""
    image_paths: List[Path] = []
    labels: List[int] = []
    groups: List[str] = []
    augmented_flags: List[bool] = []

    for class_name, label in SINIF_ETIKETI.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  [UYARI] Sinif klasoru bulunamadi: {class_dir}")
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in GORUNTU_UZANTILARI:
                image_paths.append(img_file)
                labels.append(label)
                groups.append(f"{class_name}::{kaynak_id_belirle(img_file.name)}")
                augmented_flags.append(_augmentasyon_kopyasi_mi(img_file.name))

    return image_paths, labels, groups, augmented_flags


def _summarize_grouping(groups: List[str], augmented_flags: List[bool]) -> Dict[str, Any]:
    """Dosya adindan uretilen kaynak grup kapsamini ozetle."""
    counts = Counter(groups)
    total_images = len(groups)
    multi_group_count = sum(1 for count in counts.values() if count > 1)
    images_in_multi_groups = sum(count for count in counts.values() if count > 1)
    unique_groups = len(counts)
    augmented_samples = int(sum(1 for flag in augmented_flags if flag))

    return {
        "total_images": total_images,
        "unique_groups": unique_groups,
        "multi_group_count": multi_group_count,
        "singleton_group_count": unique_groups - multi_group_count,
        "images_in_multi_groups": images_in_multi_groups,
        "multi_group_coverage": (
            images_in_multi_groups / total_images if total_images else 0.0
        ),
        "augmented_samples": augmented_samples,
        "original_samples": total_images - augmented_samples,
        "grouping_reliable": multi_group_count > 0,
    }


def _collect_class_dirs(data_dir: Path) -> set[str]:
    """Veri dizinindeki tum sinif klasorlerini topla."""
    return {entry.name for entry in data_dir.iterdir() if entry.is_dir()}


def _missing_class_names(labels: List[int], num_classes: int) -> List[str]:
    """Etiket listesinde hic temsil edilmeyen siniflari dondur."""
    class_counts = np.bincount(labels, minlength=num_classes)
    return [
        SINIF_ISIMLERI[class_id]
        for class_id, count in enumerate(class_counts)
        if count == 0
    ]


def _validate_expected_classes(data_dir: Path, split_name: str) -> None:
    """Veri dizinindeki sinif klasorlerinin beklenen yapida oldugunu dogrula."""
    expected_classes = set(SINIF_ISIMLERI)
    classes = _collect_class_dirs(data_dir)
    if not classes:
        raise FileNotFoundError(
            f"{split_name} dizininde sinif klasoru yok: {data_dir}"
        )

    missing = sorted(expected_classes - classes)
    unexpected = sorted(classes - expected_classes)
    if missing or unexpected:
        raise ValueError(
            f"{split_name} sinif isimleri eslesmiyor!\n"
            f"  {data_dir}: mevcut={sorted(classes)} | "
            f"eksik={missing or ['yok']} | "
            f"beklenmeyen={unexpected or ['yok']}"
        )


def _validate_class_match(trainval_dir: Path, test_dir: Path) -> None:
    """Iki dizindeki sinif klasorlerinin beklenen yapida oldugunu dogrula."""
    expected_classes = set(SINIF_ISIMLERI)
    tv_classes = _collect_class_dirs(trainval_dir)
    te_classes = _collect_class_dirs(test_dir)
    if not tv_classes:
        raise FileNotFoundError(
            f"Trainval dizininde sinif klasoru yok: {trainval_dir}"
        )
    if not te_classes:
        raise FileNotFoundError(
            f"Test dizininde sinif klasoru yok: {test_dir}"
        )

    def _format_details(root: Path, classes: set[str]) -> str:
        missing = sorted(expected_classes - classes)
        unexpected = sorted(classes - expected_classes)
        return (
            f"  {root}: mevcut={sorted(classes)} | "
            f"eksik={missing or ['yok']} | "
            f"beklenmeyen={unexpected or ['yok']}"
        )

    if tv_classes != expected_classes or te_classes != expected_classes:
        raise ValueError(
            f"Sinif isimleri eslesmiyor!\n"
            f"{_format_details(trainval_dir, tv_classes)}\n"
            f"{_format_details(test_dir, te_classes)}"
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
        "Train/validation kaynagi ile harici test dizini arasinda ortak kaynak grup bulundu.\n"
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
    require_all_classes_in_each_split: bool = False,
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
        if not moved and require_all_classes_in_each_split:
            raise RuntimeError(
                "Leak-free split sonrasi sinif kapsami eksik kaldi. "
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

    if require_all_classes_in_each_split:
        for split_name, idxs in (("train", split_indices[0]), ("validation", split_indices[1])):
            missing_classes = _missing_class_names([labels[idx] for idx in idxs], num_classes)
            if missing_classes:
                joined = ", ".join(missing_classes)
                raise RuntimeError(
                    f"{split_name} split'inde sinif kapsami eksik kaldi: {joined}"
                )

    return split_indices[0], split_indices[1]


def _stratified_train_val_split(
    labels: List[int],
    val_ratio: float,
    seed: int,
    num_classes: int,
    require_all_classes_in_each_split: bool = False,
) -> Tuple[List[int], List[int]]:
    """Kaynak grup cikarilamadiginda sinif-dengeli train/val bolmesi yap."""
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio 0 ile 1 arasinda olmali.")

    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    train_idxs: List[int] = []
    val_idxs: List[int] = []

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels_arr == class_id)
        if len(class_indices) == 0:
            continue

        shuffled = rng.permutation(class_indices)
        if len(shuffled) == 1:
            if require_all_classes_in_each_split:
                raise RuntimeError(
                    f"validation split'inde sinif kapsami eksik kaldi: {SINIF_ISIMLERI[class_id]}"
                )
            train_idxs.extend(int(idx) for idx in shuffled)
            continue

        val_count = int(round(len(shuffled) * val_ratio))
        val_count = max(1, min(len(shuffled) - 1, val_count))

        val_idxs.extend(int(idx) for idx in shuffled[:val_count])
        train_idxs.extend(int(idx) for idx in shuffled[val_count:])

    if not train_idxs or not val_idxs:
        raise RuntimeError("Train/Val bolmesi olusturulamadi: splitlerden biri bos kaldi.")

    if require_all_classes_in_each_split:
        for split_name, idxs in (("train", train_idxs), ("validation", val_idxs)):
            missing_classes = _missing_class_names([labels[idx] for idx in idxs], num_classes)
            if missing_classes:
                joined = ", ".join(missing_classes)
                raise RuntimeError(
                    f"{split_name} split'inde sinif kapsami eksik kaldi: {joined}"
                )

    return train_idxs, val_idxs


def _build_group_level_records(
    labels: List[int],
    groups: List[str],
) -> tuple[list[int], list[str]]:
    """Her kaynak grup icin tek bir etiket ve anahtar listesi olustur."""
    seen: Dict[str, int] = {}
    ordered_groups: list[str] = []
    ordered_labels: list[int] = []

    for label, group in zip(labels, groups):
        if group in seen:
            if seen[group] != label:
                raise RuntimeError(f"Ayni kaynak grup icin tutarsiz etiket bulundu: {group}")
            continue
        seen[group] = label
        ordered_groups.append(group)
        ordered_labels.append(label)

    return ordered_labels, ordered_groups


def _split_group_keys(
    labels: List[int],
    groups: List[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    use_group_split: bool,
) -> tuple[set[str], set[str], set[str], str]:
    """Kaynak grup anahtarlarini train/val/test olarak bol."""
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio 0 ile 1 arasinda olmali.")
    if test_ratio < 0 or test_ratio >= 1.0:
        raise ValueError("test_ratio 0 ile 1 arasinda olmali.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio 1'den kucuk olmali.")

    group_labels, group_keys = _build_group_level_records(labels, groups)
    eval_ratio = val_ratio + test_ratio
    split_fn = _group_stratified_train_val_split if use_group_split else _stratified_train_val_split
    split_strategy = "group_stratified" if use_group_split else "stratified_without_groups"

    train_group_idxs, temp_group_idxs = split_fn(
        labels=group_labels,
        groups=group_keys,
        val_ratio=eval_ratio,
        seed=seed,
        num_classes=len(SINIF_ISIMLERI),
    ) if use_group_split else split_fn(
        labels=group_labels,
        val_ratio=eval_ratio,
        seed=seed,
        num_classes=len(SINIF_ISIMLERI),
    )

    train_groups = {group_keys[idx] for idx in train_group_idxs}
    temp_labels = [group_labels[idx] for idx in temp_group_idxs]
    temp_keys = [group_keys[idx] for idx in temp_group_idxs]

    if test_ratio == 0:
        val_groups = set(temp_keys)
        test_groups: set[str] = set()
        return train_groups, val_groups, test_groups, split_strategy

    temp_test_ratio = test_ratio / eval_ratio
    val_rel_idxs, test_rel_idxs = split_fn(
        labels=temp_labels,
        groups=temp_keys,
        val_ratio=temp_test_ratio,
        seed=seed + 1,
        num_classes=len(SINIF_ISIMLERI),
    ) if use_group_split else split_fn(
        labels=temp_labels,
        val_ratio=temp_test_ratio,
        seed=seed + 1,
        num_classes=len(SINIF_ISIMLERI),
    )

    val_groups = {temp_keys[idx] for idx in val_rel_idxs}
    test_groups = {temp_keys[idx] for idx in test_rel_idxs}
    return train_groups, val_groups, test_groups, split_strategy


def _indices_for_groups(
    groups: List[str],
    augmented_flags: List[bool],
    selected_groups: set[str],
    *,
    original_only: bool,
) -> List[int]:
    """Secilen kaynak gruplari icin ornek indekslerini dondur."""
    return [
        idx for idx, group in enumerate(groups)
        if group in selected_groups and (not original_only or not augmented_flags[idx])
    ]


def create_dataloaders(
    trainval_dir: Path,
    test_dir: Path | None,
    batch_size: int = 32,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    include_test: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, Dict]:
    """
    Train, validation ve test DataLoader'lari olustur.

    - Varsayilan akista tum split'ler tek kaynaktan olusturulur.
    - Train split'i ayni kaynagin augment/turev kopyalarini kullanabilir.
    - Validation ve test split'leri yalnizca original goruntulerden kurulur.
    - ``test_dir`` verilirse harici/original test dizini olarak kullanilir.
    """
    if include_test:
        if test_ratio <= 0 or test_ratio >= 1.0:
            raise ValueError("test_ratio 0 ile 1 arasinda olmali.")
    elif test_ratio < 0 or test_ratio >= 1.0:
        raise ValueError("test_ratio 0 ile 1 arasinda olmali.")
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio 0 ile 1 arasinda olmali.")

    _validate_expected_classes(trainval_dir, "Trainval")
    using_external_test = bool(
        include_test
        and test_dir is not None
        and test_dir.resolve() != trainval_dir.resolve()
    )
    if using_external_test and test_dir is not None:
        _validate_class_match(trainval_dir, test_dir)

    tv_paths, tv_labels, tv_groups, tv_augmented = collect_images(trainval_dir)
    if len(tv_paths) == 0:
        raise FileNotFoundError(f"Trainval verisi bulunamadi: {trainval_dir}")
    tv_group_stats = _summarize_grouping(tv_groups, tv_augmented)

    paths_test: List[Path] = []
    labels_test: List[int] = []
    test_group_stats: Dict[str, Any] | None = None
    test_groups: List[str] = []
    test_augmented: List[bool] = []
    if using_external_test and test_dir is not None:
        paths_test, labels_test, test_groups, test_augmented = collect_images(test_dir)
        if len(paths_test) == 0:
            raise FileNotFoundError(f"Test verisi bulunamadi: {test_dir}")
        test_group_stats = _summarize_grouping(test_groups, test_augmented)
        _validate_dataset_separation(tv_groups, test_groups, trainval_dir, test_dir)

    print(f"  [TrainVal] Kaynak: {trainval_dir}")
    print(f"  [TrainVal] Toplam goruntu: {len(tv_paths)}")
    for name, lbl in SINIF_ETIKETI.items():
        print(f"    {name}: {tv_labels.count(lbl)}")
    print(f"  [TrainVal] Kaynak grup sayisi: {tv_group_stats['unique_groups']}")

    split_warnings = []
    if tv_group_stats["grouping_reliable"]:
        use_group_split = True
    else:
        use_group_split = False
        if tv_group_stats["augmented_samples"] > 0:
            split_warnings.append(
                "TrainVal dosya adlarindan tekrarli kaynak grup cikarilamadi; "
                "stratified split kullanildi ve augment turevleri icin leak-free garanti verilemiyor."
            )

    internal_test_ratio = 0.0 if using_external_test or not include_test else test_ratio
    train_group_keys, val_group_keys, test_group_keys_internal, split_strategy = _split_group_keys(
        labels=tv_labels,
        groups=tv_groups,
        val_ratio=val_ratio,
        test_ratio=internal_test_ratio,
        seed=seed,
        use_group_split=use_group_split,
    )

    train_idxs = _indices_for_groups(tv_groups, tv_augmented, train_group_keys, original_only=False)
    val_idxs = _indices_for_groups(tv_groups, tv_augmented, val_group_keys, original_only=True)
    internal_test_idxs = _indices_for_groups(
        tv_groups,
        tv_augmented,
        test_group_keys_internal,
        original_only=True,
    )

    paths_train = [tv_paths[i] for i in train_idxs]
    labels_train = [tv_labels[i] for i in train_idxs]
    paths_val = [tv_paths[i] for i in val_idxs]
    labels_val = [tv_labels[i] for i in val_idxs]
    train_missing_classes = _missing_class_names(labels_train, len(SINIF_ISIMLERI))
    val_missing_classes = _missing_class_names(labels_val, len(SINIF_ISIMLERI))
    if train_missing_classes:
        raise RuntimeError(f"Train split'inde eksik siniflar: {', '.join(train_missing_classes)}")
    if val_missing_classes:
        raise RuntimeError(f"Validation split'inde eksik siniflar: {', '.join(val_missing_classes)}")

    if using_external_test and test_dir is not None:
        external_test_idxs = _indices_for_groups(
            test_groups,
            test_augmented,
            set(test_groups),
            original_only=True,
        )
        paths_test = [paths_test[i] for i in external_test_idxs]
        labels_test = [labels_test[i] for i in external_test_idxs]
        if not paths_test:
            raise RuntimeError(
                "Harici test dizininde original goruntu bulunamadi; test split olusturulamiyor."
            )
        if _missing_class_names(labels_test, len(SINIF_ISIMLERI)):
            missing = _missing_class_names(labels_test, len(SINIF_ISIMLERI))
            raise RuntimeError(f"Test split'inde eksik siniflar: {', '.join(missing)}")
        print(f"  [Test] Kaynak: {test_dir}")
        print(f"  [Test] Toplam goruntu: {len(paths_test)}")
        for name, lbl in SINIF_ETIKETI.items():
            print(f"    {name}: {labels_test.count(lbl)}")
    elif include_test:
        paths_test = [tv_paths[i] for i in internal_test_idxs]
        labels_test = [tv_labels[i] for i in internal_test_idxs]
        if not paths_test:
            raise RuntimeError("Test split olusturulamadi; original test ornekleri bulunamadi.")
        test_missing_classes = _missing_class_names(labels_test, len(SINIF_ISIMLERI))
        if test_missing_classes:
            raise RuntimeError(f"Test split'inde eksik siniflar: {', '.join(test_missing_classes)}")
        print(f"  [Test] Kaynak: {trainval_dir} (internal split)")
        print(f"  [Test] Toplam goruntu: {len(paths_test)}")
        for name, lbl in SINIF_ETIKETI.items():
            print(f"    {name}: {labels_test.count(lbl)}")
    print(f"  [Split] Strateji: {split_strategy}")
    for warning in split_warnings:
        print(f"  [UYARI] {warning}")

    train_ds = MRIDataset(paths_train, labels_train, get_transforms(image_size, is_train=True))
    val_ds = MRIDataset(paths_val, labels_val, get_transforms(image_size, is_train=False))
    test_ds = (
        MRIDataset(paths_test, labels_test, get_transforms(image_size, is_train=False))
        if include_test
        else None
    )

    pin_memory = torch.cuda.is_available()

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=test_generator,
        )
        if test_ds is not None
        else None
    )

    info = {
        "num_classes": len(SINIF_ISIMLERI),
        "class_names": SINIF_ISIMLERI,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds) if test_ds is not None else 0,
        "train_labels": labels_train,
        "val_labels": labels_val,
        "test_labels": labels_test,
        "train_groups": len({tv_groups[i] for i in train_idxs}),
        "val_groups": len({tv_groups[i] for i in val_idxs}),
        "trainval_dir": str(trainval_dir),
        "test_dir": str(test_dir) if test_dir is not None else None,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio if include_test and not using_external_test else None,
        "split_strategy": split_strategy,
        "trainval_grouping": tv_group_stats,
        "test_grouping": test_group_stats,
        "split_warnings": split_warnings,
        "uses_external_test_dir": using_external_test,
    }

    return train_loader, val_loader, test_loader, info


def create_full_train_test_loaders(
    trainval_dir: Path,
    test_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Final model egitimi icin trainval'in tamami ve harici/original test diziniyle
    DataLoader'lar olustur.
    """
    _validate_expected_classes(trainval_dir, "Trainval")
    if test_dir.resolve() == trainval_dir.resolve():
        raise ValueError(
            "Full-trainval final egitim icin test dizini trainval'den farkli olmali."
        )
    _validate_class_match(trainval_dir, test_dir)

    tv_paths, tv_labels, tv_groups, tv_augmented = collect_images(trainval_dir)
    if len(tv_paths) == 0:
        raise FileNotFoundError(f"Trainval verisi bulunamadi: {trainval_dir}")
    tv_group_stats = _summarize_grouping(tv_groups, tv_augmented)

    test_paths, test_labels, test_groups, test_augmented = collect_images(test_dir)
    if len(test_paths) == 0:
        raise FileNotFoundError(f"Test verisi bulunamadi: {test_dir}")
    test_group_stats = _summarize_grouping(test_groups, test_augmented)
    _validate_dataset_separation(tv_groups, test_groups, trainval_dir, test_dir)

    train_idxs = _indices_for_groups(tv_groups, tv_augmented, set(tv_groups), original_only=False)
    test_idxs = _indices_for_groups(test_groups, test_augmented, set(test_groups), original_only=True)

    paths_train = [tv_paths[i] for i in train_idxs]
    labels_train = [tv_labels[i] for i in train_idxs]
    paths_test = [test_paths[i] for i in test_idxs]
    labels_test = [test_labels[i] for i in test_idxs]

    train_missing_classes = _missing_class_names(labels_train, len(SINIF_ISIMLERI))
    test_missing_classes = _missing_class_names(labels_test, len(SINIF_ISIMLERI))
    if train_missing_classes:
        raise RuntimeError(f"Train split'inde eksik siniflar: {', '.join(train_missing_classes)}")
    if test_missing_classes:
        raise RuntimeError(f"Test split'inde eksik siniflar: {', '.join(test_missing_classes)}")

    print(f"  [TrainVal] Kaynak: {trainval_dir}")
    print(f"  [TrainVal] Toplam goruntu: {len(paths_train)}")
    for name, lbl in SINIF_ETIKETI.items():
        print(f"    {name}: {labels_train.count(lbl)}")
    print(f"  [TrainVal] Kaynak grup sayisi: {tv_group_stats['unique_groups']}")
    print(f"  [Test] Kaynak: {test_dir}")
    print(f"  [Test] Toplam goruntu: {len(paths_test)}")
    for name, lbl in SINIF_ETIKETI.items():
        print(f"    {name}: {labels_test.count(lbl)}")
    print("  [Split] Strateji: full_trainval_external_test")

    train_ds = MRIDataset(paths_train, labels_train, get_transforms(image_size, is_train=True))
    test_ds = MRIDataset(paths_test, labels_test, get_transforms(image_size, is_train=False))

    pin_memory = torch.cuda.is_available()
    train_generator = torch.Generator().manual_seed(seed)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=test_generator,
    )

    info = {
        "num_classes": len(SINIF_ISIMLERI),
        "class_names": SINIF_ISIMLERI,
        "train_size": len(train_ds),
        "val_size": 0,
        "test_size": len(test_ds),
        "train_labels": labels_train,
        "val_labels": [],
        "test_labels": labels_test,
        "train_groups": len(set(tv_groups)),
        "val_groups": 0,
        "trainval_dir": str(trainval_dir),
        "test_dir": str(test_dir),
        "val_ratio": None,
        "test_ratio": None,
        "split_strategy": "full_trainval_external_test",
        "trainval_grouping": tv_group_stats,
        "test_grouping": test_group_stats,
        "split_warnings": [],
        "uses_external_test_dir": True,
        "full_trainval_run": True,
    }
    return train_loader, test_loader, info
