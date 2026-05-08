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
from typing import Iterator, Tuple, List, Dict, Any, Sequence
import hashlib
import json
import re
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..ayarlar import NORM_STATS_CACHE_KLASORU

SINIF_ISIMLERI = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SINIF_ETIKETI = {name: idx for idx, name in enumerate(SINIF_ISIMLERI)}
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png"}

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader olustururken persistent_workers/prefetch_factor'u guvenli ayarla.

    ``persistent_workers`` ve ``prefetch_factor`` yalnizca ``num_workers > 0``
    iken anlamlidir. ``num_workers=0`` iken bu argumanlari vermek PyTorch'tan
    hata aliriz; bu helper, koullara gore otomatik karar verir.
    """
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker'larinda tekrar uretilebilir RNG durumu kur."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Worker icindeki torchvision random transformlari da torch'un global RNG'sini
    # kullaniyor; numpy/random'a ek olarak torch RNG'sini de seed et.
    torch.manual_seed(worker_seed)


_STATS_CACHE_VERSION = 1


def _stats_cache_key(
    paths: Sequence[Path],
    image_size: int,
    max_samples: int | None,
    seed: int,
) -> str:
    """Stats cache anahtarini, path listesinin icerigine bagli olarak hashle.

    Path listesinin sirasi ve dosya adlari hash'e dahildir; ayni split + ayni
    image_size + ayni seed/max_samples icin cache hit garanti olur. Disk
    icerigi degisirse hash degismez; bu durumda cache'i bilerek invalide
    etmek icin cache klasorunu silmek yeterlidir.
    """
    hasher = hashlib.sha1()
    hasher.update(str(_STATS_CACHE_VERSION).encode("utf-8"))
    hasher.update(str(int(image_size)).encode("utf-8"))
    hasher.update(str(int(seed)).encode("utf-8"))
    hasher.update(str(max_samples if max_samples is not None else -1).encode("utf-8"))
    hasher.update(str(len(paths)).encode("utf-8"))
    for path in paths:
        hasher.update(str(path).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _load_stats_cache(cache_path: Path) -> tuple[
    Tuple[float, float, float], Tuple[float, float, float]
] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    mean = payload.get("mean")
    std = payload.get("std")
    if (
        not isinstance(mean, list) or len(mean) != 3
        or not isinstance(std, list) or len(std) != 3
    ):
        return None
    try:
        return (
            (float(mean[0]), float(mean[1]), float(mean[2])),
            (float(std[0]), float(std[1]), float(std[2])),
        )
    except (TypeError, ValueError):
        return None


def _save_stats_cache(
    cache_path: Path,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    *,
    image_size: int,
    sample_count: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _STATS_CACHE_VERSION,
        "image_size": int(image_size),
        "sample_count": int(sample_count),
        "mean": list(mean),
        "std": list(std),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError:
        pass


def compute_dataset_stats(
    image_paths: Sequence[Path],
    image_size: int,
    *,
    max_samples: int | None = 1024,
    seed: int = 42,
    cache_dir: Path | None = NORM_STATS_CACHE_KLASORU,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """RGB-replicated egitim goruntulerinden kanal bazli mean/std hesapla.

    Sonuclari ``[0, 1]`` araligindaki tensor degerleri uzerinden, yani
    ``transforms.ToTensor`` ciktisina uygun olcekte dondurur. ``cache_dir``
    verilirse path listesi + image_size + max_samples + seed kombinasyonu
    icin disk cache kullanilir; bu sayede HPO trial'lari arasi yeniden disk
    okuma elenir. Cache'i devre disi birakmak icin ``cache_dir=None``.
    """
    if not image_paths:
        raise ValueError("Stats hesaplamak icin gorseller bos olamaz.")

    paths = list(image_paths)

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_key = _stats_cache_key(paths, image_size, max_samples, seed)
        cache_path = cache_dir / f"{cache_key}.json"
        cached = _load_stats_cache(cache_path)
        if cached is not None:
            return cached

    sampled_paths = paths
    if max_samples is not None and len(sampled_paths) > max_samples:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(sampled_paths), size=max_samples, replace=False)
        sampled_paths = [sampled_paths[i] for i in sorted(idxs.tolist())]

    deterministic_ops = _resize_and_crop(image_size)

    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in sampled_paths:
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            for op in deterministic_ops:
                img = op(img)
            arr = np.asarray(img, dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        sum_ += flat.sum(axis=0)
        sum_sq += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    if pixel_count == 0:
        raise RuntimeError("Stats hesaplanamadi: bos goruntu listesi.")

    mean = sum_ / pixel_count
    var = np.maximum(sum_sq / pixel_count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)

    mean_tuple = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_tuple = (float(std[0]), float(std[1]), float(std[2]))

    if cache_path is not None:
        _save_stats_cache(
            cache_path,
            mean_tuple,
            std_tuple,
            image_size=image_size,
            sample_count=len(sampled_paths),
        )

    return mean_tuple, std_tuple


def _resize_and_crop(image_size: int) -> List[transforms.Compose]:
    """Aspect-ratio'yu koruyan resize + center crop akisi.

    ``transforms.Resize(image_size)`` (int) kisa kenari ``image_size``'a esitler;
    ardindan kare ``CenterCrop`` uzun kenardan kirpar. ``antialias=True`` torchvision
    surumleri arasi tutarlilik icin acikca verilir.
    """
    return [
        transforms.Resize(
            image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(image_size),
    ]


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


def get_transforms(
    image_size: int = 224,
    is_train: bool = True,
    hflip_p: float = 0.0,
    rotation_degrees: float = 10.0,
    color_jitter: float = 0.1,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Egitim veya degerlendirme icin goruntu donusumleri.

    Aspect-ratio koruyan resize + center crop kullanir; raw veri seti (orn.
    176x208 slice'lar) ile processed kare cikti arasinda tutarli sonuc verir.
    Normalize istatistikleri varsayilan olarak ImageNet'tir; pretrained=False
    egitimleri icin caller dataset-bazli mean/std gecebilir.
    """
    base_ops = _resize_and_crop(image_size)
    normalize = transforms.Normalize(mean=list(mean), std=list(std))
    if is_train:
        ops: list = list(base_ops)
        if hflip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=hflip_p))
        if rotation_degrees > 0:
            ops.append(transforms.RandomRotation(rotation_degrees))
        if color_jitter > 0:
            ops.append(transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter))
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)
    return transforms.Compose([*base_ops, transforms.ToTensor(), normalize])


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
    """Train/val ve test veri kaynaklarinin ayrik oldugunu dogrula.

    Tam grup anahtari (``"<class>::<kaynak_id>"``) ortakligi kontrol edilir:
    ayni denek ayni sinif altinda iki tarafta da gorunuyor olmamali.

    Cross-class stem kontrolu yapilmaz: bu veri setinde dosya numaralandirmasi
    her sinif klasorunde sifirdan basladigi icin ``27 (10).jpg`` gibi adlar
    farkli siniflarda farkli scan'leri temsil eder; (sinif, stem) cifti gercek
    kaynak kimligini olusturur.
    """
    overlap = sorted(set(trainval_groups) & set(test_groups))
    if overlap:
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

    if use_group_split:
        split_fn_kwargs = dict(labels=group_labels, groups=group_keys)
        split_strategy = "group_stratified"
    else:
        split_fn_kwargs = dict(labels=group_labels)
        split_strategy = "stratified_without_groups"

    split_fn = _group_stratified_train_val_split if use_group_split else _stratified_train_val_split

    train_group_idxs, temp_group_idxs = split_fn(
        **split_fn_kwargs,
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
    if use_group_split:
        temp_fn_kwargs = dict(labels=temp_labels, groups=temp_keys)
    else:
        temp_fn_kwargs = dict(labels=temp_labels)

    val_rel_idxs, test_rel_idxs = split_fn(
        **temp_fn_kwargs,
        val_ratio=temp_test_ratio,
        seed=seed + 1,
        num_classes=len(SINIF_ISIMLERI),
    )

    val_groups = {temp_keys[idx] for idx in val_rel_idxs}
    test_groups = {temp_keys[idx] for idx in test_rel_idxs}
    return train_groups, val_groups, test_groups, split_strategy


def _split_group_keys_kfold(
    labels: List[int],
    groups: List[str],
    *,
    n_folds: int,
    test_ratio: float,
    seed: int,
    use_group_split: bool,
) -> tuple[list[tuple[set[str], set[str]]], set[str], str]:
    """Trainval'i K stratified fold'a ayir; opsiyonel olarak once test grubunu cikar.

    Bolme her zaman grup seviyesinde calisir (her kaynak ID tek bir fold'un
    val'inde); bu sayede augment turevleri ve cross-fold sizinti engellenir.
    Sinif dengesi sklearn StratifiedKFold ile saglanir.

    Returns:
        fold_assignments: her fold icin (train_group_keys, val_group_keys) ciftleri
        test_groups: test_ratio>0 ise ayrilan test kaynak gruplari, aksi halde bos
        split_strategy: 'group_stratified_kfold' veya 'stratified_kfold_without_groups'
    """
    from sklearn.model_selection import StratifiedKFold

    if n_folds < 2:
        raise ValueError(f"n_folds en az 2 olmali (verilen: {n_folds}).")
    if test_ratio < 0.0 or test_ratio >= 1.0:
        raise ValueError("test_ratio 0 ile 1 arasinda olmali.")

    group_labels, group_keys = _build_group_level_records(labels, groups)
    if len(group_keys) < n_folds:
        raise ValueError(
            f"K-fold icin yeterli kaynak grup yok: bulunan={len(group_keys)}, n_folds={n_folds}."
        )

    test_groups: set[str] = set()
    cv_indices = list(range(len(group_keys)))
    if test_ratio > 0:
        if use_group_split:
            split_fn = _group_stratified_train_val_split
            split_kwargs: dict[str, Any] = dict(labels=group_labels, groups=group_keys)
        else:
            split_fn = _stratified_train_val_split
            split_kwargs = dict(labels=group_labels)

        cv_idxs, test_idxs = split_fn(
            **split_kwargs,
            val_ratio=test_ratio,
            seed=seed,
            num_classes=len(SINIF_ISIMLERI),
        )
        test_groups = {group_keys[i] for i in test_idxs}
        cv_indices = list(cv_idxs)

    cv_labels = [group_labels[i] for i in cv_indices]
    cv_keys = [group_keys[i] for i in cv_indices]
    if len(cv_keys) < n_folds:
        raise ValueError(
            "Test ayrildiktan sonra K-fold icin yeterli kaynak grup kalmadi: "
            f"kalan={len(cv_keys)}, n_folds={n_folds}."
        )

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(cv_keys), 1), dtype=np.float32)
    fold_assignments: list[tuple[set[str], set[str]]] = []
    for train_local_idxs, val_local_idxs in splitter.split(X_dummy, cv_labels):
        train_groups = {cv_keys[i] for i in train_local_idxs}
        val_groups = {cv_keys[i] for i in val_local_idxs}
        if train_groups & val_groups:
            raise RuntimeError("K-fold split sonrasi grup sizintisi tespit edildi.")
        fold_assignments.append((train_groups, val_groups))

    strategy = "group_stratified_kfold" if use_group_split else "stratified_kfold_without_groups"
    return fold_assignments, test_groups, strategy


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
    hflip_p: float = 0.0,
    rotation_degrees: float = 10.0,
    color_jitter: float = 0.1,
    *,
    use_dataset_stats: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader | None, Dict]:
    """
    Train, validation ve test DataLoader'lari olustur.

    - Varsayilan akista tum split'ler tek kaynaktan olusturulur.
    - Train split'i ayni kaynagin augment/turev kopyalarini kullanabilir.
    - Validation ve test split'leri yalnizca original goruntulerden kurulur.
    - ``test_dir`` verilirse harici/original test dizini olarak kullanilir.
    """
    if val_ratio <= 0 or val_ratio >= 1.0:
        raise ValueError("val_ratio 0 ile 1 arasinda olmali.")
    if test_ratio < 0 or test_ratio >= 1.0:
        raise ValueError("test_ratio 0 ile 1 arasinda olmali.")

    _validate_expected_classes(trainval_dir, "Trainval")
    using_external_test = bool(
        include_test
        and test_dir is not None
        and test_dir.resolve() != trainval_dir.resolve()
    )
    if include_test and not using_external_test and test_ratio <= 0:
        raise ValueError(
            "Harici test dizini yoksa test_ratio pozitif olmali."
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
    train_original_idxs = _indices_for_groups(
        tv_groups, tv_augmented, train_group_keys, original_only=True
    )
    val_idxs = _indices_for_groups(tv_groups, tv_augmented, val_group_keys, original_only=True)
    internal_test_idxs = _indices_for_groups(
        tv_groups,
        tv_augmented,
        test_group_keys_internal,
        original_only=True,
    )

    paths_train = [tv_paths[i] for i in train_idxs]
    labels_train = [tv_labels[i] for i in train_idxs]
    labels_train_original = [tv_labels[i] for i in train_original_idxs]
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

    if use_dataset_stats:
        norm_mean, norm_std = compute_dataset_stats(paths_train, image_size, seed=seed)
        print(
            "  [Normalize] Train-bazli mean/std kullaniliyor: "
            f"mean={[round(v, 4) for v in norm_mean]}, std={[round(v, 4) for v in norm_std]}"
        )
    else:
        norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD

    train_transform = get_transforms(
        image_size,
        is_train=True,
        hflip_p=hflip_p,
        rotation_degrees=rotation_degrees,
        color_jitter=color_jitter,
        mean=norm_mean,
        std=norm_std,
    )
    eval_transform = get_transforms(
        image_size, is_train=False, mean=norm_mean, std=norm_std
    )
    train_ds = MRIDataset(paths_train, labels_train, train_transform)
    val_ds = MRIDataset(paths_val, labels_val, eval_transform)
    test_ds = (
        MRIDataset(paths_test, labels_test, eval_transform)
        if include_test
        else None
    )

    pin_memory = torch.cuda.is_available()

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = _make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    val_loader = _make_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=val_generator,
    )
    test_loader = (
        _make_dataloader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        "train_original_labels": labels_train_original,
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
        "normalize_mean": list(norm_mean),
        "normalize_std": list(norm_std),
        "normalize_source": "dataset" if use_dataset_stats else "imagenet",
    }

    return train_loader, val_loader, test_loader, info


def iter_kfold_dataloaders(
    trainval_dir: Path,
    test_dir: Path | None,
    *,
    n_folds: int,
    batch_size: int = 32,
    image_size: int = 224,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    include_test: bool = True,
    hflip_p: float = 0.0,
    rotation_degrees: float = 10.0,
    color_jitter: float = 0.1,
    use_dataset_stats: bool = False,
) -> Iterator[Tuple[int, DataLoader, DataLoader, DataLoader | None, Dict]]:
    """K-fold cross validation icin fold basina (train_loader, val_loader, test_loader, info) uretir.

    Test seti tum fold'lar arasinda paylasilir (harici dizinden veya tek seferlik
    test_ratio ile cikarilan ic split'ten). Train/Val bolmesi her fold icin
    grup-bazli StratifiedKFold ile yapilir; ayni kaynak ID birden fazla fold'un
    val'inda gorunmez. Train fold'lari augment turevlerini icerebilirken val
    yalnizca original goruntulerden kurulur.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds en az 2 olmali (verilen: {n_folds}).")
    if test_ratio < 0 or test_ratio >= 1.0:
        raise ValueError("test_ratio 0 ile 1 arasinda olmali.")

    _validate_expected_classes(trainval_dir, "Trainval")
    using_external_test = bool(
        include_test
        and test_dir is not None
        and test_dir.resolve() != trainval_dir.resolve()
    )
    if include_test and not using_external_test and test_ratio <= 0:
        raise ValueError(
            "Harici test dizini yoksa test_ratio pozitif olmali."
        )
    if using_external_test and test_dir is not None:
        _validate_class_match(trainval_dir, test_dir)

    tv_paths, tv_labels, tv_groups, tv_augmented = collect_images(trainval_dir)
    if len(tv_paths) == 0:
        raise FileNotFoundError(f"Trainval verisi bulunamadi: {trainval_dir}")
    tv_group_stats = _summarize_grouping(tv_groups, tv_augmented)

    paths_test_external: List[Path] = []
    labels_test_external: List[int] = []
    test_group_stats: Dict[str, Any] | None = None
    if using_external_test and test_dir is not None:
        ext_paths, ext_labels, ext_groups, ext_augmented = collect_images(test_dir)
        if len(ext_paths) == 0:
            raise FileNotFoundError(f"Test verisi bulunamadi: {test_dir}")
        test_group_stats = _summarize_grouping(ext_groups, ext_augmented)
        _validate_dataset_separation(tv_groups, ext_groups, trainval_dir, test_dir)
        external_test_idxs = _indices_for_groups(
            ext_groups, ext_augmented, set(ext_groups), original_only=True
        )
        if not external_test_idxs:
            raise RuntimeError(
                "Harici test dizininde original goruntu bulunamadi; test split olusturulamiyor."
            )
        paths_test_external = [ext_paths[i] for i in external_test_idxs]
        labels_test_external = [ext_labels[i] for i in external_test_idxs]
        if _missing_class_names(labels_test_external, len(SINIF_ISIMLERI)):
            missing = _missing_class_names(labels_test_external, len(SINIF_ISIMLERI))
            raise RuntimeError(f"Test split'inde eksik siniflar: {', '.join(missing)}")

    split_warnings: List[str] = []
    use_group_split = tv_group_stats["grouping_reliable"]
    if not use_group_split and tv_group_stats["augmented_samples"] > 0:
        split_warnings.append(
            "TrainVal dosya adlarindan tekrarli kaynak grup cikarilamadi; "
            "stratified K-fold kullanildi ve augment turevleri icin leak-free garanti verilemiyor."
        )

    internal_test_ratio = 0.0 if (using_external_test or not include_test) else test_ratio
    fold_assignments, internal_test_groups, split_strategy = _split_group_keys_kfold(
        labels=tv_labels,
        groups=tv_groups,
        n_folds=n_folds,
        test_ratio=internal_test_ratio,
        seed=seed,
        use_group_split=use_group_split,
    )

    if include_test and not using_external_test:
        internal_test_idxs = _indices_for_groups(
            tv_groups, tv_augmented, internal_test_groups, original_only=True
        )
        if not internal_test_idxs:
            raise RuntimeError(
                "Test split olusturulamadi; original test ornekleri bulunamadi."
            )
        paths_test = [tv_paths[i] for i in internal_test_idxs]
        labels_test = [tv_labels[i] for i in internal_test_idxs]
        if _missing_class_names(labels_test, len(SINIF_ISIMLERI)):
            missing = _missing_class_names(labels_test, len(SINIF_ISIMLERI))
            raise RuntimeError(f"Test split'inde eksik siniflar: {', '.join(missing)}")
    elif using_external_test:
        paths_test = paths_test_external
        labels_test = labels_test_external
    else:
        paths_test = []
        labels_test = []

    pin_memory = torch.cuda.is_available()

    print(f"  [TrainVal] Kaynak: {trainval_dir}")
    print(f"  [TrainVal] Toplam goruntu: {len(tv_paths)}")
    print(f"  [TrainVal] Kaynak grup sayisi: {tv_group_stats['unique_groups']}")
    print(f"  [Split] Strateji: {split_strategy}, n_folds={n_folds}")
    if include_test:
        kaynak_label = test_dir if using_external_test else f"{trainval_dir} (internal split)"
        print(f"  [Test] Kaynak: {kaynak_label}")
        print(f"  [Test] Toplam goruntu: {len(paths_test)}")
    for warning in split_warnings:
        print(f"  [UYARI] {warning}")

    for fold_index, (train_group_keys, val_group_keys) in enumerate(fold_assignments):
        train_idxs = _indices_for_groups(
            tv_groups, tv_augmented, train_group_keys, original_only=False
        )
        train_original_idxs = _indices_for_groups(
            tv_groups, tv_augmented, train_group_keys, original_only=True
        )
        val_idxs = _indices_for_groups(
            tv_groups, tv_augmented, val_group_keys, original_only=True
        )

        paths_train = [tv_paths[i] for i in train_idxs]
        labels_train = [tv_labels[i] for i in train_idxs]
        labels_train_original = [tv_labels[i] for i in train_original_idxs]
        paths_val = [tv_paths[i] for i in val_idxs]
        labels_val = [tv_labels[i] for i in val_idxs]

        train_missing = _missing_class_names(labels_train, len(SINIF_ISIMLERI))
        val_missing = _missing_class_names(labels_val, len(SINIF_ISIMLERI))
        if train_missing:
            raise RuntimeError(
                f"Fold {fold_index} train split'inde eksik siniflar: {', '.join(train_missing)}"
            )
        if val_missing:
            raise RuntimeError(
                f"Fold {fold_index} validation split'inde eksik siniflar: {', '.join(val_missing)}"
            )

        if use_dataset_stats:
            norm_mean, norm_std = compute_dataset_stats(paths_train, image_size, seed=seed + fold_index)
        else:
            norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD

        train_transform = get_transforms(
            image_size,
            is_train=True,
            hflip_p=hflip_p,
            rotation_degrees=rotation_degrees,
            color_jitter=color_jitter,
            mean=norm_mean,
            std=norm_std,
        )
        eval_transform = get_transforms(
            image_size, is_train=False, mean=norm_mean, std=norm_std
        )

        train_ds = MRIDataset(paths_train, labels_train, train_transform)
        val_ds = MRIDataset(paths_val, labels_val, eval_transform)
        test_ds = (
            MRIDataset(paths_test, labels_test, eval_transform)
            if include_test and paths_test
            else None
        )

        fold_seed = seed + fold_index
        train_loader = _make_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(fold_seed),
        )
        val_loader = _make_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(fold_seed + 1),
        )
        test_loader = (
            _make_dataloader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                generator=torch.Generator().manual_seed(fold_seed + 2),
            )
            if test_ds is not None
            else None
        )

        info = {
            "num_classes": len(SINIF_ISIMLERI),
            "class_names": SINIF_ISIMLERI,
            "fold_index": fold_index,
            "n_folds": n_folds,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds) if test_ds is not None else 0,
            "train_labels": labels_train,
            "train_original_labels": labels_train_original,
            "val_labels": labels_val,
            "test_labels": labels_test,
            "train_groups": len({tv_groups[i] for i in train_idxs}),
            "val_groups": len({tv_groups[i] for i in val_idxs}),
            "trainval_dir": str(trainval_dir),
            "test_dir": str(test_dir) if test_dir is not None else None,
            "val_ratio": None,
            "test_ratio": test_ratio if include_test and not using_external_test else None,
            "split_strategy": split_strategy,
            "trainval_grouping": tv_group_stats,
            "test_grouping": test_group_stats,
            "split_warnings": split_warnings,
            "uses_external_test_dir": using_external_test,
            "normalize_mean": list(norm_mean),
            "normalize_std": list(norm_std),
            "normalize_source": "dataset" if use_dataset_stats else "imagenet",
        }

        yield fold_index, train_loader, val_loader, test_loader, info


def create_full_train_test_loaders(
    trainval_dir: Path,
    test_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
    hflip_p: float = 0.0,
    rotation_degrees: float = 10.0,
    color_jitter: float = 0.1,
    *,
    use_dataset_stats: bool = False,
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
    train_original_idxs = _indices_for_groups(
        tv_groups, tv_augmented, set(tv_groups), original_only=True
    )
    test_idxs = _indices_for_groups(test_groups, test_augmented, set(test_groups), original_only=True)

    paths_train = [tv_paths[i] for i in train_idxs]
    labels_train = [tv_labels[i] for i in train_idxs]
    labels_train_original = [tv_labels[i] for i in train_original_idxs]
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

    if use_dataset_stats:
        norm_mean, norm_std = compute_dataset_stats(paths_train, image_size, seed=seed)
        print(
            "  [Normalize] Train-bazli mean/std kullaniliyor: "
            f"mean={[round(v, 4) for v in norm_mean]}, std={[round(v, 4) for v in norm_std]}"
        )
    else:
        norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD

    train_transform = get_transforms(
        image_size,
        is_train=True,
        hflip_p=hflip_p,
        rotation_degrees=rotation_degrees,
        color_jitter=color_jitter,
        mean=norm_mean,
        std=norm_std,
    )
    eval_transform = get_transforms(
        image_size, is_train=False, mean=norm_mean, std=norm_std
    )
    train_ds = MRIDataset(paths_train, labels_train, train_transform)
    test_ds = MRIDataset(paths_test, labels_test, eval_transform)

    pin_memory = torch.cuda.is_available()
    train_generator = torch.Generator().manual_seed(seed)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = _make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    test_loader = _make_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=test_generator,
    )

    info = {
        "num_classes": len(SINIF_ISIMLERI),
        "class_names": SINIF_ISIMLERI,
        "train_size": len(train_ds),
        "val_size": 0,
        "test_size": len(test_ds),
        "train_labels": labels_train,
        "train_original_labels": labels_train_original,
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
        "normalize_mean": list(norm_mean),
        "normalize_std": list(norm_std),
        "normalize_source": "dataset" if use_dataset_stats else "imagenet",
    }
    return train_loader, test_loader, info
