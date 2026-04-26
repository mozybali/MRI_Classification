#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference.py
------------
Egitilmis model ile tahmin yapma (inference) scripti.
DL (.pt) ve XGBoost (.json) modellerini destekler.

Kullanim:
    python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --image test.jpg
    python model/inference.py --model-path model/ciktilar/modeller/best_xgboost.json --image test.jpg
    python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --batch ./images/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.dl.dataset import (
        GORUNTU_UZANTILARI,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SINIF_ISIMLERI,
    )
    from model.dl.utils import get_device, load_checkpoint
    from model.training_runner import build_model
    from model.sl.xgb_classifier import load_xgb_model, load_xgb_model_with_meta
    from model.sl.features import extract_features as _extract_features
else:
    from .dl.dataset import (
        GORUNTU_UZANTILARI,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SINIF_ISIMLERI,
    )
    from .dl.utils import get_device, load_checkpoint
    from .training_runner import build_model
    from .sl.xgb_classifier import load_xgb_model, load_xgb_model_with_meta
    from .sl.features import extract_features as _extract_features


def _detect_model_type(model_path: Path) -> str:
    """Dosya uzantisindan model tipini belirle."""
    suffix = model_path.suffix.lower()
    if suffix == ".json":
        return "xgboost"
    if suffix == ".pt":
        return "torch"
    raise ValueError(f"Desteklenmeyen model dosya formati: {suffix}")


def load_model(model_path: Path, device: torch.device):
    """Checkpoint'tan model yukle."""
    checkpoint = load_checkpoint(model_path, map_location=device)
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint 'model_name' eksik veya bos.")
    num_classes = checkpoint.get("num_classes", 4)

    model = build_model(model_name, num_classes, device, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = checkpoint.get("image_size", 224)
    class_names = checkpoint.get("class_names", SINIF_ISIMLERI)
    if len(class_names) != num_classes:
        raise ValueError(
            "Checkpoint metadata tutarsiz: class_names uzunlugu num_classes ile eslesmiyor."
        )

    raw_mean = checkpoint.get("normalize_mean")
    raw_std = checkpoint.get("normalize_std")
    normalize_mean = tuple(raw_mean) if raw_mean else IMAGENET_MEAN
    normalize_std = tuple(raw_std) if raw_std else IMAGENET_STD

    return model, image_size, class_names, normalize_mean, normalize_std


def _build_eval_transform(image_size: int, mean, std) -> transforms.Compose:
    """Egitimle ayni aspect-preserving resize + center crop akisi."""
    return transforms.Compose([
        transforms.Resize(
            image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def _load_image_for_inference(
    image_path: Path,
    *,
    apply_mri_preprocessing: bool,
) -> Image.Image:
    """Inference icin PIL goruntu yukle; istege bagli MRI on isleme uygula."""
    if not apply_mri_preprocessing:
        return Image.open(image_path).convert("RGB")

    # GorselIsleyici disk-yazma yan etkisi olmayan saf donusum saglar.
    # Modul yuklemesi cikartmali; hem CLI ('python model/inference.py')
    # hem de paket modu ('python -m model.inference') altinda calisir.
    from goruntu_isleme.goruntu_isleyici import GorselIsleyici

    isleyici = GorselIsleyici()
    processed = isleyici.goruntu_isle(str(image_path))
    if processed is None:
        raise RuntimeError(
            f"MRI on isleme basarisiz (kalite kontrol veya yukleme hatasi): {image_path}"
        )
    # processed: 2D uint8 -> 3 kanalli PIL'e cevir, transform pipeline RGB bekliyor
    return Image.fromarray(processed).convert("RGB")


def predict_image(
    model,
    image_path: Path,
    image_size: int,
    class_names,
    device,
    *,
    normalize_mean=IMAGENET_MEAN,
    normalize_std=IMAGENET_STD,
    apply_mri_preprocessing: bool = False,
):
    """Tek bir goruntu icin DL tahmin yap."""
    transform = _build_eval_transform(image_size, normalize_mean, normalize_std)

    image = _load_image_for_inference(
        image_path, apply_mri_preprocessing=apply_mri_preprocessing
    )
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    return {
        "dosya": str(image_path),
        "tahmin_sinif": pred_idx,
        "tahmin_adi": class_names[pred_idx],
        "guven_skoru": probs[pred_idx].item(),
        "olasiliklar": {class_names[i]: probs[i].item() for i in range(len(class_names))},
    }


# ==================== XGBoost Inference ====================


def load_xgb_model_for_inference(model_path: Path):
    """XGBoost modelini yukle ve sinif bilgilerini dondur.

    Side-car metadata dosyasi (.meta.json) varsa image_size ve class_names
    oradan okunur; yoksa varsayilan degerler kullanilir.
    """
    xgb_model, meta = load_xgb_model_with_meta(model_path)
    image_size = meta.get("image_size", 224)
    class_names = meta.get("class_names", SINIF_ISIMLERI)
    return xgb_model, image_size, class_names


def predict_image_xgb(
    model,
    image_path: Path,
    image_size: int,
    class_names,
    *,
    apply_mri_preprocessing: bool = False,
):
    """Tek bir goruntu icin XGBoost tahmin yap."""
    if apply_mri_preprocessing:
        rgb = _load_image_for_inference(image_path, apply_mri_preprocessing=True)
        gray = rgb.convert("L")
    else:
        gray = Image.open(image_path).convert("L")
    gray = gray.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(gray, dtype=np.uint8)
    features = _extract_features(arr).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "dosya": str(image_path),
        "tahmin_sinif": pred_idx,
        "tahmin_adi": class_names[pred_idx],
        "guven_skoru": float(probs[pred_idx]),
        "olasiliklar": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def collect_batch_images(batch_dir: Path) -> list[Path]:
    """Batch inference icin desteklenen goruntuleri buyuk/kucuk harf duyarli olmadan topla."""
    return sorted(
        path for path in batch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in GORUNTU_UZANTILARI
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="MRI inference (DL ve XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --image test.jpg
  python model/inference.py --model-path model/ciktilar/modeller/best_xgboost.json --image test.jpg
  python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --batch ./images/
        """,
    )
    parser.add_argument("--model-path", type=str, required=True, help=".pt veya .json model yolu")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Tek goruntu yolu")
    source_group.add_argument("--batch", type=str, help="Goruntu klasoru (batch tahmin)")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Goruntuye on-isleme pipeline'ini (kalite kontrol, denoise, CLAHE, vb.) "
            "tahmin oncesi uygula. Ham goruntu (orn. Veri_Seti/OriginalDataset) "
            "uzerinde inference yaparken egitim dagilimiyla tutarliligi korur."
        ),
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[HATA] Model bulunamadi: {model_path}")
        return 1

    try:
        model_type = _detect_model_type(model_path)
        apply_preprocess = bool(args.preprocess)

        if model_type == "xgboost":
            model, image_size, class_names = load_xgb_model_for_inference(model_path)

            def predict_fn(m, p, s, c):
                return predict_image_xgb(
                    m, p, s, c, apply_mri_preprocessing=apply_preprocess
                )
        else:
            device = get_device()
            model, image_size, class_names, norm_mean, norm_std = load_model(
                model_path, device
            )

            def predict_fn(m, p, s, c):
                return predict_image(
                    m,
                    p,
                    s,
                    c,
                    device,
                    normalize_mean=norm_mean,
                    normalize_std=norm_std,
                    apply_mri_preprocessing=apply_preprocess,
                )

        print(f"[OK] Model yuklendi: {model_path.name} (tip: {model_type})")

        if args.image:
            img_path = Path(args.image)
            if not img_path.exists():
                print(f"[HATA] Goruntu bulunamadi: {img_path}")
                return 1

            result = predict_fn(model, img_path, image_size, class_names)

            print(f"\n{'='*60}")
            print("TAHMIN SONUCU")
            print(f"{'='*60}")
            print(f"Tahmin: {result['tahmin_adi']}")
            print(f"Guven : {result['guven_skoru']:.2%}")
            print("\nSinif Olasiliklari:")
            for name, prob in sorted(
                result["olasiliklar"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                bar = "#" * int(prob * 40)
                print(f"   {name:25s}: {prob:6.2%} {bar}")
            print(f"{'='*60}\n")

        elif args.batch:
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                print(f"[HATA] Klasor bulunamadi: {batch_dir}")
                return 1

            images = collect_batch_images(batch_dir)

            if not images:
                print(f"[UYARI] Goruntu bulunamadi: {batch_dir}")
                return 1

            print(f"\n[INFO] Batch tahmin: {len(images)} goruntu")
            results = []
            for img_path in sorted(images):
                result = predict_fn(model, img_path, image_size, class_names)
                results.append(result)
                print(f"  {img_path.name}: {result['tahmin_adi']} ({result['guven_skoru']:.2%})")

            counts = Counter(r["tahmin_adi"] for r in results)
            print("\nTahmin Dagilimi:")
            for cls, cnt in counts.most_common():
                print(f"   {cls:25s}: {cnt}")

    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[HATA] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
