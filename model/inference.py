#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference.py
------------
Egitilmis derin ogrenme modeli ile tahmin yapma (inference) scripti.
Yeni MRI goruntuleri icin demans seviyesi tahmini yapar.

Kullanim:
    python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --image test.jpg
    python model/inference.py --model-path model/ciktilar/modeller/best_unet.pt --batch ./images/
"""

import argparse
from collections import Counter
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.dl.dataset import GORUNTU_UZANTILARI, SINIF_ISIMLERI
    from model.dl.models.resnet_classifier import ResNetClassifier
    from model.dl.models.unet_classifier import UNetClassifier
    from model.dl.utils import get_device, load_checkpoint
else:
    from .dl.dataset import GORUNTU_UZANTILARI, SINIF_ISIMLERI
    from .dl.models.resnet_classifier import ResNetClassifier
    from .dl.models.unet_classifier import UNetClassifier
    from .dl.utils import get_device, load_checkpoint


def load_model(model_path: Path, device: torch.device):
    """Checkpoint'tan model yukle."""
    checkpoint = load_checkpoint(model_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet")
    num_classes = checkpoint.get("num_classes", 4)

    if model_name == "resnet":
        model = ResNetClassifier(num_classes=num_classes, pretrained=False)
    elif model_name == "unet":
        model = UNetClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", 224)
    class_names = checkpoint.get("class_names", SINIF_ISIMLERI)
    if len(class_names) != num_classes:
        raise ValueError(
            "Checkpoint metadata tutarsiz: class_names uzunlugu num_classes ile eslesmiyor."
        )

    return model, image_size, class_names


def predict_image(model, image_path: Path, image_size: int, class_names, device):
    """Tek bir goruntu icin tahmin yap."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
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


def collect_batch_images(batch_dir: Path) -> list[Path]:
    """Batch inference icin desteklenen goruntuleri buyuk/kucuk harf duyarli olmadan topla."""
    return sorted(
        path for path in batch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in GORUNTU_UZANTILARI
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="MRI derin ogrenme inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python model/inference.py --model-path model/ciktilar/modeller/best_resnet.pt --image test.jpg
  python model/inference.py --model-path model/ciktilar/modeller/best_unet.pt --batch ./images/
        """,
    )
    parser.add_argument("--model-path", type=str, required=True, help=".pt checkpoint yolu")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Tek goruntu yolu")
    source_group.add_argument("--batch", type=str, help="Goruntu klasoru (batch tahmin)")
    args = parser.parse_args(argv)

    device = get_device()
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"[HATA] Model bulunamadi: {model_path}")
        return 1

    try:
        model, image_size, class_names = load_model(model_path, device)
        print(f"[OK] Model yuklendi: {model_path.name}")

        if args.image:
            img_path = Path(args.image)
            if not img_path.exists():
                print(f"[HATA] Goruntu bulunamadi: {img_path}")
                return 1

            result = predict_image(model, img_path, image_size, class_names, device)

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
                result = predict_image(model, img_path, image_size, class_names, device)
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
