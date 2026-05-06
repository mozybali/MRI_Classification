import base64
from io import BytesIO
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.conf import settings
from pathlib import Path
from PIL import Image

from .models import PredictionRecord
from .services.model_loader import registry

# .services.gradcam torch'u modül seviyesinde import ettiği için lazy yükleyeceğiz.


def explain_prediction(request, record_id):
    """
    Belirli bir tahmin kaydı için Grad-CAM ısı haritası üretir.
    """
    record = get_object_or_404(PredictionRecord, id=record_id)
    
    if record.model_type != "resnet":
        return JsonResponse({"error": "Grad-CAM sadece ResNet modelleri için desteklenir."}, status=400)

    try:
        # Torch ve ResNet transform'u sadece bu uç noktada gerekli — modül yüklemesini
        # web env'de torch yokken bile çalıştırabilmek için lazy import.
        try:
            import torch
            from model.dl.dataset import get_transforms
            from .services.gradcam import GradCAM, apply_heatmap_to_image
        except ImportError as exc:
            return JsonResponse(
                {"error": f"Grad-CAM için torch kurulu olmalı: {exc}"},
                status=503,
            )

        # 1. Modeli ve Metadataları yükle
        model_path = settings.MODEL_DIR / f"{record.model_type}_model.pt" # Varsayılan isim veya record'a eklenmeli
        # Eğer record'da model adı yoksa, klasördeki ilk .pt modelini alalım (veya daha iyisi kayıt sırasında saklanmalı)
        # Şimdilik model_loader'dan aktif olanı veya en son eğitileni bulalım.
        model_files = list(settings.MODEL_DIR.glob("*.pt"))
        if not model_files:
             return JsonResponse({"error": "Model dosyası bulunamadı."}, status=404)
        
        # En mantıklı modeli seç (basitlik için ilkini alıyoruz, gerçekte record'da saklanmalı)
        model_path = model_files[0]
        model, meta = registry.get_resnet(model_path)
        
        # 2. ResNet'in son conv katmanını bul (ResNet18 için model.layer4)
        # target_layer = model.layer4[1].conv2  # ResNet18 için tipik
        # Alternatif olarak dinamik bulalım:
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module # En son konvolüsyon katmanını bulur
        
        if not target_layer:
            return JsonResponse({"error": "Konvolüsyon katmanı bulunamadı."}, status=500)

        # 3. Görüntüyü hazırla
        img_path = Path(settings.MEDIA_ROOT) / record.image.name
        img = Image.open(img_path).convert("RGB")
        
        transform = get_transforms(
            image_size=meta["image_size"],
            is_train=False,
            mean=meta["mean"],
            std=meta["std"]
        )
        input_tensor = transform(img).unsqueeze(0).to(meta["device"])
        input_tensor.requires_grad = True

        # 4. Grad-CAM Üret
        cam_generator = GradCAM(model, target_layer)
        heatmap = cam_generator.generate(input_tensor)
        
        # 5. Görselleştirme
        result_img = apply_heatmap_to_image(img_path, heatmap)
        
        # Base64'e çevir
        buffered = BytesIO()
        result_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JsonResponse({
            "success": True,
            "heatmap_image": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)
