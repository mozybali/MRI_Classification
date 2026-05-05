import json
import os
from pathlib import Path

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import PredictionRecord
from .services.model_loader import registry

# Mevcut ML modüllerinden fonksiyonları import et
# settings.py içindeki sys.path eklemesi sayesinde çalışacak
try:
    from model.inference import predict_image, predict_image_xgb
    from model.dl.utils import get_device
except ImportError as e:
    print(f"ML modülleri import edilemedi: {e}")

def index(request):
    """Tahmin ana sayfası - Model listesini ve yükleme alanını gösterir."""
    models = registry.list_available_models()
    return render(request, "inference_app/index.html", {
        "models": models,
    })

def predict(request):
    """
    POST: MRI görüntüsü al, tahmin yap, veritabanına kaydet ve JSON döndür.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Sadece POST metodu kabul edilir."}, status=405)

    try:
        image_file = request.FILES.get("image")
        model_name = request.POST.get("model_name")
        apply_preprocess = request.POST.get("preprocess") == "true"

        if not image_file or not model_name:
            return JsonResponse({"error": "Görüntü ve model seçimi zorunludur."}, status=400)

        # 1. Model yolunu belirle ve yükle
        model_path = settings.MODEL_DIR / model_name
        if not model_path.exists():
            return JsonResponse({"error": f"Model bulunamadı: {model_name}"}, status=404)

        # 2. Görüntüyü geçici olarak kaydet (ML modülleri dosya yolu beklediği için)
        path = default_storage.save(f"tmp/{image_file.name}", ContentFile(image_file.read()))
        tmp_path = Path(settings.MEDIA_ROOT) / path

        # 3. Tahmin yap
        result = {}
        if model_name.endswith(".pt"):
            model, meta = registry.get_resnet(model_path)
            result = predict_image(
                model,
                tmp_path,
                meta["image_size"],
                meta["class_names"],
                meta["device"],
                normalize_mean=meta["mean"],
                normalize_std=meta["std"],
                apply_mri_preprocessing=apply_preprocess
            )
            model_type = "resnet"
        elif model_name.endswith(".json"):
            model, meta = registry.get_xgboost(model_path)
            result = predict_image_xgb(
                model,
                tmp_path,
                meta["image_size"],
                meta["class_names"],
                apply_mri_preprocessing=apply_preprocess
            )
            model_type = "xgboost"
        else:
            return JsonResponse({"error": "Desteklenmeyen model formatı."}, status=400)

        # 4. Veritabanına kaydet
        # Geçici dosyayı media/predictions altına taşıyalım veya direkt oradan referans verelim
        # Kolaylık için modeldeki image alanına direkt geçici dosyayı atayabiliriz (Django onu taşıyacaktır)
        record = PredictionRecord.objects.create(
            image=path,
            model_type=model_type,
            predicted_class=result["tahmin_adi"],
            confidence=result["guven_skoru"],
            probabilities=result["olasiliklar"],
            apply_preprocess=apply_preprocess
        )

        return JsonResponse({
            "success": True,
            "record_id": record.id,
            "tahmin_adi": record.predicted_class,
            "severity_label": record.severity_label,
            "css_class": record.css_class,
            "confidence": record.confidence,
            "probabilities": record.probabilities,
            "image_url": record.image.url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

def history(request):
    """Son tahmin kayıtlarını listele."""
    records = PredictionRecord.objects.all()[:20]
    return render(request, "inference_app/history.html", {
        "records": records,
    })
