import json
import os
import time
from pathlib import Path

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import PredictionRecord
from .services.model_loader import registry
from .services.xgb_inference import predict_image_xgb_local

# NOT: `model.inference` modül seviyesinde torch import ediyor. Torch web env'de
# zorunlu olmadığından (yalnızca ResNet için gerekli), ResNet ile ilgili
# importlar request anında, tek tek branch içinde lazy yapılır.

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

        request_started_at = time.perf_counter()

        # 1. Model yolunu belirle ve yükle
        model_path = settings.MODEL_DIR / model_name
        if not model_path.exists():
            return JsonResponse({"error": f"Model bulunamadı: {model_name}"}, status=404)

        # 2. Görüntüyü geçici olarak kaydet (ML modülleri dosya yolu beklediği için)
        path = default_storage.save(f"tmp/{image_file.name}", ContentFile(image_file.read()))
        tmp_path = Path(settings.MEDIA_ROOT) / path

        # 3. Tahmin yap
        result = {}
        model_load_started_at = time.perf_counter()
        yolo_meta_path = settings.MODEL_DIR / (Path(model_name).stem + ".yolo.meta.json")
        if model_name.endswith(".pt") and yolo_meta_path.exists():
            try:
                from model.yolo_inference import predict_image_yolo
            except ImportError as exc:
                return JsonResponse(
                    {"error": f"YOLO inference için ultralytics kurulu olmalı: {exc}"},
                    status=503,
                )
            model, meta = registry.get_yolo(model_path)
            model_loaded_at = time.perf_counter()
            result = predict_image_yolo(
                model,
                tmp_path,
                meta["image_size"],
                meta["class_names"],
                apply_mri_preprocessing=apply_preprocess,
            )
            model_type = "yolo"
        elif model_name.endswith(".pt"):
            try:
                from model.inference import predict_image
            except ImportError as exc:
                return JsonResponse(
                    {"error": f"ResNet inference için torch kurulu olmalı: {exc}"},
                    status=503,
                )
            model, meta = registry.get_resnet(model_path)
            model_loaded_at = time.perf_counter()
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
            model_loaded_at = time.perf_counter()
            result = predict_image_xgb_local(
                model,
                tmp_path,
                meta["image_size"],
                meta["class_names"],
                apply_mri_preprocessing=apply_preprocess
            )
            model_type = "xgboost"
        else:
            return JsonResponse({"error": "Desteklenmeyen model formatı."}, status=400)

        inference_finished_at = time.perf_counter()

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
            "image_url": record.image.url,
            "timings": {
                "model_load_seconds": round(model_loaded_at - model_load_started_at, 3),
                "inference_seconds": round(inference_finished_at - model_loaded_at, 3),
                "total_seconds": round(inference_finished_at - request_started_at, 3),
            },
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
