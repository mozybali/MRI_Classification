import zipfile
import io
import csv
import time
from pathlib import Path
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .services.model_loader import registry
from .services.xgb_inference import predict_image_xgb_local

# `model.inference` torch'u zorunlu kıldığı için ResNet branch'inde lazy import
# yapılır; XGBoost akışı `predict_image_xgb_local` ile torch'tan bağımsız çalışır.

def _is_valid_image_path(filename: str) -> bool:
    """
    macOS ZIP'lerinde oluşan __MACOSX/ ve ._ önekli metadata dosyalarını filtreler.
    Sadece gerçek görüntü dosyalarına True döner.
    """
    p = Path(filename)
    # __MACOSX klasörü altındaki her şeyi atla
    if "__MACOSX" in p.parts:
        return False
    # Gizli macOS metadata dosyaları (._filename)
    if p.name.startswith("._"):
        return False
    # Sadece bilinen görüntü uzantıları
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}


def batch_predict(request):
    """
    Bir veya birden fazla ZIP dosyası alır, içindeki tüm resimleri
    seçilen modelle analiz eder.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST gerekli"}, status=405)

    zip_files = request.FILES.getlist("zip_file")  # Çoklu ZIP desteği
    model_name = request.POST.get("model_name")
    apply_preprocess = request.POST.get("preprocess") == "true"

    if not zip_files or not model_name:
        return JsonResponse({"error": "ZIP dosyası ve model seçimi zorunludur."}, status=400)

    try:
        request_started_at = time.perf_counter()
        model_path = settings.MODEL_DIR / model_name
        results = []
        errors = []

        # Model tipini belirle (sidecar .yolo.meta.json varsa YOLO, yoksa ResNet)
        yolo_meta_path = settings.MODEL_DIR / (Path(model_name).stem + ".yolo.meta.json")
        is_yolo = model_name.endswith(".pt") and yolo_meta_path.exists()
        is_resnet = model_name.endswith(".pt") and not is_yolo

        model = None
        meta = None
        model_load_started_at = time.perf_counter()
        resnet_predict_fn = None
        yolo_predict_fn = None
        if is_yolo:
            try:
                from model.yolo_inference import predict_image_yolo as _predict_image_yolo
                yolo_predict_fn = _predict_image_yolo
            except ImportError as exc:
                return JsonResponse(
                    {"error": f"YOLO inference için ultralytics kurulu olmalı: {exc}"},
                    status=503,
                )
            model, meta = registry.get_yolo(model_path)
        elif is_resnet:
            try:
                from model.inference import predict_image as _predict_image
                resnet_predict_fn = _predict_image
            except ImportError as exc:
                return JsonResponse(
                    {"error": f"ResNet inference için torch kurulu olmalı: {exc}"},
                    status=503,
                )
            model, meta = registry.get_resnet(model_path)
        else:
            model, meta = registry.get_xgboost(model_path)

        model_loaded_at = time.perf_counter()
        processed_count = 0

        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file) as z:
                for filename in z.namelist():
                    if not _is_valid_image_path(filename):
                        continue  # __MACOSX, ._ metadata vs. atla

                    tmp_path = None
                    path = None
                    try:
                        with z.open(filename) as f:
                            content = f.read()
                            path = default_storage.save(
                                f"batch_tmp/{Path(filename).name}", ContentFile(content)
                            )
                            tmp_path = Path(settings.MEDIA_ROOT) / path

                        if is_yolo:
                            res = yolo_predict_fn(
                                model, tmp_path,
                                meta["image_size"], meta["class_names"],
                                apply_mri_preprocessing=apply_preprocess,
                            )
                        elif is_resnet:
                            res = resnet_predict_fn(
                                model, tmp_path,
                                meta["image_size"], meta["class_names"], meta["device"],
                                normalize_mean=meta["mean"],
                                normalize_std=meta["std"],
                                apply_mri_preprocessing=apply_preprocess,
                            )
                        else:
                            res = predict_image_xgb_local(
                                model, tmp_path,
                                meta["image_size"], meta["class_names"],
                                apply_mri_preprocessing=apply_preprocess,
                            )

                        results.append({
                            "filename": filename,
                            "prediction": res["tahmin_adi"],
                            "confidence": round(res["guven_skoru"] * 100, 2)
                        })
                        processed_count += 1

                    except Exception as img_err:
                        # Tek bir görüntü hatalıysa tüm batch durmasın
                        errors.append({"filename": filename, "error": str(img_err)})
                    finally:
                        if path:
                            default_storage.delete(path)

        finished_at = time.perf_counter()
        return JsonResponse({
            "success": True,
            "results": results,
            "errors": errors,
            "processed_count": processed_count,
            "timings": {
                "model_load_seconds": round(model_loaded_at - model_load_started_at, 3),
                "processing_seconds": round(finished_at - model_loaded_at, 3),
                "total_seconds": round(finished_at - request_started_at, 3),
            },
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def export_csv(request):
    """Tahmin sonuçlarını CSV olarak indir."""
    data = request.POST.get("data") # JSON string olarak bekliyoruz
    import json
    results = json.loads(data)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="mri_batch_results.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Dosya Adı', 'Tahmin', 'Güven Skoru (%)'])
    for res in results:
        writer.writerow([res['filename'], res['prediction'], res['confidence']])
        
    return response
