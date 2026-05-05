import zipfile
import io
import csv
from pathlib import Path
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .services.model_loader import registry

# ML modülleri
try:
    from model.inference import predict_image, predict_image_xgb
except ImportError:
    predict_image = None

def batch_predict(request):
    """
    ZIP dosyası alır, içindeki tüm resimleri seçilen modelle analiz eder.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST gerekli"}, status=405)

    zip_file = request.FILES.get("zip_file")
    model_name = request.POST.get("model_name")
    apply_preprocess = request.POST.get("preprocess") == "true"

    if not zip_file or not model_name:
        return JsonResponse({"error": "ZIP dosyası ve model seçimi zorunludur."}, status=400)

    try:
        # 1. Modeli yükle
        model_path = settings.MODEL_DIR / model_name
        results = []
        
        # 2. ZIP dosyasını oku
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Görüntüyü geçici olarak kaydet
                    with z.open(filename) as f:
                        content = f.read()
                        path = default_storage.save(f"batch_tmp/{filename}", ContentFile(content))
                        tmp_path = Path(settings.MEDIA_ROOT) / path
                        
                        # 3. Tahmin yap
                        if model_name.endswith(".pt"):
                            model, meta = registry.get_resnet(model_path)
                            res = predict_image(model, tmp_path, meta["image_size"], meta["class_names"], meta["device"], apply_mri_preprocessing=apply_preprocess)
                        else:
                            model, meta = registry.get_xgboost(model_path)
                            res = predict_image_xgb(model, tmp_path, meta["image_size"], meta["class_names"], apply_mri_preprocessing=apply_preprocess)
                        
                        results.append({
                            "filename": filename,
                            "prediction": res["tahmin_adi"],
                            "confidence": round(res["guven_skoru"] * 100, 2)
                        })
                        
                        # Geçici dosyayı sil (isteğe bağlı, temizlik için)
                        default_storage.delete(path)

        return JsonResponse({"success": True, "results": results})

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
