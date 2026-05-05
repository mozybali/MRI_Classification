import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Mevcut görüntü işleme modülünden fonksiyonları kullan
try:
    from goruntu_isleme.ana_islem import mri_preprocess_image
except ImportError:
    mri_preprocess_image = None

def get_preprocessing_steps(request):
    """
    Yüklenen görüntünün ön işleme adımlarını (Ham -> CLAHE -> Resize) döndürür.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST gerekli"}, status=405)

    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "Görüntü yok"}, status=400)

    try:
        # Görüntüyü oku
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        steps = []

        # 1. Ham Görüntü
        steps.append({"title": "Ham Görüntü", "img": _to_base64(img)})

        # 2. Gri Ton
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps.append({"title": "Gri Ton", "img": _to_base64(gray)})

        # 3. CLAHE (Kontrast Artırma)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(gray)
        steps.append({"title": "CLAHE Uygulandı", "img": _to_base64(img_clahe)})

        # 4. Final (Resize & Normalize Görseli)
        final = cv2.resize(img_clahe, (224, 224))
        steps.append({"title": "Boyutlandırıldı (224x224)", "img": _to_base64(final)})

        return JsonResponse({"success": True, "steps": steps})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def _to_base64(img_array):
    """OpenCV dizisini base64 PNG'ye çevirir."""
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')
