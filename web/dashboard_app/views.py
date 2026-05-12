import os
import json
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from collections import Counter

def index(request):
    """Dashboard ana sayfası."""
    return render(request, "dashboard_app/index.html")

def eda_stats(request):
    """
    Veri seti dağılımını hesapla ve JSON döndür.
    """
    dataset_path = Path(settings.BASE_DIR).parent / "Veri_Seti" / "OriginalDataset"
    
    stats = {}
    if dataset_path.exists():
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                # Gizli dosyaları (örn .DS_Store) elemek için uzantı kontrolü
                count = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                stats[class_dir.name] = count
                
    return JsonResponse({
        "success": True,
        "labels": list(stats.keys()),
        "data": list(stats.values()),
        "total": sum(stats.values())
    })

def model_reports(request):
    """
    model/ciktilar/raporlar altındaki en güncel raporları listele.
    """
    # Eski kod sadece tek seviye "raporlar/" klasorunu kontrol ediyordu.
    # Raporlar proje icinde alt dizinlerde (ör. hiperparametre_arama/.../best_run/raporlar)
    # olusturulabilir; bu nedenle recursive arama yaparak tum rapor dosyalarini toplayalim.
    root_dir = Path(settings.BASE_DIR).parent / "model" / "ciktilar"

    reports: list[dict] = []
    if root_dir.exists():
        # tum rapor_*.json dosyalarini bul ve son degisiklik tarihine gore sirala (yeniden en once)
        files = [p for p in root_dir.rglob("rapor_*.json") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                # Hata olursa bu dosyayi atla
                continue
            reports.append({
                "filename": file.name,
                "model": data.get("model", "Unknown"),
                "timestamp": data.get("timestamp"),
                "accuracy": data.get("test_metrics", {}).get("accuracy", 0),
                "f1": data.get("test_metrics", {}).get("f1_macro", 0),
                "path": str(file),
            })

    return JsonResponse({"reports": reports})

def get_report_detail(request, filename):
    """Belirli bir raporun detaylarını (karışıklık matrisi vb.) döndür."""
    root_dir = Path(settings.BASE_DIR).parent / "model" / "ciktilar"
    # Dosya alt dizinlerde olabilir; isimle arama yap
    if root_dir.exists():
        matches = [p for p in root_dir.rglob("*.json") if p.is_file() and p.name == filename]
        if matches:
            try:
                with open(matches[0], 'r', encoding='utf-8') as f:
                    return JsonResponse(json.load(f))
            except Exception:
                return JsonResponse({"error": "Failed to read report"}, status=500)
    return JsonResponse({"error": "Report not found"}, status=404)
