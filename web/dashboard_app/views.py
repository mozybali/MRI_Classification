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
    reports_dir = Path(settings.BASE_DIR).parent / "model" / "ciktilar" / "raporlar"
    
    reports = []
    if reports_dir.exists():
        for file in sorted(reports_dir.glob("rapor_*.json"), reverse=True):
            with open(file, 'r') as f:
                data = json.load(f)
                reports.append({
                    "filename": file.name,
                    "model": data.get("model", "Unknown"),
                    "timestamp": data.get("timestamp"),
                    "accuracy": data.get("test_metrics", {}).get("accuracy", 0),
                    "f1": data.get("test_metrics", {}).get("f1_macro", 0),
                })
    
    return JsonResponse({"reports": reports})

def get_report_detail(request, filename):
    """Belirli bir raporun detaylarını (karışıklık matrisi vb.) döndür."""
    reports_dir = Path(settings.BASE_DIR).parent / "model" / "ciktilar" / "raporlar"
    report_path = reports_dir / filename
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            return JsonResponse(json.load(f))
    return JsonResponse({"error": "Report not found"}, status=404)
