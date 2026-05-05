from django.shortcuts import render
from django.http import JsonResponse


def index(request):
    """EDA Dashboard ana sayfası — Sprint 3'te doldurulacak."""
    return render(request, "dashboard_app/index.html")


def eda_stats(request):
    """EDA istatistiklerini JSON olarak döndür — Sprint 3'te doldurulacak."""
    return JsonResponse({"status": "coming_soon"})


def model_reports(request):
    """Model raporlarını JSON olarak döndür — Sprint 3'te doldurulacak."""
    return JsonResponse({"status": "coming_soon"})


def eda_charts(request):
    """EDA grafik listesini döndür — Sprint 3'te doldurulacak."""
    return JsonResponse({"charts": []})
