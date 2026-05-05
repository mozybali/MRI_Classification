from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/eda-stats/", views.eda_stats, name="eda_stats"),
    path("api/model-reports/", views.model_reports, name="model_reports"),
    path("api/eda-charts/", views.eda_charts, name="eda_charts"),
]
