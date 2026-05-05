from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/eda-stats/", views.eda_stats, name="eda_stats"),
    path("api/model-reports/", views.model_reports, name="model_reports"),
    path("api/report/<str:filename>/", views.get_report_detail, name="report_detail"),
]
