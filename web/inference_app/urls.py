from django.urls import path
from . import views, xai_views, preprocess_views, batch_views

app_name = "infer"

urlpatterns = [
    path("", views.index, name="index"),
    path("predict/", views.predict, name="predict"),
    path("history/", views.history, name="history"),
    path("explain/<int:record_id>/", xai_views.explain_prediction, name="explain"),
    path("preprocess-steps/", preprocess_views.get_preprocessing_steps, name="preprocess_steps"),
    path("batch-predict/", batch_views.batch_predict, name="batch_predict"),
    path("export-csv/", batch_views.export_csv, name="export_csv"),
]
