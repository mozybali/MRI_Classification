from django.urls import path
from . import views

app_name = "infer"

urlpatterns = [
    path("", views.index, name="index"),
    path("predict/", views.predict, name="predict"),
    path("history/", views.history, name="history"),
]
