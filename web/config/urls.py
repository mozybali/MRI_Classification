"""URL yapılandırması — MRI Classification Web Uygulaması"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("infer/", include("inference_app.urls", namespace="infer")),
    path("dashboard/", include("dashboard_app.urls", namespace="dashboard")),
    # Kök URL → tahmin ekranına yönlendir
    path("", RedirectView.as_view(url="/infer/", permanent=False)),
]

# Geliştirme ortamında media dosyalarını serve et
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
