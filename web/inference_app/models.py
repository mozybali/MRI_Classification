from django.db import models


CLASS_CHOICES = [
    ("NonDemented",      "NonDemented"),
    ("VeryMildDemented", "VeryMildDemented"),
    ("MildDemented",     "MildDemented"),
    ("ModerateDemented", "ModerateDemented"),
]

MODEL_CHOICES = [
    ("resnet",   "ResNet18"),
    ("xgboost",  "XGBoost"),
    ("yolo",     "YOLOv8"),
]


class PredictionRecord(models.Model):
    """Bir MRI tahmin işleminin kalıcı kaydı."""

    image           = models.ImageField(upload_to="predictions/%Y/%m/%d/")
    model_type      = models.CharField(max_length=20, choices=MODEL_CHOICES)
    predicted_class = models.CharField(max_length=50, choices=CLASS_CHOICES)
    confidence      = models.FloatField()
    probabilities   = models.JSONField()      # {"NonDemented": 0.92, ...}
    apply_preprocess = models.BooleanField(default=False)
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Tahmin Kaydı"
        verbose_name_plural = "Tahmin Kayıtları"

    def __str__(self):
        return (
            f"[{self.model_type.upper()}] {self.predicted_class} "
            f"({self.confidence:.1%}) — {self.created_at:%Y-%m-%d %H:%M}"
        )

    @property
    def css_class(self):
        """Sınıfa göre CSS badge sınıfı."""
        return {
            "NonDemented":      "none",
            "VeryMildDemented": "very-mild",
            "MildDemented":     "mild",
            "ModerateDemented": "moderate",
        }.get(self.predicted_class, "none")

    @property
    def severity_label(self):
        return {
            "NonDemented":      "Sağlıklı",
            "VeryMildDemented": "Çok Hafif Demans",
            "MildDemented":     "Hafif Demans",
            "ModerateDemented": "Orta Demans",
        }.get(self.predicted_class, self.predicted_class)

    @property
    def confidence_percent(self):
        return round(self.confidence * 100)
