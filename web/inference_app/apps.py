from django.apps import AppConfig


class InferenceAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'inference_app'

    def ready(self):
        """Sunucu başlarken ML modellerini arka planda önbelleğe al."""
        import threading

        def _warmup():
            try:
                from django.conf import settings
                from .services.model_loader import registry
                model_dir = settings.MODEL_DIR
                if not model_dir.exists():
                    return
                for path in model_dir.iterdir():
                    if path.suffix.lower() == ".json" and not path.name.endswith(".meta.json"):
                        registry.get_xgboost(path)
                        print(f"[warmup] XGBoost yüklendi: {path.name}")
                    # ResNet için torch yüklü olması gerektiğinden atla
            except Exception as exc:
                print(f"[warmup] Model ön yükleme atlandı: {exc}")

        t = threading.Thread(target=_warmup, daemon=True)
        t.start()

