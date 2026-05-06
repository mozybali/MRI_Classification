"""
xgb_inference.py
----------------
XGBoost tahmin yardımcısı. `model.inference` torch'u modül seviyesinde import
ettiği için, web env'de torch kurulu olmasa da XGBoost akışının çalışabilmesi
için bu küçük yardımcı kullanılır.

`predict_image_xgb`'in mantığını birebir tekrarlar; ancak yalnızca XGBoost
yolunda gereken paketleri (PIL, numpy ve `model.sl.features`) import eder.
Preprocess akışı GorselIsleyici'yi (transitif olarak tqdm) gerektirir.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def predict_image_xgb_local(
    model,
    image_path: Path,
    image_size: int,
    class_names,
    *,
    apply_mri_preprocessing: bool = False,
):
    """`model.inference.predict_image_xgb` ile aynı çıktıyı üretir."""
    from model.sl.features import extract_features

    if apply_mri_preprocessing:
        from goruntu_isleme.goruntu_isleyici import GorselIsleyici

        isleyici = GorselIsleyici()
        processed = isleyici.goruntu_isle(str(image_path))
        if processed is None:
            raise RuntimeError(
                f"MRI ön işleme başarısız (kalite kontrol veya yükleme hatası): {image_path}"
            )
        gray = Image.fromarray(processed).convert("L")
    else:
        gray = Image.open(image_path).convert("L")

    gray = gray.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(gray, dtype=np.uint8)
    features = extract_features(arr).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "dosya": str(image_path),
        "tahmin_sinif": pred_idx,
        "tahmin_adi": class_names[pred_idx],
        "guven_skoru": float(probs[pred_idx]),
        "olasiliklar": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }
