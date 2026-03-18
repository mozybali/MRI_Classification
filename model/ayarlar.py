#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ayarlar.py
----------
Derin öğrenme model eğitimi için merkezi konfigürasyon dosyası.
"""

from pathlib import Path

# ==================== PROJE YOLLARI ====================
PROJE_KOK = Path(__file__).parent.parent

# Varsayılan veri dizinleri
# Augmented veri: yalnızca train + validation için kullanılır
TRAINVAL_VERI_DIZINI = PROJE_KOK / "Veri_Seti" / "AugmentedAlzheimerDataset"
# Original veri: yalnızca test için kullanılır
TEST_VERI_DIZINI = PROJE_KOK / "Veri_Seti" / "OriginalDataset"

# Geriye dönük uyumluluk
VARSAYILAN_VERI_DIZINI = TRAINVAL_VERI_DIZINI

# Model çıktıları
CIKTI_KLASORU = PROJE_KOK / "model" / "ciktilar"
MODELS_KLASORU = CIKTI_KLASORU / "modeller"
RAPORLAR_KLASORU = CIKTI_KLASORU / "raporlar"
GORSELLER_KLASORU = CIKTI_KLASORU / "gorseller"

# ==================== GENEL ====================
RASTGELE_TOHUM = 42
