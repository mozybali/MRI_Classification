#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ayarlar.py
----------
Derin ogrenme model egitimi icin merkezi konfigurasyon dosyasi.
"""

from pathlib import Path

# ==================== PROJE YOLLARI ====================
PROJE_KOK = Path(__file__).parent.parent
ISLENMIS_VERI_KLASORU = PROJE_KOK / "goruntu_isleme" / "cikti"
ISLENMIS_TRAINVAL_VERI_DIZINI = ISLENMIS_VERI_KLASORU / "trainval"
ISLENMIS_TEST_VERI_DIZINI = ISLENMIS_VERI_KLASORU / "test"

# Varsayilan veri dizinleri
# Model egitimi varsayilan olarak leak-free sekilde uretilmis islenmis split'leri kullanir.
TRAINVAL_VERI_DIZINI = ISLENMIS_TRAINVAL_VERI_DIZINI
TEST_VERI_DIZINI = ISLENMIS_TEST_VERI_DIZINI

# Model ciktilari
CIKTI_KLASORU = PROJE_KOK / "model" / "ciktilar"
MODELS_KLASORU = CIKTI_KLASORU / "modeller"
RAPORLAR_KLASORU = CIKTI_KLASORU / "raporlar"
GORSELLER_KLASORU = CIKTI_KLASORU / "gorseller"
HPO_KLASORU = CIKTI_KLASORU / "hiperparametre_arama"

# SL ozellik cache
SL_FEATURE_CACHE_KLASORU = CIKTI_KLASORU / "sl_ozellikler"

# ==================== GENEL ====================
RASTGELE_TOHUM = 42

# ==================== EGITIM ====================
VARSAYILAN_EARLY_STOPPING_SABIR = 30
