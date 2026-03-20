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
# Dogru metodoloji geregi varsayilan kaynak original veri uzerinden train/val/test split'tir.
# Islenmis veri dizinleri yalnizca ilgili CLI flag'leri ile secilir.
TRAINVAL_VERI_DIZINI = PROJE_KOK / "Veri_Seti" / "OriginalDataset"
TEST_VERI_DIZINI = PROJE_KOK / "Veri_Seti" / "OriginalDataset"

# Geriye donuk uyumluluk
VARSAYILAN_VERI_DIZINI = TRAINVAL_VERI_DIZINI

# Model ciktilari
CIKTI_KLASORU = PROJE_KOK / "model" / "ciktilar"
MODELS_KLASORU = CIKTI_KLASORU / "modeller"
RAPORLAR_KLASORU = CIKTI_KLASORU / "raporlar"
GORSELLER_KLASORU = CIKTI_KLASORU / "gorseller"
HPO_KLASORU = CIKTI_KLASORU / "hiperparametre_arama"

# ==================== GENEL ====================
RASTGELE_TOHUM = 42
