#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features.py
-----------
Tek MRI goruntusunden sabit boyutlu ozellik vektoru cikarma.

Ozellik gruplari:
- HOG (Histogram of Oriented Gradients)
- LBP histogrami (Local Binary Pattern)
- GLCM Haralick ozellikleri (contrast, homogeneity, energy, correlation)
- Gri-ton histogrami + temel istatistikler (mean, std, skew, kurtosis, entropy)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from skimage.feature import (
    graycomatrix,
    graycoprops,
    hog,
    local_binary_pattern,
)

# ==================== HOG ====================
_HOG_ORIENTATIONS = 9
_HOG_PIXELS_PER_CELL = (16, 16)
_HOG_CELLS_PER_BLOCK = (2, 2)

# ==================== LBP ====================
_LBP_RADIUS = 3
_LBP_N_POINTS = 8 * _LBP_RADIUS
_LBP_N_BINS = _LBP_N_POINTS + 2  # uniform LBP icin

# ==================== GLCM ====================
_GLCM_DISTANCES = [1, 3]
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]

# ==================== Histogram ====================
_HIST_BINS = 32


def _ensure_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Goruntunun 2D gri-ton uint8 formatinda olmasini garanti et.

    Not: uint8 giriste max 255 > 1.0 oldugundan min-max normalizasyon her zaman
    uygulanir.  Float [0,1] girdi icin de min-max kaydirma yapilir; bu tutarli
    bir davranistir ancak min > 0 ise hafif brightness shift'e neden olabilir.
    """
    if image.ndim == 3:
        # RGB -> gri-ton donusumu
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    img = image.astype(np.float64)
    if img.max() > 1.0 or img.min() < 0.0:
        img = img - img.min()
        denom = img.max()
        if denom > 0:
            img = img / denom
    return (img * 255).astype(np.uint8)


def _extract_hog(gray: np.ndarray) -> np.ndarray:
    features = hog(
        gray,
        orientations=_HOG_ORIENTATIONS,
        pixels_per_cell=_HOG_PIXELS_PER_CELL,
        cells_per_block=_HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float64)


def _extract_lbp(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, _LBP_N_POINTS, _LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=_LBP_N_BINS,
        range=(0, _LBP_N_BINS),
        density=True,
    )
    return hist.astype(np.float64)


def _extract_glcm(gray: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(
        gray,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    feats: list[float] = []
    for prop in _GLCM_PROPS:
        values = graycoprops(glcm, prop)
        feats.extend(values.ravel().tolist())
    return np.array(feats, dtype=np.float64)


def _extract_histogram_stats(gray: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray.ravel(), bins=_HIST_BINS, range=(0, 256), density=True)
    flat = gray.ravel().astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    skew = float(sp_stats.skew(flat))
    kurt = float(sp_stats.kurtosis(flat))
    # Sabit goruntuler icin skew/kurtosis NaN donebilir; 0.0 olarak ele al
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    # Shannon entropy
    hist_nonzero = hist[hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
    stats_vec = np.array([mean, std, skew, kurt, entropy], dtype=np.float64)
    return np.concatenate([hist, stats_vec])


def extract_features(image: np.ndarray) -> np.ndarray:
    """Tek goruntuden sabit boyutlu ozellik vektoru cikar.

    Parameters
    ----------
    image : np.ndarray
        2D gri-ton veya 3D RGB goruntu.

    Returns
    -------
    np.ndarray
        1D ozellik vektoru (float64).
    """
    gray = _ensure_gray_uint8(image)
    parts = [
        _extract_hog(gray),
        _extract_lbp(gray),
        _extract_glcm(gray),
        _extract_histogram_stats(gray),
    ]
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError(
            "Ozellik vektorunde NaN veya Inf degeri tespit edildi. "
            "Girdi goruntusu kontrol edilmeli."
        )
    return result
