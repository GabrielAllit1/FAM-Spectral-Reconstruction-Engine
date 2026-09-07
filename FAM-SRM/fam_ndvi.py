# ===============================================================
# fam_ndvi.py — FAM-SRM v2.0
#
# NDVI_FAM MODULE
#
# This module computes a FAM-derived normalized vegetation feature index from:
#     NDVI_FAM = (vnir - R) / (vnir + R + eps)
#
# IMPORTANT: vnir is synthesized from RGB by FAM-SRM. NDVI_FAM is therefore
# a model-derived feature index, not sensor-measured NDVI or physical NIR
# reflectance. Independent measured-NIR validation is required before treating
# it as equivalent to conventional multispectral NDVI.
#
# Includes:
#   • CPU-only safe operations
#   • Multi-colormap export
#   • TIFF/JPG/PNG safety
#   • MLA-PDF integration
#   • Normalized output [0,1]
#
# ===============================================================

import numpy as np
import cv2
import os


# ---------------------------------------------------------------
# Core NDVI_FAM computation
# ---------------------------------------------------------------
def NDVI_FAM(vnir: np.ndarray, R: np.ndarray, eps: float = 1e-6, mode: str = "linear") -> np.ndarray:
    """
    Compute the FAM-derived normalized vegetation feature index.

    ``vnir`` is a synthetic channel produced from RGB by FAM-SRM. The result
    is not sensor-measured NDVI and should not be interpreted as calibrated
    multispectral reflectance without independent validation.

    Parameters
    ----------
    vnir : np.ndarray
        Synthetic virtual-NIR feature channel in [0,1].
    R : np.ndarray
        Red channel (normalized [0,1]).
    eps : float
        Stabilizer for denominator.
    mode : str
        Output mapping: 'linear' for direct normalized remapping or 'soft'
        for visualization-oriented contrast compression.

    Returns
    -------
    np.ndarray in [0,1]
    """
    vnir = vnir.astype(np.float32)
    R = R.astype(np.float32)

    raw = (vnir - R) / (vnir + R + eps)

    # Output mapping selection. Neither mapping changes the synthetic-source
    # claim boundary of vnir/NDVI_FAM.
    if mode == "soft":
        ndvi = 0.5 + 0.5 * np.tanh(2.0 * raw)  # visualization mapping
    else:  # mode == "linear" (default)
        ndvi = (raw + 1.0) / 2.0               # direct linear remapping

    # Clip and return
    ndvi = np.clip(ndvi, 0, 1)
    return ndvi.astype(np.float32)


# ---------------------------------------------------------------
# NDVI Colormaps for MLA-PDF Pages
# ---------------------------------------------------------------
NDVI_COLORMAPS = {
    "classic": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet": cv2.COLORMAP_JET,
    "gva": cv2.COLORMAP_SUMMER,  # green → cyan → white
}


def save_ndvi_colormaps(ndvi: np.ndarray, out_prefix: str) -> dict:
    """
    Save multiple NDVI_FAM colormap variants for PDF export.

    Parameters
    ----------
    ndvi : 2D float32 [0,1]
    out_prefix : str
        Output path prefix

    Returns
    -------
    dict : {colormap_name: file_path}
    """
    if ndvi.ndim != 2:
        raise ValueError("NDVI_FAM must be a 2D single-channel array.")

    cmap_outputs = {}

    # Convert normalized index to 0–255 uint8 for display.
    ndvi8 = (np.clip(ndvi, 0, 1) * 255).astype(np.uint8)

    for name, cmap in NDVI_COLORMAPS.items():
        colored = cv2.applyColorMap(ndvi8, cmap)
        out_path = f"{out_prefix}_ndvi_{name}.png"
        cv2.imwrite(out_path, colored)
        cmap_outputs[name] = out_path

    return cmap_outputs


# ---------------------------------------------------------------
# NDVI_FAM Legend Generator
# ---------------------------------------------------------------
def save_ndvi_legend(out_path: str, cmap=cv2.COLORMAP_JET):
    """
    Save a horizontal normalized-index legend colorbar (0→1).
    """
    bar = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (35, 1))
    bar_c = cv2.applyColorMap(bar, cmap)
    cv2.imwrite(out_path, bar_c)


# ---------------------------------------------------------------
# NDVI_FAM Histogram Generator
# ---------------------------------------------------------------
def save_ndvi_histogram(ndvi: np.ndarray, out_path: str):
    """
    Save a histogram image for the normalized NDVI_FAM distribution.
    """
    arr = np.clip(ndvi.flatten(), 0, 1)
    hist = np.histogram(arr, bins=64, range=(0, 1))[0]

    h = 200
    w = 400
    img = np.ones((h, w, 3), np.uint8) * 255
    hist = hist / hist.max()

    for i in range(len(hist)):
        x = int((i / 63) * (w - 1))
        y = int(hist[i] * (h - 1))
        cv2.line(img, (x, h - 1), (x, h - 1 - y), (0, 0, 0), 1)

    cv2.imwrite(out_path, img)
