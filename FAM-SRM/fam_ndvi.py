# ===============================================================
# fam_ndvi.py — FAM-SRM v2.0
#
# NDVI_FAM MODULE
#
# This module computes the FAM-enhanced NDVI from:
#     NDVI_FAM = (vnir - R) / (vnir + R + eps)
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
    Compute FAM-enhanced NDVI.

    Parameters
    ----------
    vnir : np.ndarray
        Virtual NIR channel in [0,1]
    R : np.ndarray
        Red channel (normalized [0,1])
    eps : float
        Stabilizer for denominator
    mode : str
        Enhancement mode: 'linear' (scientific/GIS), 'soft' (visualization/PDFs)

    Returns
    -------
    np.ndarray in [0,1]
    """
    vnir = vnir.astype(np.float32)
    R = R.astype(np.float32)

    raw = (vnir - R) / (vnir + R + eps)

    # Enhancement mode selection
    if mode == "soft":
        ndvi = 0.5 + 0.5 * np.tanh(2.0 * raw)  # preserves extremes for visualization
    else:  # mode == "linear" (default, scientifically accurate)
        ndvi = (raw + 1.0) / 2.0               # clean linear mapping

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
    Save multiple NDVI colormap variants for PDF export.

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
        raise ValueError("NDVI must be a 2D single-channel array.")

    cmap_outputs = {}

    # Convert NDVI to 0–255 uint8
    ndvi8 = (np.clip(ndvi, 0, 1) * 255).astype(np.uint8)

    for name, cmap in NDVI_COLORMAPS.items():
        colored = cv2.applyColorMap(ndvi8, cmap)
        out_path = f"{out_prefix}_ndvi_{name}.png"
        cv2.imwrite(out_path, colored)
        cmap_outputs[name] = out_path

    return cmap_outputs


# ---------------------------------------------------------------
# NDVI Legend Generator
# ---------------------------------------------------------------
def save_ndvi_legend(out_path: str, cmap=cv2.COLORMAP_JET):
    """
    Save a horizontal NDVI legend colorbar (0→1).
    """
    bar = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (35, 1))
    bar_c = cv2.applyColorMap(bar, cmap)
    cv2.imwrite(out_path, bar_c)


# ---------------------------------------------------------------
# NDVI Histogram Generator
# ---------------------------------------------------------------
def save_ndvi_histogram(ndvi: np.ndarray, out_path: str):
    """
    Save a histogram image for NDVI distribution.
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
