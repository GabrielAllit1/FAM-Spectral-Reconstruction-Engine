# ===============================================================
# spectral_graphon_vegetation.py — FAM-SRM v2.0 (CPU VERSION)
#
# Computes the Graphon Structural Residual (GSR):
#     GSR = normalize( vNIR - GraphonSmooth(vNIR) )
#
# Includes:
#   • CPU-safe filter wrappers
#   • Radial 1/(1+r) Ulam-style weighting
#   • TIFF-safe, crash-proof normalization
#   • MLA-PDF compatible output in [0,1]
#
# ===============================================================

import numpy as np
import cv2


# ---------------------------------------------------------------
# Safe convolution wrapper — enforces 2D + float32
# ---------------------------------------------------------------
def safe_filter2D(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """CPU-safe filter2D with strict 2D enforcement."""
    if src.ndim != 2:
        raise ValueError("spectral_graphon_map expected a 2D single-channel array.")
    src = src.astype(np.float32)
    return cv2.filter2D(src, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------
# Graphon kernel — radial 1/(1+r) weighting
# ---------------------------------------------------------------
def graphon_kernel(size: int = 15) -> np.ndarray:
    """
    Create a radial structural kernel based on graph limit theory.

    The kernel emphasizes near-center weights and decays as 1/(1+r),
    stabilizing vegetation texture on orthomosaic-scale imagery.
    """
    if size % 2 == 0:
        raise ValueError("Graphon kernel size must be odd.")

    c = size // 2
    k = np.zeros((size, size), np.float32)

    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - c)**2 + (j - c)**2)
            k[i, j] = 1.0 / (1.0 + r)

    k /= k.sum()
    return k


# ---------------------------------------------------------------
# Main Graphon Structural Residual operator
# ---------------------------------------------------------------
def spectral_graphon_map(vnir: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Compute Graphon Structural Residual (GSR) from vNIR.

    Parameters
    ----------
    vnir : np.ndarray
        Virtual NIR image, float32 in [0,1], 2D
    k : int
        Graphon kernel size (odd). Recommended: 13–17 for UAV orthos.

    Returns
    -------
    np.ndarray
        Residual map normalized to [0,1], MLA-PDF safe.
    """
    # Validate input
    if vnir.ndim != 2:
        raise ValueError("vnir must be a 2D single-channel array.")

    vnir = vnir.astype(np.float32)

    # Build kernel
    kern = graphon_kernel(k)

    # Smooth using graphon kernel
    smooth = safe_filter2D(vnir, kern)

    # Structural residual
    residual = vnir - smooth

    # Normalize to [0,1] for PDF/report export
    out = cv2.normalize(residual, None, 0, 1, cv2.NORM_MINMAX)
    return out.astype(np.float32)
