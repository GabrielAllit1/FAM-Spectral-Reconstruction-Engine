# ===============================================================
# vegetation_fractal_stress.py — FAM-SRM v2.0 (CPU VERSION)
#
# Computes the Fractal Stress Index (FSI):
#     FSI = normalize( FractalDetail(vNIR) - SmoothBackground )
#
# FSI highlights:
#   • Vegetation thinning
#   • Crown anomalies
#   • Expansion joint stress patterns
#   • High-frequency canopy disruptions
#
# ---------------------------------------------------------------
# SAFE FOR:
#   • TIFF orthomosaics
#   • JPG/PNG RGB images
#   • MLA-PDF export
#   • CPU-only environments
# ===============================================================

import numpy as np
import cv2


# ---------------------------------------------------------------
# Safe convolution wrapper
# ---------------------------------------------------------------
def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure the image is 2D single-channel."""
    if img.ndim != 2:
        raise ValueError("FSI requires a 2D vNIR image.")
    return img.astype(np.float32)


# ---------------------------------------------------------------
# Fractal Detail Extraction
# ---------------------------------------------------------------
def fractal_enhance(img: np.ndarray) -> np.ndarray:
    """
    Extract local fractal texture using Laplacian magnitude,
    then stabilize with Gaussian smoothing.
    """
    img = _ensure_gray(img)

    # Local curvature (2nd derivative)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)

    # Suppress noise, retain structural texture
    sm = cv2.GaussianBlur(mag, (7, 7), 0)

    # Normalize to [0,1] for report export
    out = cv2.normalize(sm, None, 0, 1, cv2.NORM_MINMAX)
    return out.astype(np.float32)


# ---------------------------------------------------------------
# FSI COMPUTATION
# ---------------------------------------------------------------
def vegetation_FSI(vnir: np.ndarray) -> np.ndarray:
    """
    Compute Fractal Stress Index (FSI) from vNIR.

    Parameters
    ----------
    vnir : np.ndarray
        2D virtual NIR array (float32 [0,1])

    Returns
    -------
    np.ndarray
        FSI map in [0,1]
    """
    vnir = _ensure_gray(vnir)

    # Step 1: Local fractal signal
    f1 = fractal_enhance(vnir)

    # Step 2: Coarse background (removes slow spatial trends)
    smooth = cv2.GaussianBlur(f1, (21, 21), 0)

    # Step 3: Structural residual
    residual = f1 - smooth

    # Step 4: Normalize to [0,1] as required by MLA-PDF engine
    fsi = cv2.normalize(residual, None, 0, 1, cv2.NORM_MINMAX)

    return fsi.astype(np.float32)
