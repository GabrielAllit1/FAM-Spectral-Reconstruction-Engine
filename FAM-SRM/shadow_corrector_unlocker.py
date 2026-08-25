# ===============================================================
# shadow_corrector_unlocker.py — FAM-SRM v2.0 (CPU VERSION, FAST)
#
# Shadow Corrector + Luminance Unlocker
#
# Uses:
#   • RGB-based illumination field
#   • Variance-based entropy proxy (no slow loops)
# ===============================================================

import numpy as np
import cv2


# ---------------------------------------------------------------
# Utility: Ensure float32 2D image
# ---------------------------------------------------------------
def _ensure_float32_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("ShadowCorrector expected single-channel 2D vNIR.")
    return img.astype(np.float32)


# ---------------------------------------------------------------
# Compute a local illumination field using Gaussian smoothing
# ---------------------------------------------------------------
def _estimate_illumination(rgb: np.ndarray, sigma: int = 31) -> np.ndarray:
    """
    Estimate a soft illumination field from the RGB brightness.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB input must be H×W×3.")

    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0

    # Per-pixel brightness
    b = np.mean(rgb[:, :, :3], axis=2).astype(np.float32)

    # Smooth to estimate large-scale illumination component
    ill = cv2.GaussianBlur(b, (sigma, sigma), 0)

    # Normalize illumination to [0,1]
    ill_n = cv2.normalize(ill, None, 0, 1, cv2.NORM_MINMAX)
    return ill_n.astype(np.float32)


# ---------------------------------------------------------------
# FAST entropy-based luminance unlocking
# ---------------------------------------------------------------
def _entropy_unlocker(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Enhances local texture detail using a variance-based entropy proxy.
    Complexity is O(N) using Gaussian blurs, no nested Python loops.
    """
    img = _ensure_float32_gray(img)
    img = np.clip(img, 0, 1)

    ksize = 11  # neighborhood for variance/entropy proxy
    mean = cv2.GaussianBlur(img, (ksize, ksize), 0)
    mean_sq = cv2.GaussianBlur(img * img, (ksize, ksize), 0)
    var = mean_sq - mean * mean
    var = np.clip(var, 0, None)

    # Normalize variance to [0,1] as entropy-like field
    v_min, v_max = var.min(), var.max()
    if v_max > v_min:
        ent_proxy = (var - v_min) / (v_max - v_min)
    else:
        ent_proxy = np.zeros_like(var, np.float32)

    out = (1.0 - strength) * img + strength * ent_proxy
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------
# MAIN SHADOW CORRECTOR
# ---------------------------------------------------------------
def shadow_corrector(vnir: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Shadow-aware correction of vNIR using RGB illumination cues.

    Parameters
    ----------
    vnir : np.ndarray
        Virtual NIR proxy (2D float32 [0,1])
    rgb : np.ndarray
        Raw RGB image (3-channel)
    """

    vnir = _ensure_float32_gray(vnir)

    # Step 1 — Estimate large-scale illumination
    illum = _estimate_illumination(rgb, sigma=31)

    # Step 2 — Normalize vNIR by the illumination field
    corrected = vnir / (illum + 1e-6)

    # Step 3 — Re-normalize to [0,1]
    corrected = cv2.normalize(corrected, None, 0, 1, cv2.NORM_MINMAX)

    # Step 4 — Local entropy/variance unlocker (enhances canopy structure)
    unlocked = _entropy_unlocker(corrected, strength=0.30)

    # Step 5 — Final squeeze into [0,1]
    out = np.clip(unlocked, 0, 1).astype(np.float32)

    return out
