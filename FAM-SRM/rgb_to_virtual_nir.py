# ===============================================================
# rgb_to_virtual_nir.py — FAM-SRM v2.0 (CPU VERSION, FAST)
#
# Safe, crash-proof virtual NIR generator implementing:
#   • KRO (Kolakoski Recursive Operator)
#   • TMBE (Thue–Morse Bias Equalizer)
#   • USK (Ulam Spiral Kernel)
#   • BES (Beatty Entropy Scaling)  [FAST, VARIANCE-BASED]
#   • FDM (Fractal Dimension Modulator)
# ===============================================================

import numpy as np
import cv2


# ---------------------------------------------------------------
# Safeguard: Enforce 2D float32 for all convolution operators
# ---------------------------------------------------------------
def safe_filter2D(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Enforce 2D, float32, CPU-safe filter2D."""
    if src.ndim != 2:
        raise ValueError("safe_filter2D received multi-channel input — expected 2D single channel.")
    src = src.astype(np.float32)
    return cv2.filter2D(src, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------
# Kolakoski kernel generator
# ---------------------------------------------------------------
def kolakoski_kernel(size: int = 7) -> np.ndarray:
    seq = [1, 2]
    while len(seq) < size * size:
        last = seq[-1]
        nxt = 1 if last == 2 else 2
        seq.append(nxt)

    arr = np.array(seq[: size * size], np.float32)
    arr = arr.reshape(size, size)
    arr /= arr.sum()
    return arr


# ---------------------------------------------------------------
# Thue–Morse parity mask
# ---------------------------------------------------------------
def thue_morse_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            x = i * w + j
            mask[i, j] = bin(x).count("1") % 2
    return mask


# ---------------------------------------------------------------
# Ulam spiral radial kernel
# ---------------------------------------------------------------
def ulam_kernel(size: int = 11) -> np.ndarray:
    c = size // 2
    k = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - c)**2 + (j - c)**2)
            k[i, j] = 1.0 / (1.0 + r)
    k /= k.sum()
    return k


# ---------------------------------------------------------------
# FAST Beatty Entropy Scaling (variance-based local entropy proxy)
# ---------------------------------------------------------------
def beatty_entropy_scale(img: np.ndarray) -> np.ndarray:
    """
    Fast BES implementation.

    Uses local variance from Gaussian-blurred img and img^2 as an
    entropy proxy, then mixes with a Beatty sequence field.
    Complexity is O(N) instead of O(N * window^2).
    """
    img = img.astype(np.float32)
    img = np.clip(img, 0, 1)

    h, w = img.shape
    # Local mean and variance as entropy proxy
    ksize = 11  # odd
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

    # Beatty sequence field
    phi = 1.61803398875
    B = ((np.arange(h * w, dtype=np.float32) * phi) % 1.0).reshape(h, w)

    # Mix original with Beatty * entropy proxy
    mix = 0.5 * img + 0.5 * B * ent_proxy
    return np.clip(mix, 0, 1).astype(np.float32)


# ---------------------------------------------------------------
# Fractal Dimension Modulator (FDM)
# ---------------------------------------------------------------
def fractal_dimension_modulator(img: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(img, cv2.CV_32F)
    mag = np.abs(lap)
    sm = cv2.GaussianBlur(mag, (7, 7), 0)
    out = cv2.normalize(sm, None, 0, 1, cv2.NORM_MINMAX)
    return out.astype(np.float32)


# ---------------------------------------------------------------
# MAIN OPERATOR
# ---------------------------------------------------------------
def rgb_to_vnir(rgb: np.ndarray) -> np.ndarray:
    """
    CPU-safe vNIR generator.
    Crash-proof for TIFF and RGB images.
    """

    rgb = rgb.astype(np.float32)

    # Normalize to [0,1]
    if rgb.max() > 1.0:
        rgb /= 255.0

    # Enforce 3 channels only
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB must be H×W×3.")

    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]

    # Split channels
    B = rgb[:, :, 0]
    G = rgb[:, :, 1]
    R = rgb[:, :, 2]

    # Y must be 2D for all kernels
    Y = (0.45 * G + 0.35 * R + 0.20 * B).astype(np.float32)

    # === KRO ===
    kro_k = kolakoski_kernel(7)
    Y1 = safe_filter2D(Y, kro_k)

    # === TMBE ===
    tm = thue_morse_mask(*Y.shape)
    Y2 = np.clip(Y1 * (0.5 + 0.5 * tm), 0, 1)

    # === USK ===
    usk = ulam_kernel(11)
    Y3 = safe_filter2D(Y2, usk)

    # === BES (fast) ===
    Y4 = beatty_entropy_scale(Y3)

    # === FDM ===
    vnir = fractal_dimension_modulator(Y4)

    return np.clip(vnir, 0, 1).astype(np.float32)
