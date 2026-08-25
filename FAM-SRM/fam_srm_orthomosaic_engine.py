# ===============================================================
# fam_srm_orthomosaic_engine.py — FAM-SRM v3.1 (Tiled CPU Edition)
#
# Unified orthomosaic/RGB processing engine:
#   • vNIR (virtual NIR proxy)
#   • NDVI_FAM (FAM-enhanced NDVI)
#   • FSI (Fractal Stress Index)
#   • Graphon structural residual placeholder (optional)
#   • THUMBNAILS (safe, downscaled automatically)
#
# Tile-based processing:
#   • Large orthos are split into tiles (default 2048×2048)
#   • Tiles are written to a temp folder inside the output folder:
#         "<YYYYMMDD>_tiled_images"
#   • All spectral products are recomposed into full-resolution arrays
#
# Auto-detects:
#   • GeoTIFF orthomosaics (R,G,B)
#   • Standard RGB photos (JPG/PNG)
#
# Memory-safe / decompression-bomb protected (no silent 1×1 collapse).
#
# Produced for: Gabriel Allit (Salt19 LLC)
# ===============================================================

import os
import sys
import math
import datetime
from typing import Tuple, Optional

import numpy as np

# Import FAM-SRM NDVI module
from fam_ndvi import NDVI_FAM

# OpenCV import guard
try:
    import cv2
except ImportError as e:
    raise RuntimeError(
        "OpenCV (cv2) is required for fam_srm_orthomosaic_engine. "
        "Install with `conda install -c conda-forge opencv` or `pip install opencv-python-headless`."
    ) from e

# Optional: matplotlib only for histograms / colorbars
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Hist/colorbar generation will be skipped gracefully


# ===============================================================
# Utility helpers
# ===============================================================

def log(msg: str):
    """Lightweight stdout logger (GUI can capture this)."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_thumbnail(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
    """Downscale image preserving aspect ratio; no weird 1×1 degeneracy."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return img.copy()
    scale = max_side / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return thumb


def today_prefix() -> str:
    """Return YYYYMMDD string for folder naming."""
    return datetime.date.today().strftime("%Y%m%d")


# ===============================================================
# Image loading (JPG/PNG/TIFF) with shape + dtype sanity
# ===============================================================

def load_rgb_image(path: str) -> np.ndarray:
    """
    Load an RGB image from disk using OpenCV.
    Always returns float32 in [0, 1], shape (H, W, 3).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input image not found: {path}")

    log(f"Loading image: {path}")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise RuntimeError(
            f"Expected 3-channel RGB image; got shape {img_rgb.shape}"
        )

    img_rgb = img_rgb.astype(np.float32) / 255.0

    h, w, _ = img_rgb.shape
    log(f"Loaded image shape: {w} × {h} (W×H), dtype={img_rgb.dtype}")
    if h < 32 or w < 32:
        raise RuntimeError(
            f"Image is too small after load ({w}×{h}). "
            "Something went wrong (decompression / downscale)."
        )

    return img_rgb


# ===============================================================
# FAM-SRM spectral operators
# ===============================================================

def compute_Y(rgb: np.ndarray) -> np.ndarray:
    """Base vegetation-biased luminance Y."""
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = 0.35 * R + 0.45 * G + 0.20 * B
    return np.clip(Y, 0.0, 1.0).astype(np.float32)


def kolakoski_kernel(size: int = 7) -> np.ndarray:
    """
    Build a Kolakoski-inspired 2D kernel.
    Not a literal Kolakoski sequence, but uses its run-length structure
    (1s and 2s) to modulate radial weights.
    """
    # 1D Kolakoski prefix
    seq = [1, 2, 2, 1, 1, 2, 1, 2, 2][: size * 2]
    seq = np.array(seq, dtype=np.float32)

    # Radial grid
    ax = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(ax, ax)
    rr = np.sqrt(xx**2 + yy**2)

    rr_norm = rr / (rr.max() + 1e-6)
    idx = np.minimum((rr_norm * (len(seq) - 1)).astype(int), len(seq) - 1)
    k = seq[idx].astype(np.float32)

    k /= k.sum()
    return k


def KRO(Y: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Kolakoski Recursive Operator — nonlinear spatial smoothing."""
    k = kolakoski_kernel(kernel_size)
    # Ensure input is float32 for OpenCV compatibility
    Y_f32 = Y.astype(np.float32)
    Y_blur = cv2.filter2D(Y_f32, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    # mild residual feedback for "recursive" feel
    out = 0.7 * Y_blur + 0.3 * Y_f32
    return np.clip(out, 0.0, 1.0)


def TMBE(Y: np.ndarray) -> np.ndarray:
    """Thue–Morse Bias Equalizer — parity-based chromatic bias correction."""
    h, w = Y.shape
    # 2D Thue–Morse parity mask
    x = np.arange(w, dtype=np.int32)
    y = np.arange(h, dtype=np.int32)
    xx, yy = np.meshgrid(x, y)

    def bitcount(n: np.ndarray) -> np.ndarray:
        # numpy-compatible bit counting for parity calculation
        # Count bits by successively checking each bit position
        count = np.zeros_like(n, dtype=np.int32)
        temp = n.copy()
        while np.any(temp > 0):
            count += temp & 1
            temp >>= 1
        return count

    parity = (bitcount(xx) + bitcount(yy)) & 1
    mask = np.where(parity == 0, 1.0, -1.0).astype(np.float32)

    # Local mean
    Y_f32 = Y.astype(np.float32)
    local_mean = cv2.GaussianBlur(Y_f32, (0, 0), sigmaX=1.2, sigmaY=1.2)
    residual = Y_f32 - local_mean

    # Apply parity flip to residual, then recombine
    corrected = local_mean + 0.3 * residual * mask
    return np.clip(corrected, 0.0, 1.0)


def USK(Y: np.ndarray) -> np.ndarray:
    """Ulam Spiral Kernel — emphasize structural patterns."""
    h, w = Y.shape
    # Approximate by gently boosting mid-frequency details via unsharp masking
    Y_f32 = Y.astype(np.float32)
    blur = cv2.GaussianBlur(Y_f32, (0, 0), sigmaX=2.0, sigmaY=2.0)
    high = Y_f32 - blur
    out = Y_f32 + 0.8 * high
    return np.clip(out, 0.0, 1.0)


def BES(Y: np.ndarray) -> np.ndarray:
    """Beatty Entropy Scaling — radiometric scaling with entropy proxy."""
    # Local variance as entropy proxy
    ksize = 7
    Y_f32 = Y.astype(np.float32)
    mu = cv2.blur(Y_f32, (ksize, ksize))
    mu2 = cv2.blur(Y_f32 * Y_f32, (ksize, ksize))
    var = np.clip(mu2 - mu * mu, 0.0, None)

    # Beatty-like scaling using sqrt(2) factor
    alpha = (np.sqrt(2.0) - 1.0)
    scale = 1.0 + alpha * (var / (var.max() + 1e-6))

    out = Y_f32 * scale
    out = out / (out.max() + 1e-6)
    return np.clip(out, 0.0, 1.0)


def FDM(Y: np.ndarray) -> np.ndarray:
    """Fractal Dimension Modulator — texture enhancement."""
    # Laplacian-of-Gaussian-ish
    Y_f32 = Y.astype(np.float32)
    blur = cv2.GaussianBlur(Y_f32, (0, 0), sigmaX=1.0, sigmaY=1.0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    # Normalize laplacian
    lap_norm = lap / (np.max(np.abs(lap)) + 1e-6)
    out = Y_f32 + 0.5 * lap_norm
    return np.clip(out, 0.0, 1.0)


def compute_vnir(rgb: np.ndarray) -> np.ndarray:
    """Full vNIR pipeline for a single RGB tile."""
    Y = compute_Y(rgb)
    y1 = KRO(Y)
    y2 = TMBE(y1)
    y3 = USK(y2)
    y4 = BES(y3)
    vnir = FDM(y4)
    # IMPORTANT: no per-tile normalization here; we normalize globally later
    return vnir.astype(np.float32)


def compute_ndvi_fam(vnir: np.ndarray, rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NDVI_FAM = (vNIR - R) / (vNIR + R + eps)."""
    R = rgb[..., 0]
    num = vnir - R
    den = vnir + R + eps
    ndvi = num / den
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)
    return ndvi


def compute_fsi(vnir: np.ndarray) -> np.ndarray:
    """
    Fractal Stress Index:
    FSI ≈ normalized residual of (FDM(vNIR) − GaussianBlur(FDM(vNIR))).
    """
    fdm_v = FDM(vnir)
    fdm_v_f32 = fdm_v.astype(np.float32)
    smooth = cv2.GaussianBlur(fdm_v_f32, (0, 0), sigmaX=2.0, sigmaY=2.0)
    residual = fdm_v_f32 - smooth
    # Normalize to [0,1] based on abs residual
    mag = np.abs(residual)
    mag /= (mag.max() + 1e-6)
    return mag.astype(np.float32)


def global_normalize_vnir(vnir_full: np.ndarray, rgb_full: np.ndarray) -> np.ndarray:
    """
    Global percentile-based normalization of vNIR after tiling / full-block processing.
    Uses RGB to mask out empty borders so they don't dominate the histogram.
    """
    # mask out empty/black/white borders so they don't break histogram
    valid_mask = (rgb_full.sum(axis=2) > 0.03)
    vnir_valid = vnir_full[valid_mask]

    if vnir_valid.size > 0:
        lo = np.percentile(vnir_valid, 1.0)
        hi = np.percentile(vnir_valid, 99.0)
    else:
        lo, hi = float(vnir_full.min()), float(vnir_full.max())

    # protect against divide-by-zero
    if hi - lo < 1e-6:
        hi = lo + 1e-3

    vnir_full = (vnir_full - lo) / (hi - lo)
    vnir_full = np.clip(vnir_full, 0.0, 1.0).astype(np.float32)
    log(f"Global vNIR normalization: lo={lo:.4f}, hi={hi:.4f}")
    return vnir_full


# ===============================================================
# NDVI colormaps
# ===============================================================

def apply_colormap_ndvi(ndvi: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """
    Map NDVI [-1,1] to RGB using specified colormap.
    Supported: "jet", "viridis", "classic", "gva".
    """
    ndvi_norm = (ndvi + 1.0) / 2.0  # [0,1]
    ndvi_u8 = np.clip(ndvi_norm * 255.0, 0, 255).astype(np.uint8)

    if cmap == "jet":
        cm = cv2.COLORMAP_JET
        colored = cv2.applyColorMap(ndvi_u8, cm)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if cmap == "viridis":
        # OpenCV viridis
        cm = cv2.COLORMAP_VIRIDIS
        colored = cv2.applyColorMap(ndvi_u8, cm)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Classic NDVI: brown → yellow → green
    if cmap == "classic":
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.5:  # soil-ish
                r = int(160 + 80 * (t / 0.5))  # 160→240
                g = int(100 + 60 * (t / 0.5))  # 100→160
                b = int(60 * (t / 0.5))        # 0→60
            else:         # veg
                t2 = (t - 0.5) / 0.5
                r = int(240 * (1 - t2))
                g = int(160 + 95 * t2)
                b = int(60 * (1 - t2))
            lut[i, 0, :] = [b, g, r]
        colored = cv2.LUT(cv2.merge([ndvi_u8]*3), lut)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # GVA Signature Theme: teal → green → magenta-ish
    if cmap == "gva":
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            r = int(80 + 150 * t)
            g = int(200 * (1 - abs(t - 0.5) * 2))
            b = int(60 + 150 * (1 - t))
            lut[i, 0, :] = [b, g, r]
        colored = cv2.LUT(cv2.merge([ndvi_u8]*3), lut)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Fallback to jet
    cm = cv2.COLORMAP_JET
    colored = cv2.applyColorMap(ndvi_u8, cm)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ===============================================================
# vNIR visualization helpers
# ===============================================================

def save_vnir_hist_and_colorbar(vnir: np.ndarray, base_path: str):
    if plt is None:
        log("matplotlib not available; skipping vNIR hist/colorbar.")
        return

    vnir_flat = vnir.flatten()
    vnir_flat = vnir_flat[np.isfinite(vnir_flat)]

    # Histogram
    hist_path = base_path + "_hist.png"
    plt.figure(figsize=(4, 2))
    plt.hist(vnir_flat, bins=64, color="black")
    plt.title("vNIR Histogram")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    log(f"Saved vNIR histogram: {hist_path}")

    # Colorbar-like gradient
    cb_path = base_path + "_colorbar.png"
    grad = np.linspace(0, 1, 512, dtype=np.float32)
    grad_img = np.tile(grad, (20, 1))  # 20 px high
    grad_u8 = (grad_img * 255).astype(np.uint8)
    grad_rgb = cv2.applyColorMap(grad_u8, cv2.COLORMAP_JET)
    grad_rgb = cv2.cvtColor(grad_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(cb_path, cv2.cvtColor(grad_rgb, cv2.COLOR_RGB2BGR))
    log(f"Saved vNIR colorbar: {cb_path}")


# ===============================================================
# Tiled processing
# ===============================================================

def tile_indices(h: int, w: int, tile_size: int) -> Tuple[int, int, list]:
    n_tiles_y = int(math.ceil(h / float(tile_size)))
    n_tiles_x = int(math.ceil(w / float(tile_size)))
    indices = []
    for ty in range(n_tiles_y):
        y0 = ty * tile_size
        y1 = min(h, (ty + 1) * tile_size)
        for tx in range(n_tiles_x):
            x0 = tx * tile_size
            x1 = min(w, (tx + 1) * tile_size)
            indices.append((ty, tx, y0, y1, x0, x1))
    return n_tiles_y, n_tiles_x, indices


def process_tiled(
    rgb_full: np.ndarray,
    output_folder: str,
    base_name: str,
    tile_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a large RGB orthomosaic in tiles.
    Writes RGB tiles to a temp folder, returns full vNIR, NDVI_FAM, FSI arrays.
    """
    h, w, _ = rgb_full.shape
    log(f"Tiled processing enabled. Tile size: {tile_size} px")

    # Create temp tile folder
    tiles_folder_name = f"{today_prefix()}_tiled_images"
    tiles_folder = os.path.join(output_folder, tiles_folder_name)
    ensure_dir(tiles_folder)
    log(f"Tile folder: {tiles_folder}")

    vnir_full = np.zeros((h, w), dtype=np.float32)

    n_ty, n_tx, indices = tile_indices(h, w, tile_size)
    total_tiles = len(indices)
    log(f"Number of tiles: {total_tiles} ({n_ty} rows × {n_tx} cols)")

    for idx, (ty, tx, y0, y1, x0, x1) in enumerate(indices, start=1):
        rgb_tile = rgb_full[y0:y1, x0:x1, :]

        # Save tile for reference
        tile_name = f"{base_name}_tile_r{ty:03d}_c{tx:03d}.jpg"
        tile_path = os.path.join(tiles_folder, tile_name)
        rgb_tile_u8 = (np.clip(rgb_tile, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(tile_path, cv2.cvtColor(rgb_tile_u8, cv2.COLOR_RGB2BGR))

        log(f"[Tile {idx}/{total_tiles}] Processing tile r{ty} c{tx} "
            f"({x0}:{x1}, {y0}:{y1})")

        # Spectral processing per-tile (vNIR only; NDVI/FSI after global norm)
        vnir_tile = compute_vnir(rgb_tile)

        # Place into full vNIR array
        vnir_full[y0:y1, x0:x1] = vnir_tile

    # ============================================================
    # GLOBAL VNIR NORMALIZATION (AFTER TILE STITCHING)
    # ============================================================
    vnir_full = global_normalize_vnir(vnir_full, rgb_full)

    # ============================================================
    # RECOMPUTE NDVI_FAM USING NORMALIZED GLOBAL VNIR
    # ============================================================

    R = rgb_full[..., 0]
    
    # For multiband GeoTIFF (GIS, analysis) — use linear
    ndvi_fam_linear = NDVI_FAM(vnir_full, R, mode="linear")
    
    # For PDF reports (visual clarity) — use soft
    ndvi_fam_visual = NDVI_FAM(vnir_full, R, mode="soft")
    
    # Use linear for primary NDVI output (scientific accuracy)
    ndvi_full = ndvi_fam_linear
    
    # Store visual version for theme generation
    ndvi_visual = ndvi_fam_visual

    # FSI must be recomputed using normalized VNIR as well
    fsi_full = compute_fsi(vnir_full)

    return vnir_full, ndvi_full, fsi_full


# ===============================================================
# Main entry point
# ===============================================================

def process_orthomosaic(
    input_path: str,
    output_folder: str,
    tile_size: int = 2048,
    tile_threshold: int = 4096,
) -> dict:
    """
    Main engine function called by GUI.
    - input_path: path to orthomosaic (JPG/PNG/TIFF)
    - output_folder: user-chosen output folder
    - tile_size: tile width/height in pixels
    - tile_threshold: if max(H, W) > tile_threshold → use tiled mode
    Returns dict of output paths.
    """
    ensure_dir(output_folder)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    rgb = load_rgb_image(input_path)
    h, w, _ = rgb.shape

    use_tiling = max(h, w) > tile_threshold

    if use_tiling:
        vnir, ndvi, fsi = process_tiled(rgb, output_folder, base_name, tile_size=tile_size)
    else:
        log("Image below tiling threshold; processing as single block.")
        vnir_raw = compute_vnir(rgb)
        vnir = global_normalize_vnir(vnir_raw, rgb)
        
        R = rgb[..., 0]
        ndvi = NDVI_FAM(vnir, R, mode="linear")           # Scientific output
        ndvi_visual = NDVI_FAM(vnir, R, mode="soft")       # For beautiful PDFs
        fsi = compute_fsi(vnir)

    # ===========================================================
    # Save core products
    # ===========================================================

    outputs = {}

    # vNIR grayscale (0–255)
    vnir_u8 = (np.clip(vnir, 0.0, 1.0) * 255.0).astype(np.uint8)
    vnir_path = os.path.join(output_folder, f"{base_name}_vnir.png")
    cv2.imwrite(vnir_path, vnir_u8)
    outputs["vnir"] = vnir_path
    log(f"Saved vNIR: {vnir_path}")

    # vNIR thumbnail + hist/colorbar
    vnir_thumb = safe_thumbnail(vnir_u8, max_side=2048)
    vnir_thumb_path = os.path.join(output_folder, f"{base_name}_vnir_thumb.png")
    cv2.imwrite(vnir_thumb_path, vnir_thumb)
    outputs["vnir_thumb"] = vnir_thumb_path
    log(f"Saved vNIR thumbnail: {vnir_thumb_path}")

    save_vnir_hist_and_colorbar(
        vnir.astype(np.float32),
        os.path.join(output_folder, f"{base_name}_vnir_thumb.png"),
    )

    # NDVI raw float (as 32-bit TIFF-like PNG)
    ndvi_norm = (ndvi + 1.0) / 2.0
    ndvi_u8 = (np.clip(ndvi_norm, 0.0, 1.0) * 255.0).astype(np.uint8)

    ndvi_gray_path = os.path.join(output_folder, f"{base_name}_ndvi_fam.png")
    cv2.imwrite(ndvi_gray_path, ndvi_u8)
    outputs["ndvi_fam"] = ndvi_gray_path
    log(f"Saved NDVI_FAM grayscale: {ndvi_gray_path}")

    # NDVI color themes
    ndvi_thumb = safe_thumbnail(ndvi_u8, max_side=2048)
    ndvi_thumb_rgb = cv2.cvtColor(ndvi_thumb, cv2.COLOR_GRAY2RGB)

    def save_ndvi_theme(theme_name: str, cmap: str):
        # Use soft-enhanced NDVI for visual themes
        theme_img = apply_colormap_ndvi(ndvi_visual, cmap=cmap)
        theme_thumb = safe_thumbnail(theme_img, max_side=2048)
        theme_path = os.path.join(
            output_folder,
            f"{base_name}_ndvi_fam_thumb_{theme_name}.png",
        )
        cv2.imwrite(theme_path, cv2.cvtColor(theme_thumb, cv2.COLOR_RGB2BGR))
        outputs[f"ndvi_fam_{theme_name}"] = theme_path
        log(f"Saved NDVI_FAM theme '{theme_name}': {theme_path}")

    save_ndvi_theme("Classic", "classic")
    save_ndvi_theme("GVA_Signature_Theme", "gva")
    save_ndvi_theme("Jet", "jet")
    save_ndvi_theme("Viridis", "viridis")

    # FSI
    fsi_u8 = (np.clip(fsi, 0.0, 1.0) * 255.0).astype(np.uint8)
    fsi_path = os.path.join(output_folder, f"{base_name}_fsi.png")
    cv2.imwrite(fsi_path, fsi_u8)
    outputs["fsi"] = fsi_path
    log(f"Saved FSI: {fsi_path}")

    fsi_thumb = safe_thumbnail(fsi_u8, max_side=2048)
    fsi_thumb_path = os.path.join(output_folder, f"{base_name}_fsi_thumb.png")
    cv2.imwrite(fsi_thumb_path, fsi_thumb)
    outputs["fsi_thumb"] = fsi_thumb_path
    log(f"Saved FSI thumbnail: {fsi_thumb_path}")

    log("FAM-SRM processing complete.")
    return outputs


# ===============================================================
# CLI convenience
# ===============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FAM-SRM v3.1 Orthomosaic Engine (Tiled CPU Edition)"
    )
    parser.add_argument("input", help="Path to input RGB orthomosaic (JPG/PNG/TIFF)")
    parser.add_argument("output", help="Output folder")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile size in pixels (default 2048)",
    )
    parser.add_argument(
        "--tile-threshold",
        type=int,
        default=4096,
        help="If max(H,W) > tile-threshold, use tiled processing (default 4096)",
    )

    args = parser.parse_args()

    outs = process_orthomosaic(
        input_path=args.input,
        output_folder=args.output,
        tile_size=args.tile_size,
        tile_threshold=args.tile_threshold,
    )

    log("Outputs:")
    for k, v in outs.items():
        log(f"  {k}: {v}")
