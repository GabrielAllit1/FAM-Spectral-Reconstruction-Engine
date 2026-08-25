# ===============================================================
# fam_srm_multiband_export.py
#
# Engineer-grade multiband GeoTIFF writer for FAM-SRM.
#
# Band order (Option B):
#   1: R
#   2: G
#   3: B
#   4: vNIR
#   5: NDVI_FAM
#   6: FSI        (optional)
#   7: Graphon    (optional)
#
# If FSI or Graphon are None, those bands are omitted.
# ===============================================================

from typing import Optional, Dict

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    Affine = None
    RASTERIO_AVAILABLE = False


def _build_profile(
    width: int,
    height: int,
    count: int,
    ref_profile: Optional[dict] = None,
) -> dict:
    """
    Build a GeoTIFF profile based on an existing profile (if present)
    or create a generic one otherwise.
    """
    if ref_profile is not None:
        profile = ref_profile.copy()
    else:
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "dtype": "float32",
            "compress": "deflate",
            "tiled": True,
        }

    # Enforce float32, correct count
    profile["dtype"] = "float32"
    profile["count"] = count

    # If no transform, fall back to identity
    if "transform" not in profile or profile["transform"] is None:
        if Affine is not None:
            profile["transform"] = Affine.translation(0, 0) * Affine.scale(1, -1)
        else:
            profile["transform"] = None

    return profile


def export_multiband_geotiff(
    out_path: str,
    rgb_bgr: np.ndarray,
    vnir: np.ndarray,
    ndvi_fam: np.ndarray,
    fsi: Optional[np.ndarray] = None,
    graphon: Optional[np.ndarray] = None,
    ref_profile: Optional[dict] = None,
) -> Optional[str]:
    """
    Export a multiband GeoTIFF with band order:

        1: R
        2: G
        3: B
        4: vNIR
        5: NDVI_FAM
        6: FSI        (if provided)
        7: Graphon    (if provided)

    All bands are stored as float32. RGB is normalized to [0,1].
    NDVI_FAM may be in [-1,1], which is fine as float32.

    Parameters
    ----------
    out_path : str
        Destination path for the multiband GeoTIFF.
    rgb_bgr : np.ndarray
        Source BGR image, uint8 or float32.
    vnir : np.ndarray
        Virtual NIR proxy, float32.
    ndvi_fam : np.ndarray
        NDVI_FAM map, float32.
    fsi : np.ndarray, optional
        FSI map, float32.
    graphon : np.ndarray, optional
        Graphon structural residual, float32.
    ref_profile : dict, optional
        Source raster profile (if reading from GeoTIFF).
        Used to propagate CRS, transform, etc.

    Returns
    -------
    str or None
        Path to written file, or None if rasterio is unavailable.
    """

    if not RASTERIO_AVAILABLE:
        # Silent and clean if rasterio not present
        return None

    # Ensure float32 and consistent shapes
    rgb = rgb_bgr.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0

    R = rgb[:, :, 2]
    G = rgb[:, :, 1]
    B = rgb[:, :, 0]

    h, w = R.shape

    vnir = vnir.astype(np.float32)
    ndvi_fam = ndvi_fam.astype(np.float32)

    band_list = [R, G, B, vnir, ndvi_fam]
    band_descriptions = [
        "Red (normalized 0-1)",
        "Green (normalized 0-1)",
        "Blue (normalized 0-1)",
        "Virtual NIR (FAM-SRM)",
        "NDVI_FAM (FAM-enhanced NDVI)",
    ]

    if fsi is not None:
        band_list.append(fsi.astype(np.float32))
        band_descriptions.append("FSI (Fractal Stress Index)")

    if graphon is not None:
        band_list.append(graphon.astype(np.float32))
        band_descriptions.append("Graphon structural residual")

    count = len(band_list)

    profile = _build_profile(
        width=w,
        height=h,
        count=count,
        ref_profile=ref_profile,
    )

    # Write the multiband GeoTIFF
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, band in enumerate(band_list, start=1):
            dst.write(band.astype(np.float32), i)
            dst.set_band_description(i, band_descriptions[i-1])

        # Global metadata
        dst.update_tags(
            1,  # just attach to first band
            FAM_SRM="True",
            BAND_ORDER="R,G,B,vNIR,NDVI_FAM,FSI(optional),Graphon(optional)",
            NOTE="Engineer-grade FAM-SRM multiband GeoTIFF",
        )

    return out_path
