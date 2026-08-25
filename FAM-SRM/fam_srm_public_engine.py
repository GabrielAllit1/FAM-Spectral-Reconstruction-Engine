# ===============================================================
# fam_srm_public_engine.py — FAM-SRM v2.0
#
# PUBLIC-FACING ENGINE WRAPPER
#
# This module provides a simple, stable, safe public API:
#
#       run_fam_srm(input_path, output_dir, settings)
#
# Returns a dictionary containing:
#   • paths to vNIR / NDVI / FSI / Graphon PNGs
#   • path to MLA-formatted scientific PDF
#   • optional colormap products
#
# Designed for:
#   • GUI consumption
#   • CLI integration
#   • Flask/FastAPI services
#   • External automation
# ===============================================================

import os
import numpy as np
import cv2

from fam_srm_orthomosaic_engine import (
    load_rgb_orthomosaic,
    process_rgb_array,
    save_png,
    save_geotiff_single,
    save_geotiff_multiband,
)

from pdf_report_fam_srm import generate_fam_srm_pdf
from fam_ndvi import save_ndvi_colormaps, save_ndvi_histogram, save_ndvi_legend

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except:
    RASTERIO_AVAILABLE = False


# ============================================================================
# Main public API function
# ============================================================================
def run_fam_srm(
    input_path: str,
    output_dir: str,
    compute_graphon: bool = True,
    compute_fsi: bool = True,
    save_multiband_geotiff: bool = True,
):
    """
    Full public FAM-SRM engine.

    Parameters
    ----------
    input_path : str
        Path to RGB orthomosaic or JPG/PNG.
    output_dir : str
        Directory for all output deliverables.
    compute_graphon : bool
        Include Graphon map.
    compute_fsi : bool
        Include Fractal Stress Index.
    save_multiband_geotiff : bool
        Save multiband .tif if GeoTIFF input.

    Returns
    -------
    dict
        Paths to all outputs.
    """

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Base prefix for writing deliverables
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_prefix = os.path.join(output_dir, base)

    # ----------------------------------------------------------
    # Load RGB (Auto-detect TIFF vs JPG/PNG)
    # ----------------------------------------------------------
    data = load_rgb_orthomosaic(input_path)
    rgb_bgr = data["rgb_bgr"]
    profile = data["profile"]

    # ----------------------------------------------------------
    # Run core FAM-SRM operators
    # ----------------------------------------------------------
    outputs = process_rgb_array(
        rgb_bgr,
        compute_graphon=compute_graphon,
        compute_fsi=compute_fsi,
    )

    # ----------------------------------------------------------
    # Save outputs (PNG + optional GeoTIFF)
    # ----------------------------------------------------------
    output_files = {}

    for key, arr in outputs.items():
        png_path = f"{out_prefix}_{key}.png"
        tif_path = f"{out_prefix}_{key}.tif"

        save_png(png_path, arr)
        output_files[key] = png_path

        if profile is not None and RASTERIO_AVAILABLE:
            save_geotiff_single(tif_path, arr, profile)
            output_files[key + "_tif"] = tif_path

    # ----------------------------------------------------------
    # Save multiband TIFF (optional)
    # ----------------------------------------------------------
    if profile is not None and RASTERIO_AVAILABLE and save_multiband_geotiff:
        multi_tif = f"{out_prefix}_fam_srm_products.tif"
        save_geotiff_multiband(multi_tif, outputs, profile)
        output_files["multiband"] = multi_tif

    # ----------------------------------------------------------
    # SAVE NDVI COLORMAP VARIANTS
    # ----------------------------------------------------------
    ndvi_cmaps = save_ndvi_colormaps(outputs["ndvi_fam"], out_prefix)
    output_files["ndvi_colormaps"] = ndvi_cmaps

    # ----------------------------------------------------------
    # NDVI Histogram + Legend
    # ----------------------------------------------------------
    ndvi_hist = f"{out_prefix}_ndvi_hist.png"
    ndvi_legend = f"{out_prefix}_ndvi_legend.png"

    save_ndvi_histogram(outputs["ndvi_fam"], ndvi_hist)
    save_ndvi_legend(ndvi_legend)

    output_files["ndvi_hist"] = ndvi_hist
    output_files["ndvi_legend"] = ndvi_legend

    # ----------------------------------------------------------
    # MLA-Formatted PDF Report
    # ----------------------------------------------------------
    pdf_path = f"{out_prefix}_FAM_SRM_Report.pdf"

    generate_fam_srm_pdf(
        out_pdf_path=pdf_path,
        input_file=input_path,
        out_prefix=out_prefix,
        vnir_path=output_files["vnir"],
        ndvi_path=output_files["ndvi_fam"],
        fsi_path=output_files.get("fsi"),
        graphon_path=output_files.get("graphon"),
        metadata={
            "Operator Chain": "KRO → TMBE → USK → BES → FDM → ShadowCorrector",
            "Source GeoTIFF": str(profile is not None),
            "Compute FSI": str(compute_fsi),
            "Compute Graphon": str(compute_graphon),
            "Output Folder": output_dir,
        },
        logo_path="LOGO.png"
    )

    output_files["pdf"] = pdf_path

    return output_files


# ============================================================================
# CLI wrapper (optional)
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Public FAM-SRM v2.0 API — Produce vNIR, NDVI, FSI, Graphon + MLA PDF"
    )
    parser.add_argument("input", help="Input image (GeoTIFF/JPG/PNG)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--no-graphon", action="store_true")
    parser.add_argument("--no-fsi", action="store_true")
    parser.add_argument("--no-multiband", action="store_true")

    args = parser.parse_args()

    results = run_fam_srm(
        input_path=args.input,
        output_dir=args.out_dir,
        compute_graphon=not args.no_graphon,
        compute_fsi=not args.no_fsi,
        save_multiband_geotiff=not args.no_multiband,
    )

    print("\n=== FAM-SRM OUTPUTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
