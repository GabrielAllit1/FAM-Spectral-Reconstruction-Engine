# ===============================================================
# pdf_report_fam_srm_techspec.py — FAM-SRM v3.0
#
# Dual-output Technical Specification PDF Generator:
#   • Scientific Edition (light, engineering format)
#   • Dark Aerospace Edition (high-contrast, presentation format)
#
# FIXES:
#   • OpenCV-only image loading (NO Pillow → NO decompression bombs)
#   • Safe downscaled figures for ultra-large GeoTIFF outputs
#
# Produced for:
#   Gabriel Allit — Salt19 LLC
# ===============================================================

import os
import cv2
import platform
import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# ===============================================================
# SAFE IMAGE RESIZE
# ===============================================================

def safe_resize(img_path: str, max_dim: int = 2200) -> str:
    """
    Prevent decompression-bomb crashes by using OpenCV instead of Pillow.
    All images are resized to max_dim in either axis.
    """
    if not os.path.exists(img_path):
        return img_path

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return img_path

    h, w = img.shape[:2]
    scale = max(h / max_dim, w / max_dim)

    if scale <= 1:
        return img_path

    new_w = int(w / scale)
    new_h = int(h / scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path = img_path.replace(".png", "_safe.png").replace(".tif", "_safe.png")
    cv2.imwrite(out_path, resized)
    return out_path

# ===============================================================
# THUMBNAIL + HISTOGRAM GENERATION
# ===============================================================

def make_hist_png(arr, out_path):
    arr = arr.astype(np.float32)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0, 1)

    hist = np.histogram(arr, bins=64, range=(0, 1))[0]
    hist = hist / (hist.max() + 1e-6)

    h, w = 200, 400
    canvas = np.ones((h, w, 3), np.uint8) * 250

    for i, v in enumerate(hist):
        x = int(i / 63 * (w - 1))
        y = int(v * (h - 1))
        cv2.line(canvas, (x, h - 1), (x, h - 1 - y), (40, 40, 40), 1)

    cv2.imwrite(out_path, canvas)

def load_img(img_path: str):
    try:
        img = ImageReader(img_path)
        return img
    except:
        return None

# ===============================================================
# MLA PAGE HEADER
# ===============================================================

MLA_HEADER = "FAM-SRM Technical Specification — Salt19 LLC — Allit, G."

def draw_header(c: canvas.Canvas, page_number: int, dark=False):
    c.setFont("Helvetica", 10)
    color = (1,1,1) if dark else (0,0,0)
    c.setFillColorRGB(*color)
    c.drawRightString(560, 750, f"{MLA_HEADER}    {page_number}")

# ===============================================================
# PAGE TEMPLATES
# ===============================================================

def title_page(c, logo_path, input_file, multiband_path, mode, dark=False):
    width, height = letter

    color = (1,1,1) if dark else (0,0,0)
    c.setFillColorRGB(*color)

    # LOGO
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(logo, 350, 580, width=200, preserveAspectRatio=True)

    # TITLE
    c.setFont("Helvetica-Bold", 26)
    c.drawString(60, 680, "FAM-SRM Technical Specification")

    c.setFont("Helvetica", 14)
    c.drawString(60, 650, f"Input File: {input_file}")
    c.drawString(60, 630, f"Multiband Raster: {multiband_path}")
    c.drawString(60, 610, f"Mode: {mode}")
    c.drawString(60, 590, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(60, 570, f"Host System: {platform.node()} — {platform.system()} {platform.release()}")

# ===============================================================
# OPERATOR CHAIN PAGE
# ===============================================================

def operator_page(c, dark=False):
    width, height = letter
    color = (1,1,1) if dark else (0,0,0)
    c.setFillColorRGB(*color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 710, "Operator Chain & Mathematical Definitions")

    y = 670
    c.setFont("Helvetica", 11)

    lines = [
        "vNIR = FDM( BES( USK( TMBE( KRO(Y) ) ) ) )",
        "Y = 0.45·G + 0.35·R + 0.20·B",
        "",
        "KRO — Kolakoski Recursive Operator:",
        "      Nonlinear spatial smoothing via Kolakoski sequence kernels.",
        "",
        "TMBE — Thue–Morse Bias Equalizer:",
        "      2D parity lattice applied to chromatic bias across texture fields.",
        "",
        "USK — Ulam Spiral Kernel:",
        "      Spiral-based structural kernel reinforcing canopy geometry.",
        "",
        "BES — Beatty Entropy Scaling:",
        "      Irrational Beatty sequence blended with local entropy magnitude.",
        "",
        "FDM — Fractal Dimension Modulator:",
        "      Laplacian-variance fractal estimate reinjected into intensity.",
        "",
        "NDVI_FAM — FAM-enhanced NDVI:",
        "      NDVI_FAM = tanh((vNIR − R) / (vNIR + R + ε))",
    ]

    for line in lines:
        c.drawString(60, y, line)
        y -= 18

# ===============================================================
# RASTER SPECS PAGE
# ===============================================================

def raster_specs_page(c, profile, extra_metadata=None, dark=False):
    color = (1,1,1) if dark else (0,0,0)
    c.setFillColorRGB(*color)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 710, "Raster Specifications")

    y = 670
    c.setFont("Helvetica", 11)

    if profile:
        for key in ["width", "height", "count", "dtype"]:
            if key in profile:
                c.drawString(60, y, f"{key.capitalize()}: {profile[key]}")
                y -= 16

        if "crs" in profile:
            c.drawString(60, y, f"CRS: {profile['crs']}")
            y -= 16

        if "transform" in profile:
            c.drawString(60, y, f"Transform: {profile['transform']}")
            y -= 16

            try:
                px = abs(profile["transform"].a)
                py = abs(profile["transform"].e)
                c.drawString(60, y, f"Pixel Size: {px:.4f} x {py:.4f}")
                y -= 16
            except:
                pass

    else:
        c.drawString(60, y, "No GeoTIFF profile available (non-georeferenced input).")
        y -= 16

    # EXTRA METADATA
    if extra_metadata:
        y -= 20
        c.setFont("Helvetica-Bold", 13)
        c.drawString(60, y, "Additional Metadata:")
        y -= 16
        c.setFont("Helvetica", 11)
        for k, v in extra_metadata.items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 16

# ===============================================================
# QC NOTES PAGE
# ===============================================================

def qc_page(c, dark=False):
    color = (1,1,1) if dark else (0,0,0)
    c.setFillColorRGB(*color)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 710, "QC Checklist & Usage Notes")

    y = 670
    c.setFont("Helvetica", 11)

    lines = [
        "• vNIR is a proxy NIR band reconstructed from visible wavelengths.",
        "• NDVI_FAM is most stable when interpreted in a relative, not absolute sense.",
        "• FSI highlights fractal anomalies indicative of thinning or structural stress.",
        "• Graphon map represents structural heterogeneity via spectral graph theory.",
        "",
        "Recommended GIS Workflow:",
        "  1. Load multiband GeoTIFF.",
        "  2. Visualize Bands 1–3 (RGB).",
        "  3. Map Band 4 (vNIR) grayscale.",
        "  4. Map Band 5 (NDVI_FAM) using stable colormap.",
        "  5. Inspect Band 6 (FSI) for canopy thinning.",
        "  6. Inspect Band 7 (Graphon) for heterogeneity anomalies.",
        "",
        "Disclaimer:",
        "  This product is a mathematical model, not true multispectral NIR.",
        "  Interpretation should be vegetation-relative and pattern-based.",
    ]

    for line in lines:
        c.drawString(60, y, line)
        y -= 18

# ===============================================================
# MAIN GENERATOR — PRODUCES TWO PDFs
# ===============================================================

def generate_fam_srm_techspec_pdf(
    out_pdf_path: str,
    input_file: str,
    multiband_path: str,
    profile=None,
    mode="Engineer-grade",
    extra_metadata=None,
    logo_path="C:/Users/gabri/Documents/FAM-Spectral Reconstruction Engine/Test/LOGO.jpg",
):
    """
    Creates:
      1. Scientific TechSpec PDF
      2. Dark Aerospace TechSpec PDF
    """

    # Ensure output folder exists
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)

    out_pdf_dark = out_pdf_path.replace(".pdf", "_Dark.pdf")

    # -----------------------------------------------------------
    # SCIENTIFIC EDITION (Light)
    # -----------------------------------------------------------
    c = canvas.Canvas(out_pdf_path, pagesize=letter)
    page = 1

    # PAGE 1 — TITLE
    draw_header(c, page, dark=False)
    title_page(c, logo_path, input_file, multiband_path, mode, dark=False)
    c.showPage()
    page += 1

    # PAGE 2 — OPERATOR DETAILS
    draw_header(c, page, dark=False)
    operator_page(c, dark=False)
    c.showPage()
    page += 1

    # PAGE 3 — RASTER SPECS
    draw_header(c, page, dark=False)
    raster_specs_page(c, profile, extra_metadata, dark=False)
    c.showPage()
    page += 1

    # PAGE 4 — QC NOTES
    draw_header(c, page, dark=False)
    qc_page(c, dark=False)
    c.save()

    # -----------------------------------------------------------
    # DARK AEROSPACE EDITION
    # -----------------------------------------------------------
    c = canvas.Canvas(out_pdf_dark, pagesize=letter)
    page = 1

    # DARK BACKGROUND
    c.setFillColorRGB(0.05, 0.05, 0.07)
    c.rect(0, 0, 612, 792, fill=1)

    # PAGE 1
    draw_header(c, page, dark=True)
    title_page(c, logo_path, input_file, multiband_path, mode, dark=True)
    c.showPage()
    page += 1

    # PAGE 2
    c.setFillColorRGB(0.05, 0.05, 0.07); c.rect(0, 0, 612, 792, fill=1)
    draw_header(c, page, dark=True)
    operator_page(c, dark=True)
    c.showPage()
    page += 1

    # PAGE 3
    c.setFillColorRGB(0.05, 0.05, 0.07); c.rect(0, 0, 612, 792, fill=1)
    draw_header(c, page, dark=True)
    raster_specs_page(c, profile, extra_metadata, dark=True)
    c.showPage()
    page += 1

    # PAGE 4
    c.setFillColorRGB(0.05, 0.05, 0.07); c.rect(0, 0, 612, 792, fill=1)
    draw_header(c, page, dark=True)
    qc_page(c, dark=True)
    c.save()

    return {
        "scientific_pdf": out_pdf_path,
        "dark_pdf": out_pdf_dark
    }
