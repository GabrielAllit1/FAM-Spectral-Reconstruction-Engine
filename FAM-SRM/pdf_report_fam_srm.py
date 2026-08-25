# ===============================================================
# pdf_report_fam_srm.py — FAM-SRM v3.0 SAFE EDITION
#
# FIXED:
#   • Pillow Decompression Bomb error
#   • Huge vNIR/NDVI images causing crashes
#
# METHOD:
#   • All images are force downscaled safely before PDF embedding
#   • Uses OpenCV for resizing (does NOT trigger PIL bomb check)
#
# FEATURES:
#   • MLA-formatted multi-page PDF
#   • vNIR, NDVI_FAM (4 maps), FSI, Graphon
#   • Histograms + colorbars
#   • Metadata page
#
# Produced for: Gabriel Allit (Salt19 LLC)
# ===============================================================

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import numpy as np
import cv2

# ===============================================================
# MLA HEADER
# ===============================================================

MLA_HEADER = (
    "FAM-SRM: Virtual Multispectral Reconstruction & Vegetation Index Modeling — Produced by Gabriel Allit"
)

def draw_mla_header(c: canvas.Canvas, page_num: int):
    c.setFont("Times-Roman", 10)
    hdr = f"{MLA_HEADER}    {page_num}"
    c.drawRightString(560, 750, hdr)

# ===============================================================
# SAFE IMAGE RESIZE — FIX FOR DECOMPRESSION BOMB
# ===============================================================

def safe_resize_for_pdf(img_path: str, max_dim: int = 2000) -> str:
    """
    Loads ANY size image safely using OpenCV, resizes if too large,
    and returns path to resized temporary file.
    Prevents Pillow decompression-bomb crashes.
    """
    if not os.path.exists(img_path):
        return img_path

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return img_path

    h, w = img.shape[:2]
    scale = max(h / max_dim, w / max_dim)

    if scale <= 1.0:
        return img_path  # Already safe

    new_w = int(w / scale)
    new_h = int(h / scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    safe_path = img_path.replace(".png", "_safe.png").replace(".tif", "_safe.png")
    cv2.imwrite(safe_path, resized)
    return safe_path

# ===============================================================
# SCRIPTING UTILITIES
# ===============================================================

def add_title(c, text, y):
    c.setFont("Times-Bold", 16)
    c.drawString(72, y, text)
    return y - 30

def add_text(c, text, y):
    c.setFont("Times-Roman", 12)
    for line in text.split("\n"):
        c.drawString(72, y, line)
        y -= 18
    return y

def make_hist_png(arr, out_path):
    arr = arr.astype(np.float32).flatten()
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0, 1)

    hist = np.histogram(arr, bins=64, range=(0, 1))[0]
    hist = hist / (hist.max() + 1e-6)

    h, w = 200, 400
    img = np.ones((h, w, 3), np.uint8) * 255

    for i, v in enumerate(hist):
        x = int(i / 63 * (w - 1))
        y = int(v * (h - 1))
        cv2.line(img, (x, h - 1), (x, h - 1 - y), (0, 0, 0), 1)

    cv2.imwrite(out_path, img)

# ===============================================================
# PAGE ADDER (image + histogram + colorbar)
# ===============================================================

def add_image_page(c, page, title, img_path):
    c.showPage()
    draw_mla_header(c, page)
    page += 1

    y = add_title(c, title, 710)

    # SAFE RESIZE FIX
    img_path = safe_resize_for_pdf(img_path, max_dim=2000)

    # Preview image
    if os.path.exists(img_path):
        img = ImageReader(img_path)
        c.drawImage(img, 72, 330, width=480, preserveAspectRatio=True)

        # Histogram
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        hist_path = img_path + "_hist.png"
        make_hist_png(arr, hist_path)
        c.drawImage(hist_path, 72, 150, width=300)

        # Colorbar
        bar = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (20, 1))
        bar = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
        bar_path = img_path + "_colorbar.png"
        cv2.imwrite(bar_path, bar)
        c.drawImage(bar_path, 400, 150, width=150)

    return page

# ===============================================================
# MAIN REPORT GENERATOR
# ===============================================================

def generate_fam_srm_pdf(
    out_pdf_path: str,
    input_file: str,
    out_prefix: str,
    vnir_path: str,
    ndvi_path: str,
    fsi_path: str = None,
    graphon_path: str = None,
    metadata: dict = None,
    logo_path: str = "LOGO.png",
):

    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)

    c = canvas.Canvas(out_pdf_path, pagesize=letter)
    width, height = letter
    page = 1

    # ===========================================================
    # COVER PAGE
    # ===========================================================
    draw_mla_header(c, page)

    c.setFont("Times-Bold", 26)
    c.drawString(72, 680, "FAM-SRM Scientific Report")

    c.setFont("Times-Roman", 14)
    c.drawString(72, 650, f"Source File: {input_file}")
    c.drawString(72, 630, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Logo
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(logo, 350, 500, width=200, preserveAspectRatio=True)

    c.setFont("Times-Roman", 12)
    c.drawString(72, 580, "Operator Chain:")
    c.drawString(72, 560, "KRO → TMBE → USK → BES → FDM → ShadowCorrector → FSI → Graphon")

    c.showPage()
    page += 1

    # ===========================================================
    # PAGE: vNIR
    # ===========================================================
    page = add_image_page(c, page, "Virtual NIR (vNIR)", vnir_path)

    # ===========================================================
    # NDVI_FAM — 4 Colormaps
    # ===========================================================
    ndvi_gray = cv2.imread(ndvi_path, cv2.IMREAD_GRAYSCALE)
    ndvi_norm = (ndvi_gray.astype(np.float32) / 255.0)

    cmaps = {
        "NDVI_FAM — Classic": cv2.COLORMAP_JET,
        "NDVI_FAM — Viridis": cv2.COLORMAP_VIRIDIS,
        "NDVI_FAM — Jet": cv2.COLORMAP_JET,
        "NDVI_FAM — GVA Signature Theme": cv2.COLORMAP_SUMMER,
    }

    for name, cmap in cmaps.items():
        mapped = cv2.applyColorMap((ndvi_norm * 255).astype(np.uint8), cmap)
        out_path = f"{ndvi_path}_{name.replace(' ', '_')}.png"
        cv2.imwrite(out_path, mapped)
        page = add_image_page(c, page, name, out_path)

    # ===========================================================
    # FSI Page
    # ===========================================================
    if fsi_path and os.path.exists(fsi_path):
        page = add_image_page(c, page, "Fractal Stress Index (FSI)", fsi_path)

    # ===========================================================
    # Graphon Page
    # ===========================================================
    if graphon_path and os.path.exists(graphon_path):
        page = add_image_page(c, page, "Graphon Structural Map", graphon_path)

    # ===========================================================
    # METADATA PAGE
    # ===========================================================
    c.showPage()
    draw_mla_header(c, page)

    y = add_title(c, "Metadata & Processing Notes", 710)

    if metadata:
        for k, v in metadata.items():
            y = add_text(c, f"{k}: {v}", y)
    else:
        y = add_text(c, "No metadata provided.", y)

    c.save()
