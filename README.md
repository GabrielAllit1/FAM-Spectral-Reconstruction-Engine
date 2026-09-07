# FAM Spectral Reconstruction Engine

**Experimental RGB-derived spectral feature reconstruction and vegetation-analysis toolkit**

FAM-SRM is a Python research prototype for transforming RGB imagery and orthomosaics into a family of derived spectral-style products, structural feature maps, geospatial rasters, and technical reports.

The current implementation combines deterministic image operators, spatial kernels, local-variance structure, fractal-style edge response, and derived vegetation indexing into a CPU-oriented desktop/GIS workflow.

> **Scientific claim boundary:** FAM-SRM does **not** measure near-infrared reflectance. Its `vNIR` channel is synthesized from RGB imagery. Therefore `NDVI_FAM` is a model-derived index, not sensor-measured NDVI and not a substitute for calibrated multispectral or hyperspectral data. These outputs should be treated as experimental feature maps until validated against independent measured-NIR reference data.

## What the engine does

Given an RGB GeoTIFF, TIFF, JPEG, or PNG, the current pipeline can generate:

- synthetic/virtual NIR (`vNIR`) derived from RGB
- a normalized FAM-derived vegetation index (`NDVI_FAM`)
- fractal stress / local structural feature products
- graphon-style structural heterogeneity maps
- PNG visualizations and colormap variants
- single-band GeoTIFF outputs when georeferencing is available
- optional multiband GeoTIFF products
- histograms and legends
- PDF technical/reporting output

For GeoTIFF inputs, the processing path preserves the source geospatial profile for compatible derived raster exports.

## Operator chain

The current virtual-NIR path is implemented as a deterministic RGB transformation:

```text
RGB
 │
 ├─ weighted luminance-like field
 │
 ├─ KRO   Kolakoski-derived spatial kernel
 ├─ TMBE  Thue-Morse parity modulation
 ├─ USK   Ulam-style radial kernel
 ├─ BES   local-variance / Beatty-field scaling
 └─ FDM   Laplacian + smoothing structural modulator
        │
        ▼
      vNIR
        │
        ├─ FAM-derived vegetation index
        ├─ fractal stress features
        └─ graphon-style structural map
```

The implementation is intentionally inspectable. The core virtual-NIR transform is in:

`FAM-SRM/rgb_to_virtual_nir.py`

The derived vegetation index is computed as:

```text
raw_index = (vNIR - R) / (vNIR + R + eps)
```

and then normalized for output. Because `vNIR` is synthetic, this equation must not be interpreted as conventional sensor-derived NDVI.

## Main components

```text
FAM-SRM/
├── rgb_to_virtual_nir.py          # RGB -> synthetic vNIR transformation
├── fam_ndvi.py                    # FAM-derived vegetation index + visualization
├── vegetation_fractal_stress.py   # structural/fractal stress feature map
├── spectral_graphon_vegetation.py # graphon-style heterogeneity map
├── shadow_corrector_unlocker.py   # shadow/illumination correction stage
├── fam_srm_orthomosaic_engine.py  # raster/orthomosaic processing and export
├── fam_srm_multiband_export.py    # multiband output support
├── fam_srm_public_engine.py       # stable wrapper / CLI-oriented API
├── fam_srm_gui.py                 # PyQt6 desktop interface
├── pdf_report_fam_srm.py          # report generation
└── environment_fam_srm.yml        # reproducible Conda environment
```

## Technology stack

The checked-in environment targets:

- Python 3.10
- NumPy 1.26.4
- Rasterio
- OpenCV
- PyQt6
- ReportLab
- Pillow

The engine is currently CPU-oriented.

## Installation

Using Conda:

```bash
conda env create -f FAM-SRM/environment_fam_srm.yml
conda activate fam_srm_env
```

## Run from the public engine wrapper

```bash
python FAM-SRM/fam_srm_public_engine.py INPUT_IMAGE --out-dir OUTPUT_DIRECTORY
```

Optional switches:

```bash
--no-graphon
--no-fsi
--no-multiband
```

Example:

```bash
python FAM-SRM/fam_srm_public_engine.py orthomosaic.tif --out-dir outputs
```

## Desktop GUI

```bash
python FAM-SRM/fam_srm_gui.py
```

The GUI is intended for interactive scientific/GIS exploration of the same underlying operators and export paths.

## Programmatic API

The main public wrapper is:

```python
run_fam_srm(
    input_path,
    output_dir,
    compute_graphon=True,
    compute_fsi=True,
    save_multiband_geotiff=True,
)
```

It returns a dictionary containing generated output paths.

## Example outputs

The repository includes example generated products from the current engine.

![FAM-SRM example output](FAM-SRM%20output.jpg)

Additional thumbnails in the repository show FSI and several visualization palettes for the FAM-derived vegetation index.

## Interpretation guidance

FAM-SRM products are best understood as **derived image features**, not direct spectral observations.

Reasonable current uses include:

- exploratory RGB vegetation analysis
- visual prioritization and segmentation support
- structural texture/heterogeneity comparison
- comparative processing of repeated RGB captures under controlled conditions
- generation of candidate features for downstream statistical or ML validation

The following claims require independent validation and should not be inferred from the current implementation alone:

- physical NIR reflectance recovery
- equivalence to a multispectral camera
- calibrated NDVI equivalence
- crop-health, disease, nutrient, moisture, or stress diagnosis
- cross-camera or cross-season spectral invariance
- agronomic decision thresholds

## Validation roadmap

A scientifically stronger validation program should compare FAM-SRM outputs against synchronized, georegistered measured data rather than relying only on visual plausibility.

Priority tests are:

1. **Measured NIR comparison** — RGB and real NIR captured over the same targets under matched geometry and illumination.
2. **Index correlation** — compare `NDVI_FAM` against conventional NDVI from calibrated red/NIR measurements.
3. **Holdout testing** — evaluate across cameras, sites, seasons, vegetation classes, illumination conditions, and flight altitudes not used during development.
4. **Baseline comparison** — compare against simpler RGB vegetation indices and learned RGB-to-NIR baselines.
5. **Error characterization** — report RMSE/correlation/rank consistency, spatial failure modes, uncertainty, and cases where the transform is misleading.
6. **Downstream validation** — only after spectral comparison, test whether derived features improve a real task such as segmentation or stress classification.

Until those tests are complete, FAM-SRM should be described as an **experimental RGB-derived spectral reconstruction/feature-engineering system**, not a replacement for physical multispectral sensing.

## Project status

Research prototype. The repository contains working image-processing, raster-export, GUI, and reporting components, but the spectral interpretation remains experimental and requires independent measured-data validation.

Built by Gabriel Allit / SALT19.
