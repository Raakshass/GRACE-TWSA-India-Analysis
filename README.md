# GRACE/GRACE-FO Terrestrial Water Storage Anomaly (TWSA) Analysis over India

> **B.Tech Project (BTP)** — Analysis of long-term terrestrial water storage changes over India using GRACE/GRACE-FO satellite gravimetry data (April 2002 – June 2026).

---

## Overview

This repository contains all analysis scripts for studying terrestrial water storage anomaly (TWSA) trends over India using:

- **GRACE/GRACE-FO CSR RL06.3 Mascon** solutions (April 2002 – June 2026)
- **IMD gridded daily precipitation** data (0.25° × 0.25°)
- **ESA WorldCover 2021 v200** land use/land cover data (10 m resolution)

The study identifies **hotspot regions** of significant water storage depletion and gain, reconstructs the 11-month GRACE/GRACE-FO inter-mission data gap (July 2017 – May 2018), and analyzes the relationship between precipitation patterns and water storage changes.

---

## Repository Structure

```
GRACE-TWSA-India-Analysis/
│
├── 01_grace_mascon_analysis/       # GRACE data loading, spatial maps, time series
│   ├── plot_grace_mascon.py        → Figure 1 (All-India TWSA time series)
│   └── trend_map.py               → Figure 7 (TWSA trend map, standalone)
│
├── 02_imd_precipitation/           # IMD precipitation download, processing, mapping
│   ├── download_imd.py            → IMD data download script
│   ├── wateryear_pipeline.py      → Water-year (June–May) aggregation
│   ├── spatial_maps.py            → Figures 2–6 (precipitation & TWSA spatial maps)
│   └── rainfall_timeseries.py     → Figure 6 (ROI precipitation bar charts)
│
├── 03_hotspot_identification/      # Hotspot delineation and temporal analysis
│   ├── hotspot_extraction.py      → Figures 7, 8 (trend map + hotspot contours)
│   └── hotspot_temporal.py        → Figure 9 (ROI TWSA time series, 5-panel)
│
├── 04_gap_filling/                 # Inter-mission gap reconstruction
│   ├── grace_gap_fill.py          → Figure 10 (gap-filled All-India TWSA)
│   └── gapfilled_figures.py       → Figures 11–16 (gap-filled spatial + temporal)
│
├── 05_lulc_analysis/               # Land use/land cover analysis
│   ├── extract_lulc.py            → Table 4 (LULC statistics per ROI)
│   └── plot_lulc_maps.py          → Figure 18 (LULC maps, 4 ROIs)
│
├── 06_punjab_haryana_analysis/     # Regional case study
│   └── regional_analysis.py       → Figures 18–20 (Punjab-Haryana analysis)
│
├── data/                           # Data directory (see data/README.md)
│   └── README.md
│
├── outputs/                        # Generated figures (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Figure–Code Mapping

| Figure | Description | Script |
|--------|-------------|--------|
| **Fig 1** | All-India area-weighted TWSA time series (2002–2026) | `01_grace_mascon_analysis/plot_grace_mascon.py` |
| **Fig 2** | Mean annual precipitation spatial map | `02_imd_precipitation/spatial_maps.py` |
| **Fig 3** | Precipitation trend map (mm/yr) | `02_imd_precipitation/spatial_maps.py` |
| **Fig 4** | Standardized rainfall anomaly (year-wise panels) | `02_imd_precipitation/spatial_maps.py` |
| **Fig 5** | TWSA year-wise spatial evolution (water years) | `02_imd_precipitation/spatial_maps.py` |
| **Fig 6** | ROI annual precipitation bar charts | `02_imd_precipitation/rainfall_timeseries.py` |
| **Fig 7** | TWSA linear trend map over India | `03_hotspot_identification/hotspot_extraction.py` |
| **Fig 8** | Hotspot regions with trend thresholds | `03_hotspot_identification/hotspot_extraction.py` |
| **Fig 9** | Monthly TWSA time series (All-India + 4 ROIs) | `03_hotspot_identification/hotspot_temporal.py` |
| **Fig 10** | Gap-filled All-India TWSA time series | `04_gap_filling/grace_gap_fill.py` |
| **Fig 11** | Gap-filled TWSA year-wise panels (WY 2002–2025) | `04_gap_filling/gapfilled_figures.py` |
| **Fig 12** | ROI1 spatial mean TWSA + trend | `04_gap_filling/gapfilled_figures.py` |
| **Fig 13** | ROI2 spatial mean TWSA + trend | `04_gap_filling/gapfilled_figures.py` |
| **Fig 14** | ROI3 spatial mean TWSA + trend | `04_gap_filling/gapfilled_figures.py` |
| **Fig 15** | ROI4 spatial mean TWSA + trend | `04_gap_filling/gapfilled_figures.py` |
| **Fig 16** | Gap-filled ROI temporal TWSA (5-panel) | `04_gap_filling/gapfilled_figures.py` |
| **Fig 18** | LULC maps of 4 hotspot ROIs | `05_lulc_analysis/plot_lulc_maps.py` |
| **Table 1** | GRACE data gap summary | `04_gap_filling/grace_gap_fill.py` |
| **Table 2** | Hotspot region summary | `03_hotspot_identification/hotspot_extraction.py` |
| **Table 4** | LULC composition (%) per ROI | `05_lulc_analysis/extract_lulc.py` |

---

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| **GRACE/GRACE-FO CSR RL06.3 Mascons** | [UT CSR](https://www2.csr.utexas.edu/grace/RL0603_mascons.html) | 0.5° × 0.5°, monthly |
| **IMD Gridded Daily Rainfall** | [IMD Pune](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | 0.25° × 0.25°, daily |
| **ESA WorldCover 2021 v200** | [ESA](https://esa-worldcover.org/en) | 10 m |
| **TWSA Reconstruction (gap-filling)** | [Zenodo](https://doi.org/10.5281/zenodo.15107065) | 1° × 1°, monthly |
| **India Boundary** | [DataMeet](https://github.com/datameet/maps) | Vector (GeoJSON) |

---

## Setup & Installation

### Prerequisites
- Python >= 3.9
- ~2 GB disk space for GRACE NetCDF + IMD data

### Installation

```bash
# Clone the repository
git clone https://github.com/Raakshass/GRACE-TWSA-India-Analysis.git
cd GRACE-TWSA-India-Analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download GRACE mascon data and India boundary (see `data/README.md`)
2. Place files in the `data/` directory
3. Run scripts in numerical order (01 → 06)

---

## Execution Order

Scripts are numbered to indicate the recommended execution order:

```bash
# Step 1: GRACE mascon analysis (spatial mean, time series, trend)
python 01_grace_mascon_analysis/plot_grace_mascon.py

# Step 2: IMD precipitation (download → aggregate → map)
python 02_imd_precipitation/download_imd.py
python 02_imd_precipitation/wateryear_pipeline.py
python 02_imd_precipitation/spatial_maps.py

# Step 3: Hotspot identification
python 03_hotspot_identification/hotspot_extraction.py
python 03_hotspot_identification/hotspot_temporal.py

# Step 4: Gap filling & gap-filled figures
python 04_gap_filling/grace_gap_fill.py
python 04_gap_filling/gapfilled_figures.py

# Step 5: LULC analysis
python 05_lulc_analysis/extract_lulc.py
python 05_lulc_analysis/plot_lulc_maps.py

# Step 6: Regional analysis
python 06_punjab_haryana_analysis/regional_analysis.py
```

---

## Key Findings

- **All-India TWSA trend**: −0.56 cm/yr (April 2002 – June 2026)
- **ROI1** (North-West India): −5.78 cm/yr — severe groundwater depletion in Punjab, Haryana, western UP
- **ROI2** (North-East India): −2.74 cm/yr — glacial/snow mass loss in Eastern Himalayas
- **ROI3** (Central-Eastern India): −1.59 cm/yr — combined agricultural extraction in Bihar, Bengal
- **ROI4** (Southern Peninsula): +1.05 cm/yr — water storage gain, likely from watershed management

---

## Methodology

1. **GRACE Mascon Processing**: CSR RL06.3 mascon solutions with all corrections applied, masked to India boundary using official political boundary shapefile
2. **Trend Estimation**: Pixel-wise OLS linear regression with significance testing (p < 0.05)
3. **Hotspot Delineation**: Contour-based shapefile generation at defined TWSA trend thresholds
4. **Gap Reconstruction**: 11-month inter-mission gap (Jul 2017 – May 2018) filled using ensemble reconstruction dataset, regridded from 1° to 0.5° via nearest-neighbor interpolation
5. **Water Year Convention**: Indian water year (June–May), yielding 24 complete water years (WY 2002–WY 2025)

---

## Citation

If you use this code, please cite:

```
Save, H., S. Bettadpur, and B.D. Tapley (2016), High-resolution CSR GRACE RL05 mascons,
Journal of Geophysical Research: Solid Earth, 121, doi:10.1002/2016JB013007.
```

---

## License

This project is for academic/research purposes as part of a B.Tech Project.
