"""
GRACE Gap-Filling using Pre-Computed Reconstructed TWSA
=======================================================
Uses the already-reconstructed TWSA dataset published by:
  Anjaneyulu, Garai, Abhishek & Zhao (2026)
  DOI: https://doi.org/10.1007/s00190-026-02098-x
  Zenodo: https://zenodo.org/records/18732257

Data format: Excel file with rows = grid points (lon, lat),
  columns = monthly TWSA values (1980-01 to 2024-12)

Google Colab:
  Cell 1: !pip install -q cartopy geopandas xarray netCDF4 scipy shapely tqdm openpyxl
  Cell 2: Upload TWSA_Reconstruction_Locally_Adaptive_Ensemble_1980-2024_v1.xlsx
  Cell 3: Paste this entire script
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (works in terminal + Colab)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy import stats
import pandas as pd
import os
import urllib.request
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("all imports successful.")

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
# STEP 1: LOAD GRACE MASCON DATA + INDIA MASK
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading GRACE data and creating India mask")
print("=" * 60)

NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
NC_URL = 'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
INDIA_GEOJSON = 'india_boundary.geojson'
INDIA_URL = 'https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson'

if not os.path.exists(NC_FILE):
    print(f"downloading {NC_FILE} (~107 MB)...")
    urllib.request.urlretrieve(NC_URL, NC_FILE)
    print("done.")
else:
    print(f"{NC_FILE} exists.")

if not os.path.exists(INDIA_GEOJSON):
    print("downloading india boundary...")
    urllib.request.urlretrieve(INDIA_URL, INDIA_GEOJSON)
    print("done.")

# load india boundary
india_gdf = gpd.read_file(INDIA_GEOJSON)
try:
    india_geom = india_gdf.union_all()
except AttributeError:
    india_geom = india_gdf.unary_union  # geopandas < 1.0 fallback
prepared_india = prep(india_geom)

# load grace mascon
ds_grace = xr.open_dataset(NC_FILE)
time_ref = np.datetime64('2002-01-01')
time_days = ds_grace['time'].values
time_dates = np.array([time_ref + np.timedelta64(int(d), 'D') for d in time_days])
ds_grace = ds_grace.assign_coords(time=time_dates)

grace_india = ds_grace['lwe_thickness'].sel(
    lat=slice(6.0, 38.0),
    lon=slice(67.0, 98.0),
    time=slice(np.datetime64('2002-04-01'), np.datetime64('2026-07-01'))
)
print(f"GRACE shape: {grace_india.shape}")
print(f"time: {grace_india.time.values[0]} to {grace_india.time.values[-1]}")

# create india mask
mask_file = os.path.join(SAVE_DIR, 'india_mask.npy')
if os.path.exists(mask_file):
    india_mask = np.load(mask_file)
    print(f"loaded cached india mask: {india_mask.sum()} pixels")
else:
    print("creating india mask (takes ~1 minute)...")
    lon2d, lat2d = np.meshgrid(grace_india.lon.values, grace_india.lat.values)
    india_mask = np.zeros_like(lon2d, dtype=bool)
    for i in range(india_mask.shape[0]):
        for j in range(india_mask.shape[1]):
            india_mask[i, j] = prepared_india.contains(Point(lon2d[i, j], lat2d[i, j]))
    np.save(mask_file, india_mask)
    print(f"india mask created: {india_mask.sum()} pixels")

grace_masked = grace_india.where(india_mask)
n_lat = len(grace_india.lat)
n_lon = len(grace_india.lon)
grace_lats = grace_india.lat.values
grace_lons = grace_india.lon.values


# ================================================================
# STEP 2: READ RECONSTRUCTED TWSA FROM EXCEL
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Reading reconstructed TWSA from Zenodo Excel file")
print("=" * 60)
print("Source: Anjaneyulu et al. (2026), J. Geodesy")
print("DOI: https://doi.org/10.1007/s00190-026-02098-x\n")

# find the excel file
ENSEMBLE_FILE = 'TWSA_Reconstruction_Locally_Adaptive_Ensemble_1980-2024_v1.xlsx'

# also check in gap_fill_results/zenodo_data/ (from previous download)
if not os.path.exists(ENSEMBLE_FILE):
    alt_path = os.path.join(SAVE_DIR, 'zenodo_data', ENSEMBLE_FILE)
    if os.path.exists(alt_path):
        ENSEMBLE_FILE = alt_path

if not os.path.exists(ENSEMBLE_FILE):
    print(f"ERROR: {ENSEMBLE_FILE} not found!")
    print("please upload the file to Colab using:")
    print("  from google.colab import files")
    print("  uploaded = files.upload()")
    raise FileNotFoundError(ENSEMBLE_FILE)

fsize_mb = os.path.getsize(ENSEMBLE_FILE) / (1024 * 1024)
print(f"reading: {os.path.basename(ENSEMBLE_FILE)} ({fsize_mb:.0f} MB)")
print("this is a large Excel file — loading takes 2-5 minutes...\n")

# file layout: rows = grid points, cols = [lon, lat, 1980-01, ..., 2024-12]
df = pd.read_excel(ENSEMBLE_FILE, engine='openpyxl')
print(f"loaded: {df.shape[0]} grid points × {df.shape[1]} columns")
print(f"time range: {df.columns[2]} to {df.columns[-1]}")
print(f"lon range: {df['lon'].min()} to {df['lon'].max()}")
print(f"lat range: {df['lat'].min()} to {df['lat'].max()}")


# ================================================================
# STEP 3: EXTRACT GAP MONTHS FOR INDIA
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Extracting 11 gap months (Jul 2017 – May 2018) for India")
print("=" * 60)

# filter to india bounding box
india_rows = (df['lat'] >= 6.0) & (df['lat'] <= 38.0) & \
             (df['lon'] >= 67.0) & (df['lon'] <= 98.0)
df_india = df[india_rows].copy()
print(f"india grid points: {len(df_india)}")

# gap month column names
gap_month_cols = ['2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
                  '2018-01', '2018-02', '2018-03', '2018-04', '2018-05']
n_gap = len(gap_month_cols)

# verify columns exist
missing = [c for c in gap_month_cols if c not in df.columns]
if missing:
    print(f"WARNING: missing columns: {missing}")
    gap_month_cols = [c for c in gap_month_cols if c in df.columns]
    n_gap = len(gap_month_cols)

print(f"gap months to extract: {n_gap}")
for c in gap_month_cols:
    print(f"  {c}")

# extract gap data
gap_lons = df_india['lon'].values
gap_lats = df_india['lat'].values
gap_values = df_india[gap_month_cols].values  # (n_points, n_gap_months)

# get unique sorted coordinates
unique_lats = np.sort(np.unique(gap_lats))
unique_lons = np.sort(np.unique(gap_lons))
print(f"\nreconstructed grid: {len(unique_lats)} lats × {len(unique_lons)} lons")
print(f"lat: {unique_lats[0]} to {unique_lats[-1]}, spacing: {np.diff(unique_lats[:3])}")
print(f"lon: {unique_lons[0]} to {unique_lons[-1]}, spacing: {np.diff(unique_lons[:3])}")

# build 3D array: (n_gap_months, n_lat, n_lon)
gap_3d = np.full((n_gap, len(unique_lats), len(unique_lons)), np.nan)

for p_idx in range(len(gap_lats)):
    lat_idx = np.searchsorted(unique_lats, gap_lats[p_idx])
    lon_idx = np.searchsorted(unique_lons, gap_lons[p_idx])
    gap_3d[:, lat_idx, lon_idx] = gap_values[p_idx, :]

# UNIT CONVERSION: Zenodo reconstructed data is in mm, GRACE is in cm
# divide by 10 to convert mm → cm
gap_3d = gap_3d / 10.0
print("\nunit conversion: mm → cm (divided by 10)")

# create xarray
gap_times_dt64 = np.array([np.datetime64(f'{c}-15') for c in gap_month_cols])

recon_gap = xr.DataArray(
    gap_3d,
    dims=['time', 'lat', 'lon'],
    coords={'time': gap_times_dt64, 'lat': unique_lats, 'lon': unique_lons}
)
print(f"reconstructed gap xarray: {recon_gap.shape}")

# regrid to match GRACE 0.5° mascon grid
print("regridding to GRACE 0.5° grid (nearest neighbor)...")
recon_gap_regridded = recon_gap.interp(lat=grace_lats, lon=grace_lons, method='nearest')
print(f"regridded shape: {recon_gap_regridded.shape}")

# apply india mask
gap_filled_masked = np.where(india_mask[np.newaxis, :, :],
                             recon_gap_regridded.values, np.nan)

print(f"\ngap-filled data ready: {gap_filled_masked.shape}")
print(f"india spatial mean during gap:")
for k, c in enumerate(gap_month_cols):
    val = np.nanmean(gap_filled_masked[k])
    print(f"  {c}: {val:.2f} cm")


# ================================================================
# STEP 4: MERGE GAP-FILLED DATA INTO GRACE TIME SERIES
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: Merging reconstructed gap into GRACE time series")
print("=" * 60)

grace_full_times = grace_india.time.values
grace_full_data = grace_masked.values

# merge time arrays (remove duplicates)
all_times = np.sort(np.unique(np.concatenate([grace_full_times, gap_times_dt64])))

# create merged data array
merged_data = np.full((len(all_times), n_lat, n_lon), np.nan)

# insert GRACE data
for t_idx, t in enumerate(grace_full_times):
    merged_idx = np.where(all_times == t)[0]
    if len(merged_idx) > 0:
        merged_data[merged_idx[0]] = grace_full_data[t_idx]

# insert gap-filled data (from reconstructed TWSA)
for g_idx, t in enumerate(gap_times_dt64):
    merged_idx = np.where(all_times == t)[0]
    if len(merged_idx) > 0:
        merged_data[merged_idx[0]] = gap_filled_masked[g_idx]

# apply india mask
merged_masked = np.where(india_mask[np.newaxis, :, :], merged_data, np.nan)

print(f"merged time series: {len(all_times)} months")
print(f"  original GRACE: {len(grace_full_times)} months")
print(f"  gap-filled added: {len(gap_times_dt64)} months")

# area-weighted india mean
weights = np.cos(np.deg2rad(grace_lats))
weights_2d = weights[:, np.newaxis] * np.ones((1, n_lon))
weights_masked = np.where(india_mask, weights_2d, np.nan)


# ================================================================
# STEP 5: COMPUTE TRENDS (BEFORE vs AFTER GAP-FILLING)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: Mathematical trend analysis")
print("=" * 60)

# --- A. ORIGINAL GRACE time series (WITH gap = NaN months excluded) ---
grace_only_masked = np.where(india_mask[np.newaxis, :, :], grace_full_data, np.nan)
grace_ts_original = np.nansum(grace_only_masked * weights_masked[np.newaxis, :, :], axis=(1, 2)) / \
                    np.nansum(weights_masked)

grace_time_num_orig = (grace_full_times - grace_full_times[0]).astype('timedelta64[D]').astype(float)
valid_orig = ~np.isnan(grace_ts_original)

slope_orig, intercept_orig, r_orig, p_orig, se_orig = stats.linregress(
    grace_time_num_orig[valid_orig], grace_ts_original[valid_orig])
slope_yr_orig = slope_orig * 365.25

print(f"\n{'─' * 70}")
print(f"A. ORIGINAL GRACE TREND (gap months excluded)")
print(f"{'─' * 70}")
print(f"  n data points:  {valid_orig.sum()} months (out of {len(grace_full_times)})")
print(f"  time span:      {grace_full_times[0]} to {grace_full_times[-1]}")
print(f"  slope (per day): {slope_orig:.6f} cm/day")
print(f"  slope (per yr):  {slope_yr_orig:.4f} cm/year")
print(f"  intercept:       {intercept_orig:.4f} cm")
print(f"  R:               {r_orig:.6f}")
print(f"  R²:              {r_orig**2:.6f}")
print(f"  p-value:         {p_orig:.4e}")
print(f"  std error:       {se_orig:.6e}")

# --- B. GAP-FILLED COMPLETE time series ---
india_ts_filled = np.nansum(merged_masked * weights_masked[np.newaxis, :, :], axis=(1, 2)) / \
                  np.nansum(weights_masked)

time_num_filled = (all_times - all_times[0]).astype('timedelta64[D]').astype(float)
valid_filled = ~np.isnan(india_ts_filled)

slope_filled, intercept_filled, r_filled, p_filled, se_filled = stats.linregress(
    time_num_filled[valid_filled], india_ts_filled[valid_filled])
slope_yr_filled = slope_filled * 365.25
trend_line_filled = intercept_filled + slope_filled * time_num_filled

print(f"\n{'─' * 70}")
print(f"B. GAP-FILLED TREND (11 reconstructed months included)")
print(f"{'─' * 70}")
print(f"  n data points:  {valid_filled.sum()} months (out of {len(all_times)})")
print(f"  time span:      {all_times[0]} to {all_times[-1]}")
print(f"  slope (per day): {slope_filled:.6f} cm/day")
print(f"  slope (per yr):  {slope_yr_filled:.4f} cm/year")
print(f"  intercept:       {intercept_filled:.4f} cm")
print(f"  R:               {r_filled:.6f}")
print(f"  R²:              {r_filled**2:.6f}")
print(f"  p-value:         {p_filled:.4e}")
print(f"  std error:       {se_filled:.6e}")

# --- C. COMPARISON ---
print(f"\n{'─' * 70}")
print(f"C. COMPARISON: BEFORE vs AFTER GAP-FILLING")
print(f"{'─' * 70}")
print(f"  {'Metric':<25} {'Before (gap excluded)':<25} {'After (gap filled)':<25} {'Change'}")
print(f"  {'─'*25} {'─'*25} {'─'*25} {'─'*15}")
print(f"  {'n months':<25} {valid_orig.sum():<25} {valid_filled.sum():<25} +{valid_filled.sum() - valid_orig.sum()}")
print(f"  {'Trend (cm/yr)':<25} {slope_yr_orig:<25.4f} {slope_yr_filled:<25.4f} {slope_yr_filled - slope_yr_orig:+.4f}")
print(f"  {'R²':<25} {r_orig**2:<25.6f} {r_filled**2:<25.6f} {r_filled**2 - r_orig**2:+.6f}")
print(f"  {'p-value':<25} {p_orig:<25.4e} {p_filled:<25.4e}")
print(f"  {'Std error':<25} {se_orig:<25.6e} {se_filled:<25.6e}")


# ================================================================
# STEP 6: PLOT — UPDATED TEMPORAL TWSA (COMPLETE SERIES)
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: Plotting UPDATED TWSA time series")
print("=" * 60)

# identify which months are gap-filled
is_gap = np.array([t in gap_times_dt64 for t in all_times])

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

GAP_START = np.datetime64('2017-07-01')
GAP_END   = np.datetime64('2018-06-01')

fig, ax = plt.subplots(figsize=(16, 5))

# --- 0. PINK SHADED GAP BAND ---
ax.axvspan(GAP_START, GAP_END, color='#ffcccc', alpha=0.4, zorder=0)

# --- 1. COMPLETE CONTINUOUS BLUE LINE ---
ax.plot(all_times, india_ts_filled, 'o-', color='#2166ac',
        linewidth=2.8, markersize=3, alpha=0.7)

# --- 2. 12-MONTH RUNNING MEAN ---
kernel = np.ones(12) / 12
smoothed = np.convolve(india_ts_filled, kernel, mode='valid')
smooth_t = all_times[5:-6]
ax.plot(smooth_t, smoothed, color='black', linewidth=4, zorder=5)

# --- 3. LINEAR TREND ---
ax.plot(all_times, trend_line_filled, '--', color='#1b7837', linewidth=3.5, zorder=4)

# --- Trend annotation box ---
ax.text(0.98, 0.92, f'Trend: {slope_yr_filled:.2f} cm/yr',
        transform=ax.transAxes, fontsize=10, fontweight='bold',
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9))

# --- Auto y-axis limits ---
ts_min = np.nanmin(india_ts_filled)
ts_max = np.nanmax(india_ts_filled)
y_range = ts_max - ts_min
y_pad = y_range * 0.12
ax.set_ylim(ts_min - y_pad, ts_max + y_pad)

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_ylabel('TWSA (cm)', fontsize=11)
ax.set_title('All-India', fontsize=13, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)
ax.set_xlim(all_times[0], all_times[-1])
ax.set_xlabel('Year', fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# --- Bottom-centered legend ---
legend_handles = [
    Line2D([0], [0], color='#2166ac', linewidth=2.8, marker='o', markersize=3,
           alpha=0.7, label='GRACE/GRACE-FO TWSA (gap-filled)'),
    Line2D([0], [0], color='black', linewidth=4, label='12-month running mean'),
    Line2D([0], [0], color='#1b7837', linewidth=3.5, linestyle='--',
           label='Linear trend'),
    mpatches.Patch(color='#ffcccc', alpha=0.6,
                   label='GRACE/GRACE-FO gap (Jul 2017 – May 2018)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(SAVE_DIR, 'india_twsa_gap_filled_timeseries.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# print caption separately (not in figure)
print(f"\n{'─' * 70}")
print(f"Figure Caption:")
print(f"{'─' * 70}")
print(f"All-India area-weighted TWSA time series (Apr 2002 – Jun 2026).")
print(f"Blue line: continuous GRACE/GRACE-FO TWSA (CSR RL06.3 Mascons)")
print(f"with 11-month gap (Jul 2017 – May 2018) filled using reconstructed")
print(f"TWSA from Anjaneyulu et al. (2026),")
print(f"DOI: https://doi.org/10.1007/s00190-026-02098-x")
print(f"Red squares: reconstructed gap months.")
print(f"Dark red: 12-month running mean on complete series.")
print(f"Green dashed: linear trend = {slope_yr_filled:.2f} cm/year")
print(f"  (before gap-filling: {slope_yr_orig:.2f} cm/year)")
print(f"  R² = {r_filled**2:.4f}, p = {p_filled:.2e}")
print(f"{'─' * 70}")

ds_grace.close()

print("\n" + "=" * 60)
print("ALL DONE. Figure saved in:", SAVE_DIR)
print("=" * 60)
