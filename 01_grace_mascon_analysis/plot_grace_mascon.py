#!/usr/bin/env python3
"""
GRACE/GRACE-FO RL06.3 Mascon Data Plotter for India
====================================================
Plots CSR GRACE/GRACE-FO RL0603 Mascons (all corrections applied)
for the Indian subcontinent, Apr 2002 - May 2026.

Uses India boundary shapefile (datameet/maps) for correct political boundary.
Mascon data clipped to India boundary and masked to show only land within India.

Data source: https://www2.csr.utexas.edu/grace/RL0603_mascons.html
Citation: Save, H., S. Bettadpur, and B.D. Tapley (2016),
          doi:10.1002/2016JB013007
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import mapping
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_FILE = Path(__file__).parent / 'data' / 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
SHAPEFILE = Path(__file__).parent / 'data' / 'india_boundary.geojson'
OUTPUT_DIR = Path(__file__).parent / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# India geographic bounds (extended to cover full J&K per official boundary)
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 67.0
INDIA_LON_MAX = 98.0

# Reference date for time coordinate
TIME_REFERENCE = np.datetime64('2002-01-01')

# Time range
TIME_START = np.datetime64('2002-04-01')
TIME_END = np.datetime64('2026-06-01')

# GRACE/GRACE-FO gap
GRACE_GAP_START = np.datetime64('2017-07-01')
GRACE_GAP_END = np.datetime64('2018-06-01')

# ============================================================
# Load India boundary shapefile
# ============================================================
print("Loading India boundary shapefile...")
india_gdf = gpd.read_file(SHAPEFILE)
india_geom = india_gdf.union_all()
print(f"  Boundary bounds: {india_gdf.total_bounds}")

# ============================================================
# Load GRACE mascon data
# ============================================================
print("Loading GRACE mascon data...")
ds = xr.open_dataset(DATA_FILE)

# Convert time (days since 2002-01-01, stored with capital 'Units')
time_days = ds['time'].values
time_dates = np.array([TIME_REFERENCE + np.timedelta64(int(d), 'D') for d in time_days])
ds = ds.assign_coords(time=time_dates)

main_var = 'lwe_thickness'
units_label = 'cm'

print(f"  Variable: {main_var}, Shape: {ds[main_var].shape}")
print(f"  Time range: {time_dates[0]} to {time_dates[-1]}, {len(time_dates)} timesteps")

# ============================================================
# Subset to India region and time range
# ============================================================
india = ds[main_var].sel(
    lat=slice(INDIA_LAT_MIN, INDIA_LAT_MAX),
    lon=slice(INDIA_LON_MIN, INDIA_LON_MAX)
)
india = india.sel(time=slice(TIME_START, TIME_END))

print(f"  India subset: {india.shape}, {len(india.time)} timesteps")

# ============================================================
# Mask data to India boundary using shapefile
# ============================================================
print("Masking data to India boundary...")
lon2d, lat2d = np.meshgrid(india.lon.values, india.lat.values)

from shapely.geometry import Point
from shapely.prepared import prep

prepared_geom = prep(india_geom)
mask = np.zeros_like(lon2d, dtype=bool)
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        mask[i, j] = prepared_geom.contains(Point(lon2d[i, j], lat2d[i, j]))

print(f"  Grid points inside India: {mask.sum()} / {mask.size}")

# Apply mask to all timesteps
india_masked = india.where(mask)

# ============================================================
# Area weights (cosine of latitude)
# ============================================================
lat_vals = india.lat.values
weights = np.cos(np.deg2rad(lat_vals))
weights_da = xr.DataArray(weights, dims=['lat'], coords={'lat': lat_vals})


def add_india_map(ax):
    """Add India shapefile boundary and map features."""
    ax.add_geometries(
        india_gdf.geometry, crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor='black', linewidth=1.5
    )
    ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
    ax.set_extent([INDIA_LON_MIN, INDIA_LON_MAX, INDIA_LAT_MIN, INDIA_LAT_MAX],
                  crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(70, 100, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 40, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')


# ============================================================
# FIGURE 1: Spatial Map — Temporal Mean TWSA
# ============================================================
print("\n[1/3] Generating Figure 1: mean TWSA spatial map...")

temporal_mean = india_masked.mean(dim='time')

fig, ax = plt.subplots(1, 1, figsize=(10, 12),
                       subplot_kw={'projection': ccrs.PlateCarree()})

vmax_mean = max(abs(float(temporal_mean.quantile(0.02))),
                abs(float(temporal_mean.quantile(0.98))))
im = ax.pcolormesh(
    india.lon.values, india.lat.values, temporal_mean.values,
    transform=ccrs.PlateCarree(),
    cmap='RdBu',
    vmin=-vmax_mean, vmax=vmax_mean,
    shading='auto'
)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04,
                     shrink=0.75, aspect=30)
cbar.set_label(f'Liquid Water Equivalent Thickness ({units_label})', fontsize=11)

add_india_map(ax)

# Figure caption as suptitle + subtitle
fig.suptitle('Figure 1', fontsize=14, fontweight='bold', y=0.95)
ax.set_title('Mean Terrestrial Water Storage Anomaly over India\n'
             '(April 2002 – May 2026, baseline: 2004.0–2010.0)\n'
             'Data: CSR GRACE/GRACE-FO RL06.3 Mascons',
             fontsize=12, pad=10)

# Figure caption below the plot
fig.text(0.5, 0.01,
         'Figure 1. Temporal mean of liquid water equivalent thickness anomaly over India\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions (April 2002 – May 2026).\n'
         'Anomalies are relative to the 2004.0–2010.0 mean baseline. Units: centimeters of water equivalent.',
         ha='center', va='bottom', fontsize=9, fontstyle='italic', wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(OUTPUT_DIR / 'figure1_india_twsa_spatial_mean.png', dpi=200, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'figure1_india_twsa_spatial_mean.png'}")
plt.close()

# ============================================================
# FIGURE 2: Time Series — Area-Weighted Average TWSA
# ============================================================
print("[2/3] Generating Figure 2: TWSA time series...")

weighted_mean = india_masked.weighted(weights_da).mean(dim=['lat', 'lon'])

fig, ax = plt.subplots(figsize=(14, 6))

times = weighted_mean.time.values
values = weighted_mean.values

ax.plot(times, values, 'o-', color='#2166ac', linewidth=0.8, markersize=2.5,
        alpha=0.7, label='Monthly TWSA')

if len(values) > 12:
    kernel = np.ones(12) / 12
    smoothed = np.convolve(values, kernel, mode='valid')
    smooth_times = times[5:-6]
    ax.plot(smooth_times, smoothed, color='#b2182b', linewidth=2.5,
            label='12-month running mean', zorder=5)

ax.axvspan(GRACE_GAP_START, GRACE_GAP_END, alpha=0.15, color='gray',
           label='GRACE/GRACE-FO gap')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.5)

time_numeric = (times - times[0]).astype('timedelta64[D]').astype(float)
valid = ~np.isnan(values)
slope, intercept, r, p, se = stats.linregress(time_numeric[valid], values[valid])
trend_line = intercept + slope * time_numeric
slope_per_year = slope * 365.25
ax.plot(times, trend_line, '--', color='#4daf4a', linewidth=1.5,
        label=f'Linear trend: {slope_per_year:.2f} {units_label}/year', zorder=4)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel(f'TWSA ({units_label})', fontsize=12)
ax.set_title('Figure 2', fontsize=14, fontweight='bold', loc='left')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

mid_gap = GRACE_GAP_START + (GRACE_GAP_END - GRACE_GAP_START) // 2
ymin, ymax = ax.get_ylim()
ax.annotate('GRACE \u2192 GRACE-FO\ntransition',
            xy=(mid_gap, ymax * 0.9),
            ha='center', va='top', fontsize=9, color='gray',
            fontstyle='italic')

# Figure caption
fig.text(0.5, -0.02,
         'Figure 2. Area-weighted terrestrial water storage anomaly time series over India\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions (April 2002 – May 2026).\n'
         'Anomalies relative to 2004.0–2010.0 baseline. Grey shading marks the GRACE to GRACE-FO data gap.',
         ha='center', va='top', fontsize=9, fontstyle='italic', wrap=True)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'figure2_india_twsa_timeseries.png', dpi=200, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'figure2_india_twsa_timeseries.png'}")
plt.close()

# ============================================================
# FIGURE 3: Spatial Trend Map — Linear TWSA Trend (cm/year)
# ============================================================
print("[3/3] Generating Figure 3: TWSA trend map...")

time_days_arr = (india.time.values - india.time.values[0]).astype('timedelta64[D]').astype(float)
time_years_arr = time_days_arr / 365.25

nlat = india.shape[1]
nlon = india.shape[2]
trend_map = np.full((nlat, nlon), np.nan)
pvalue_map = np.full_like(trend_map, np.nan)

data_vals = india_masked.values
print(f"  Computing pixel-wise trends ({nlat}\u00d7{nlon} grid)...")
for i in range(nlat):
    for j in range(nlon):
        ts = data_vals[:, i, j]
        valid_mask = ~np.isnan(ts)
        if valid_mask.sum() > 10:
            sl, _, _, pv, _ = stats.linregress(time_years_arr[valid_mask], ts[valid_mask])
            trend_map[i, j] = sl
            pvalue_map[i, j] = pv

print("  Trend computation complete.")

fig, ax = plt.subplots(1, 1, figsize=(10, 12),
                       subplot_kw={'projection': ccrs.PlateCarree()})

vmax = np.nanpercentile(np.abs(trend_map), 95)
if vmax < 0.1:
    vmax = 0.5

im = ax.pcolormesh(
    india.lon.values, india.lat.values, trend_map,
    transform=ccrs.PlateCarree(),
    cmap='RdBu',
    vmin=-vmax, vmax=vmax,
    shading='auto'
)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04,
                     shrink=0.75, aspect=30)
cbar.set_label(f'TWSA Linear Trend ({units_label}/year)', fontsize=11)

add_india_map(ax)

# Stipple for significance
sig_mask = pvalue_map < 0.05
if sig_mask.any():
    sig_lons, sig_lats = np.meshgrid(india.lon.values, india.lat.values)
    step = 4
    sig_mask_sub = np.zeros_like(sig_mask, dtype=bool)
    sig_mask_sub[::step, ::step] = sig_mask[::step, ::step]
    ax.scatter(sig_lons[sig_mask_sub], sig_lats[sig_mask_sub],
               marker='.', s=0.5, color='black', alpha=0.4,
               transform=ccrs.PlateCarree())

fig.suptitle('Figure 3', fontsize=14, fontweight='bold', y=0.95)
ax.set_title('Linear Trend in Terrestrial Water Storage Anomaly over India\n'
             '(April 2002 – May 2026)\n'
             'Data: CSR GRACE/GRACE-FO RL06.3 Mascons',
             fontsize=12, pad=10)

fig.text(0.5, 0.01,
         'Figure 3. Pixel-wise linear trend in liquid water equivalent thickness anomaly over India\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions (April 2002 – May 2026).\n'
         'Red indicates water storage loss; blue indicates gain. Units: centimeters per year.',
         ha='center', va='bottom', fontsize=9, fontstyle='italic', wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(OUTPUT_DIR / 'figure3_india_twsa_trend.png', dpi=200, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'figure3_india_twsa_trend.png'}")
plt.close()

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Data file: {DATA_FILE.name}")
print(f"Variable: {main_var} (Liquid Water Equivalent Thickness)")
print(f"Units: {units_label} (centimeters of water equivalent)")
print(f"India boundary: datameet/maps (india-composite.geojson)")
print(f"Time period: {str(india.time.values[0])[:10]} to {str(india.time.values[-1])[:10]}")
print(f"Monthly solutions: {len(india.time)}")
print(f"Anomaly baseline: 2004.0 – 2010.0 mean")
print(f"Grid resolution: 0.25 degree")
print(f"Grid points inside India: {mask.sum()}")
print()
print(f"Area-weighted mean TWSA: {float(weighted_mean.mean()):.2f} {units_label}")
print(f"Std dev TWSA: {float(weighted_mean.std()):.2f} {units_label}")
print(f"Min TWSA: {float(weighted_mean.min()):.2f} {units_label} "
      f"({str(weighted_mean.time.values[weighted_mean.argmin().values])[:7]})")
print(f"Max TWSA: {float(weighted_mean.max()):.2f} {units_label} "
      f"({str(weighted_mean.time.values[weighted_mean.argmax().values])[:7]})")
print(f"Area-weighted linear trend: {slope_per_year:.3f} {units_label}/year")
print(f"Domain-mean pixel trend: {float(np.nanmean(trend_map)):.3f} {units_label}/year")
print()
print(f"Saved:")
print(f"  Figure 1: {OUTPUT_DIR / 'figure1_india_twsa_spatial_mean.png'}")
print(f"  Figure 2: {OUTPUT_DIR / 'figure2_india_twsa_timeseries.png'}")
print(f"  Figure 3: {OUTPUT_DIR / 'figure3_india_twsa_trend.png'}")
print("\nDone!")

ds.close()
