"""
IMD Spatial Maps + Year-by-Year Anomaly Panels
================================================
Replicating Figure 2 and Figure 3 from Panda & Wahr (2016, WRR)

Figure 2: (a) Mean annual precipitation  (b) Precipitation trend map
Figure 3: (a) Standardized rainfall anomaly panels per water year
          (b) GRACE TWSA anomaly panels per water year
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'imd_precipitation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

# India boundary
INDIA_GEOJSON = 'india_boundary.geojson'
india_gdf = gpd.read_file(INDIA_GEOJSON)
india_geom = india_gdf.union_all()

# Water-year annual rainfall
# Find the water-year file
wy_files = [f for f in os.listdir(OUTPUT_DIR) if 'wateryear' in f and f.endswith('.nc')]
if wy_files:
    wy_file = os.path.join(OUTPUT_DIR, sorted(wy_files)[-1])  # Latest
else:
    raise FileNotFoundError("No water-year NetCDF found! Run imd_wateryear_pipeline.py first.")

print(f"  Loading water-year rainfall: {wy_file}")
imd_wy = xr.open_dataset(wy_file)
water_years = imd_wy.water_year.values
print(f"  Water years: {water_years[0]}-{water_years[-1]} ({len(water_years)} years)")

imd_lat = imd_wy.lat.values
imd_lon = imd_wy.lon.values

# GRACE data
NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
ds_grace = xr.open_dataset(NC_FILE)
time_ref = np.datetime64('2002-01-01')
time_days = ds_grace['time'].values
time_dates = np.array([time_ref + np.timedelta64(int(d), 'D') for d in time_days])
ds_grace = ds_grace.assign_coords(time=time_dates)
grace_india = ds_grace['lwe_thickness'].sel(
    lat=slice(6.0, 38.0), lon=slice(67.0, 98.0),
    time=slice(np.datetime64('2002-04-01'), np.datetime64('2026-07-01'))
)
grace_lat = grace_india.lat.values
grace_lon = grace_india.lon.values

# India mask for GRACE
india_mask = np.load('gap_fill_results/india_mask.npy')

# Create India mask for IMD grid
print("  Creating IMD India mask...")
prepared_india = prep(india_geom)
lon2d_imd, lat2d_imd = np.meshgrid(imd_lon, imd_lat)
imd_india_mask = np.zeros_like(lon2d_imd, dtype=bool)
for i in range(imd_india_mask.shape[0]):
    for j in range(imd_india_mask.shape[1]):
        imd_india_mask[i, j] = prepared_india.contains(Point(lon2d_imd[i, j], lat2d_imd[i, j]))

# Also mask out NaN pixels from the data
sample = imd_wy['rain'].isel(water_year=0).values
data_mask = ~np.isnan(sample) & (sample >= 0)
imd_india_mask = imd_india_mask & data_mask
print(f"  IMD India pixels: {imd_india_mask.sum()}")

# ============================================================
# COMPUTE GRACE WATER-YEAR ANNUAL MEANS
# ============================================================
print("\n" + "=" * 60)
print("Computing GRACE water-year annual TWSA...")
print("=" * 60)

import pandas as pd

grace_months = pd.DatetimeIndex(grace_india.time.values)
grace_wy = np.where(grace_months.month >= 6, grace_months.year, grace_months.year - 1)
grace_india_wy = grace_india.assign_coords(water_year=('time', grace_wy))

# Group and mean
grace_wy_annual = grace_india_wy.groupby('water_year').mean(dim='time')

# Filter to valid water years that have at least 8 months
grace_valid_wys = []
for wy in np.unique(grace_wy):
    subset = grace_india_wy.where(grace_india_wy.water_year == wy, drop=True)
    if len(subset.time) >= 8:
        grace_valid_wys.append(wy)

grace_wy_annual = grace_wy_annual.sel(water_year=grace_valid_wys)
print(f"  GRACE water years: {grace_valid_wys[0]}-{grace_valid_wys[-1]} ({len(grace_valid_wys)} years)")

# Apply India mask to GRACE
grace_wy_masked = grace_wy_annual.where(india_mask)

# ============================================================
# FIGURE 2: SPATIAL MAPS — MEAN + TREND
# ============================================================
print("\n" + "=" * 60)
print("Creating Figure 2: Mean + Trend spatial maps...")
print("=" * 60)

rain_data = imd_wy['rain'].values  # shape: (n_wy, nlat, nlon)

# (a) Mean annual precipitation
mean_rain = np.nanmean(rain_data, axis=0)
mean_rain_masked = np.where(imd_india_mask, mean_rain, np.nan)

# (b) Linear trend at each pixel
n_wy = len(water_years)
trend_rain = np.full((len(imd_lat), len(imd_lon)), np.nan)
pval_rain = np.full_like(trend_rain, np.nan)

for i in range(len(imd_lat)):
    for j in range(len(imd_lon)):
        if imd_india_mask[i, j]:
            ts = rain_data[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() > 5:
                sl, _, _, pv, _ = stats.linregress(
                    np.arange(n_wy)[valid], ts[valid]
                )
                trend_rain[i, j] = sl
                pval_rain[i, j] = pv

print(f"  Mean rainfall range: {np.nanmin(mean_rain_masked):.0f} to {np.nanmax(mean_rain_masked):.0f} mm")
print(f"  Trend range: {np.nanmin(trend_rain):.1f} to {np.nanmax(trend_rain):.1f} mm/yr")

# Plot Figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10),
                                subplot_kw={'projection': ccrs.PlateCarree()})

for ax in [ax1, ax2]:
    ax.add_geometries(
        india_gdf.geometry, crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor='black', linewidth=1.2
    )
    ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
    ax.set_extent([67.0, 98.0, 6.0, 38.0], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(70, 100, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 40, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

# Panel (a): Mean annual precipitation
im1 = ax1.pcolormesh(
    imd_lon, imd_lat, mean_rain_masked,
    transform=ccrs.PlateCarree(),
    cmap='YlGnBu', shading='auto',
    vmin=0, vmax=3000
)
cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.7, pad=0.02)
cbar1.set_label('Mean Annual Precipitation (mm)', fontsize=11)
ax1.set_title(f'(a) Mean Annual Precipitation\n(WY {water_years[0]}–{water_years[-1]+1})',
              fontsize=13, fontweight='bold')

# Panel (b): Precipitation trend
vmax_t = np.nanpercentile(np.abs(trend_rain[~np.isnan(trend_rain)]), 95)
im2 = ax2.pcolormesh(
    imd_lon, imd_lat, trend_rain,
    transform=ccrs.PlateCarree(),
    cmap='BrBG', shading='auto',
    vmin=-vmax_t, vmax=vmax_t
)

# Stipple significant trends (p < 0.05)
sig_mask = pval_rain < 0.05
sig_lats, sig_lons = np.where(sig_mask)
if len(sig_lats) > 0:
    ax2.scatter(
        imd_lon[sig_lons], imd_lat[sig_lats],
        s=0.3, c='black', alpha=0.4, transform=ccrs.PlateCarree(),
        marker='.'
    )

cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.7, pad=0.02)
cbar2.set_label('Precipitation Trend (mm/yr)', fontsize=11)
ax2.set_title(f'(b) Linear Trend in Precipitation\n(WY {water_years[0]}–{water_years[-1]+1})',
              fontsize=13, fontweight='bold')

sig_pct = np.nansum(sig_mask) / np.nansum(~np.isnan(trend_rain)) * 100
ax2.text(0.02, 0.02, f'Stippled: p < 0.05 ({sig_pct:.0f}% pixels)',
         transform=ax2.transAxes, fontsize=8, ha='left', va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('IMD Gridded Precipitation (0.25° × 0.25°) over India',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

fig2_path = os.path.join(OUTPUT_DIR, 'spatial_mean_and_trend_precipitation.png')
fig.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig2_path}")

# ============================================================
# FIGURE 3a: STANDARDIZED RAINFALL ANOMALY — YEAR-BY-YEAR
# ============================================================
print("\n" + "=" * 60)
print("Creating Figure 3a: Rainfall anomaly spatial evolution...")
print("=" * 60)

# Compute standardized anomaly at each pixel
std_rain = np.nanstd(rain_data, axis=0)
std_rain[std_rain == 0] = np.nan  # Avoid division by zero

n_panels = len(water_years)
ncols = 5
nrows = int(np.ceil(n_panels / ncols))

fig3a, axes3a = plt.subplots(nrows, ncols, figsize=(22, nrows * 4.5),
                              subplot_kw={'projection': ccrs.PlateCarree()})

for idx in range(nrows * ncols):
    row, col = divmod(idx, ncols)
    ax = axes3a[row, col] if nrows > 1 else axes3a[col]
    
    if idx < n_panels:
        wy = water_years[idx]
        annual_rain = rain_data[idx]
        z_score = (annual_rain - mean_rain) / std_rain
        z_score_masked = np.where(imd_india_mask, z_score, np.nan)
        
        im = ax.pcolormesh(
            imd_lon, imd_lat, z_score_masked,
            transform=ccrs.PlateCarree(),
            cmap='RdBu', shading='auto',
            vmin=-3, vmax=3
        )
        ax.add_geometries(
            india_gdf.geometry, crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black', linewidth=0.8
        )
        ax.set_extent([67.0, 98.0, 6.0, 38.0], crs=ccrs.PlateCarree())
        ax.set_title(f'{wy}–{str(wy+1)[-2:]}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7)
        
        # Minimal axis labels
        if col == 0:
            ax.set_yticks(np.arange(10, 40, 10), crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
        else:
            ax.set_yticks([])
        if row == nrows - 1:
            ax.set_xticks(np.arange(70, 100, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
        else:
            ax.set_xticks([])
    else:
        ax.set_visible(False)

# Shared colorbar
cbar_ax = fig3a.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig3a.colorbar(im, cax=cbar_ax)
cbar.set_label('Rainfall Anomaly (Standardized)', fontsize=12)

fig3a.suptitle(
    'Spatial Evolution of Standardized Annual Rainfall Anomaly\n'
    f'(Water Year: Jun–May, {water_years[0]}–{water_years[-1]+1})',
    fontsize=16, fontweight='bold', y=1.02
)
fig3a.subplots_adjust(wspace=0.05, hspace=0.15)

fig3a_path = os.path.join(OUTPUT_DIR, 'fig3a_rainfall_anomaly_spatial_evolution.png')
fig3a.savefig(fig3a_path, dpi=200, bbox_inches='tight')
plt.close(fig3a)
print(f"  Saved: {fig3a_path}")

# ============================================================
# FIGURE 3b: TWSA ANOMALY — YEAR-BY-YEAR
# ============================================================
print("\n" + "=" * 60)
print("Creating Figure 3b: TWSA anomaly spatial evolution...")
print("=" * 60)

# Use GRACE water-year means
twsa_data = grace_wy_masked.values  # (n_wy, nlat, nlon)
twsa_wys = grace_valid_wys

n_panels_g = len(twsa_wys)
ncols_g = 5
nrows_g = int(np.ceil(n_panels_g / ncols_g))

fig3b, axes3b = plt.subplots(nrows_g, ncols_g, figsize=(22, nrows_g * 4.5),
                              subplot_kw={'projection': ccrs.PlateCarree()})

# Determine color range from data
vmax_twsa = np.nanpercentile(np.abs(twsa_data), 95)
if vmax_twsa < 1:
    vmax_twsa = 10

for idx in range(nrows_g * ncols_g):
    row, col = divmod(idx, ncols_g)
    ax = axes3b[row, col] if nrows_g > 1 else axes3b[col]
    
    if idx < n_panels_g:
        wy = twsa_wys[idx]
        wy_idx = list(twsa_wys).index(wy)
        twsa_map = twsa_data[wy_idx]
        
        im = ax.pcolormesh(
            grace_lon, grace_lat, twsa_map,
            transform=ccrs.PlateCarree(),
            cmap='RdBu', shading='auto',
            vmin=-vmax_twsa, vmax=vmax_twsa
        )
        ax.add_geometries(
            india_gdf.geometry, crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black', linewidth=0.8
        )
        ax.set_extent([67.0, 98.0, 6.0, 38.0], crs=ccrs.PlateCarree())
        ax.set_title(f'{wy}–{str(wy+1)[-2:]}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7)
        
        if col == 0:
            ax.set_yticks(np.arange(10, 40, 10), crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
        else:
            ax.set_yticks([])
        if row == nrows_g - 1:
            ax.set_xticks(np.arange(70, 100, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
        else:
            ax.set_xticks([])
    else:
        ax.set_visible(False)

# Shared colorbar
cbar_ax2 = fig3b.add_axes([0.92, 0.15, 0.015, 0.7])
cbar2 = fig3b.colorbar(im, cax=cbar_ax2)
cbar2.set_label('TWSA (cm of water equivalent thickness)', fontsize=12)

fig3b.suptitle(
    'Spatial Evolution of GRACE-Derived Terrestrial Water Storage Anomaly\n'
    f'(Water Year: Jun–May, {twsa_wys[0]}–{twsa_wys[-1]+1})',
    fontsize=16, fontweight='bold', y=1.02
)
fig3b.subplots_adjust(wspace=0.05, hspace=0.15)

fig3b_path = os.path.join(OUTPUT_DIR, 'fig3b_twsa_anomaly_spatial_evolution.png')
fig3b.savefig(fig3b_path, dpi=200, bbox_inches='tight')
plt.close(fig3b)
print(f"  Saved: {fig3b_path}")

# ============================================================
# CLEANUP
# ============================================================
ds_grace.close()
imd_wy.close()

print("\n" + "=" * 60)
print("ALL SPATIAL MAPS COMPLETE")
print(f"  Fig 2: {fig2_path}")
print(f"  Fig 3a: {fig3a_path}")
print(f"  Fig 3b: {fig3b_path}")
print("=" * 60)
