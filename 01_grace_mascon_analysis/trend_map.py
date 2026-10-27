#!/usr/bin/env python3
"""
Pixel-wise TWSA Linear Trend Map — Full Standalone Google Colab Script
======================================================================
Paste this entire script into a single Colab cell.
First cell: !pip install -q cartopy geopandas xarray netCDF4 scipy shapely

Each pixel shows the slope (cm/year) of the TWSA time series at that location.
Red = water storage loss, Blue = water storage gain.
"""

# !pip install -q cartopy geopandas xarray netCDF4 scipy shapely

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Download data
# ============================================================
NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
NC_URL = 'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'

if not os.path.exists(NC_FILE):
    print(f"downloading {NC_FILE} (~107 megabytes)...")
    import urllib.request
    urllib.request.urlretrieve(NC_URL, NC_FILE)
    print("download complete.")
else:
    print(f"{NC_FILE} already exists.")

INDIA_GEOJSON = 'india_boundary.geojson'
INDIA_URL = 'https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson'

if not os.path.exists(INDIA_GEOJSON):
    print("downloading india boundary...")
    import urllib.request
    urllib.request.urlretrieve(INDIA_URL, INDIA_GEOJSON)
    print("done.")

# ============================================================
# 2. Load data
# ============================================================
print("loading data...")
india_gdf = gpd.read_file(INDIA_GEOJSON)
india_geom = india_gdf.union_all()
prepared_india = prep(india_geom)

ds = xr.open_dataset(NC_FILE)
time_ref = np.datetime64('2002-01-01')
time_days = ds['time'].values
time_dates = np.array([time_ref + np.timedelta64(int(d), 'D') for d in time_days])
ds = ds.assign_coords(time=time_dates)

india_data = ds['lwe_thickness'].sel(
    lat=slice(6.0, 38.0),
    lon=slice(67.0, 98.0),
    time=slice(np.datetime64('2002-04-01'), np.datetime64('2026-07-01'))
)
print(f"shape: {india_data.shape}, timesteps: {len(india_data.time)}")
print(f"time: {india_data.time.values[0]} to {india_data.time.values[-1]}")

# ============================================================
# 3. Create India boundary mask
# ============================================================
print("creating india mask (takes ~1 minute)...")
lon2d, lat2d = np.meshgrid(india_data.lon.values, india_data.lat.values)
india_mask = np.zeros_like(lon2d, dtype=bool)
for i in range(india_mask.shape[0]):
    for j in range(india_mask.shape[1]):
        india_mask[i, j] = prepared_india.contains(Point(lon2d[i, j], lat2d[i, j]))

india_masked = india_data.where(india_mask)
print(f"grid points inside india: {india_mask.sum()}")

# ============================================================
# 4. Compute pixel-wise linear trend (slope of TWSA vs time)
# ============================================================
print("computing pixel-wise trends (takes 1-2 minutes)...")

time_days_arr = (india_masked.time.values - india_masked.time.values[0]).astype('timedelta64[D]').astype(float)
time_years_arr = time_days_arr / 365.25

nlat = india_masked.shape[1]
nlon = india_masked.shape[2]
trend_map = np.full((nlat, nlon), np.nan)
pvalue_map = np.full_like(trend_map, np.nan)
data_vals = india_masked.values

for i in range(nlat):
    for j in range(nlon):
        ts = data_vals[:, i, j]
        valid_mask = ~np.isnan(ts)
        if valid_mask.sum() > 10:
            sl, _, _, pv, _ = stats.linregress(time_years_arr[valid_mask], ts[valid_mask])
            trend_map[i, j] = sl
            pvalue_map[i, j] = pv

print("done.")
print(f"mean trend: {np.nanmean(trend_map):.3f} cm/year")
print(f"strongest loss: {np.nanmin(trend_map):.3f} cm/year")
print(f"strongest gain: {np.nanmax(trend_map):.3f} cm/year")

# ============================================================
# 5. Plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 14),
                       subplot_kw={'projection': ccrs.PlateCarree()})

vmax = np.nanpercentile(np.abs(trend_map), 95)
if vmax < 0.1:
    vmax = 0.5

im = ax.pcolormesh(
    india_data.lon.values, india_data.lat.values, trend_map,
    transform=ccrs.PlateCarree(),
    cmap='RdBu',
    vmin=-vmax, vmax=vmax,
    shading='auto'
)

# india boundary
ax.add_geometries(
    india_gdf.geometry, crs=ccrs.PlateCarree(),
    facecolor='none', edgecolor='black', linewidth=1.5
)
ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
ax.set_extent([67.0, 98.0, 6.0, 38.0], crs=ccrs.PlateCarree())
ax.set_xticks(np.arange(70, 100, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(10, 40, 5), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
ax.tick_params(labelsize=9)
ax.grid(True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

# vertical colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                     shrink=0.6, aspect=25, pad=0.02)
cbar.set_label('TWSA Linear Trend (centimeters/year)', fontsize=12)

ax.set_title('Pixel-wise Linear Trend in Terrestrial Water Storage Anomaly\n'
             'over India (April 2002 – June 2026)\n'
             'Data: CSR GRACE/GRACE-FO RL06.3 Mascons',
             fontsize=13, fontweight='bold')

fig.text(0.5, 0.01,
         'Figure. Linear trend (slope of TWSA vs time) at each 0.25 degree pixel over India\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions (April 2002 – June 2026).\n'
         'Red = water storage loss over time. Blue = water storage gain. Units: centimeters per year.',
         ha='center', va='bottom', fontsize=9, fontstyle='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('india_twsa_trend_map.png', dpi=200, bbox_inches='tight')
plt.show()

sig_pct = np.sum(pvalue_map < 0.05) / np.sum(~np.isnan(pvalue_map)) * 100
print(f"\nstatistically significant pixels (p < 0.05): {sig_pct:.1f}%")

ds.close()
print("done.")
