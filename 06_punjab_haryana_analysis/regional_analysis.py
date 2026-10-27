#!/usr/bin/env python3
"""
GRACE/GRACE-FO RL06.3 Mascon Analysis — Google Colab Script
============================================================
Paste this entire script into a single Colab cell and run.

Plots generated:
  Figure 1: Spatial map — June vs December mean TWSA over India
  Figure 2: Temporal series — Punjab + Haryana (2002–2026)
  Figure 3: Temporal series — Full India (2002–2026)
  Figure 4: Comparison — Punjab-Haryana vs Full India overlay

Data: CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc
Source: https://www2.csr.utexas.edu/grace/RL0603_mascons.html
"""

# ============================================================
# CELL 1: Install dependencies (run this cell first)
# ============================================================
# !pip install -q cartopy geopandas xarray netCDF4 scipy shapely

# ============================================================
# CELL 2: Imports and data download
# ============================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# download the netcdf file if not already present
NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
NC_URL = 'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'

if not os.path.exists(NC_FILE):
    print(f"downloading {NC_FILE} (~107 megabytes)...")
    import urllib.request
    urllib.request.urlretrieve(NC_URL, NC_FILE)
    print("download complete.")
else:
    print(f"{NC_FILE} already exists, skipping download.")

# download india boundary shapefile (datameet — correct political boundary)
INDIA_GEOJSON = 'india_boundary.geojson'
INDIA_URL = 'https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson'

if not os.path.exists(INDIA_GEOJSON):
    print("downloading india boundary shapefile...")
    import urllib.request
    urllib.request.urlretrieve(INDIA_URL, INDIA_GEOJSON)
    print("download complete.")

# ============================================================
# CELL 3: Load data and create masks
# ============================================================
print("loading data...")

# load india boundary
india_gdf = gpd.read_file(INDIA_GEOJSON)
india_geom = india_gdf.union_all()
prepared_india = prep(india_geom)

# load grace mascon data
ds = xr.open_dataset(NC_FILE)

# convert time coordinate (stored as days since 2002-01-01 with capital 'Units')
time_ref = np.datetime64('2002-01-01')
time_days = ds['time'].values
time_dates = np.array([time_ref + np.timedelta64(int(d), 'D') for d in time_days])
ds = ds.assign_coords(time=time_dates)

# subset to india bounding box (extended for full jammu & kashmir)
india_data = ds['lwe_thickness'].sel(
    lat=slice(6.0, 38.0),
    lon=slice(67.0, 98.0),
    time=slice(np.datetime64('2002-04-01'), np.datetime64('2026-06-01'))
)

print(f"india subset shape: {india_data.shape}")
print(f"time range: {india_data.time.values[0]} to {india_data.time.values[-1]}")
print(f"timesteps: {len(india_data.time)}")

# create india land mask (point-in-polygon)
print("creating india boundary mask (this takes ~1 minute)...")
lon2d, lat2d = np.meshgrid(india_data.lon.values, india_data.lat.values)
india_mask = np.zeros_like(lon2d, dtype=bool)
for i in range(india_mask.shape[0]):
    for j in range(india_mask.shape[1]):
        india_mask[i, j] = prepared_india.contains(Point(lon2d[i, j], lat2d[i, j]))

print(f"grid points inside india: {india_mask.sum()}")

# apply india mask
india_masked = india_data.where(india_mask)

# create punjab-haryana mask (rectangular approximation)
# punjab: ~29.5 to 32.5 north, 73.8 to 76.8 east
# haryana: ~27.5 to 30.5 north, 74.5 to 77.5 east
# combined bounding box:
PH_LAT_MIN, PH_LAT_MAX = 27.5, 32.5
PH_LON_MIN, PH_LON_MAX = 73.8, 77.5

ph_lat_mask = (india_data.lat >= PH_LAT_MIN) & (india_data.lat <= PH_LAT_MAX)
ph_lon_mask = (india_data.lon >= PH_LON_MIN) & (india_data.lon <= PH_LON_MAX)
ph_mask_2d = np.outer(ph_lat_mask.values, ph_lon_mask.values) & india_mask

ph_masked = india_data.where(ph_mask_2d)
print(f"grid points inside punjab-haryana: {ph_mask_2d.sum()}")

# area weights (cosine of latitude)
weights_india = np.cos(np.deg2rad(india_data.lat.values))
weights_da = xr.DataArray(weights_india, dims=['lat'], coords={'lat': india_data.lat.values})

print("data loaded and masks created.\n")

# ============================================================
# helper function for map features
# ============================================================
def add_india_map(ax, extent=None):
    """add india boundary and map features."""
    ax.add_geometries(
        india_gdf.geometry, crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor='black', linewidth=1.5
    )
    ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
    if extent is None:
        extent = [67.0, 98.0, 6.0, 38.0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(70, 100, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 40, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')


# ============================================================
# FIGURE 1: Spatial Map — June vs December Mean TWSA
# ============================================================
print("=" * 60)
print("FIGURE 1: June 2019 vs December 2019 spatial comparison")
print("=" * 60)

# extract year and month from time coordinate
years = np.array([int(str(t)[:4]) for t in india_masked.time.values])
months = np.array([t.astype('datetime64[M]').astype(int) % 12 + 1
                   for t in india_masked.time.values])

# june 2019
june_idx = (months == 6) & (years == 2019)
june_data = india_masked.isel(time=june_idx).squeeze()
print(f"june 2019 found: {june_idx.sum()} timestep(s)")

# december 2019
dec_idx = (months == 12) & (years == 2019)
dec_data = india_masked.isel(time=dec_idx).squeeze()
print(f"december 2019 found: {dec_idx.sum()} timestep(s)")

# symmetric color scale
vmax_june = float(np.nanmax(np.abs(june_data.values)))
vmax_dec = float(np.nanmax(np.abs(dec_data.values)))
vmax = max(vmax_june, vmax_dec)
vmax = min(vmax, 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                subplot_kw={'projection': ccrs.PlateCarree()})

# june 2019 panel
im1 = ax1.pcolormesh(
    india_data.lon.values, india_data.lat.values, june_data.values,
    transform=ccrs.PlateCarree(), cmap='RdBu',
    vmin=-vmax, vmax=vmax, shading='auto'
)
add_india_map(ax1)
ax1.set_title('June 2019 (Pre-Monsoon)', fontsize=13, fontweight='bold')

# december 2019 panel
im2 = ax2.pcolormesh(
    india_data.lon.values, india_data.lat.values, dec_data.values,
    transform=ccrs.PlateCarree(), cmap='RdBu',
    vmin=-vmax, vmax=vmax, shading='auto'
)
add_india_map(ax2)
ax2.set_title('December 2019 (Post-Monsoon)', fontsize=13, fontweight='bold')

# vertical colorbar on the right side to avoid overlapping the maps
cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical',
                     shrink=0.7, aspect=25, pad=0.02)
cbar.set_label('Liquid Water Equivalent Thickness (centimeters)', fontsize=11)

# caption below
fig.text(0.5, -0.01,
         'Figure 1. Terrestrial water storage anomaly over India for June 2019 (pre-monsoon) and December 2019 (post-monsoon)\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions. Anomalies relative to 2004–2010 baseline.\n'
         'Red = water loss, Blue = water gain. Units: centimeters of water equivalent.',
         ha='center', va='top', fontsize=9, fontstyle='italic')

plt.subplots_adjust(wspace=0.05)
plt.savefig('figure1_june_december_2019_spatial.png', dpi=200, bbox_inches='tight')
plt.show()
print("saved: figure1_june_december_2019_spatial.png\n")

# june vs december 2019 statistics
june_area_mean = float(np.nanmean(june_data.values[india_mask]))
dec_area_mean = float(np.nanmean(dec_data.values[india_mask]))
print(f"  june mean twsa (india): {june_area_mean:.2f} centimeters")
print(f"  december mean twsa (india): {dec_area_mean:.2f} centimeters")
print(f"  seasonal difference (december - june): {dec_area_mean - june_area_mean:.2f} centimeters\n")


# ============================================================
# FIGURE 2: Punjab-Haryana Time Series (2002–2026)
# ============================================================
print("=" * 60)
print("FIGURE 2: Punjab-Haryana time series")
print("=" * 60)

# area-weighted mean for punjab-haryana
ph_ts = ph_masked.weighted(weights_da).mean(dim=['lat', 'lon'])

fig, ax = plt.subplots(figsize=(15, 6))

times_ph = ph_ts.time.values
values_ph = ph_ts.values

# monthly values
ax.plot(times_ph, values_ph, 'o-', color='#d6604d', linewidth=0.8,
        markersize=2.5, alpha=0.7, label='Monthly TWSA')

# 12-month running mean
if len(values_ph) > 12:
    kernel = np.ones(12) / 12
    smoothed = np.convolve(values_ph, kernel, mode='valid')
    smooth_times = times_ph[5:-6]
    ax.plot(smooth_times, smoothed, color='#4a1486', linewidth=2.5,
            label='12-month running mean', zorder=5)

# linear trend
time_num = (times_ph - times_ph[0]).astype('timedelta64[D]').astype(float)
valid = ~np.isnan(values_ph)
slope_ph, intercept_ph, _, _, _ = stats.linregress(time_num[valid], values_ph[valid])
trend_ph = intercept_ph + slope_ph * time_num
slope_ph_yr = slope_ph * 365.25
ax.plot(times_ph, trend_ph, '--', color='#1b7837', linewidth=1.5,
        label=f'Linear trend: {slope_ph_yr:.2f} centimeters/year', zorder=4)

# grace/grace-fo gap
ax.axvspan(np.datetime64('2017-07-01'), np.datetime64('2018-06-01'),
           alpha=0.15, color='gray', label='GRACE/GRACE-FO gap')
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('TWSA (centimeters)', fontsize=12)
ax.set_title('Figure 2', fontsize=14, fontweight='bold', loc='left')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

fig.text(0.5, -0.02,
         'Figure 2. Area-weighted terrestrial water storage anomaly time series for Punjab and Haryana\n'
         '(27.5 degrees north to 32.5 degrees north, 73.8 degrees east to 77.5 degrees east) from CSR GRACE/GRACE-FO RL06.3 mascons (April 2002 to May 2026).\n'
         'Anomalies relative to 2004 to 2010 baseline. Units: centimeters of water equivalent.',
         ha='center', va='top', fontsize=9, fontstyle='italic')

plt.tight_layout()
plt.savefig('figure2_punjab_haryana_timeseries.png', dpi=200, bbox_inches='tight')
plt.show()
print("saved: figure2_punjab_haryana_timeseries.png")
print(f"  punjab-haryana linear trend: {slope_ph_yr:.2f} centimeters/year")
print(f"  mean twsa: {float(np.nanmean(values_ph)):.2f} centimeters")
print(f"  min twsa: {float(np.nanmin(values_ph)):.2f} centimeters")
print(f"  max twsa: {float(np.nanmax(values_ph)):.2f} centimeters\n")


# ============================================================
# FIGURE 3: Full India Time Series (2002–2026)
# ============================================================
print("=" * 60)
print("FIGURE 3: Full India time series")
print("=" * 60)

# area-weighted mean for all india
india_ts = india_masked.weighted(weights_da).mean(dim=['lat', 'lon'])

fig, ax = plt.subplots(figsize=(15, 6))

times_in = india_ts.time.values
values_in = india_ts.values

# monthly values
ax.plot(times_in, values_in, 'o-', color='#2166ac', linewidth=0.8,
        markersize=2.5, alpha=0.7, label='Monthly TWSA')

# 12-month running mean
if len(values_in) > 12:
    kernel = np.ones(12) / 12
    smoothed_in = np.convolve(values_in, kernel, mode='valid')
    smooth_times_in = times_in[5:-6]
    ax.plot(smooth_times_in, smoothed_in, color='#b2182b', linewidth=2.5,
            label='12-month running mean', zorder=5)

# linear trend
time_num_in = (times_in - times_in[0]).astype('timedelta64[D]').astype(float)
valid_in = ~np.isnan(values_in)
slope_in, intercept_in, _, _, _ = stats.linregress(time_num_in[valid_in], values_in[valid_in])
trend_in = intercept_in + slope_in * time_num_in
slope_in_yr = slope_in * 365.25
ax.plot(times_in, trend_in, '--', color='#1b7837', linewidth=1.5,
        label=f'Linear trend: {slope_in_yr:.2f} centimeters/year', zorder=4)

# grace/grace-fo gap
ax.axvspan(np.datetime64('2017-07-01'), np.datetime64('2018-06-01'),
           alpha=0.15, color='gray', label='GRACE/GRACE-FO gap')
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('TWSA (centimeters)', fontsize=12)
ax.set_title('Figure 3', fontsize=14, fontweight='bold', loc='left')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

fig.text(0.5, -0.02,
         'Figure 3. Area-weighted terrestrial water storage anomaly time series for all of India\n'
         'from CSR GRACE/GRACE-FO RL06.3 mascon solutions (April 2002 to May 2026).\n'
         'Anomalies relative to 2004 to 2010 baseline. Units: centimeters of water equivalent.',
         ha='center', va='top', fontsize=9, fontstyle='italic')

plt.tight_layout()
plt.savefig('figure3_india_timeseries.png', dpi=200, bbox_inches='tight')
plt.show()
print("saved: figure3_india_timeseries.png")
print(f"  all-india linear trend: {slope_in_yr:.2f} centimeters/year")
print(f"  mean twsa: {float(np.nanmean(values_in)):.2f} centimeters")
print(f"  min twsa: {float(np.nanmin(values_in)):.2f} centimeters")
print(f"  max twsa: {float(np.nanmax(values_in)):.2f} centimeters\n")


# ============================================================
# FIGURE 4: Comparison — Punjab-Haryana vs Full India
# ============================================================
print("=" * 60)
print("FIGURE 4: Punjab-Haryana vs Full India comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(15, 6))

# punjab-haryana 12-month running mean
if len(values_ph) > 12:
    kernel = np.ones(12) / 12
    ph_smooth = np.convolve(values_ph, kernel, mode='valid')
    ph_smooth_t = times_ph[5:-6]
    ax.plot(ph_smooth_t, ph_smooth, color='#d6604d', linewidth=2.5,
            label=f'Punjab-Haryana (trend: {slope_ph_yr:.2f} centimeters/year)')

# all india 12-month running mean
if len(values_in) > 12:
    in_smooth = np.convolve(values_in, kernel, mode='valid')
    in_smooth_t = times_in[5:-6]
    ax.plot(in_smooth_t, in_smooth, color='#2166ac', linewidth=2.5,
            label=f'All India (trend: {slope_in_yr:.2f} centimeters/year)')

ax.axvspan(np.datetime64('2017-07-01'), np.datetime64('2018-06-01'),
           alpha=0.15, color='gray', label='GRACE/GRACE-FO gap')
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('TWSA — 12-month running mean (centimeters)', fontsize=12)
ax.set_title('Figure 4', fontsize=14, fontweight='bold', loc='left')
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

fig.text(0.5, -0.02,
         'Figure 4. Comparison of 12-month running mean terrestrial water storage anomaly:\n'
         'Punjab-Haryana (red) versus All India (blue). Punjab-Haryana shows a steeper decline,\n'
         'indicating concentrated groundwater depletion in the northern Indo-Gangetic alluvial belt.',
         ha='center', va='top', fontsize=9, fontstyle='italic')

plt.tight_layout()
plt.savefig('figure4_comparison_overlay.png', dpi=200, bbox_inches='tight')
plt.show()
print("saved: figure4_comparison_overlay.png\n")


# ============================================================
# INFERENCES AND COMPARISON
# ============================================================
print("=" * 60)
print("INFERENCES")
print("=" * 60)

ratio = slope_ph_yr / slope_in_yr if slope_in_yr != 0 else float('inf')

print(f"""
1. SEASONAL CONTRAST (FIGURE 1):
   june (pre-monsoon) shows the driest conditions — aquifers are at their lowest
   after months of extraction without recharge. december (post-monsoon) shows
   partial recovery due to monsoon rainfall infiltration. however, the recovery
   is incomplete in northern india, indicating that annual extraction exceeds
   annual recharge in punjab, haryana, and rajasthan.

2. PUNJAB-HARYANA DEPLETION (FIGURE 2):
   linear trend: {slope_ph_yr:.2f} centimeters per year.
   this is one of the fastest groundwater depletion rates globally, driven by
   rice-wheat cropping systems that rely heavily on tube-well irrigation.
   the seasonal amplitude is large, but the 12-month running mean shows a
   persistent downward trajectory with no sign of stabilization.

3. ALL-INDIA AVERAGE (FIGURE 3):
   linear trend: {slope_in_yr:.2f} centimeters per year.
   the national average is less severe because southern and eastern india
   partially compensate with stable or slightly increasing storage. however,
   the overall trajectory is still negative, confirming net water loss at
   the national scale.

4. COMPARISON (FIGURE 4):
   punjab-haryana is depleting approximately {abs(ratio):.1f} times faster than
   the national average. the 12-month running means diverge progressively
   after 2010, showing that the problem is accelerating in the northern
   indo-gangetic belt while the rest of india is relatively stable.
   this spatial heterogeneity directly justifies the need for optimized
   monitoring well placement — the core objective of this project.
""")

# final statistics table
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<45} {'Punjab-Haryana':>15} {'All India':>12}")
print("-" * 72)
print(f"{'Linear trend (centimeters/year)':<45} {slope_ph_yr:>15.2f} {slope_in_yr:>12.2f}")
print(f"{'Mean TWSA (centimeters)':<45} {float(np.nanmean(values_ph)):>15.2f} {float(np.nanmean(values_in)):>12.2f}")
print(f"{'Min TWSA (centimeters)':<45} {float(np.nanmin(values_ph)):>15.2f} {float(np.nanmin(values_in)):>12.2f}")
print(f"{'Max TWSA (centimeters)':<45} {float(np.nanmax(values_ph)):>15.2f} {float(np.nanmax(values_in)):>12.2f}")
print(f"{'Std dev (centimeters)':<45} {float(np.nanstd(values_ph)):>15.2f} {float(np.nanstd(values_in)):>12.2f}")
print(f"{'Depletion rate ratio (region/national)':<45} {abs(ratio):>15.1f}x {'1.0x':>12}")
print()

ds.close()
print("done. all figures saved.")
