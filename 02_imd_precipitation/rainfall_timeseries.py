"""
IMD Precipitation Analysis — Water Year Edition
=================================================
1. Annual rainfall time series (bar chart) by water year (Jun-May)
2. ROI-wise rainfall statistics
3. Correlation between IMD rainfall and GRACE TWSA (water-year means)
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = 'imd_precipitation'
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

# Water-year annual rainfall
wy_files = sorted([f for f in os.listdir(SAVE_DIR) if 'wateryear' in f and f.endswith('.nc')])
wy_file = os.path.join(SAVE_DIR, wy_files[-1])
imd_annual = xr.open_dataset(wy_file)
water_years = imd_annual.water_year.values
print(f"IMD water-year annual: {len(water_years)} years (WY {water_years[0]}-{water_years[-1]})")

# Monthly data (for reference)
monthly_file = os.path.join(SAVE_DIR, 'imd_rainfall_monthly_india_2002_2026.nc')
imd_monthly = xr.open_dataset(monthly_file)
print(f"IMD monthly: {len(imd_monthly.time)} records")

# GRACE
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

# India mask & ROIs
india_mask = np.load('gap_fill_results/india_mask.npy')
gdf_hot = gpd.read_file('all_hotspots.geojson')
print(f"Loaded {len(gdf_hot)} ROIs")

# ============================================================
# BUILD ROI MASKS
# ============================================================
print("\n" + "=" * 60)
print("Building ROI masks...")
print("=" * 60)

imd_lat = imd_annual.lat.values
imd_lon = imd_annual.lon.values
grace_lat = grace_india.lat.values
grace_lon = grace_india.lon.values

# IMD India mask
sample = imd_annual['rain'].isel(water_year=0).values
imd_india_mask = ~np.isnan(sample) & (sample >= 0)
print(f"IMD India pixels: {imd_india_mask.sum()}")

# IMD ROI masks
imd_roi_masks = {}
for _, row in gdf_hot.iterrows():
    roi_id = row['roi_id']
    poly = row.geometry
    mask = np.zeros((len(imd_lat), len(imd_lon)), dtype=bool)
    for i in range(len(imd_lat)):
        for j in range(len(imd_lon)):
            if imd_india_mask[i, j] and poly.contains(Point(imd_lon[j], imd_lat[i])):
                mask[i, j] = True
    imd_roi_masks[roi_id] = mask
    print(f"  {roi_id}: {mask.sum()} IMD pixels")

# GRACE ROI masks
grace_weights = np.cos(np.deg2rad(grace_lat))
grace_weights_2d = grace_weights[:, np.newaxis] * np.ones((1, len(grace_lon)))
grace_masked = grace_india.where(india_mask)

grace_roi_masks = {}
for _, row in gdf_hot.iterrows():
    roi_id = row['roi_id']
    poly = row.geometry
    mask = np.zeros_like(india_mask, dtype=bool)
    for i in range(len(grace_lat)):
        for j in range(len(grace_lon)):
            if india_mask[i, j] and poly.contains(Point(grace_lon[j], grace_lat[i])):
                mask[i, j] = True
    grace_roi_masks[roi_id] = mask

# ============================================================
# COMPUTE TIME SERIES
# ============================================================
print("\n" + "=" * 60)
print("Computing time series...")
print("=" * 60)

# IMD water-year annual rainfall (area-weighted)
def imd_annual_ts(mask):
    weights = np.cos(np.deg2rad(imd_lat))[:, np.newaxis] * np.ones((1, len(imd_lon)))
    w = np.where(mask, weights, np.nan)
    rain = imd_annual['rain'].values  # (n_wy, nlat, nlon)
    ts = np.nansum(rain * w[np.newaxis, :, :], axis=(1, 2)) / np.nansum(w)
    return ts

india_rain_annual = imd_annual_ts(imd_india_mask)
roi_rain_annual = {}
for roi_id, mask in imd_roi_masks.items():
    roi_rain_annual[roi_id] = imd_annual_ts(mask)

print(f"Water years: {water_years[0]}-{water_years[-1]}")

# GRACE water-year TWSA (area-weighted)
grace_times = grace_india.time.values
grace_months = pd.DatetimeIndex(grace_times)
grace_wy = np.where(grace_months.month >= 6, grace_months.year, grace_months.year - 1)

grace_india_weights = np.where(india_mask, grace_weights_2d, np.nan)
india_twsa = np.nansum(grace_masked.values * grace_india_weights[np.newaxis, :, :], axis=(1, 2)) / \
             np.nansum(grace_india_weights)

# GRACE annual by water year
grace_df = pd.DataFrame({'time': grace_times, 'twsa': india_twsa, 'wy': grace_wy})
grace_annual = grace_df.groupby('wy')['twsa'].mean()

roi_twsa = {}
for roi_id, mask in grace_roi_masks.items():
    w = np.where(mask, grace_weights_2d, np.nan)
    roi_twsa[roi_id] = np.nansum(grace_masked.values * w[np.newaxis, :, :], axis=(1, 2)) / np.nansum(w)

roi_grace_annual = {}
for roi_id in roi_twsa:
    df = pd.DataFrame({'time': grace_times, 'twsa': roi_twsa[roi_id], 'wy': grace_wy})
    roi_grace_annual[roi_id] = df.groupby('wy')['twsa'].mean()

# ============================================================
# FIGURE 1: ANNUAL RAINFALL TIME SERIES (water year)
# ============================================================
print("\n" + "=" * 60)
print("Plotting annual rainfall time series (water year)...")
print("=" * 60)

labels = ['All-India', 'ROI1', 'ROI2', 'ROI3', 'ROI4']
rain_series = [india_rain_annual] + [roi_rain_annual[f'ROI{i}'] for i in range(1, 5)]
wy_labels_str = [f'{y}\n–{str(y+1)[-2:]}' for y in water_years]

fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)

for i, (ax, label, rain) in enumerate(zip(axes, labels, rain_series)):
    colors = ['#e74c3c' if r < np.mean(rain) else '#3498db' for r in rain]
    ax.bar(range(len(water_years)), rain, color=colors, alpha=0.8, width=0.7,
           edgecolor='white', linewidth=0.5)

    mean_rain = np.mean(rain)
    ax.axhline(y=mean_rain, color='black', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'Mean: {mean_rain:.0f} mm')

    sl, ic, r, p, se = stats.linregress(np.arange(len(water_years)), rain)
    trend_line = ic + sl * np.arange(len(water_years))
    ax.plot(range(len(water_years)), trend_line, color='#2ecc71', linewidth=2.5,
            label=f'Trend: {sl:.1f} mm/yr (p={p:.3f})')

    ax.set_ylabel('Rainfall (mm)', fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold', loc='left')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    min_idx = np.argmin(rain)
    max_idx = np.argmax(rain)
    ax.annotate(f'{rain[min_idx]:.0f}', xy=(min_idx, rain[min_idx]),
                ha='center', va='top', fontsize=8, fontweight='bold', color='#c0392b')
    ax.annotate(f'{rain[max_idx]:.0f}', xy=(max_idx, rain[max_idx]),
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2980b9')

axes[-1].set_xlabel('Water Year (Jun–May)', fontsize=12)
axes[-1].set_xticks(range(len(water_years)))
axes[-1].set_xticklabels(wy_labels_str, rotation=0, ha='center', fontsize=8)

plt.suptitle(
    f'IMD Annual Rainfall by Water Year (Jun–May)\n'
    f'Study Period: April 2002 – June 2026',
    fontsize=16, fontweight='bold', y=1.01
)
plt.tight_layout()
fig1_path = os.path.join(SAVE_DIR, 'annual_rainfall_timeseries_wateryear.png')
fig.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig1_path}")

# ============================================================
# FIGURE 2: RAINFALL-TWSA CORRELATION (water year)
# ============================================================
print("\n" + "=" * 60)
print("Computing rainfall-TWSA correlation (water year)...")
print("=" * 60)

fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.5))

all_rain_annual = [india_rain_annual] + [roi_rain_annual[f'ROI{i}'] for i in range(1, 5)]
all_grace_series = [grace_annual] + [roi_grace_annual[f'ROI{i}'] for i in range(1, 5)]

for i, (ax, label) in enumerate(zip(axes2, labels)):
    common_wys = sorted(set(water_years) & set(all_grace_series[i].index))
    rain_vals = [all_rain_annual[i][list(water_years).index(y)] for y in common_wys]
    twsa_vals = [all_grace_series[i][y] for y in common_wys]

    ax.scatter(rain_vals, twsa_vals, c='#3498db', s=50, alpha=0.8,
               edgecolors='white', linewidth=0.5)

    for yr, rx, tx in zip(common_wys, rain_vals, twsa_vals):
        ax.annotate(f'{yr}', (rx, tx), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', alpha=0.7)

    sl, ic, r_val, p_val, se = stats.linregress(rain_vals, twsa_vals)
    x_fit = np.linspace(min(rain_vals), max(rain_vals), 100)
    ax.plot(x_fit, ic + sl * x_fit, color='#e74c3c', linewidth=2)

    ax.set_title(f'{label}\nr={r_val:.2f}, p={p_val:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Rainfall (mm)', fontsize=10)
    if i == 0:
        ax.set_ylabel('TWSA (cm)', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Annual IMD Rainfall vs GRACE TWSA Correlation (Water Year, Jun–May)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(SAVE_DIR, 'rainfall_twsa_correlation_wateryear.png')
fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {fig2_path}")

# ============================================================
# PRINT STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("ROI-wise Rainfall Statistics (Water Year)")
print("=" * 60)

print(f"\n{'Region':12s} | {'Mean (mm)':>10s} | {'Std (mm)':>10s} | {'Min (mm)':>10s} | {'Max (mm)':>10s} | {'Trend (mm/yr)':>14s} | {'p-value':>8s}")
print("-" * 90)

for label, rain in zip(labels, all_rain_annual):
    sl, ic, r, p, se = stats.linregress(np.arange(len(water_years)), rain)
    print(f"{label:12s} | {np.mean(rain):10.0f} | {np.std(rain):10.0f} | {np.min(rain):10.0f} | {np.max(rain):10.0f} | {sl:+14.1f} | {p:8.4f}")

print("\n" + "=" * 60)
print("Rainfall-TWSA Correlation (Water Year)")
print("=" * 60)

print(f"\n{'Region':12s} | {'Pearson r':>10s} | {'p-value':>10s} | {'Interpretation':30s}")
print("-" * 75)

for i, label in enumerate(labels):
    common_wys = sorted(set(water_years) & set(all_grace_series[i].index))
    rain_vals = [all_rain_annual[i][list(water_years).index(y)] for y in common_wys]
    twsa_vals = [all_grace_series[i][y] for y in common_wys]
    r_val, p_val = stats.pearsonr(rain_vals, twsa_vals)

    if abs(r_val) > 0.7:
        interp = "Strong"
    elif abs(r_val) > 0.4:
        interp = "Moderate"
    else:
        interp = "Weak"
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"{label:12s} | {r_val:+10.3f} | {p_val:10.4f} | {interp} ({sig})")

ds_grace.close()
imd_annual.close()
imd_monthly.close()

print("\n" + "=" * 60)
print("ALL DONE!")
print(f"  Figure 1: {fig1_path}")
print(f"  Figure 2: {fig2_path}")
print("=" * 60)
