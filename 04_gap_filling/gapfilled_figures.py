"""
Panda & Wahr (2016, WRR) Style Figures — Using Gap-Filled GRACE Data
=====================================================================
1. Year-wise India TWSA spatial anomaly panels (Figure 3 style)
2. Spatial plots (mean + trend) for each ROI after gap-filling
3. Temporal time series for ROIs (gap-filled, continuous)

All figures use the 11-month gap-filled GRACE/GRACE-FO data.
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy import stats
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

print("All imports successful.")

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# STEP 1: LOAD GRACE DATA + RECONSTRUCT GAP + MERGE
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading GRACE data + gap-filled reconstruction")
print("=" * 60)

NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
india_gdf = gpd.read_file('india_boundary.geojson')
india_mask = np.load(os.path.join(SAVE_DIR, 'india_mask.npy'))

ds = xr.open_dataset(NC_FILE)
time_ref = np.datetime64('2002-01-01')
time_days = ds['time'].values
time_dates = np.array([time_ref + np.timedelta64(int(d), 'D') for d in time_days])
ds = ds.assign_coords(time=time_dates)

grace_india = ds['lwe_thickness'].sel(
    lat=slice(6.0, 38.0), lon=slice(67.0, 98.0),
    time=slice(np.datetime64('2002-04-01'), np.datetime64('2026-07-01'))
)
lat_vals = grace_india.lat.values
lon_vals = grace_india.lon.values
grace_times = grace_india.time.values
print(f"Original GRACE: {grace_india.shape}, {grace_times[0]} to {grace_times[-1]}")

# --- Load reconstructed gap from Zenodo Excel ---
print("\nReading reconstructed TWSA from Zenodo Excel (takes 2-5 min)...")
EXCEL_FILE = 'TWSA_Reconstruction_Locally_Adaptive_Ensemble_1980-2024_v1.xlsx'
df = pd.read_excel(EXCEL_FILE, engine='openpyxl')

# Extract India grid points
india_lats = np.arange(6.25, 38.25, 0.5)
india_lons = np.arange(67.25, 98.25, 0.5)
india_df = df[(df['lat'] >= 6.0) & (df['lat'] <= 38.0) &
              (df['lon'] >= 67.0) & (df['lon'] <= 98.0)]

gap_months = pd.date_range('2017-07', '2018-05', freq='MS')
gap_cols = [f'{d.year}-{d.month:02d}' for d in gap_months]

# Build gap grid
gap_lats = np.sort(india_df['lat'].unique())
gap_lons = np.sort(india_df['lon'].unique())
gap_data = np.full((len(gap_months), len(gap_lats), len(gap_lons)), np.nan)

for _, row in india_df.iterrows():
    lat_i = np.searchsorted(gap_lats, row['lat'])
    lon_i = np.searchsorted(gap_lons, row['lon'])
    for t, col in enumerate(gap_cols):
        if col in row.index:
            gap_data[t, lat_i, lon_i] = row[col] / 10.0  # mm → cm

# Regrid to GRACE resolution
from scipy.interpolate import RegularGridInterpolator
gap_regridded = np.full((len(gap_months), len(lat_vals), len(lon_vals)), np.nan)
for t in range(len(gap_months)):
    valid = ~np.isnan(gap_data[t])
    if valid.any():
        interp = RegularGridInterpolator(
            (gap_lats, gap_lons), gap_data[t],
            method='nearest', bounds_error=False, fill_value=np.nan
        )
        pts = np.array([(la, lo) for la in lat_vals for lo in lon_vals])
        gap_regridded[t] = interp(pts).reshape(len(lat_vals), len(lon_vals))

# Apply India mask
gap_regridded = np.where(india_mask[np.newaxis, :, :], gap_regridded, np.nan)
print(f"Gap-filled grid: {gap_regridded.shape}")

# --- Merge into continuous time series ---
gap_times_dt64 = np.array([np.datetime64(f'{d.year}-{d.month:02d}-15')
                           for d in gap_months])

# Create merged time + data arrays
all_times_list = list(grace_times) + list(gap_times_dt64)
all_data_list = list(grace_india.values) + list(gap_regridded)

# Sort by time
sort_idx = np.argsort(all_times_list)
all_times = np.array(all_times_list)[sort_idx]
all_data = np.array(all_data_list)[sort_idx]

print(f"Merged: {all_data.shape[0]} months, {all_times[0]} to {all_times[-1]}")

# Apply India mask
all_data_masked = np.where(india_mask[np.newaxis, :, :], all_data, np.nan)

# ================================================================
# STEP 2: WATER-YEAR AGGREGATION (Jun–May) on GAP-FILLED data
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Water-year aggregation on gap-filled data")
print("=" * 60)

all_months_dt = pd.DatetimeIndex(all_times)
wy_labels = np.where(all_months_dt.month >= 6, all_months_dt.year, all_months_dt.year - 1)

unique_wys = np.sort(np.unique(wy_labels))
valid_wys = []
wy_means = []

for wy in unique_wys:
    mask_wy = wy_labels == wy
    n_months = mask_wy.sum()
    if n_months >= 8:  # need at least 8 months for a valid water year
        wy_mean = np.nanmean(all_data_masked[mask_wy], axis=0)
        wy_means.append(wy_mean)
        valid_wys.append(wy)
        print(f"  WY {wy}: {n_months} months")

wy_means = np.array(wy_means)
valid_wys = np.array(valid_wys)
print(f"\nTotal valid water years: {len(valid_wys)} ({valid_wys[0]}–{valid_wys[-1]})")

# ================================================================
# FIGURE 1: YEAR-WISE TWSA SPATIAL ANOMALY PANELS (Figure 3 style)
# ================================================================
print("\n" + "=" * 60)
print("FIGURE 1: Year-wise TWSA spatial anomaly panels")
print("=" * 60)

n_panels = len(valid_wys)
ncols = 5
nrows = int(np.ceil(n_panels / ncols))

fig1, axes1 = plt.subplots(nrows, ncols, figsize=(22, nrows * 4.5),
                            subplot_kw={'projection': ccrs.PlateCarree()})

vmax_twsa = np.nanpercentile(np.abs(wy_means), 95)
if vmax_twsa < 1:
    vmax_twsa = 10

for idx in range(nrows * ncols):
    row, col = divmod(idx, ncols)
    ax = axes1[row, col] if nrows > 1 else axes1[col]

    if idx < n_panels:
        wy = valid_wys[idx]
        twsa_map = wy_means[idx]

        im = ax.pcolormesh(
            lon_vals, lat_vals, twsa_map,
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
        if row == nrows - 1:
            ax.set_xticks(np.arange(70, 100, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
        else:
            ax.set_xticks([])
    else:
        ax.set_visible(False)

cbar_ax = fig1.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig1.colorbar(im, cax=cbar_ax)
cbar.set_label('TWSA (cm of water equivalent thickness)', fontsize=12)

fig1.suptitle(
    'Spatial Evolution of GRACE-Derived TWSA (Gap-Filled)\n'
    f'(Water Year: Jun–May, {valid_wys[0]}–{valid_wys[-1]+1})',
    fontsize=16, fontweight='bold', y=1.02
)
fig1.subplots_adjust(wspace=0.05, hspace=0.15)

path1 = os.path.join(SAVE_DIR, 'fig3_twsa_spatial_evolution_gapfilled.png')
fig1.savefig(path1, dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {path1}")

# ================================================================
# STEP 3: LOAD ROI POLYGONS
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Loading ROI polygons")
print("=" * 60)

gdf_hot = gpd.read_file('all_hotspots.geojson')
print(f"Loaded {len(gdf_hot)} ROIs")

# Build ROI masks
weights_2d = np.cos(np.deg2rad(lat_vals))[:, np.newaxis] * np.ones((1, len(lon_vals)))

roi_masks = {}
for _, row in gdf_hot.iterrows():
    rid = row['roi_id']
    poly = row.geometry
    rmask = np.zeros_like(india_mask, dtype=bool)
    for i in range(rmask.shape[0]):
        for j in range(rmask.shape[1]):
            if india_mask[i, j]:
                if poly.contains(Point(lon_vals[j], lat_vals[i])):
                    rmask[i, j] = True
    roi_masks[rid] = rmask
    print(f"  {rid}: {rmask.sum()} pixels")

# ================================================================
# FIGURE 2: SPATIAL MEAN + TREND FOR EACH ROI (gap-filled data)
# ================================================================
print("\n" + "=" * 60)
print("FIGURE 2: ROI-wise spatial mean + trend maps")
print("=" * 60)

# Compute pixel-wise trend on the full monthly gap-filled data
t_num = (all_times - all_times[0]).astype('timedelta64[D]').astype(float)
t_years = t_num / 365.25

trend_map = np.full((len(lat_vals), len(lon_vals)), np.nan)
mean_map = np.nanmean(all_data_masked, axis=0)

for i in range(len(lat_vals)):
    for j in range(len(lon_vals)):
        if india_mask[i, j]:
            ts = all_data_masked[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() > 20:
                sl, _, _, _, _ = stats.linregress(t_years[valid], ts[valid])
                trend_map[i, j] = sl

print(f"  Trend range: {np.nanmin(trend_map):.2f} to {np.nanmax(trend_map):.2f} cm/yr")

# Create 2-column figure per ROI (mean TWSA + trend)
for _, row in gdf_hot.iterrows():
    rid = row['roi_id']
    rtype = row['type']
    thresh = row['threshold']
    rmask = roi_masks[rid]
    poly = row.geometry
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    pad = 2.0
    extent = [bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad]
    # Clamp to India
    extent = [max(extent[0], 67), min(extent[1], 98),
              max(extent[2], 6), min(extent[3], 38)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                    subplot_kw={'projection': ccrs.PlateCarree()})

    for ax in [ax1, ax2]:
        ax.add_geometries(india_gdf.geometry, crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='gray', linewidth=0.5)
        ax.add_geometries([poly], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='red', linewidth=2.5)
        ax.coastlines(resolution='50m', linewidth=0.3, color='gray')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.grid(True, linewidth=0.3, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=9)

    # (a) Mean TWSA
    roi_mean = np.where(rmask, mean_map, np.nan)
    vabs = max(abs(np.nanmin(roi_mean)), abs(np.nanmax(roi_mean)))
    im1 = ax1.pcolormesh(lon_vals, lat_vals, roi_mean,
                          transform=ccrs.PlateCarree(),
                          cmap='RdBu', shading='auto',
                          vmin=-vabs, vmax=vabs)
    cb1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', shrink=0.8, pad=0.08)
    cb1.set_label('Mean TWSA (cm)', fontsize=10)
    ax1.set_title(f'{rid} — Mean TWSA\n(Apr 2002 – Jun 2026, gap-filled)',
                  fontsize=12, fontweight='bold')

    # (b) TWSA Trend
    roi_trend = np.where(rmask, trend_map, np.nan)
    vabs_t = max(abs(np.nanmin(roi_trend)), abs(np.nanmax(roi_trend)))
    im2 = ax2.pcolormesh(lon_vals, lat_vals, roi_trend,
                          transform=ccrs.PlateCarree(),
                          cmap='RdBu', shading='auto',
                          vmin=-vabs_t, vmax=vabs_t)
    cb2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', shrink=0.8, pad=0.08)
    cb2.set_label('TWSA Trend (cm/yr)', fontsize=10)

    # Compute mean trend value within ROI
    mean_trend = np.nanmean(roi_trend)
    ax2.set_title(f'{rid} — TWSA Trend (mean: {mean_trend:.2f} cm/yr)\n'
                  f'(Apr 2002 – Jun 2026, gap-filled)',
                  fontsize=12, fontweight='bold')

    fig.suptitle(f'{rid}: {rtype.capitalize()} Hotspot (threshold = {thresh:+.1f} cm/yr)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    roi_path = os.path.join(SAVE_DIR, f'roi_spatial_{rid.lower()}_gapfilled.png')
    fig.savefig(roi_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {roi_path}")

# ================================================================
# FIGURE 3: TEMPORAL TIME SERIES FOR ALL ROIS (gap-filled)
# ================================================================
print("\n" + "=" * 60)
print("FIGURE 3: ROI temporal time series (gap-filled)")
print("=" * 60)

GAP_START = np.datetime64('2017-07-01')
GAP_END = np.datetime64('2018-06-01')

# Compute All-India time series
weights_india = np.where(india_mask, weights_2d, np.nan)
india_ts = np.nansum(all_data_masked * weights_india[np.newaxis, :, :], axis=(1, 2)) / \
           np.nansum(weights_india)

# Compute ROI time series
roi_ts_dict = {}
for rid, rmask in roi_masks.items():
    roi_weights = np.where(rmask, weights_2d, np.nan)
    roi_ts = np.nansum(all_data_masked * roi_weights[np.newaxis, :, :], axis=(1, 2)) / \
             np.nansum(roi_weights)
    roi_ts_dict[rid] = roi_ts

def plot_twsa_panel(ax, times, ts, title, show_legend=False):
    """Plot a TWSA panel with consistent style."""
    ax.axvspan(GAP_START, GAP_END, color='#ffcccc', alpha=0.4, zorder=0)

    ax.plot(times, ts, 'o-', color='#2166ac',
            linewidth=2.8, markersize=3, alpha=0.7)

    if len(ts) > 12:
        kernel = np.ones(12) / 12
        sm = np.convolve(ts, kernel, mode='valid')
        sm_t = times[5:-6]
        ax.plot(sm_t, sm, color='black', linewidth=4, zorder=5)

    t_num_ts = (times - times[0]).astype('timedelta64[D]').astype(float)
    valid = ~np.isnan(ts)
    if valid.sum() > 10:
        sl, ic, _, _, _ = stats.linregress(t_num_ts[valid], ts[valid])
        sl_yr = sl * 365.25
        ax.plot(times, ic + sl * t_num_ts, '--', color='#1b7837', linewidth=3.5, zorder=4)
        ax.text(0.98, 0.92, f'Trend: {sl_yr:.2f} cm/yr',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

    ts_min, ts_max = np.nanmin(ts), np.nanmax(ts)
    y_pad = (ts_max - ts_min) * 0.12
    ax.set_ylim(ts_min - y_pad, ts_max + y_pad)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel('TWSA (cm)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)

# Combined 5-subplot figure
n_plots = 5
fig3, axes = plt.subplots(n_plots, 1, figsize=(16, 3.8 * n_plots), sharex=True)

labels = ['All-India', 'ROI1', 'ROI2', 'ROI3', 'ROI4']
ts_list = [india_ts] + [roi_ts_dict[f'ROI{i}'] for i in range(1, 5)]

for i in range(n_plots):
    plot_twsa_panel(axes[i], all_times, ts_list[i], labels[i])

axes[-1].set_xlim(all_times[0], all_times[-1])
axes[-1].set_xlabel('Year', fontsize=12)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

legend_handles = [
    Line2D([0], [0], color='#2166ac', linewidth=2.8, marker='o', markersize=3,
           alpha=0.7, label='GRACE/GRACE-FO TWSA (gap-filled)'),
    Line2D([0], [0], color='black', linewidth=4, label='12-month running mean'),
    Line2D([0], [0], color='#1b7837', linewidth=3.5, linestyle='--',
           label='Linear trend'),
    mpatches.Patch(color='#ffcccc', alpha=0.6,
                   label='GRACE/GRACE-FO gap (Jul 2017 – May 2018)'),
]
fig3.legend(handles=legend_handles, loc='lower center', ncol=4,
            fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 1])
path3 = os.path.join(SAVE_DIR, 'figure_roi_temporal_gapfilled.png')
fig3.savefig(path3, dpi=200, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {path3}")

# Also save standalone All-India
fig_ai, ax_ai = plt.subplots(figsize=(16, 5))
plot_twsa_panel(ax_ai, all_times, india_ts, 'All-India')
ax_ai.set_xlim(all_times[0], all_times[-1])
ax_ai.set_xlabel('Year', fontsize=12)
ax_ai.xaxis.set_major_locator(mdates.YearLocator(2))
ax_ai.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax_ai.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig_ai.legend(handles=legend_handles, loc='lower center', ncol=4,
              fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.06, 1, 1])
path_ai = os.path.join(SAVE_DIR, 'figure_allIndia_twsa_gapfilled.png')
fig_ai.savefig(path_ai, dpi=200, bbox_inches='tight')
plt.close(fig_ai)
print(f"  Saved: {path_ai}")

ds.close()

print("\n" + "=" * 60)
print("ALL DONE. Gap-filled figures saved in:", SAVE_DIR)
print("=" * 60)
