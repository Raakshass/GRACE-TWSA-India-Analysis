"""
Figure 1 (All-India standalone) + Figure 4 (5-subplot combined)
================================================================
- Figure 1 style matches Figure 4 exactly
- Only the 11-month inter-mission gap (Jul 2017 – May 2018) is highlighted
- Different y-axis limits per subplot in Figure 4
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

print("all imports successful.")

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# STEP 1: LOAD GRACE DATA + INDIA MASK
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading GRACE data and India mask")
print("=" * 60)

NC_FILE = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'

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

mask_file = os.path.join(SAVE_DIR, 'india_mask.npy')
india_mask = np.load(mask_file)
print(f"India mask: {india_mask.sum()} pixels")

grace_masked = grace_india.where(india_mask)
lat_vals = grace_india.lat.values
lon_vals = grace_india.lon.values
times = grace_india.time.values

# ================================================================
# STEP 2: LOAD HOTSPOT POLYGONS
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Loading hotspot polygons")
print("=" * 60)

gdf_hot = gpd.read_file('all_hotspots.geojson')
print(f"Loaded {len(gdf_hot)} ROIs")

# ================================================================
# STEP 3: COMPUTE TIME SERIES
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Computing time series")
print("=" * 60)

weights = np.cos(np.deg2rad(lat_vals))
weights_2d = weights[:, np.newaxis] * np.ones((1, len(lon_vals)))

weights_india = np.where(india_mask, weights_2d, np.nan)
india_ts = np.nansum(grace_masked.values * weights_india[np.newaxis, :, :], axis=(1, 2)) / \
           np.nansum(weights_india)
print(f"All-India: {india_mask.sum()} pixels")

roi_data = []
for idx, row in gdf_hot.iterrows():
    roi_id = row['roi_id']
    poly = row.geometry
    roi_mask = np.zeros_like(india_mask, dtype=bool)
    for i in range(roi_mask.shape[0]):
        for j in range(roi_mask.shape[1]):
            if india_mask[i, j]:
                if poly.contains(Point(lon_vals[j], lat_vals[i])):
                    roi_mask[i, j] = True
    n_pixels = roi_mask.sum()
    print(f"{roi_id}: {n_pixels} pixels")
    roi_weights = np.where(roi_mask, weights_2d, np.nan)
    roi_ts = np.nansum(grace_masked.values * roi_weights[np.newaxis, :, :], axis=(1, 2)) / \
             np.nansum(roi_weights)
    roi_data.append({'roi_id': roi_id, 'ts': roi_ts})

# 11-month inter-mission gap only
GAP_START = np.datetime64('2017-07-01')
GAP_END   = np.datetime64('2018-06-01')

# ================================================================
# HELPER: plot one TWSA subplot (shared style for Fig 1 & Fig 4)
# ================================================================
def plot_twsa_panel(ax, times, ts, title, show_legend_entries=False):
    """Plot a single TWSA panel with consistent style."""
    data_color = '#2166ac'

    # --- 11-month gap highlight ---
    ax.axvspan(GAP_START, GAP_END, color='#ffcccc', alpha=0.4, zorder=0)

    # --- TWSA line ---
    ax.plot(times, ts, 'o-', color=data_color,
            linewidth=2.8, markersize=3, alpha=0.7,
            label='TWSA' if show_legend_entries else '_nolegend_')

    # --- 12-month running mean ---
    if len(ts) > 12:
        kernel = np.ones(12) / 12
        sm = np.convolve(ts, kernel, mode='valid')
        sm_t = times[5:-6]
        ax.plot(sm_t, sm, color='black', linewidth=4,
                label='12-month running mean' if show_legend_entries else '_nolegend_')

    # --- Linear trend ---
    t_num = (times - times[0]).astype('timedelta64[D]').astype(float)
    valid = ~np.isnan(ts)
    sl_yr = np.nan
    if valid.sum() > 10:
        sl, ic, r_val, p_val, _ = stats.linregress(t_num[valid], ts[valid])
        sl_yr = sl * 365.25
        ax.plot(times, ic + sl * t_num, '--', color='#1b7837', linewidth=3.5,
                label='Linear trend' if show_legend_entries else '_nolegend_')

        ax.text(0.98, 0.92, f'Trend: {sl_yr:.2f} cm/yr',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

    # --- Auto y-axis limits ---
    ts_min = np.nanmin(ts)
    ts_max = np.nanmax(ts)
    y_range = ts_max - ts_min
    y_pad = y_range * 0.12
    ax.set_ylim(ts_min - y_pad, ts_max + y_pad)

    # --- Formatting ---
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel('TWSA (cm)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)

    return sl_yr

# shared legend handles
def get_legend_handles():
    return [
        Line2D([0], [0], color='#2166ac', linewidth=2.8, marker='o', markersize=3,
               alpha=0.7, label='TWSA'),
        Line2D([0], [0], color='black', linewidth=4, label='12-month running mean'),
        Line2D([0], [0], color='#1b7837', linewidth=3.5, linestyle='--',
               label='Linear trend'),
        mpatches.Patch(color='#ffcccc', alpha=0.6,
                       label='GRACE/GRACE-FO gap (Jul 2017 – May 2018)'),
    ]

# ================================================================
# FIGURE 1: STANDALONE ALL-INDIA (same style as Figure 4)
# ================================================================
print("\n" + "=" * 60)
print("FIGURE 1: Standalone All-India TWSA")
print("=" * 60)

fig1, ax1 = plt.subplots(figsize=(16, 5))

trend_india = plot_twsa_panel(ax1, times, india_ts, 'All-India', show_legend_entries=True)

ax1.set_xlim(times[0], times[-1])
ax1.set_xlabel('Year', fontsize=12)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

fig1.legend(handles=get_legend_handles(), loc='lower center', ncol=4,
            fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig1_path = os.path.join(SAVE_DIR, 'figure1_allIndia_twsa.png')
fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {fig1_path}")

# ================================================================
# FIGURE 4: COMBINED 5-SUBPLOT (different y-axis limits)
# ================================================================
print("\n" + "=" * 60)
print("FIGURE 4: Combined 5-subplot TWSA")
print("=" * 60)

n_plots = 5
fig4, axes = plt.subplots(n_plots, 1, figsize=(16, 3.8 * n_plots), sharex=True)

subplot_labels = ['All-India', 'ROI1', 'ROI2', 'ROI3', 'ROI4']
subplot_ts = [india_ts] + [r['ts'] for r in roi_data]

for i in range(n_plots):
    trend = plot_twsa_panel(axes[i], times, subplot_ts[i], subplot_labels[i],
                            show_legend_entries=(i == 0))

axes[-1].set_xlim(times[0], times[-1])
axes[-1].set_xlabel('Year', fontsize=12)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

fig4.legend(handles=get_legend_handles(), loc='lower center', ncol=4,
            fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig4_path = os.path.join(SAVE_DIR, 'figure4_hotspot_temporal_twsa.png')
fig4.savefig(fig4_path, dpi=200, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: {fig4_path}")

ds_grace.close()

print("\n" + "=" * 60)
print("ALL DONE.")
print(f"  Figure 1: {fig1_path}")
print(f"  Figure 4: {fig4_path}")
print("=" * 60)
