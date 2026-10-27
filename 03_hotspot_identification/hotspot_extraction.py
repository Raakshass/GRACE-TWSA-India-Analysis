"""
Hotspot Region Extraction using Contour-Based Shapefile Generation
===================================================================
Paste this cell AFTER the trend map cell in your Colab notebook.
It uses the `trend_map` and `india_mask` variables already computed.

Method: Contour-Based Shapefile Generation
Reference: Li et al. (2019), Xie et al. (2025), Zhang et al. (2021)
as described in Section 2.1.2 of
https://www.sciencedirect.com/science/article/pii/S0301479725044317

Steps:
  1. Extract contour lines from the trend map at multiple thresholds
  2. Filter for closed contour polygons
  3. Convert to shapefiles (GeoJSON)
  4. Plot hotspot regions on the map
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import os

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# STEP 1: Identify appropriate contour thresholds for India
# ================================================================
print("=" * 60)
print("HOTSPOT EXTRACTION: Contour-Based Shapefile Generation")
print("=" * 60)

# fill NaN pixels with 0 for smooth contouring (NaN breaks contour finder)
trend_filled = np.where(np.isnan(trend_map), 0, trend_map)

# print trend statistics to help pick thresholds
print(f"\ntrend map statistics (india pixels only):")
india_trends = trend_map[~np.isnan(trend_map)]
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {pct}th percentile: {np.percentile(india_trends, pct):.2f} cm/year")

# ================================================================
# STEP 2: Extract closed contour polygons
# ================================================================
# define thresholds for hotspot identification
# negative thresholds = water loss hotspots
# positive thresholds = water gain hotspots
# we try multiple thresholds and keep the largest closed contour for each

NEGATIVE_THRESHOLDS = [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
POSITIVE_THRESHOLDS = [0.3, 0.5, 0.7, 1.0]

lon_vals = india_data.lon.values
lat_vals = india_data.lat.values

def extract_contour_polygons(trend_grid, lon, lat, threshold, min_area_deg2=2.0):
    """
    extract closed contour polygons at a given threshold.

    for negative thresholds: find regions where trend < threshold
    for positive thresholds: find regions where trend > threshold

    returns list of shapely Polygons with area > min_area_deg2 square degrees.
    """
    fig_temp, ax_temp = plt.subplots()

    if threshold < 0:
        # for negative: contour at threshold, regions below it are hotspots
        cs = ax_temp.contour(lon, lat, trend_grid, levels=[threshold])
    else:
        # for positive: contour at threshold, regions above it are hotspots
        cs = ax_temp.contour(lon, lat, trend_grid, levels=[threshold])

    plt.close(fig_temp)

    polygons = []
    if len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            if len(seg) < 4:
                continue
            # check if contour is closed (first point ≈ last point)
            dist = np.sqrt((seg[0, 0] - seg[-1, 0])**2 + (seg[0, 1] - seg[-1, 1])**2)
            if dist < 1.0:  # close enough to be considered closed (within 1 degree)
                try:
                    poly = Polygon(seg)
                    if poly.is_valid and poly.area >= min_area_deg2:
                        polygons.append(poly)
                except Exception:
                    continue

    return polygons


# extract polygons for all thresholds
print("\nextracting contour polygons...")
all_hotspots = []

for thresh in NEGATIVE_THRESHOLDS:
    polys = extract_contour_polygons(trend_filled, lon_vals, lat_vals, thresh)
    for p in polys:
        # verify the enclosed region actually has trends below the threshold
        # (contour direction matters)
        centroid = p.centroid
        ci = np.argmin(np.abs(lat_vals - centroid.y))
        cj = np.argmin(np.abs(lon_vals - centroid.x))
        center_trend = trend_map[ci, cj] if not np.isnan(trend_map[ci, cj]) else 0

        if center_trend < thresh:
            all_hotspots.append({
                'polygon': p,
                'threshold': thresh,
                'type': 'depletion',
                'center_trend': center_trend,
                'area_deg2': p.area,
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x
            })

for thresh in POSITIVE_THRESHOLDS:
    polys = extract_contour_polygons(trend_filled, lon_vals, lat_vals, thresh)
    for p in polys:
        centroid = p.centroid
        ci = np.argmin(np.abs(lat_vals - centroid.y))
        cj = np.argmin(np.abs(lon_vals - centroid.x))
        center_trend = trend_map[ci, cj] if not np.isnan(trend_map[ci, cj]) else 0

        if center_trend > thresh:
            all_hotspots.append({
                'polygon': p,
                'threshold': thresh,
                'type': 'gain',
                'center_trend': center_trend,
                'area_deg2': p.area,
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x
            })

print(f"total candidate hotspot polygons found: {len(all_hotspots)}")

# ================================================================
# STEP 3: Select major hotspot regions (largest polygon per zone)
# ================================================================
# group hotspots by approximate geographic location (cluster nearby polygons)
# then pick the most extreme (largest area at tightest threshold) for each cluster

from scipy.spatial.distance import cdist

if len(all_hotspots) > 0:
    centroids = np.array([[h['centroid_lat'], h['centroid_lon']] for h in all_hotspots])

    # simple clustering: merge polygons whose centroids are within 5 degrees
    visited = [False] * len(all_hotspots)
    clusters = []

    for idx in range(len(all_hotspots)):
        if visited[idx]:
            continue
        cluster = [idx]
        visited[idx] = True
        for jdx in range(idx + 1, len(all_hotspots)):
            if visited[jdx]:
                continue
            if all_hotspots[idx]['type'] != all_hotspots[jdx]['type']:
                continue
            dist = np.sqrt((centroids[idx, 0] - centroids[jdx, 0])**2 +
                           (centroids[idx, 1] - centroids[jdx, 1])**2)
            if dist < 5.0:
                cluster.append(jdx)
                visited[jdx] = True
        clusters.append(cluster)

    # for each cluster, pick the polygon with the most extreme threshold
    # (tightest contour that still forms a closed polygon = most extreme hotspot)
    selected_hotspots = []
    for cluster in clusters:
        cluster_entries = [all_hotspots[i] for i in cluster]
        if cluster_entries[0]['type'] == 'depletion':
            # pick the most negative threshold
            best = min(cluster_entries, key=lambda x: x['threshold'])
        else:
            # pick the most positive threshold
            best = max(cluster_entries, key=lambda x: x['threshold'])
        selected_hotspots.append(best)

    # sort by absolute trend (most extreme first)
    selected_hotspots.sort(key=lambda x: abs(x['center_trend']), reverse=True)

    print(f"\nselected {len(selected_hotspots)} distinct hotspot regions:")
    for idx, h in enumerate(selected_hotspots):
        print(f"  ROI{idx+1}: {h['type']}, threshold={h['threshold']:.1f} cm/yr, "
              f"center_trend={h['center_trend']:.2f} cm/yr, "
              f"area={h['area_deg2']:.1f} deg², "
              f"centroid=({h['centroid_lat']:.1f}°N, {h['centroid_lon']:.1f}°E)")
else:
    selected_hotspots = []
    print("no hotspot polygons found. check trend map thresholds.")


# ================================================================
# STEP 4: Save hotspot shapefiles
# ================================================================
print("\nsaving hotspot shapefiles...")

for idx, h in enumerate(selected_hotspots):
    roi_name = f"ROI{idx+1}_{h['type']}"
    gdf = gpd.GeoDataFrame(
        {
            'roi_id': [f'ROI{idx+1}'],
            'type': [h['type']],
            'threshold_cm_yr': [h['threshold']],
            'center_trend_cm_yr': [round(h['center_trend'], 2)],
            'area_deg2': [round(h['area_deg2'], 1)],
            'centroid_lat': [round(h['centroid_lat'], 2)],
            'centroid_lon': [round(h['centroid_lon'], 2)],
        },
        geometry=[h['polygon']],
        crs='EPSG:4326'
    )

    # save as geojson
    outpath = os.path.join(SAVE_DIR, f'{roi_name}.geojson')
    gdf.to_file(outpath, driver='GeoJSON')
    print(f"  saved: {outpath}")

    # also save as shapefile
    shp_dir = os.path.join(SAVE_DIR, f'{roi_name}_shp')
    os.makedirs(shp_dir, exist_ok=True)
    gdf.to_file(os.path.join(shp_dir, f'{roi_name}.shp'))
    print(f"  saved: {shp_dir}/{roi_name}.shp")

# save all hotspots as a single combined file
if selected_hotspots:
    all_gdf = gpd.GeoDataFrame(
        {
            'roi_id': [f'ROI{i+1}' for i in range(len(selected_hotspots))],
            'type': [h['type'] for h in selected_hotspots],
            'threshold': [h['threshold'] for h in selected_hotspots],
            'center_trend': [round(h['center_trend'], 2) for h in selected_hotspots],
        },
        geometry=[h['polygon'] for h in selected_hotspots],
        crs='EPSG:4326'
    )
    all_gdf.to_file(os.path.join(SAVE_DIR, 'all_hotspots.geojson'), driver='GeoJSON')
    print(f"\n  combined file: {SAVE_DIR}/all_hotspots.geojson")


# ================================================================
# STEP 5: Plot hotspot regions on the trend map
# ================================================================
print("\nplotting hotspot regions on trend map...")

fig, ax = plt.subplots(1, 1, figsize=(14, 16),
                       subplot_kw={'projection': ccrs.PlateCarree()})

vmax = np.nanpercentile(np.abs(trend_map), 95)

im = ax.pcolormesh(
    lon_vals, lat_vals, trend_map,
    transform=ccrs.PlateCarree(), cmap='RdBu',
    vmin=-vmax, vmax=vmax, shading='auto'
)

# india boundary
ax.add_geometries(india_gdf.geometry, crs=ccrs.PlateCarree(),
                  facecolor='none', edgecolor='black', linewidth=1.5)
ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
ax.set_extent([67.0, 98.0, 6.0, 38.0], crs=ccrs.PlateCarree())
ax.set_xticks(np.arange(70, 100, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(10, 40, 5), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°E'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f°N'))
ax.tick_params(labelsize=9)
ax.grid(True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

# overlay hotspot polygons
colors_depletion = ['#ff0000', '#ff4444', '#ff8888', '#ffaaaa']
colors_gain = ['#0000ff', '#4444ff', '#8888ff', '#aaaaff']
legend_patches = []

for idx, h in enumerate(selected_hotspots):
    if h['type'] == 'depletion':
        color = colors_depletion[min(idx, len(colors_depletion)-1)]
        linestyle = '-'
    else:
        color = colors_gain[min(idx, len(colors_gain)-1)]
        linestyle = '--'

    ax.add_geometries(
        [h['polygon']], crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor=color, linewidth=3, linestyle=linestyle
    )

    # label the ROI
    cx, cy = h['centroid_lon'], h['centroid_lat']
    ax.text(cx, cy, f'ROI{idx+1}', transform=ccrs.PlateCarree(),
            fontsize=11, fontweight='bold', color=color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))

    legend_patches.append(
        mpatches.Patch(facecolor='none', edgecolor=color, linewidth=2,
                       label=f'ROI{idx+1}: {h["type"]} ({h["threshold"]:.1f} cm/yr)')
    )

ax.legend(handles=legend_patches, loc='lower left', fontsize=10, framealpha=0.9)

cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, aspect=25, pad=0.02)
cbar.set_label('TWSA Linear Trend (centimeters/year)', fontsize=12)

ax.set_title('Hotspot Regions Identified via Contour-Based Shapefile Generation\n'
             'TWSA Trend Map (April 2002 – June 2026)\n'
             'Data: CSR GRACE/GRACE-FO RL06.3 Mascons',
             fontsize=13, fontweight='bold')

fig.text(0.5, 0.01,
         'Hotspot regions extracted using closed contour polygons at specified trend thresholds.\n'
         'Depletion hotspots (red contours) and gain hotspots (blue dashed contours).\n'
         'Method: Contour-based shapefile generation (Li et al., 2019; Xie et al., 2025).',
         ha='center', va='bottom', fontsize=9, fontstyle='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(SAVE_DIR, 'india_hotspot_regions.png'), dpi=200, bbox_inches='tight')
plt.show()


# ================================================================
# STEP 6: Temporal TWSA for each hotspot region
# ================================================================
print("\n" + "=" * 60)
print("Temporal TWSA for each hotspot region")
print("=" * 60)

fig, axes = plt.subplots(len(selected_hotspots), 1,
                         figsize=(16, 4 * len(selected_hotspots)), sharex=True)
if len(selected_hotspots) == 1:
    axes = [axes]

for idx, h in enumerate(selected_hotspots):
    ax = axes[idx]
    poly = h['polygon']

    # create mask for this ROI
    roi_mask = np.zeros_like(india_mask, dtype=bool)
    for i in range(roi_mask.shape[0]):
        for j in range(roi_mask.shape[1]):
            if india_mask[i, j]:
                pt = Point(lon_vals[j], lat_vals[i])
                if poly.contains(pt):
                    roi_mask[i, j] = True

    n_roi_pixels = roi_mask.sum()
    print(f"ROI{idx+1}: {n_roi_pixels} pixels inside polygon")

    if n_roi_pixels == 0:
        ax.text(0.5, 0.5, 'No pixels in ROI', transform=ax.transAxes, ha='center')
        continue

    # compute area-weighted mean for this ROI
    weights_roi = np.cos(np.deg2rad(lat_vals))
    roi_ts = []
    for t in range(india_masked.shape[0]):
        vals = india_masked.values[t][roi_mask]
        wts = weights_roi[:, np.newaxis] * np.ones((1, len(lon_vals)))
        wts_roi = wts[roi_mask]
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            roi_ts.append(np.average(vals[valid], weights=wts_roi[valid]))
        else:
            roi_ts.append(np.nan)

    roi_ts = np.array(roi_ts)
    times = india_masked.time.values

    # plot
    color = '#d6604d' if h['type'] == 'depletion' else '#2166ac'
    ax.plot(times, roi_ts, 'o-', color=color, linewidth=0.8, markersize=2, alpha=0.7)

    # 12-month running mean
    if len(roi_ts) > 12:
        kernel = np.ones(12) / 12
        sm = np.convolve(roi_ts, kernel, mode='valid')
        sm_t = times[5:-6]
        ax.plot(sm_t, sm, color='black', linewidth=2, label='12-month mean')

    # trend
    t_num = (times - times[0]).astype('timedelta64[D]').astype(float)
    valid = ~np.isnan(roi_ts)
    if valid.sum() > 10:
        sl, ic, _, _, _ = stats.linregress(t_num[valid], roi_ts[valid])
        sl_yr = sl * 365.25
        ax.plot(times, ic + sl * t_num, '--', color='green', linewidth=1.5,
                label=f'Trend: {sl_yr:.2f} cm/yr')

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel('TWSA (cm)', fontsize=10)
    ax.set_title(f'ROI{idx+1} — {h["type"].capitalize()} hotspot '
                 f'(threshold: {h["threshold"]:.1f} cm/yr, {n_roi_pixels} pixels)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Year', fontsize=11)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'hotspot_temporal_twsa.png'), dpi=200, bbox_inches='tight')
plt.show()

print("\ndone. all hotspot shapefiles and plots saved in:", SAVE_DIR)
