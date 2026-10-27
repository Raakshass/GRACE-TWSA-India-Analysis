# ============================================================
# LULC MAP FOR 4 HOTSPOT REGIONS — ESA WorldCover 2021 v200
# ============================================================
# Paste this cell in Google Colab AFTER the hotspot extraction cell.
# It uses the `all_hotspots.geojson` file saved earlier.
#
# Source: ESA WorldCover 2021 v200 (10m resolution)
# https://viewer.esa-worldcover.org/worldcover/
# ============================================================

# --- Install dependencies (Colab) ---
import subprocess, sys
for pkg in ['rasterio']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
import geopandas as gpd
from shapely.geometry import mapping
import os, urllib.request, warnings, gc
warnings.filterwarnings('ignore')

print("all imports successful.")

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)
TILE_DIR = os.path.join(SAVE_DIR, 'worldcover_tiles')
os.makedirs(TILE_DIR, exist_ok=True)

# ── ESA WorldCover 2021 v200 official classes & colors ──
LULC_CLASSES = {
    10: ('Tree cover',              '#006400'),
    20: ('Shrubland',               '#ffbb22'),
    30: ('Grassland',               '#ffff4c'),
    40: ('Cropland',                '#f096ff'),
    50: ('Built-up',                '#fa0000'),
    60: ('Bare / sparse vegetation','#b4b4b4'),
    70: ('Snow and ice',            '#f0f0f0'),
    80: ('Permanent water bodies',  '#0064c8'),
    90: ('Herbaceous wetland',      '#0096a0'),
    95: ('Mangroves',               '#00cf75'),
    100:('Moss and lichen',         '#fae6a0'),
}

# build colormap
class_ids = sorted(LULC_CLASSES.keys())
colors_list = [LULC_CLASSES[c][1] for c in class_ids]
boundaries = [class_ids[0] - 5]
for i, c in enumerate(class_ids):
    if i < len(class_ids) - 1:
        boundaries.append((c + class_ids[i+1]) / 2)
    else:
        boundaries.append(c + 5)
cmap = ListedColormap(colors_list)
norm = BoundaryNorm(boundaries, cmap.N)

# ── Load hotspot polygons ──
print("\nLoading ROI polygons...")
# Try multiple locations
for geojson_path in ['all_hotspots.geojson',
                     os.path.join(SAVE_DIR, 'all_hotspots.geojson')]:
    if os.path.exists(geojson_path):
        gdf_hot = gpd.read_file(geojson_path)
        break
print(f"Loaded {len(gdf_hot)} ROIs")

# ── Tile helper ──
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

def get_tiles_for_bounds(minlon, minlat, maxlon, maxlat):
    tiles = []
    lat_start = int(np.floor(minlat / 3) * 3)
    lat_end = int(np.floor(maxlat / 3) * 3)
    lon_start = int(np.floor(minlon / 3) * 3)
    lon_end = int(np.floor(maxlon / 3) * 3)
    for lat in range(lat_start, lat_end + 3, 3):
        for lon in range(lon_start, lon_end + 3, 3):
            lat_p = 'N' if lat >= 0 else 'S'
            lon_p = 'E' if lon >= 0 else 'W'
            name = f"ESA_WorldCover_10m_2021_v200_{lat_p}{abs(lat):02d}{lon_p}{abs(lon):03d}_Map.tif"
            tiles.append(name)
    return tiles

def download_tile(tile_name):
    tile_path = os.path.join(TILE_DIR, tile_name)
    if os.path.exists(tile_path):
        return tile_path
    url = f"{BASE_URL}/{tile_name}"
    print(f"    downloading {tile_name}...", end='', flush=True)
    try:
        urllib.request.urlretrieve(url, tile_path)
        size_mb = os.path.getsize(tile_path) / 1e6
        print(f" done ({size_mb:.0f} MB)")
        return tile_path
    except Exception as e:
        print(f" FAILED: {e}")
        if os.path.exists(tile_path):
            os.remove(tile_path)
        return None

# ================================================================
# GENERATE LULC MAP FOR EACH ROI (one at a time to save memory)
# ================================================================
roi_images = []  # store (roi_id, image_path)

for idx, row in gdf_hot.iterrows():
    roi_id = row['roi_id']
    roi_type = row['type']
    roi_geom = row.geometry
    bounds = roi_geom.bounds  # (minlon, minlat, maxlon, maxlat)

    print(f"\n{'=' * 50}")
    print(f"Processing {roi_id} ({roi_type})")
    print(f"{'=' * 50}")

    # 1. determine and download tiles
    tiles_needed = get_tiles_for_bounds(*bounds)
    print(f"  tiles needed: {len(tiles_needed)}")
    tile_paths = []
    for tn in tiles_needed:
        tp = download_tile(tn)
        if tp:
            tile_paths.append(tp)

    if not tile_paths:
        print(f"  ERROR: no tiles for {roi_id}")
        continue

    # 2. open tiles and merge
    print(f"  merging {len(tile_paths)} tile(s)...")
    datasets = [rasterio.open(tp) for tp in tile_paths]

    if len(datasets) == 1:
        src = datasets[0]
    else:
        mosaic, mosaic_transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': mosaic_transform,
            'driver': 'GTiff',
        })
        memfile = MemoryFile()
        src = memfile.open(**profile)
        src.write(mosaic)
        del mosaic
        gc.collect()

    # 3. clip to ROI polygon
    print(f"  clipping to {roi_id} polygon...")
    clipped, clipped_transform = rio_mask(
        src, [mapping(roi_geom)], crop=True, nodata=0, filled=True
    )
    lulc = clipped[0]
    print(f"  clipped shape: {lulc.shape}, pixels: {(lulc > 0).sum():,}")

    # close all datasets
    for ds in datasets:
        ds.close()

    # 4. compute extent
    nrows, ncols = lulc.shape
    extent = [
        clipped_transform.c,
        clipped_transform.c + clipped_transform.a * ncols,
        clipped_transform.f + clipped_transform.e * nrows,
        clipped_transform.f
    ]

    # 5. plot individual QGIS-style map (NO axes, NO ticks — clean map only)
    lulc_masked = np.ma.masked_where(lulc == 0, lulc)

    # compute figure aspect ratio from data
    data_width = extent[1] - extent[0]
    data_height = extent[3] - extent[2]
    aspect = data_width / data_height
    fig_height = 10
    fig_width = max(fig_height * aspect, 8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(lulc_masked, extent=extent, origin='upper',
              cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')

    # ROI boundary (thick black)
    if roi_geom.geom_type == 'Polygon':
        xs, ys = roi_geom.exterior.xy
        ax.plot(xs, ys, 'k-', linewidth=2.5)
    elif roi_geom.geom_type == 'MultiPolygon':
        for poly in roi_geom.geoms:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, 'k-', linewidth=2.5)

    # QGIS style: remove ALL axis decorations — pure map
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # title above the map (bold, like QGIS layout)
    ax.set_title(roi_id, fontsize=18, fontweight='bold', pad=10)

    plt.tight_layout()
    img_path = os.path.join(SAVE_DIR, f'{roi_id.lower()}_lulc_map.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    roi_images.append((roi_id, img_path))
    print(f"  saved: {img_path}")

    # free memory
    del lulc, lulc_masked, clipped
    gc.collect()

# ================================================================
# CREATE COMBINED 2×2 FIGURE WITH LEGEND COMPLETELY BELOW
# ================================================================
print(f"\n{'=' * 50}")
print("Creating combined 2×2 figure with legend below")
print(f"{'=' * 50}")

from PIL import Image

# load individual images
imgs = []
for roi_id, path in roi_images:
    img = Image.open(path)
    imgs.append((roi_id, img))
    print(f"  {roi_id}: {img.size[0]}x{img.size[1]} px")

# create 2x2 composite using gridspec
# height_ratios: big space for maps, small space for legend
fig = plt.figure(figsize=(24, 28))
gs = gridspec.GridSpec(3, 2, height_ratios=[10, 10, 1.2],
                       hspace=0.08, wspace=0.05)

for i, (roi_id, img) in enumerate(imgs):
    r, c = divmod(i, 2)
    ax = fig.add_subplot(gs[r, c])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

# legend in the bottom row — COMPLETELY OUTSIDE all maps
ax_legend = fig.add_subplot(gs[2, :])
ax_legend.set_axis_off()

legend_patches = []
for class_id in class_ids:
    name, color = LULC_CLASSES[class_id]
    legend_patches.append(
        mpatches.Patch(facecolor=color, edgecolor='black',
                       linewidth=1, label=name)
    )

ax_legend.legend(handles=legend_patches, loc='center', ncol=6,
                 fontsize=14, frameon=True, framealpha=1,
                 edgecolor='black', fancybox=False,
                 handlelength=2, handleheight=1.5,
                 columnspacing=1.5, labelspacing=0.8)

fig.suptitle('Land Use / Land Cover of Hotspot Regions\n'
             'ESA WorldCover 2021 v200 (10m resolution)',
             fontsize=22, fontweight='bold', y=0.99)

combined_path = os.path.join(SAVE_DIR, 'hotspot_lulc_maps_combined.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

for _, img in imgs:
    img.close()

print(f"\nCombined figure saved: {combined_path}")
print("\nIndividual maps saved:")
for roi_id, path in roi_images:
    print(f"  {roi_id}: {path}")
print("\n" + "=" * 50)
print("ALL DONE!")
print("=" * 50)
