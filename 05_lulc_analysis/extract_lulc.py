"""
Extract ESA WorldCover 2021 v200 (10m) Land Cover for 4 Hotspot ROIs
=====================================================================
Reads Cloud Optimized GeoTIFFs from ESA's S3 bucket, clips to each ROI,
and computes land cover class statistics (% area).

WorldCover Classes:
  10 = Tree cover       20 = Shrubland         30 = Grassland
  40 = Cropland         50 = Built-up          60 = Bare/sparse veg
  70 = Snow and ice     80 = Permanent water   90 = Herbaceous wetland
  95 = Mangroves       100 = Moss and lichen
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from shapely.geometry import box, mapping
import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

print("all imports successful.")

SAVE_DIR = 'gap_fill_results'
os.makedirs(SAVE_DIR, exist_ok=True)
TILE_DIR = os.path.join(SAVE_DIR, 'worldcover_tiles')
os.makedirs(TILE_DIR, exist_ok=True)

# ESA WorldCover class definitions
LULC_CLASSES = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen',
}

LULC_COLORS = {
    10: '#006400', 20: '#ffbb22', 30: '#ffff4c', 40: '#f096ff',
    50: '#fa0000', 60: '#b4b4b4', 70: '#f0f0f0', 80: '#0064c8',
    90: '#0096a0', 95: '#00cf75', 100: '#fae6a0',
}

# ================================================================
# 1. Load hotspot polygons
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading hotspot ROI polygons")
print("=" * 60)

gdf_hot = gpd.read_file('/Users/siddhantjain/btp/all_hotspots.geojson')
print(f"Loaded {len(gdf_hot)} ROIs")
for _, row in gdf_hot.iterrows():
    bounds = row.geometry.bounds
    print(f"  {row['roi_id']}: {row['type']}, bounds=({bounds[0]:.1f}E, {bounds[1]:.1f}N) to ({bounds[2]:.1f}E, {bounds[3]:.1f}N)")

# ================================================================
# 2. Determine which WorldCover tiles we need
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Identifying required WorldCover tiles")
print("=" * 60)

BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

def get_tile_name(lat_sw, lon_sw):
    """Generate ESA WorldCover tile filename from SW corner (3°×3° grid)."""
    lat_prefix = 'N' if lat_sw >= 0 else 'S'
    lon_prefix = 'E' if lon_sw >= 0 else 'W'
    return f"ESA_WorldCover_10m_2021_v200_{lat_prefix}{abs(lat_sw):02d}{lon_prefix}{abs(lon_sw):03d}_Map.tif"

def get_tiles_for_bounds(minlon, minlat, maxlon, maxlat):
    """Get all tile names covering a bounding box."""
    tiles = []
    # tiles are on a 3° grid, SW corner aligned to multiples of 3
    lat_start = int(np.floor(minlat / 3) * 3)
    lat_end = int(np.floor(maxlat / 3) * 3)
    lon_start = int(np.floor(minlon / 3) * 3)
    lon_end = int(np.floor(maxlon / 3) * 3)

    for lat in range(lat_start, lat_end + 3, 3):
        for lon in range(lon_start, lon_end + 3, 3):
            tiles.append(get_tile_name(lat, lon))
    return tiles

# collect all needed tiles across all ROIs
all_tiles = set()
roi_tiles = {}
for _, row in gdf_hot.iterrows():
    bounds = row.geometry.bounds  # (minlon, minlat, maxlon, maxlat)
    tiles = get_tiles_for_bounds(*bounds)
    roi_tiles[row['roi_id']] = tiles
    all_tiles.update(tiles)
    print(f"  {row['roi_id']}: needs {len(tiles)} tile(s): {', '.join(tiles)}")

print(f"\nTotal unique tiles to download: {len(all_tiles)}")

# ================================================================
# 3. Download tiles
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Downloading WorldCover tiles")
print("=" * 60)

for tile_name in sorted(all_tiles):
    tile_path = os.path.join(TILE_DIR, tile_name)
    if os.path.exists(tile_path):
        size_mb = os.path.getsize(tile_path) / 1e6
        print(f"  exists: {tile_name} ({size_mb:.1f} MB)")
        continue

    url = f"{BASE_URL}/{tile_name}"
    print(f"  downloading: {tile_name}...", end='', flush=True)
    try:
        urllib.request.urlretrieve(url, tile_path)
        size_mb = os.path.getsize(tile_path) / 1e6
        print(f" done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f" FAILED: {e}")
        # remove partial file
        if os.path.exists(tile_path):
            os.remove(tile_path)

# ================================================================
# 4. Extract land cover statistics for each ROI
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: Extracting land cover statistics for each ROI")
print("=" * 60)

results = []

for _, row in gdf_hot.iterrows():
    roi_id = row['roi_id']
    roi_type = row['type']
    roi_geom = row.geometry
    tiles = roi_tiles[roi_id]

    print(f"\n--- {roi_id} ({roi_type}) ---")

    # open all tiles for this ROI
    tile_datasets = []
    for tile_name in tiles:
        tile_path = os.path.join(TILE_DIR, tile_name)
        if os.path.exists(tile_path):
            tile_datasets.append(rasterio.open(tile_path))

    if not tile_datasets:
        print(f"  ERROR: no tiles available for {roi_id}")
        continue

    # if multiple tiles, merge them first
    if len(tile_datasets) == 1:
        src = tile_datasets[0]
    else:
        # merge tiles into a virtual mosaic
        from rasterio.io import MemoryFile
        mosaic, mosaic_transform = merge(tile_datasets)
        # create an in-memory raster
        profile = tile_datasets[0].profile.copy()
        profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': mosaic_transform,
        })
        memfile = MemoryFile()
        src = memfile.open(**profile)
        src.write(mosaic)

    # clip to ROI polygon
    try:
        clipped, clipped_transform = rio_mask(
            src, [mapping(roi_geom)], crop=True, nodata=0, filled=True
        )
        lulc_data = clipped[0]  # single band
    except Exception as e:
        print(f"  ERROR clipping: {e}")
        for ds in tile_datasets:
            ds.close()
        continue

    # close datasets
    for ds in tile_datasets:
        ds.close()

    # compute statistics
    valid_pixels = lulc_data[lulc_data > 0]
    total_pixels = len(valid_pixels)
    print(f"  total pixels: {total_pixels:,} (10m resolution)")

    class_stats = {}
    for class_id, class_name in LULC_CLASSES.items():
        count = np.sum(valid_pixels == class_id)
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        if pct > 0.01:  # skip negligible classes
            class_stats[class_id] = {
                'name': class_name,
                'count': count,
                'pct': pct
            }

    # sort by percentage (descending)
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['pct'], reverse=True)

    print(f"  land cover breakdown:")
    for class_id, info in sorted_classes:
        print(f"    {info['name']:<30} {info['pct']:6.2f}%  ({info['count']:>12,} pixels)")

    results.append({
        'roi_id': roi_id,
        'type': roi_type,
        'total_pixels': total_pixels,
        'classes': sorted_classes,
    })

# ================================================================
# 5. Print formatted table for Google Doc
# ================================================================
print("\n\n" + "=" * 80)
print("TABLE FOR GOOGLE DOC")
print("=" * 80)

for r in results:
    print(f"\n{r['roi_id']} ({r['type']}):")
    print(f"  {'Land Cover Class':<30} {'Percentage':>10}")
    print(f"  {'─'*30} {'─'*10}")
    for class_id, info in r['classes']:
        if info['pct'] >= 0.1:
            print(f"  {info['name']:<30} {info['pct']:>9.2f}%")

print("\n\n" + "=" * 80)
print("COMPACT TABLE (all ROIs)")
print("=" * 80)

# get all unique classes across ROIs
all_classes_present = set()
for r in results:
    for class_id, info in r['classes']:
        if info['pct'] >= 0.5:
            all_classes_present.add(class_id)

all_classes_sorted = sorted(all_classes_present)

header = f"{'Land Cover Class':<30}"
for r in results:
    header += f" {r['roi_id']:>8}"
print(f"\n{header}")
print(f"{'─'*30}" + " " + "─"*8 * len(results))

for class_id in all_classes_sorted:
    row_str = f"{LULC_CLASSES[class_id]:<30}"
    for r in results:
        found = False
        for cid, info in r['classes']:
            if cid == class_id:
                row_str += f" {info['pct']:>7.1f}%"
                found = True
                break
        if not found:
            row_str += f" {'—':>8}"
    print(row_str)

print("\n\nDone! All land cover statistics extracted.")
