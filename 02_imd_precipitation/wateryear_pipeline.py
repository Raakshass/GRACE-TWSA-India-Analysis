"""
IMD Precipitation: Complete Pipeline
=====================================
1. Download real-time daily IMD rainfall for Jan–Jun 2026
2. Merge with existing 2002–2025 monthly data
3. Create water-year (Jun–May) annual aggregation
4. Update time-series plots & correlation with GRACE
"""

import numpy as np
import xarray as xr
import imdlib as imd
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'imd_data'
RT_DIR = 'imd_data_2026'
OUTPUT_DIR = 'imd_precipitation'
os.makedirs(RT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: Download real-time daily IMD data for Jan–Jun 2026
# ============================================================
print("=" * 60)
print("STEP 1: Download real-time IMD rainfall (Jan–Jun 2026)")
print("=" * 60)

# Download month by month to avoid timeout issues
import calendar
months_2026 = [(2026, m) for m in range(1, 7)]  # Jan through Jun

for year, month in months_2026:
    days_in_month = calendar.monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{days_in_month:02d}'
    
    # Check if we already have files for this month
    test_file = os.path.join(RT_DIR, f'rain_ind0_25_{start_date}.grd')
    if os.path.exists(test_file):
        print(f"  {year}-{month:02d}: already downloaded, skipping")
        continue
    
    print(f"  Downloading {year}-{month:02d} ({start_date} to {end_date})...")
    try:
        data = imd.get_real_data('rain', start_date, end_date, file_dir=RT_DIR)
        print(f"    -> {data.data.shape[0]} days downloaded")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        print(f"    -> Will try to continue with available data")

print("\nVerifying downloaded files...")
rt_files = sorted([f for f in os.listdir(RT_DIR) if f.endswith('.grd')])
print(f"  Total real-time .grd files: {len(rt_files)}")
if rt_files:
    print(f"  First: {rt_files[0]}")
    print(f"  Last:  {rt_files[-1]}")

# ============================================================
# STEP 2: Load real-time 2026 data into xarray
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Load 2026 real-time data")
print("=" * 60)

rt_monthly_india = None

if rt_files:
    import re
    dates = []
    for f in rt_files:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', f)
        if match:
            dates.append(match.group(1))
    
    if dates:
        start_rt = min(dates)
        end_rt = max(dates)
        print(f"  Available real-time data: {start_rt} to {end_rt}")
        
        rt_data = imd.open_real_data('rain', start_rt, end_rt, file_dir=RT_DIR)
        rt_xr = rt_data.get_xarray()
        print(f"  Shape: {rt_xr['rain'].shape}")
        
        # Clean negative values
        rt_xr = rt_xr.where(rt_xr['rain'] >= 0)
        
        # Aggregate to monthly
        rt_monthly = rt_xr.resample(time='ME').sum(skipna=True)
        rt_count = rt_xr.resample(time='ME').count()
        rt_monthly = rt_monthly.where(rt_count['rain'] > 0)
        
        # Subset to India extent
        rt_monthly_india = rt_monthly.sel(
            lat=slice(6.0, 38.0), 
            lon=slice(67.0, 98.0)
        )
        print(f"  Monthly 2026 records: {len(rt_monthly_india.time)}")
        for t in rt_monthly_india.time.values:
            val = float(rt_monthly_india['rain'].sel(time=t).mean(skipna=True))
            print(f"    {str(t)[:7]}: mean={val:.1f} mm")

# ============================================================
# STEP 3: Merge with existing monthly data
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Merge 2002-2025 + 2026 monthly data")
print("=" * 60)

existing = xr.open_dataset(os.path.join(OUTPUT_DIR, 'imd_rainfall_monthly_india_2002_2025.nc'))
print(f"  Existing: {len(existing.time)} months ({str(existing.time.values[0])[:7]} to {str(existing.time.values[-1])[:7]})")

if rt_monthly_india is not None:
    # Interpolate 2026 data to match existing grid
    rt_interp = rt_monthly_india.interp(
        lat=existing.lat, 
        lon=existing.lon, 
        method='nearest'
    )
    
    # Concatenate along time
    merged = xr.concat([existing, rt_interp], dim='time')
    merged = merged.sortby('time')
    
    # Remove any duplicate times
    _, unique_idx = np.unique(merged.time.values, return_index=True)
    merged = merged.isel(time=unique_idx)
else:
    merged = existing
    print("  WARNING: No 2026 data available, using 2002-2025 only")

print(f"  Merged: {len(merged.time)} months ({str(merged.time.values[0])[:7]} to {str(merged.time.values[-1])[:7]})")

# Save merged monthly
merged_file = os.path.join(OUTPUT_DIR, 'imd_rainfall_monthly_india_2002_2026.nc')
merged.to_netcdf(merged_file)
print(f"  Saved: {merged_file}")

# ============================================================
# STEP 4: Water-year (Jun-May) annual aggregation
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Water-year (Jun-May) annual aggregation")
print("=" * 60)

def assign_water_year(ds):
    """Assign water year: Jun(6)-Dec(12) -> year, Jan(1)-May(5) -> year-1."""
    months = ds.time.dt.month.values
    years = ds.time.dt.year.values
    wy = np.where(months >= 6, years, years - 1)
    return wy

wy_labels = assign_water_year(merged)
merged_wy = merged.assign_coords(water_year=('time', wy_labels))

# Group by water year and sum
wy_groups = merged_wy.groupby('water_year')
wy_annual = wy_groups.sum(dim='time', skipna=True)

# Check completeness
print(f"\n  Water Year | Months | Status")
print(f"  -----------|--------|-------")
for wy in sorted(np.unique(wy_labels)):
    rain_wy = merged_wy.where(merged_wy.water_year == wy, drop=True)
    n_months = len(rain_wy.time)
    status = "Complete" if n_months >= 12 else f"Partial ({n_months} months)"
    months_available = rain_wy.time.dt.month.values
    has_monsoon = all(m in months_available for m in [6, 7, 8, 9])
    if not has_monsoon and n_months < 12:
        status += " (missing monsoon)"
    print(f"  WY {wy:4d}     | {n_months:6d} | {status}")

# Identify valid water years (at least 10 months of data)
valid_wys = []
for wy in sorted(np.unique(wy_labels)):
    rain_wy = merged_wy.where(merged_wy.water_year == wy, drop=True)
    if len(rain_wy.time) >= 10:
        valid_wys.append(wy)

print(f"\n  Valid water years: {valid_wys[0]}-{valid_wys[-1]} ({len(valid_wys)} years)")

# Keep only valid water years
wy_annual_valid = wy_annual.sel(water_year=valid_wys)

# Save
wy_file = os.path.join(OUTPUT_DIR, f'imd_rainfall_annual_wateryear_{valid_wys[0]}_{valid_wys[-1]}.nc')
wy_annual_valid.to_netcdf(wy_file)
print(f"  Saved: {wy_file}")

# Print summary table
print(f"\n  Water Year | All-India Mean Rainfall (mm)")
print(f"  -----------|-----------------------------")
for wy in valid_wys:
    val = float(wy_annual_valid['rain'].sel(water_year=wy).mean(skipna=True))
    print(f"  {wy}-{str(wy+1)[-2:]}    | {val:.0f}")

existing.close()
print("\n" + "=" * 60)
print("STEP 4 COMPLETE - Water-year aggregation done")
print("=" * 60)
