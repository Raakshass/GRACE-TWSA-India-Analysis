"""
Fast parallel IMD rainfall download + aggregation
===================================================
Downloads all years in parallel (8 threads), then aggregates.
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'imd_data'
OUTPUT_DIR = 'imd_precipitation'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2002
END_YEAR = 2026

# ============================================================
# STEP 1: Parallel download
# ============================================================
print("=" * 60)
print(f"STEP 1: Parallel download of IMD rainfall ({START_YEAR}–{END_YEAR})")
print("=" * 60)

from concurrent.futures import ThreadPoolExecutor, as_completed
from imdlib.util import ARCHIVE_URLS, ARCHIVE_GRID, GRIDS, save_download, post, LeapYear

url = ARCHIVE_URLS['rain'][0]

def download_year(year):
    """Download one year of IMD rainfall data."""
    fname = os.path.join(DATA_DIR, f'Rainfall_ind{year}_rfp25.grd')
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        size_mb = os.path.getsize(fname) / 1024 / 1024
        return year, f"already exists ({size_mb:.1f} MB)"
    
    try:
        data = {'__EVENTTARGET': '', '__EVENTARGUMENT': '', 
                'ctl00$ContentPlaceHolder1$txtYear': str(year),
                'ctl00$ContentPlaceHolder1$btnDowload': 'Download'}
        response = post(url, data, f"rain {year}")
        
        days_in_year = 366 if LeapYear(int(year)) else 365
        expected_size = GRIDS[ARCHIVE_GRID['rain']].file_size(days_in_year)
        
        save_download(response.content, fname, expected_size, f"rain {year}",
                       empty_msg=f"rain {year} not published yet by IMD.")
        
        size_mb = os.path.getsize(fname) / 1024 / 1024
        return year, f"downloaded ({size_mb:.1f} MB)"
    except Exception as e:
        return year, f"FAILED: {e}"

# Check existing
existing = []
to_download = []
for yr in range(START_YEAR, END_YEAR + 1):
    fname = os.path.join(DATA_DIR, f'Rainfall_ind{yr}_rfp25.grd')
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        existing.append(yr)
    else:
        to_download.append(yr)

print(f"  Already have: {len(existing)} years {existing}")
print(f"  To download:  {len(to_download)} years")

if to_download:
    print(f"\n  Downloading with 8 parallel threads...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_year, yr): yr for yr in to_download}
        results = []
        for future in as_completed(futures):
            yr, status = future.result()
            results.append((yr, status))
            print(f"    {yr}: {status}")
        # Print sorted summary
        results.sort()

# Verify
print("\n  Verification:")
all_ok = True
for yr in range(START_YEAR, END_YEAR + 1):
    fname = os.path.join(DATA_DIR, f'Rainfall_ind{yr}_rfp25.grd')
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        size_mb = os.path.getsize(fname) / 1024 / 1024
        print(f"    {yr}: {size_mb:.1f} MB ✓")
    else:
        print(f"    {yr}: MISSING ✗")
        all_ok = False

# ============================================================
# STEP 2: Load + aggregate
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading into xarray")
print("=" * 60)

import imdlib as imd
import xarray as xr

# open_data expects files in file_dir with standard naming
data = imd.open_data('rain', START_YEAR, END_YEAR, DATA_DIR)
print(f"  Shape: {data.data.shape}")

ds = data.get_xarray()

# Clean
ds_clean = ds.where(ds['rain'] >= 0)
ds_clean = ds_clean.sel(time=slice('2002-04-01', '2026-05-31'))
print(f"  Daily records: {len(ds_clean.time)}")

# ============================================================
# STEP 3: Monthly totals
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Monthly aggregation")
print("=" * 60)

monthly = ds_clean.resample(time='ME').sum(skipna=True)
monthly_count = ds_clean.resample(time='ME').count()
monthly = monthly.where(monthly_count['rain'] > 0)

monthly_india = monthly.sel(lat=slice(6.0, 38.0), lon=slice(67.0, 98.0))
monthly_file = os.path.join(OUTPUT_DIR, 'imd_rainfall_monthly_india_2002_2026.nc')
monthly_india.to_netcdf(monthly_file)
print(f"  {len(monthly.time)} months → {monthly_file}")

# ============================================================
# STEP 4: Annual totals
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Annual aggregation")
print("=" * 60)

annual = ds_clean.resample(time='YE').sum(skipna=True)
annual_count = ds_clean.resample(time='YE').count()
annual = annual.where(annual_count['rain'] > 0)

annual_india = annual.sel(lat=slice(6.0, 38.0), lon=slice(67.0, 98.0))
annual_file = os.path.join(OUTPUT_DIR, 'imd_rainfall_annual_india_2002_2026.nc')
annual_india.to_netcdf(annual_file)

print(f"  Year | All-India Mean Rainfall (mm)")
print(f"  -----|-----------------------------")
for t in annual_india.time.values:
    yr = str(t)[:4]
    val = float(annual_india['rain'].sel(time=t).mean(dim=['lat', 'lon']))
    print(f"  {yr} | {val:.0f} mm")
print(f"\n  Saved: {annual_file}")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  Monthly: {monthly_file}")
print(f"  Annual:  {annual_file}")
print("=" * 60)
