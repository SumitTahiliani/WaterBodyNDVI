# data_processing.py
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from mapminer.miners import Sentinel2Miner
from dask.diagnostics import ProgressBar

# --- Configuration ---
# Define months for seasonal analysis
DRY_MONTHS = [1, 2, 3, 4]
MONSOON_MONTHS = [7, 8, 9, 10]
MONTH_ORDER = DRY_MONTHS + MONSOON_MONTHS

def get_data_for_location(lon, lat, start_date="2018-01-01", end_date="2024-12-31"):
    """
    Orchestrates data fetching and processing for a given lat/lon using mapminer.

    Returns:
        A tuple of (np.array, dict) containing the NDVI data and its raster profile,
        or (None, None) if it fails.
    """
    print("Initializing Sentinel2Miner...")
    try:
        miner = Sentinel2Miner()
    except Exception as e:
        print(f"Error initializing Sentinel2Miner: {e}")
        return None, None

    print(f"Fetching Sentinel-2 data at 20m resolution for coordinates ({lon}, {lat})...")
    try:
        # OPTIMIZATION 1: Fetch at 20m resolution
        dataset = miner.fetch(
            lat=lat,
            lon=lon,
            radius=5000, # 5km radius
            daterange=(start_date, end_date),
        )
        if dataset is None or 'B04' not in dataset.data_vars:
            print("No data returned from mapminer for the given location and date range.")
            return None, None
    except Exception as e:
        print(f"Error fetching data with mapminer: {e}")
        return None, None

    print("Creating in-memory NDVI composite... (This may take a moment)")
    try:
        # Use Dask's ProgressBar to show computation progress
        with ProgressBar():
            print("  - Step 1: Pre-filtering dataset to months of interest...")
            # OPTIMIZATION 2: Filter data BEFORE grouping
            monthly_data = dataset.where(dataset['time.month'].isin(MONTH_ORDER), drop=True)

            print("  - Step 2: Creating cloud mask from SCL band...")
            clear_pixels_mask = monthly_data['SCL'].isin([2, 4, 5, 6, 7, 11])

            print("  - Step 3: Applying mask and calculating NDVI...")
            spectral_bands = monthly_data[['B04', 'B08']]
            stack_masked = spectral_bands.where(clear_pixels_mask)

            red = stack_masked['B04']
            nir = stack_masked['B08']
            ndvi = (nir - red) / (nir + red + 1e-9)

            print("  - Step 4: Grouping by month and calculating mean NDVI...")
            monthly_ndvi = ndvi.groupby("time.month").mean(dim="time", skipna=True)
            
            # The data is already filtered, so we just need to ensure order
            composite = monthly_ndvi.sel(month=MONTH_ORDER)
            
            if len(composite.month) != len(MONTH_ORDER):
                print(f"  - Warning: Found data for only {len(composite.month)} of the {len(MONTH_ORDER)} requested months.")
                composite = composite.reindex({"month": MONTH_ORDER}, fill_value=np.nan)

            print("  - Step 5: Finalizing composite. Computing now...")
            composite = composite.compute()

        print("Composite creation complete.")
        
        # --- Prepare Output ---
        profile = {
            'crs': composite.rio.crs,
            'transform': composite.rio.transform(),
            'height': composite.rio.height,
            'width': composite.rio.width,
            'count': len(composite.month),
            'dtype': str(composite.dtype)
        }
        
        return composite.values, profile

    except Exception as e:
        print(f"Error during NDVI composite creation: {e}")
        return None, None

if __name__ == '__main__':
    pichola_lon, pichola_lat = 73.679, 24.572
    
    print("--- Running Data Processing Test with mapminer ---")
    ndvi_data, raster_profile = get_data_for_location(pichola_lon, pichola_lat)

    if ndvi_data is not None:
        print("\n--- Data Processing Test Successful ---")
        print(f"Data shape: {ndvi_data.shape}")
        print(f"Raster profile: {raster_profile}")
        print("This module is ready to be imported by 'analysis.py'.")
    else:
        print("\n--- Data Processing Test Failed ---")
