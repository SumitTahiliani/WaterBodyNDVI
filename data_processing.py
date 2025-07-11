# data_processing.py
import numpy as np
import xarray as xr
from mapminer.miners import Sentinel2Miner
from dask.diagnostics import ProgressBar

# --- Configuration ---
DRY_MONTHS = [1, 2, 3, 4]
MONSOON_MONTHS = [7, 8, 9, 10]
MONTH_ORDER = DRY_MONTHS + MONSOON_MONTHS

def get_data_for_location(lon, lat, start_date="2018-01-01", end_date="2024-12-31", progress_callback=None):
    """
    Orchestrates data fetching and processing for a given lat/lon using mapminer.
    Uses a callback to report progress to a UI.
    """
    def report_progress(value, text):
        if progress_callback:
            progress_callback(value, text)
        print(text)

    report_progress(0, "Initializing Sentinel2Miner...")
    try:
        miner = Sentinel2Miner()
    except Exception as e:
        report_progress(0, f"Error initializing Sentinel2Miner: {e}")
        return None, None

    report_progress(10, f"Fetching Sentinel-2 data at 20m resolution for coordinates ({lon}, {lat})...")
    try:
        dataset = miner.fetch(
            lat=lat,
            lon=lon,
            radius=5000,
            daterange=(start_date, end_date),
        )
        if dataset is None or 'B04' not in dataset.data_vars:
            report_progress(10, "No data returned from mapminer for the given location and date range.")
            return None, None
    except Exception as e:
        report_progress(10, f"Error fetching data with mapminer: {e}")
        return None, None

    report_progress(25, "Creating in-memory NDVI composite...")
    try:
        report_progress(30, "  - Step 1: Pre-filtering dataset to months of interest...")
        monthly_data = dataset.where(dataset['time.month'].isin(MONTH_ORDER), drop=True)

        report_progress(40, "  - Step 2: Creating cloud mask from SCL band...")
        clear_pixels_mask = monthly_data['SCL'].isin([2, 4, 5, 6, 7, 11])

        report_progress(50, "  - Step 3: Applying mask and calculating NDVI...")
        spectral_bands = monthly_data[['B04', 'B08']]
        stack_masked = spectral_bands.where(clear_pixels_mask)

        red = stack_masked['B04']
        nir = stack_masked['B08']
        ndvi = (nir - red) / (nir + red + 1e-9)

        report_progress(60, "  - Step 4: Grouping by month and calculating mean NDVI...")
        monthly_ndvi = ndvi.groupby("time.month").mean(dim="time", skipna=True)
        
        composite = monthly_ndvi.sel(month=MONTH_ORDER)
        
        if len(composite.month) != len(MONTH_ORDER):
            report_progress(70, f"  - Warning: Found data for only {len(composite.month)} of the {len(MONTH_ORDER)} requested months.")
            composite = composite.reindex({"month": MONTH_ORDER}, fill_value=np.nan)

        report_progress(80, "  - Step 5: Finalizing composite. Computing now...")
        with ProgressBar():
            computed_composite = composite.compute()
        
        report_progress(100, "Composite creation complete.")
        
        profile = {
            'crs': computed_composite.rio.crs,
            'transform': computed_composite.rio.transform(),
            'height': computed_composite.rio.height,
            'width': computed_composite.rio.width,
            'count': len(computed_composite.month),
            'dtype': str(computed_composite.dtype)
        }
        
        return computed_composite.values, profile

    except Exception as e:
        report_progress(0, f"Error during NDVI composite creation: {e}")
        return None, None

if __name__ == '__main__':
    pichola_lon, pichola_lat = 73.679, 24.572
    
    # Example of using the callback in a non-UI script
    def simple_progress_tracker(value, text):
        print(f"PROGRESS: {value}% - {text}")

    print("--- Running Data Processing Test with mapminer ---")
    ndvi_data, raster_profile = get_data_for_location(
        pichola_lon, pichola_lat, progress_callback=simple_progress_tracker
    )

    if ndvi_data is not None:
        print("\n--- Data Processing Test Successful ---")
        print(f"Data shape: {ndvi_data.shape}")
        print(f"Raster profile: {raster_profile}")
    else:
        print("\n--- Data Processing Test Failed ---")
