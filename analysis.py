# analysis.py
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point
import rasterio.transform

# --- Core Analysis Functions ---

def ols_slope(stack):
    """
    Calculates the Ordinary Least Squares slope for a time-series stack.
    Assumes stack shape is (time, rows, cols).
    """
    # Create a time index array matching the stack's time dimension
    t = np.arange(stack.shape[0])[:, None, None]
    
    # Create a mask for valid (non-NaN) pixels
    mask = ~np.isnan(stack)
    
    # Calculate means, ignoring NaNs
    t_mean = np.nanmean(np.where(mask, t, np.nan), axis=0, keepdims=True)
    y_mean = np.nanmean(stack, axis=0, keepdims=True)
    
    # Demean the data
    t_d = t - t_mean
    y_d = stack - y_mean
    
    # Calculate numerator and denominator of the slope formula
    num = np.nansum(t_d * y_d, axis=0)
    den = np.nansum(t_d ** 2, axis=0)
    
    # Calculate slope, avoiding division by zero
    slope = np.where(den == 0, np.nan, num / den)
    return slope

def create_buffer_zones(lon, lat, buffer_distances_m, target_crs):
    """
    Creates concentric buffer zones around a point.
    Returns a list of Shapely geometry objects in the target CRS.
    """
    # Create a GeoDataFrame for the central point
    pt_wgs = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    
    pt_utm = pt_wgs.to_crs(epsg=32644)
    
    buffers_reprojected = []
    for dist in buffer_distances_m:
        # Create buffer in the metric CRS
        buffer_utm = pt_utm.geometry.iloc[0].buffer(dist)
        
        # Reproject the buffer back to the target CRS (matching the raster)
        buffer_reprojected = gpd.GeoSeries([buffer_utm], crs=pt_utm.crs).to_crs(target_crs)
        buffers_reprojected.append(buffer_reprojected.iloc[0])
        
    return buffers_reprojected

def run_analysis(ndvi_data, raster_profile, lon, lat, buffer_dists):
    """
    Performs the full trend analysis on in-memory NDVI data.
    
    Args:
        ndvi_data (np.array): The 8-band NDVI data array.
        raster_profile (dict): A dictionary with CRS and transform info.
        lon (float): Longitude of the lake center.
        lat (float): Latitude of the lake center.
        buffer_dists (list): List of buffer distances in meters.

    Returns:
        A pandas DataFrame with the analysis results.
    """
    print("Starting analysis...")
    
    # The transform can be an Affine object or a dict, convert to Affine
    transform = rasterio.transform.guard_transform(raster_profile['transform'])

    # Split data into dry and monsoon seasons
    # Bands 1, 3, 5, 7 are dry (indices 0, 2, 4, 6)
    # Bands 2, 4, 6, 8 are monsoon (indices 1, 3, 5, 7)
    ndvi_dry = ndvi_data[[0, 2, 4, 6], :, :]
    ndvi_monsoon = ndvi_data[[1, 3, 5, 7], :, :]

    # Compute temporal slopes for each pixel
    print("Calculating temporal slopes...")
    slope_dry = ols_slope(ndvi_dry)
    slope_monsoon = ols_slope(ndvi_monsoon)

    # Create buffer geometries
    print("Creating buffer zones...")
    buffers_geom = create_buffer_zones(lon, lat, buffer_dists, raster_profile['crs'])

    analysis_results = []
    print("Calculating zonal statistics for each buffer...")
    for dist, geom in zip(buffer_dists, buffers_geom):
        try:
            # Calculate mean slope within each buffer zone
            zs_dry = zonal_stats([geom], slope_dry, affine=transform, stats='mean', nodata=np.nan)[0]['mean']
            zs_monsoon = zonal_stats([geom], slope_monsoon, affine=transform, stats='mean', nodata=np.nan)[0]['mean']
            
            analysis_results.append({
                'buffer_m': dist,
                'ndvi_dry_slope': zs_dry,
                'ndvi_monsoon_slope': zs_monsoon,
            })
        except Exception as e:
            print(f"  - Could not process buffer {dist}m. Error: {e}")
            analysis_results.append({
                'buffer_m': dist,
                'ndvi_dry_slope': None,
                'ndvi_monsoon_slope': None,
            })
            
    print("Analysis complete.")
    return pd.DataFrame(analysis_results)

if __name__ == '__main__':
    # Example usage for testing the module
    # This requires running data_processing.py first or importing from it
    from data_processing import get_data_for_location

    pichola_lon, pichola_lat = 73.679, 24.572
    BUFFER_DISTS_TEST = [100, 500, 1000, 2000, 3000, 5000]

    # 1. Get data from the processing module
    ndvi_data_test, profile_test = get_data_for_location(pichola_lon, pichola_lat)

    if ndvi_data_test is not None:
        # 2. Run analysis on the in-memory data
        results_df = run_analysis(ndvi_data_test, profile_test, pichola_lon, pichola_lat, BUFFER_DISTS_TEST)
        
        print("\n--- Analysis Test Successful ---")
        print("Results DataFrame:")
        print(results_df.head())
        print("\nThis module is ready to be imported by 'app.py'.")
    else:
        print("\n--- Analysis Test Failed: Could not retrieve data. ---")
