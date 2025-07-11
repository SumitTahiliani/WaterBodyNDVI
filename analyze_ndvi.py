import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
from shapely.geometry import Point

# --- Core Analysis Functions ---

def ols_slope(stack):
    """
    Calculates the Ordinary Least Squares slope for a time-series stack.
    Assumes stack shape is (time, rows, cols).
    """
    t = np.arange(stack.shape[0])[:, None, None]
    mask = ~np.isnan(stack)
    
    t_mean = np.nanmean(np.where(mask, t, np.nan), axis=0, keepdims=True)
    y_mean = np.nanmean(stack, axis=0, keepdims=True)
    
    t_d = t - t_mean
    y_d = stack - y_mean
    
    num = np.nansum(t_d * y_d, axis=0)
    den = np.nansum(t_d ** 2, axis=0)
    
    # Avoid division by zero
    slope = np.where(den == 0, np.nan, num / den)
    return slope

def create_buffer_zones(lon, lat, buffer_distances_m, target_crs):
    """
    Creates concentric buffer zones around a point.
    Returns a list of GeoPandas GeoSeries objects in the target CRS.
    """
    pt_wgs = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    
    # Project to a suitable UTM zone for metric buffering (e.g., UTM 44N)
    pt_utm = pt_wgs.to_crs(epsg=32644)
    
    buffers_reprojected = []
    for dist in buffer_distances_m:
        # Create buffer in metric CRS
        buffer_utm = pt_utm.geometry.iloc[0].buffer(dist)
        
        # Create a GeoSeries for the buffer and reproject to target CRS
        buffer_reprojected = gpd.GeoSeries([buffer_utm], crs=pt_utm.crs).to_crs(target_crs)
        buffers_reprojected.append(buffer_reprojected.iloc[0])
        
    return buffers_reprojected

def analyze_lake_trends(lake_name, lon, lat, ndvi_path, buffer_dists):
    """
    Performs the full trend analysis for a single lake.
    """
    print(f"Processing lake: {lake_name}")
    
    with rasterio.open(ndvi_path) as src:
        ndvi_stack = src.read()
        transform = src.transform
        profile = src.profile

    # Assuming 8 bands: 4 dry, 4 monsoon, interleaved
    # Bands 1, 3, 5, 7 are dry (indices 0, 2, 4, 6)
    # Bands 2, 4, 6, 8 are monsoon (indices 1, 3, 5, 7)
    ndvi_dry = ndvi_stack[[0, 2, 4, 6], :, :]
    ndvi_monsoon = ndvi_stack[[1, 3, 5, 7], :, :]

    # Compute temporal slopes for each pixel
    slope_dry = ols_slope(ndvi_dry)
    slope_monsoon = ols_slope(ndvi_monsoon)

    # Create buffer geometries
    buffers_geom = create_buffer_zones(lon, lat, buffer_dists, profile['crs'])

    lake_results = []
    for dist, geom in zip(buffer_dists, buffers_geom):
        try:
            # Calculate mean slope within each buffer zone
            zs_dry = zonal_stats([geom], slope_dry, affine=transform, stats='mean', nodata=np.nan)[0]['mean']
            zs_monsoon = zonal_stats([geom], slope_monsoon, affine=transform, stats='mean', nodata=np.nan)[0]['mean']
            
            lake_results.append({
                'lake': lake_name,
                'buffer_m': dist,
                'ndvi_dry_slope': zs_dry,
                'ndvi_monsoon_slope': zs_monsoon,
            })
        except Exception as e:
            print(f"  - Could not process buffer {dist}m for {lake_name}. Error: {e}")
            lake_results.append({
                'lake': lake_name,
                'buffer_m': dist,
                'ndvi_dry_slope': None,
                'ndvi_monsoon_slope': None,
            })
            
    return lake_results

# --- Visualization Functions ---

def plot_distance_decay(df, output_dir):
    """Plots NDVI slope vs. buffer distance for each lake."""
    for lake in df['lake'].unique():
        sub = df[df['lake'] == lake].sort_values('buffer_m')
        plt.figure(figsize=(8, 5))
        plt.plot(sub['buffer_m'], sub['ndvi_dry_slope'], marker='o', label='Dry Season')
        plt.plot(sub['buffer_m'], sub['ndvi_monsoon_slope'], marker='s', label='Monsoon Season')
        plt.xlabel('Buffer Distance (m)')
        plt.ylabel('NDVI Slope (change per year)')
        plt.title(f'{lake} - NDVI Trend vs. Distance from Shore')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{lake}_NDVI_distance_decay.png'), dpi=300)
        plt.close()

def plot_seasonal_contrast_bar(df, output_dir, buffer_target=1000):
    """Plots a bar chart comparing seasonal slopes at a specific buffer distance."""
    sub = df[df['buffer_m'] == buffer_target]
    if sub.empty:
        print(f"No data available for buffer target {buffer_target}m. Skipping seasonal contrast plot.")
        return

    lakes = sub['lake'].unique()
    x = np.arange(len(lakes))
    bar_width = 0.35

    dry_vals = sub['ndvi_dry_slope']
    mon_vals = sub['ndvi_monsoon_slope']

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, dry_vals, width=bar_width, label='Dry Season')
    plt.bar(x + bar_width/2, mon_vals, width=bar_width, label='Monsoon Season')
    
    plt.xticks(ticks=x, labels=lakes, rotation=45, ha="right")
    plt.ylabel('NDVI Slope (change per year)')
    plt.title(f'Seasonal NDVI Trend Comparison at {buffer_target}m Buffer')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_contrast_bar.png'), dpi=300)
    plt.close()

# --- Main Execution ---

def main():
    # Configuration
    NDVI_STACKS_DIR = 'data/ndvi_stacks'
    OUTPUT_DIR = 'figures'
    RESULTS_CSV = 'lake_buffer_slope_summary.csv'
    BUFFER_DISTS = [100, 500, 1000, 2000, 3000, 5000]

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find available NDVI stacks and derive lake info
    available_files = os.listdir(NDVI_STACKS_DIR)
    lake_info = {}
    for f in available_files:
        if f.startswith('NDVI_') and f.endswith('.tif'):
            try:
                # Example filename: NDVI_Pichola_8band_2018-2024_composite.tif
                name = f.split('_')[1]
                # A real implementation would get coords from a lookup, here we hardcode for now
                coords_lookup = {
                    'Pichola': (73.679, 24.572),
                    'Chilika': (85.3, 19.5),
                    'Tungabhadra': (76.333, 15.3),
                    'Sukhna': (76.817, 30.733)
                }
                if name in coords_lookup:
                    lon, lat = coords_lookup[name]
                    lake_info[name] = {'lon': lon, 'lat': lat, 'path': os.path.join(NDVI_STACKS_DIR, f)}
            except IndexError:
                print(f"Could not parse lake name from filename: {f}")

    if not lake_info:
        print(f"No valid NDVI GeoTIFFs found in '{NDVI_STACKS_DIR}'. Exiting.")
        return

    # Run analysis for all lakes
    all_results = []
    for name, info in lake_info.items():
        results = analyze_lake_trends(name, info['lon'], info['lat'], info['path'], BUFFER_DISTS)
        all_results.extend(results)

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Analysis complete. Results saved to {RESULTS_CSV}")

    # Generate and save plots
    plot_distance_decay(df, OUTPUT_DIR)
    plot_seasonal_contrast_bar(df, OUTPUT_DIR)
    print(f"Visualizations saved in '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    main()
