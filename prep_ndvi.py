# prep_ndvi.py
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import Point
from datetime import datetime

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Define months for seasonal analysis
DRY_MONTHS = [1, 2, 3, 4]
MONSOON_MONTHS = [7, 8, 9, 10]
MONTH_ORDER = DRY_MONTHS + MONSOON_MONTHS

def fetch_s2_items(aoi_geom, start_date="2018-01-01", end_date="2024-12-31"):
    """Fetches Sentinel-2 L2A items for a given geometry and date range."""
    catalog = pystac_client.Client.open(PC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        geometry=aoi_geom,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 25}}  # A bit higher to get more scenes
    )
    return list(search.items())

def create_cloud_masked_ndvi(items, aoi_geom):
    """
    Creates a cloud-masked, monthly median NDVI composite from STAC items.
    """
    # Define the required assets
    assets = ["B04", "B08", "SCL"]

    # Create a data stack
    stack = stackstac.stack(
        items,
        assets=assets,
        resolution=10,
        bounds=aoi_geom.bounds,
        fill_value=np.nan,
    ).rename({"band": "spectral"}).astype("float32")

    # Create a cloud mask from the Scene Classification Layer (SCL)
    # SCL values to mask: 4 (vegetation), 5 (not vegetated), 6 (water), 7 (unclassified)
    # are considered clear. We mask out clouds, shadows, etc.
    clear_pixels = stack.sel(spectral="SCL").isin([2, 4, 5, 6, 7, 11])

    # Apply the mask
    stack_masked = stack.where(clear_pixels)

    # Calculate NDVI
    red = stack_masked.sel(spectral="B04")
    nir = stack_masked.sel(spectral="B08")
    
    # Adding a small epsilon to avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-9)

    # Create monthly median composites
    monthly_ndvi = ndvi.groupby("time.month").median(dim="time", skipna=True)
    
    # Filter for and order the months of interest
    return monthly_ndvi.sel(month=MONTH_ORDER)

def main(lake_name, lon, lat, out_tif):
    """
    Main function to generate and save a multi-band NDVI GeoTIFF for a given lake.
    """
    # Create a GeoDataFrame for the AOI
    point_geom = Point(lon, lat)
    aoi_gdf = gpd.GeoSeries([point_geom], crs="EPSG:4326")

    # Define a buffered area for fetching data (~5-km box)
    fetch_aoi = aoi_gdf.buffer(0.05, cap_style=3).iloc[0]

    print(f"Fetching items for {lake_name}...")
    items = fetch_s2_items(fetch_aoi)
    if not items:
        print(f"No items found for {lake_name}. Exiting.")
        return

    print(f"Creating monthly NDVI composite for {lake_name}...")
    # Use the original point geometry for stacking to keep it centered
    composite = create_cloud_masked_ndvi(items, aoi_gdf.iloc[0].buffer(0.045, cap_style=3))
    
    # Set CRS and save to raster
    composite = composite.rio.write_crs("EPSG:4326")
    composite.rio.to_raster(out_tif, dtype="float32", compress="deflate", windowed=True)
    print(f"✅ Saved NDVI composite to {out_tif}")

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Generate monthly NDVI composites for a given location.")
    parser.add_argument("--name", required=True, help="Name of the lake or location.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the location.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the location.")
    args = parser.parse_args()

    out_dir = pathlib.Path("data/ndvi_stacks")
    out_dir.mkdir(exist_ok=True, parents=True)
    output_tif = out_dir / f"NDVI_{args.name}_8band_2018-2024_composite.tif"
    
    main(args.name, args.lon, args.lat, output_tif)
