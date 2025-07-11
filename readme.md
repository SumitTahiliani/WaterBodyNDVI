# Lake NDVI Trend Analysis

A comprehensive analysis tool for studying vegetation trends (NDVI) in concentric buffer zones around water bodies using Sentinel-2 satellite data.

For example results and detailed analysis, refer to `bufferTrend.ipynb`.

## Overview

This project analyzes Normalized Difference Vegetation Index (NDVI) trends around lakes using incremental concentric buffers. The analysis examines how vegetation patterns change with distance from water bodies, providing insights into the ecological impact of water bodies on surrounding vegetation.

## Methodology

- **Data Source**: Sentinel-2 satellite imagery from Microsoft Planetary Computer
- **Analysis Period**: Multi-temporal analysis with seasonal separation (dry vs monsoon)
- **Buffer Analysis**: Concentric buffer zones at configurable distances (100m to 5000m)
- **Trend Calculation**: Ordinary Least Squares (OLS) slope analysis for temporal trends
- **Seasonal Comparison**: Separate analysis for dry and monsoon seasons

## Project Structure

```
WaterBodyNDVI/
├── analysis.py              # Core analysis functions and OLS slope calculations
├── app.py                   # Streamlit web application interface
├── data_processing.py       # Data fetching and preprocessing from Planetary Computer
├── visualization.py         # Plotting and visualization functions
├── analyze_ndvi.py          # Comprehensive NDVI analysis pipeline
├── prep_ndvi.py            # NDVI data preparation utilities
├── bufferTrend.ipynb       # Initial proof-of-concept analysis notebook
├── lake_buffer_slope_summary.csv  # Analysis results summary
├── figures/                # Generated visualization outputs
└── data/                   # Data storage directory (not included in the repository)
```

## Key Features

- **Automated Data Processing**: Fetches Sentinel-2 data and performs local analysis
- **Interactive Web Interface**: Streamlit-based application for easy analysis
- **Multi-lake Support**: Pre-configured analysis for major Indian lakes
- **Custom Location Analysis**: Geocoding support for any global location
- **Seasonal Analysis**: Separate trend analysis for dry and monsoon periods
- **Distance Decay Visualization**: Plots showing NDVI trends with distance from water bodies

## Results

The analysis produces:
- Temporal NDVI slope values for each buffer zone
- Seasonal comparison between dry and monsoon periods
- Distance decay plots showing vegetation trends
- Statistical summaries for multiple lakes
