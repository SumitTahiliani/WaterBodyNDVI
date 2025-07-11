# app.py
import streamlit as st
from geopy.geocoders import Nominatim
import pandas as pd
import time

# Import our custom modules
from data_processing import get_data_for_location
from analysis import run_analysis
from visualization import plot_distance_decay, plot_seasonal_contrast_bar

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Lake NDVI Trend Analysis",
    page_icon="💧",
    layout="wide"
)

# --- Helper Functions ---
def search_location(location_name):
    """Uses Nominatim to geocode a location name to lat/lon."""
    try:
        geolocator = Nominatim(user_agent="lake_ndvi_analyzer")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
    return None, None

# --- Main App UI ---
st.title("Lake NDVI Trend Analysis")
st.markdown("""
This app analyzes the vegetation trend (NDVI) in concentric buffer zones around a chosen lake. 
It uses Sentinel-2 satellite data from the Microsoft Planetary Computer.
""")

# --- User Input ---
st.sidebar.header("Location Input")
LAKE_OPTIONS = {
    "Pichola, Udaipur, India": (24.572, 73.679),
    "Chilika, Odisha, India": (19.5, 85.3),
    "Tungabhadra, Karnataka, India": (15.3, 76.333),
    "Sukhna, Chandigarh, India": (30.733, 76.817),
    "Custom": None
}

selection = st.sidebar.selectbox("Choose a lake or select 'Custom'", list(LAKE_OPTIONS.keys()))

lat, lon = None, None
lake_name = ""

if selection == "Custom":
    custom_location = st.sidebar.text_input("Enter a location name (e.g., 'Vembanad Lake, India')")
    if custom_location:
        lat, lon = search_location(custom_location)
        if lat and lon:
            st.sidebar.success(f"Found {custom_location} at ({lat:.4f}, {lon:.4f})")
            lake_name = custom_location.split(',')[0]
        else:
            st.sidebar.error("Could not find the specified location.")
else:
    lat, lon = LAKE_OPTIONS[selection]
    lake_name = selection.split(',')[0]
    st.sidebar.info(f"Using coordinates for {lake_name}: ({lat}, {lon})")

st.sidebar.header("Analysis Parameters")
BUFFER_DISTS = st.sidebar.multiselect(
    "Select buffer distances (meters)",
    options=[100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000],
    default=[100, 500, 1000, 2000, 3000, 5000]
)

# --- Analysis Execution ---
if st.sidebar.button("Run Analysis", type="primary") and lat and lon:
    st.subheader("1. Data Processing")
    
    # --- Progress Bar and Status Text ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(value, text):
        # Clamp value between 0 and 100
        value = max(0, min(100, value))
        progress_bar.progress(value)
        status_text.info(text)

    # --- Call the data processing function with the callback ---
    ndvi_data, raster_profile = get_data_for_location(
        lon, lat, progress_callback=progress_callback
    )

    if ndvi_data is not None:
        progress_bar.progress(100)
        status_text.success("In-memory NDVI data composite created successfully.")
        time.sleep(2) # Give user time to read the success message
        status_text.empty() # Clear the status text
        progress_bar.empty() # Clear the progress bar

        # 2. Analysis
        st.subheader("2. Trend Analysis")
        with st.spinner("Calculating buffer zones and temporal trends..."):
            results_df = run_analysis(ndvi_data, raster_profile, lon, lat, BUFFER_DISTS)
        st.success("Trend analysis complete.")
        
        # 3. Display Results
        st.subheader("3. Results")
        st.dataframe(results_df)
        
        # 4. Visualization
        st.subheader("4. Visualizations")
        with st.spinner("Generating plots..."):
            fig_decay = plot_distance_decay(results_df, lake_name)
            fig_bar = plot_seasonal_contrast_bar(results_df, lake_name)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_decay)
        with col2:
            if fig_bar:
                st.pyplot(fig_bar)
            else:
                st.warning("Could not generate seasonal contrast plot (missing data for 1000m buffer).")

    else:
        st.error("Analysis failed. Could not retrieve or process data for the selected location.")
        progress_bar.empty()
        status_text.empty()
else:
    st.info("Select a location and click 'Run Analysis' to begin.")
