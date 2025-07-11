# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_distance_decay(df, lake_name):
    """
    Plots NDVI slope vs. buffer distance.
    
    Args:
        df (pd.DataFrame): DataFrame with analysis results.
        lake_name (str): The name of the lake for the plot title.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sub = df.sort_values('buffer_m')
    
    ax.plot(sub['buffer_m'], sub['ndvi_dry_slope'], marker='o', linestyle='-', label='Dry Season')
    ax.plot(sub['buffer_m'], sub['ndvi_monsoon_slope'], marker='s', linestyle='-', label='Monsoon Season')
    
    ax.set_xlabel('Buffer Distance (m)')
    ax.set_ylabel('NDVI Slope (change per year)')
    ax.set_title(f'{lake_name} - NDVI Trend vs. Distance from Shore')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_seasonal_contrast_bar(df, lake_name, buffer_target=1000):
    """
    Plots a bar chart comparing seasonal slopes at a specific buffer distance.
    
    Args:
        df (pd.DataFrame): DataFrame with analysis results.
        lake_name (str): The name of the lake.
        buffer_target (int): The buffer distance to compare.

    Returns:
        matplotlib.figure.Figure: The generated plot figure, or None if data is missing.
    """
    sub = df[df['buffer_m'] == buffer_target]
    if sub.empty:
        print(f"No data for buffer target {buffer_target}m. Skipping bar plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    
    dry_slope = sub['ndvi_dry_slope'].iloc[0]
    monsoon_slope = sub['ndvi_monsoon_slope'].iloc[0]
    
    labels = ['Dry Season', 'Monsoon Season']
    values = [dry_slope, monsoon_slope]
    
    ax.bar(labels, values, color=['sandybrown', 'skyblue'])
    
    ax.set_ylabel('NDVI Slope (change per year)')
    ax.set_title(f'{lake_name} - Seasonal NDVI Trend at {buffer_target}m Buffer')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    from data_processing import get_data_for_location
    from analysis import run_analysis

    pichola_lon, pichola_lat = 73.679, 24.572
    LAKE_NAME_TEST = "Pichola"
    BUFFER_DISTS_TEST = [100, 500, 1000, 2000, 3000, 5000]

    # 1. Get data
    ndvi_data_test, profile_test = get_data_for_location(pichola_lon, pichola_lat)
    
    if ndvi_data_test is not None:
        # 2. Run analysis
        results_df = run_analysis(ndvi_data_test, profile_test, pichola_lon, pichola_lat, BUFFER_DISTS_TEST)
        
        print("\n--- Visualization Test ---")
        
        # 3. Generate plots (but don't display them in a script)
        fig1 = plot_distance_decay(results_df, LAKE_NAME_TEST)
        fig2 = plot_seasonal_contrast_bar(results_df, LAKE_NAME_TEST)
        
        if fig1 and fig2:
            print("Successfully created Figure objects for:")
            print("- Distance Decay Plot")
            print("- Seasonal Contrast Bar Plot")
            # In a real app, you would pass these figures to Streamlit's st.pyplot()
            # For testing, we can save them
            fig1.savefig("test_distance_decay.png")
            fig2.savefig("test_seasonal_contrast_bar.png")
            print("Saved test plots to 'test_distance_decay.png' and 'test_seasonal_contrast_bar.png'")
            plt.close(fig1) # Close figures to prevent them from popping up
            plt.close(fig2)
            print("\nThis module is ready to be imported by 'app.py'.")
        else:
            print("Failed to generate one or more plots.")
    else:
        print("\n--- Visualization Test Failed: Could not retrieve data. ---")
