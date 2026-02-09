import warnings,sys,os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import geohexgrid as ghg
import seaborn as sns
import shapely
from shapely import Point
import igraph as ig
from pathlib import Path
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Rectangle,Patch
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from exactextract import exact_extract
from matplotlib.lines import Line2D  # For custom legend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as cx                                  
from simplify import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning) # exactextract gives a warning that is invalid


class NetworkConfig:
    """Configuration for accesibility analysis and visualization."""

    # Input paths
    data_path = Path('input_files')
    intermediate_results_path = Path('intermediate_results')
    Path_RoadNetwork = data_path / "base_network_SRB_basins.parquet"
    Path_FactoryFile = data_path / "2_Factory_Company_geolocations.xlsx"
    path_to_Borders = data_path / "Borders_geocoded.xlsx"
    Path_AgriFile = data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    path_to_Sinks =data_path / "Borders_Ports_Rail_geocoded.xlsx"
    Path_SettlementData_Excel = data_path / "population_NEW_settlement_geocoded.xlsx"
    firefighters = data_path / "6_Firefighters_geocoded.xlsx"


    #Paths that are input and output
    Path_firefighter_accessibilty = intermediate_results_path / 'firefighter_settle_results.parquet'
    Path_firefighters_sink = intermediate_results_path / 'firefighters.parquet'
    

    #Output path
    figure_path = Path('figures')
    

    #Flag to set whether plots will be shown in a pop up window or not
    show_figures = True




def load_accessibility_results_firefighters(config: NetworkConfig) -> gpd.GeoDataFrame:


    gdf = gpd.read_parquet(config.Path_firefighter_accessibilty)

    return gdf


def load_fire_stations(config: NetworkConfig) -> gpd.GeoDataFrame:

    Sink = gpd.read_parquet(config.Path_firefighters_sink)

    return Sink


def plot_access_curve(df_worldpop: gpd.GeoDataFrame, config: NetworkConfig, emergency_service) -> None:

    fig, ax_curve = plt.subplots(1, 1, figsize=(8, 4))


    # Panel B: Calculate access curves
    None_INF_Pop_in_Zone_Sample = df_worldpop.copy()
    total_population = df_worldpop['population'].sum()
    thresholds = np.arange(0, 6.1, 1/6)  

    percentage_population_within_threshold_2 = []


    for threshold in thresholds:
        population_sum_2 = df_worldpop.loc[
            df_worldpop['closest_sink_total_fft'] <= threshold, 'population'
        ].sum()
        population_percentage_2 = (population_sum_2 / total_population) * 100
        percentage_population_within_threshold_2.append(population_percentage_2)

    # Find 100% thresholds
    threshold_100_2 = next((threshold for i, threshold in enumerate(thresholds) 
                        if percentage_population_within_threshold_2[i] == 100), None)

    # Plot access curves
    ax_curve.plot(thresholds, percentage_population_within_threshold_2, linestyle='-', 
                color='#003049', linewidth=2, label='Normal condition')

    if emergency_service == "firefighters":
        ax_curve.set_xlabel('Access time to closest fire station (hours)', fontsize=12)
    elif emergency_service == "hospitals":
        ax_curve.set_xlabel('Access time to closest health facilities (hours)', fontsize=12)
    elif emergency_service == "police":
        ax_curve.set_xlabel('Access time to closest police station (hours)', fontsize=12)
    else:
        raise ValueError(
            f"Invalid emergency_service '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    ax_curve.set_ylabel('Population with access (%)', fontsize=12)
    ax_curve.legend(fontsize=10)
    ax_curve.minorticks_on()
    ax_curve.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Match Panel A's height by setting aspect ratio
    ax_curve.set_aspect('auto', adjustable='box')

    if threshold_100_2 is not None:
        ax_curve.axvline(x=threshold_100_2, color='#003049', linestyle='--', linewidth=1, alpha=0.7)
        ax_curve.plot(threshold_100_2, 100, 'o', color='#003049', markersize=6)
        ax_curve.text(threshold_100_2, 94, f'{threshold_100_2:.1f}h', color='black', ha='left', fontsize=10)

    plt.savefig(config.figure_path / 'baseline_accessibility_fire_disruptions.png',dpi=150,transparent=True)


    if config.show_figures == True:
        plt.show()


def plot_access_time_fire_station_map(df_worldpop: gpd.GeoDataFrame, Sink: pd.DataFrame, config: NetworkConfig) -> gpd.GeoDataFrame:
    # Prepare data
    #edges_gdf = gpd.GeoDataFrame(edges, geometry='geometry')
    df_worldpop_plot = gpd.GeoDataFrame(df_worldpop, geometry='geometry', crs="EPSG:4326").to_crs(3857)
    Sink_fire = gpd.GeoDataFrame(Sink, geometry='geometry', crs="EPSG:4326").to_crs(3857)

    # Create bins for categories (1-hour intervals)
    bins = [0, 0.25, 0.5, 1, 1.5, 2, float('inf')]
    labels = ['0-15', '15-30', '30-60', '60-90', '90-120', '>120']
    colors = ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#c51b8a']

    # Assign categories
    df_worldpop_plot['category'] = pd.cut(df_worldpop_plot['closest_sink_total_fft'], 
                                        bins=bins, labels=labels, right=False)

    # Convert to object type to allow mixed values
    df_worldpop_plot['category'] = df_worldpop_plot['category'].astype('object')

    # Handle NaN values as "Not Accessible"
    df_worldpop_plot.loc[df_worldpop_plot['category'].isna(), 'category'] = 'Not Accessible'

    # Add "Not Accessible" to color mapping (using gray)
    color_map = dict(zip(labels, colors))
    color_map['Not Accessible'] = '#bdbdbd'  # Gray color

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 14))

    # Plot by category
    for category, color in color_map.items():
        data = df_worldpop_plot[df_worldpop_plot['category'] == category]
        if not data.empty:
            data.plot(ax=ax, color=color, legend=False,linewidth=0.1,edgecolor='grey')

    Sink_fire.plot(ax=ax, color='red', markersize=100, marker='+')
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    # Enhance the plot styling
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axis for cleaner look

    # Create legend patches (add "Not Accessible" at the end)
    legend_patches = [mpatches.Patch(color=color, label=f'{label} minutes') 
                    for label, color in zip(labels, colors)]
    legend_patches.append(mpatches.Patch(color='#bdbdbd', label='Not Accessible'))
    legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='Fire departments', markersize=15))

    # Add legend
    ax.legend(handles=legend_patches, 
            loc='upper right',
            fontsize=12,
            title='Access Time',
            title_fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95)

    plt.savefig(config.figure_path / 'firefighter_access.png', dpi=200, bbox_inches='tight')

    if config.show_figures == True:
        plt.show()

    return df_worldpop_plot


def plot_fire_station_accessibility_chart(df_worldpop_plot):

    # Calculate total population per category
    pop_by_category = df_worldpop_plot.groupby('category')['population'].sum()/1e6

    # Define the order of categories and their colors
    category_order = ['0-15', '15-30', '30-60', '60-90', '90-120', '>120']
    category_colors = ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#c51b8a', '#bdbdbd']
    color_dict = dict(zip(category_order, category_colors))

    # Reindex to ensure all categories are present and in correct order
    pop_by_category = pop_by_category.reindex(category_order, fill_value=0)

    # Reverse order for horizontal plot (so 0-0.5 is at top)
    pop_by_category_reversed = pop_by_category[::-1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 7))

    # Create horizontal bar chart with narrower bars
    bars = ax.barh(range(len(pop_by_category_reversed)), pop_by_category_reversed.values, 
                height=0.5,
                color=[color_dict[cat] for cat in pop_by_category_reversed.index],
                edgecolor='black', linewidth=1.5)

    # Customize the plot
    ax.set_ylabel('Access Time (hours)', fontsize=14, fontweight='bold', labelpad=-40)
    ax.set_xlabel('Population (in millions)', fontsize=14, fontweight='bold')

    # Set y-axis labels
    ax.set_yticks(range(len(pop_by_category_reversed)))
    ax.set_yticklabels(pop_by_category_reversed.index, fontsize=12)

    # Format x-axis with thousands separator
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.tick_params(axis='x', labelsize=12)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Create inset pie chart
    ax_inset = fig.add_axes([0.3, 0.15, 0.6, 0.6])  # [left, bottom, width, height]

    # Pie chart with same order as original (not reversed) and matching colors
    pie_colors = [color_dict[cat] for cat in pop_by_category.index]
    wedges, texts = ax_inset.pie(pop_by_category.values, 
                                colors=pie_colors,
                                startangle=90,
                                counterclock=False,
                                wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})




def main():
    """
    Main function to orchestrate all accessibilty calculations.
    """
    # Initialize configuration
    config = NetworkConfig()


    # =============================================================================
    # 3. Plot accessibility results for firefighters
    # =============================================================================

    #Load results of acessibility analysis to fire departments    
    print("Loading results of fire station accessibility analysis...")
    df_firefighter_accessibility = load_accessibility_results_firefighters(config)

    #Load location of fire stations
    print("Loading location data of fire stations...")
    df_fire_stations = load_fire_stations(config)

    # Plot the cumulative curve of the access times to the closest fire station   
    plot_access_curve_fire_station(df_firefighter_accessibility, config)

    #Plot a map that shows the accessibility time of each settlement to the clostest fire station 
    df_worldpop_plot = plot_access_time_fire_station_map(df_firefighter_accessibility, df_fire_stations, config)

    #Create a combined bar and pie chart to show the percentage of the population that fall in each access time category
    plot_fire_station_accessibility_chart(df_worldpop_plot)





if __name__ == "__main__":
    main()