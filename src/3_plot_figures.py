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
from config.network_config import NetworkConfig

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning) # exactextract gives a warning that is invalid




def load_accessibility_results(config: NetworkConfig, facility_type) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        
    if facility_type == "firefighters":
        gdf = gpd.read_parquet(config.Path_firefighter_accessibilty)
        Sink = gpd.read_parquet(config.Path_firefighters_sink)

    elif facility_type == "hospitals":
        gdf = gpd.read_parquet(config.Path_hospital_accessibilty)
        Sink = gpd.read_parquet(config.Path_hospital_sink)

    elif facility_type == "police":
        gdf = gpd.read_parquet(config.Path_police_accessibilty)
        Sink = gpd.read_parquet(config.Path_police_sink)

    elif facility_type == "factories":
        gdf = gpd.read_parquet(config.Path_factory_accessibility)
        Sink = gpd.read_parquet(config.Path_factory_sink)

    elif facility_type == "agriculture":
        gdf = gpd.read_parquet(config.Path_agriculture_accessibility)
        Sink = gpd.read_parquet(config.Path_agriculture_sink)

    else:
        raise ValueError(
            f"Invalid sink_type '{facility_type}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police', 'factories', 'agriculture'."
        )
    
    return gdf, Sink


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
        ax_curve.set_xlabel('Access time to closest health care facility (hours)', fontsize=12)
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


def plot_access_times_factories(df_factories: pd.DataFrame, Sink: pd.DataFrame, config: NetworkConfig) -> None:
    """
    Visualize the average access times for factories to reach all border crossings.
    
    Args:
        df_factories: data frame with industrial centers in Serbia, Sink: border crossings, config
        
    Returns:
        Nothing
    """

    df_factories_plot = df_factories.to_crs(3857)
    Sink_plot = gpd.GeoDataFrame(Sink, geometry='geometry', crs="EPSG:4326").to_crs(3857)

    bins = [1, 2, 3, 4, 5, float('inf')]
    labels = ['1-2', '2-3', '3-4', '4-5', '5+']
    colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']

    df_factories_plot['category'] = pd.cut(
        df_factories_plot['avg_access_time'], 
        bins=bins, labels=labels, right=False
    )
    df_factories_plot['category'] = df_factories_plot['category'].astype('object')
    df_factories_plot.loc[df_factories_plot['category'].isna(), 'category'] = 'Not Accessible'

    color_map = dict(zip(labels, colors))
    color_map['Not Accessible'] = '#bdbdbd'

    fig, ax = plt.subplots(figsize=(24, 14))

    for category, color in color_map.items():
        data = df_factories_plot[df_factories_plot['category'] == category]
        if not data.empty:
            data.plot(ax=ax, color=color, legend=False, linewidth=0.1, edgecolor='grey', markersize=200)

    Sink_plot.plot(ax=ax, color='black', markersize=200, marker='^')
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    ax.set_aspect('equal')
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=color, label=f'{label} hours') 
                    for label, color in zip(labels, colors)]
    legend_patches.append(Line2D([0], [0], marker='^', color='black', lw=0, 
                                label='Border Crossings', markersize=15))

    ax.legend(handles=legend_patches, 
            loc='upper right',
            fontsize=12,
            title='Average Travel Time',
            title_fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95)

    plt.savefig(config.figure_path / 'factory_access_avg.png', dpi=200, bbox_inches='tight')
    plt.show()


#move this function to plotting script
def plot_access_time_agriculture(df_agri: pd.DataFrame, Sinks: pd.DataFrame, config: NetworkConfig) -> None:
    """
    Create visualization of the average access times from agricultural areas to borders, ports and rail 
    
    Args:
        Pandas DataFrame with agricultural areas, Pandas DataFrame with borders, ports and rail locations
        
    Returns:
        Nothing
    """
    df_agri_plot = df_agri.to_crs(3857)
    Sinks_plot = gpd.GeoDataFrame(Sinks, geometry='geometry', crs="EPSG:4326").to_crs(3857)

    bins = [1, 2, 3, 4, 5, float('inf')]
    labels_cat = ['1-2', '2-3', '3-4', '4-5', '5+']
    colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
    color_map = dict(zip(labels_cat, colors))

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for ax, col, title in zip(axes, 
                            ['avg_access_road', 'avg_access_port', 'avg_access_rail'],
                            ["A","B","C"]):
                            #['Road Border Crossings', 'Ports', 'Rail Terminals']):
        
        df_plot = df_agri_plot.copy()
        df_plot['category'] = pd.cut(df_plot[col], bins=bins, labels=labels_cat, right=False)
        df_plot['category'] = df_plot['category'].astype('object')
        
        for category, color in color_map.items():
            data = df_plot[df_plot['category'] == category]
            if not data.empty:
                data.plot(ax=ax, color=color, legend=False, linewidth=0.1, edgecolor='grey', markersize=50)
        
        # Plot relevant sinks
        if 'road' in col:
            sink_subset = Sinks_plot[Sinks_plot['type'] == 'road']
            marker = '^'
        elif 'port' in col:
            sink_subset = Sinks_plot[Sinks_plot['type'] == 'port']
            marker = 's'
        else:
            sink_subset = Sinks_plot[Sinks_plot['type'] == 'rail']
            marker = 'o'
        
        sink_subset.plot(ax=ax, color='black', markersize=100, marker=marker)
        
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

        # Add letter label
        ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=20, 
                fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_aspect('equal')
        ax.axis('off')
        #ax.set_title(title, fontsize=18, fontweight='bold')

    # Shared legend
    legend_patches = [mpatches.Patch(color=color, label=f'{label} hours') 
                    for label, color in zip(labels_cat, colors)]
    legend_patches.extend([
        Line2D([0], [0], marker='^', color='black', lw=0, label='Road Borders', markersize=12),
        Line2D([0], [0], marker='s', color='black', lw=0, label='Ports', markersize=12),
        Line2D([0], [0], marker='o', color='black', lw=0, label='Rail Terminals', markersize=12),
    ])

    fig.legend(handles=legend_patches, loc='lower center', ncol=8, fontsize=12, 
            title='Average Access Time', title_fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(config.figure_path / 'agriculture_access_by_type.png', dpi=200, bbox_inches='tight')
    plt.show()



def plot_access_time_map(df_worldpop: gpd.GeoDataFrame, Sink: pd.DataFrame, config: NetworkConfig, emergency_service) -> gpd.GeoDataFrame:
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


    if emergency_service == "firefighters":
        legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='Fire departments', markersize=15))
        file_name = 'firefighter_access.png'

    elif emergency_service == "hospitals":
        legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='hospitals', markersize=15))
        file_name = 'hospital_access.png'

    elif emergency_service == "police":
        legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='police stations', markersize=15))
        file_name = 'police_station_access.png'

    else:
        raise ValueError(
            f"Invalid sink_type '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    
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

    plt.savefig(config.figure_path / file_name, dpi=200, bbox_inches='tight')

    if config.show_figures == True:
        plt.show()

    return df_worldpop_plot



def plot_accessibility_chart(df_worldpop_plot, config, emergency_service):

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
    fig, ax = plt.subplots(figsize=(4.5, 7))

    # Create horizontal bar chart with narrower bars
    bars = ax.barh(range(len(pop_by_category_reversed)), pop_by_category_reversed.values, 
                height=0.5,
                color=[color_dict[cat] for cat in pop_by_category_reversed.index],
                edgecolor='black', linewidth=1.5)

    # Customize the plot
    if emergency_service == "firefighters":
        ax.set_ylabel('Access Time to closest fire station (hours)', fontsize=14, fontweight='bold', labelpad=10)
        file_name = 'firefighter_access_chart.png'

    elif emergency_service == "hospitals":
        ax.set_ylabel('Access Time to closest health care facility (hours)', fontsize=14, fontweight='bold', labelpad=10)
        file_name = 'hospital_access_chart.png'

    elif emergency_service == "police":
        ax.set_ylabel('Access Time to closest police station (hours)', fontsize=14, fontweight='bold', labelpad=10)
        file_name = 'police_station_access_chart.png'

    else:
        raise ValueError(
            f"Invalid sink_type '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )
    
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
    
    plt.savefig(config.figure_path / file_name, dpi=200, bbox_inches='tight')
    
    if config.show_figures == True:
        plt.show()




def main():
    """
    Main function to orchestrate all accessibilty calculations.
    """
    # Initialize configuration
    config = NetworkConfig()

    # =============================================================================
    # 1. Plot accessibility results for factories
    # =============================================================================

    #Load results of acessibility analysis to factories to road borders
    print("Loading results of factory accessibility analysis...")
    df_factory_accessibility, df_factory_sinks = load_accessibility_results(config, "factories")

    #create map of the access times of factories to road border crossings
    plot_access_times_factories(df_factory_accessibility, df_factory_sinks, config)

    
    # =============================================================================
    # 2. Plot accessibility results for agricultural areas
    # =============================================================================

    #Load results of acessibility analysis to agricultural areas to road border crossings, rail terminals and ports 
    print("Loading results of accessibility analysis of agricultural areas...")
    df_agriculture_accessibility, df_agriculture_sinks = load_accessibility_results(config, "agriculture")

    #create map of the access times of factories to road border crossings
    plot_access_time_agriculture(df_agriculture_accessibility, df_agriculture_sinks, config)

    # =============================================================================
    # 3. Plot accessibility results for firefighters
    # =============================================================================

    #Load results of acessibility analysis to fire departments    
    print("Loading results of fire station accessibility analysis...")
    df_firefighter_accessibility, df_fire_stations = load_accessibility_results(config, "firefighters")

    # Plot the cumulative curve of the access times to the closest fire station   
    plot_access_curve(df_firefighter_accessibility, config, "firefighters")

    #Plot a map that shows the accessibility time of each settlement to the clostest fire station 
    df_worldpop_plot = plot_access_time_map(df_firefighter_accessibility, df_fire_stations, config, "firefighters")

    #Create a combined bar and pie chart to show the percentage of the population that fall in each access time category
    plot_accessibility_chart(df_worldpop_plot, config, "firefighters")

    # =============================================================================
    # 4. Plot accessibility results for hospitals
    # =============================================================================

    #Load results of acessibility analysis to hospitals    
    print("Loading results of hospital accessibility analysis...")
    df_hospital_accessibility, df_hospitals = load_accessibility_results(config, "hospitals")

    # Plot the cumulative curve of the access times to the closest fire station   
    plot_access_curve(df_hospital_accessibility, config, "hospitals")

    #Plot a map that shows the accessibility time of each settlement to the clostest fire station 
    df_worldpop_plot = plot_access_time_map(df_hospital_accessibility, df_hospitals, config, "hospitals")

    #Create a combined bar and pie chart to show the percentage of the population that fall in each access time category
    plot_accessibility_chart(df_worldpop_plot, config, "hospitals")


    # =============================================================================
    # 5. Plot accessibility results for police stations
    # =============================================================================

    #Load results of acessibility analysis to hospitals    
    print("Loading results of police station accessibility analysis...")
    df_police_accessibility, df_police = load_accessibility_results(config, "police")

    # Plot the cumulative curve of the access times to the closest fire station   
    plot_access_curve(df_police_accessibility, config, "police")

    #Plot a map that shows the accessibility time of each settlement to the clostest fire station 
    df_worldpop_plot = plot_access_time_map(df_police_accessibility, df_police, config, "police")

    #Create a combined bar and pie chart to show the percentage of the population that fall in each access time category
    plot_accessibility_chart(df_worldpop_plot, config, "police")





if __name__ == "__main__":
    main()