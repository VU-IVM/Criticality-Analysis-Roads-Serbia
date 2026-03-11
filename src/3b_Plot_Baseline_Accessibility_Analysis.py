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
    """
    Load accessibility analysis results and corresponding facility locations for a
    specified facility type.

    Parameters
    ----------
    config : NetworkConfig
        Configuration object containing facility-specific input paths.
    facility_type : str
        One of {"firefighters", "hospitals", "police", "factories", "agriculture"}.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (gdf, Sink) where `gdf` contains accessibility results and `Sink` contains
        facility locations for the selected facility type.

    Raises
    ------
    ValueError
        If an unsupported facility_type is provided.
    """

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
    """
    Plot an accessibility curve showing the share of population that has acces to  
    to the nearest emergency service within a given travel-time threshold, and save the PNG.

    Parameters
    ----------
    df_worldpop : gpd.GeoDataFrame
        Settlement data with location and population of the settlements within the study region
        Must include 'closest_sink_total_fft' (hours to nearest facility) and 'population'.
    config : NetworkConfig
        Provides 'figure_path' for saving and 'show_figures' flag to optionally display.
    emergency_service : str
        One of {"firefighters", "hospitals", "police"}; sets x‑axis label and filename.

    Behavior
    --------
    Computes cumulative population coverage across thresholds (0-3 h, step 20 min),
    marks the threshold where 100% coverage is reached (if any), styles the plot,
    and saves a transparent PNG to `config.figure_path`.
    """

    fig, ax_curve = plt.subplots(1, 1, figsize=(5, 5))


    # Panel B: Calculate access curves
    None_INF_Pop_in_Zone_Sample = df_worldpop.copy()
    total_population = df_worldpop['population'].sum()
    thresholds = np.arange(0, 3.1, 1/3) 

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
        file_name = config.Path_baseline_accessibility_fire_stations
    elif emergency_service == "hospitals":
        ax_curve.set_xlabel('Access time to closest health care facility (hours)', fontsize=12)
        file_name = config.Path_baseline_accessibility_hospitals
    elif emergency_service == "police":
        ax_curve.set_xlabel('Access time to closest police station (hours)', fontsize=12)
        file_name = config.Path_baseline_accessibility_police_stations
    else:
        raise ValueError(
            f"Invalid emergency_service '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    ax_curve.set_ylabel('Population with access (%)', fontsize=12)
    ax_curve.legend(fontsize=12)
    ax_curve.minorticks_on()
    ax_curve.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Match Panel A's height by setting aspect ratio
    ax_curve.set_aspect('auto', adjustable='box')

    if threshold_100_2 is not None:
        ax_curve.axvline(x=threshold_100_2, color='#003049', linestyle='--', linewidth=1, alpha=0.7)
        ax_curve.plot(threshold_100_2, 100, 'o', color='#003049', markersize=6)
        ax_curve.text(threshold_100_2+0.05, 94, f'{threshold_100_2:.1f}h', color='black', ha='left', fontsize=12)

    plt.savefig(file_name,dpi=150,transparent=True)


    if config.show_figures == True:
        plt.show()


def plot_access_times_factories(df_factories: pd.DataFrame, Sink: pd.DataFrame, config: NetworkConfig) -> None:
    """
    Plot average access times from factories to border crossings and save the map.

    Parameters
    ----------
    df_factories : GeoDataFrame
        Factory locations with 'avg_access_time' and geometric coordinates.
    Sink : GeoDataFrame or DataFrame
        Border-crossing locations; converted to GeoDataFrame for plotting.
    config : NetworkConfig
        Provides output path (`figure_path`) and display settings.

    Behavior
    --------
    Classifies factories into travel-time bands, colors them on a basemap, highlights
    border crossings, adds a legend, and saves the figure as
    'factory_access_avg.png'. Optionally displays the plot when enabled.
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
    if config.show_figures == True:
        plt.show()


def plot_accessibility_curves_agriculture(df_agri: pd.DataFrame, config: NetworkConfig) -> None:
    """
    Plot 3x2 accessibility curves for agricultural areas (road, port, rail; nearest vs. average)
    and save the figure as a transparent PNG.

    Parameters
    ----------
    df_agri : pd.DataFrame
        Must include UAL (area/weight) and six time columns:
        ['nearest_access_road','nearest_access_port','nearest_access_rail',
        'avg_access_road','avg_access_port','avg_access_rail'] (hours).
    config : NetworkConfig
        Provides output path (`figure_path`) and `show_figures` flag.

    Behavior
    --------
    For each sink/metric, computes cumulative % of UAL within thresholds (0-8 h, 0.5-h step),
    marks the ~100% threshold if reached, adds panel labels (A-F), and saves
    'baseline_accessibility_agri_road_port_rail_3x2.png'.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

    # Define thresholds (shared)
    thresholds = np.arange(0, 8.1, 0.5)  # Extended range based on your data (up to ~5.8h)

    # Total agricultural land
    total_ual = df_agri['UAL'].sum()

    # Sink types configuration
    # Top row: nearest | Bottom row: avg
    # text_offset: (x_offset, y_position) for the threshold label
    sink_configs = [
        # Top row - Nearest
        {'ax': axes[0, 0], 'col': 'nearest_access_road', 'label': 'A', 'title': 'road border crossings', 
        'metric': 'Nearest', 'show_ylabel': True, 'text_offset': (0.1, 94)},
        {'ax': axes[0, 1], 'col': 'nearest_access_port', 'label': 'B', 'title': 'ports', 
        'metric': 'Nearest', 'show_ylabel': False, 'text_offset': (0.1, 94)},
        {'ax': axes[0, 2], 'col': 'nearest_access_rail', 'label': 'C', 'title': 'rail terminals', 
        'metric': 'Nearest', 'show_ylabel': False, 'text_offset': (0.1, 94)},
        
        # Bottom row - Average
        {'ax': axes[1, 0], 'col': 'avg_access_road', 'label': 'D', 'title': 'road border crossings', 
        'metric': 'Avg.', 'show_ylabel': True, 'text_offset': (-1, 92)},
        {'ax': axes[1, 1], 'col': 'avg_access_port', 'label': 'E', 'title': 'ports', 
        'metric': 'Avg.', 'show_ylabel': False, 'text_offset': (-0.9, 94)},
        {'ax': axes[1, 2], 'col': 'avg_access_rail', 'label': 'F', 'title': 'rail terminals', 
        'metric': 'Avg.', 'show_ylabel': False, 'text_offset': (0.1, 94)},
    ]

    for sink_config in sink_configs:
        ax = sink_config['ax']
        col = sink_config['col']
        
        # Calculate percentage of UAL within each threshold
        percentage_ual = []
        for threshold in thresholds:
            ual_sum = df_agri.loc[df_agri[col] <= threshold, 'UAL'].sum()
            ual_percentage = (ual_sum / total_ual) * 100
            percentage_ual.append(ual_percentage)
        
        # Find 100% threshold
        threshold_100 = next((threshold for i, threshold in enumerate(thresholds) 
                            if percentage_ual[i] >= 99.9), None)
        
        # Plot
        ax.plot(thresholds, percentage_ual, linestyle='-', 
                color='#003049', linewidth=2, label='Normal condition')
        ax.set_xlabel(f'{sink_config["metric"]} access time to \n {sink_config["title"]} (hours)', fontsize=11)
        
        # Only show y-axis label on first column
        if sink_config['show_ylabel']:
            ax.set_ylabel('Agricultural land with access (%)', fontsize=11)
        
        ax.minorticks_on()
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_aspect('auto', adjustable='box')
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(thresholds))
        
        # Add panel label
        ax.text(0.05, 0.95, sink_config['label'], transform=ax.transAxes, fontsize=16,
                fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        if threshold_100 is not None:
            x_offset, y_pos = sink_config['text_offset']
            ax.axvline(x=threshold_100, color='#003049', linestyle='--', linewidth=1, alpha=0.7)
            ax.plot(threshold_100, 100, 'o', color='#003049', markersize=6)
            ax.text(threshold_100 + x_offset, y_pos, f'{threshold_100:.1f}h', 
                    color='black', ha='left', fontsize=14)

    # Add shared legend to the last panel (bottom right)
    axes[1, 2].legend(fontsize=12, loc='lower right')

    # Final layout
    plt.tight_layout()
    plt.savefig(config.figure_path / 'baseline_accessibility_agri_road_port_rail_3x2.png', dpi=150, transparent=True)
    if config.show_figures == True:
        plt.show()


def plot_access_time_agriculture_map(df_agri: pd.DataFrame, Sinks: pd.DataFrame, config: NetworkConfig) -> None:
    """
    Plot three maps of average access time from agricultural areas to road border crossings,
    ports, and rail terminals, and save the figure as a PNG.

    Parameters
    ----------
    df_agri : pd.DataFrame or gpd.GeoDataFrame
        Agricultural features with average access-time columns:
        ['avg_access_road', 'avg_access_port', 'avg_access_rail'] (hours) and geometry.
    Sinks : pd.DataFrame or gpd.GeoDataFrame
        Locations of sinks with 'type' in {'road','port','rail'} and a 'geometry' column.
    config : NetworkConfig
        Provides output path (`figure_path`) and `show_figures` toggle.

    Behavior
    --------
    Classifies access times into bands (1-2, 2-3, 3-4, 4-5, 5+ hours), renders three side-by-side
    maps on a basemap, adds panel labels (A-C) and a shared legend, and saves
    'agriculture_access_by_type.png'. Optionally displays the plot when enabled.
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
    plt.savefig(config.Path_agriculture_access_by_type, dpi=200, bbox_inches='tight')
    if config.show_figures == True:
        plt.show()



def plot_access_time_map(df_worldpop: gpd.GeoDataFrame, Sink: pd.DataFrame, config: NetworkConfig, emergency_service) -> gpd.GeoDataFrame:
    """
    Plot an access-time map for an emergency service and save the figure as a PNG.

    Parameters
    ----------
    df_worldpop : gpd.GeoDataFrame
        Settlement or population points with 'closest_sink_total_fft' (hours) and geometry.
    Sink : pd.DataFrame or gpd.GeoDataFrame
        Facility locations (fire, hospital, police) with a 'geometry' column.
    config : NetworkConfig
        Holds output directory (`figure_path`) and `show_figures` flag.
    emergency_service : str
        One of {"firefighters", "hospitals", "police"}; sets legend label and filename.

    Behavior
    --------
    Classifies travel times into bands (0-15, 15-30, 30-60, >60 min), renders
    population points colored by category, overlays facility markers, adds a styled
    legend, saves the output map, and optionally displays it.
    """
    
    # Prepare data
    df_worldpop_plot = gpd.GeoDataFrame(df_worldpop, geometry='geometry', crs="EPSG:4326").to_crs(3857)
    Sink_fire = gpd.GeoDataFrame(Sink, geometry='geometry', crs="EPSG:4326").to_crs(3857)

    # Create bins for categories (1-hour intervals)
    bins = [0, 0.25, 0.5, 1, float('inf')]
    labels = ['0-15', '15-30', '30-60', '>60']
    colors = ['#4cc9f0','#4895ef', '#4361ee', '#3f37c9', '#3a0ca3']

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
    fig, ax = plt.subplots(figsize=(18, 12))

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
        file_name = config.Path_firefighter_access_map

    elif emergency_service == "hospitals":
        legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='hospitals', markersize=15))
        file_name = config.Path_hospital_access_map

    elif emergency_service == "police":
        legend_patches.append(Line2D([0], [0], marker='+', color='red', lw=0, 
                                label='police stations', markersize=15))
        file_name = config.Path_police_station_access_map

    else:
        raise ValueError(
            f"Invalid sink_type '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    
    # Add legend
    legend  = ax.legend(handles=legend_patches, 
          loc='upper right',
          fontsize=12,
          title='Access Time',
          title_fontsize=14,
          frameon=True,
          fancybox=True,
          shadow=True,
          framealpha=0.95)

    legend.get_title().set_fontweight('bold')

    plt.savefig(file_name, dpi=200, bbox_inches='tight')

    if config.show_figures == True:
        plt.show()

    return df_worldpop_plot



def plot_accessibility_chart(df_worldpop_plot: gpd.GeoDataFrame, config: NetworkConfig, emergency_service) -> None:
    """
    Plot a horizontal bar chart (with inset pie) showing population distribution by
    access-time category for an emergency service, and save the figure as a PNG.

    Parameters
    ----------
    df_worldpop_plot : gpd.GeoDataFrame
        Must contain 'population' and a categorical 'category' column with
        labels like ['0-15','15-30','30-60','60-90','90-120','>120'].
    config : NetworkConfig
        Provides output directory (`figure_path`) and `show_figures` toggle.
    emergency_service : str
        One of {"firefighters","hospitals","police"}; sets y-axis label and filename.

    Behavior
    --------
    Aggregates population (in millions) per category, renders a styled horizontal
    bar chart (reversed order for readability) with a matching inset pie,
    then saves a service-specific PNG to `figure_path` and optionally displays it.
    """

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
    Run the post-processing and visualization workflow for accessibility results.
    Loads facility-specific outputs (factories, agriculture, firefighters, hospitals,
    police) from configured paths, generates cumulative curves, maps, and summary
    charts, and saves all figures to `config.figure_path`. Progress messages are
    printed during execution; the function returns no value.
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

    #plot baseline accessibility curves for agricultural areas (shortest and average to borders, rail terminals and ports)
    plot_accessibility_curves_agriculture(df_agriculture_accessibility, config)

    #create map of the access times of factories to road border crossings
    plot_access_time_agriculture_map(df_agriculture_accessibility, df_agriculture_sinks, config)
    
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