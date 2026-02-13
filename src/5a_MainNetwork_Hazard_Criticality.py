# Standard library
import os
import re
import sys
import warnings
from pathlib import Path

# Third-party - Data and scientific computing
import contextily as cx
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from tqdm import tqdm
from typing import Tuple
from damagescanner.core import DamageScanner

#  Shapely-specific imports for spatial analysis
import shapely
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, snap

# Matplotlib-specific imports for figures
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
from config.network_config import NetworkConfig


def load_data(config: NetworkConfig) -> Tuple[gpd.GeoDataFrame, xr.Dataset]:
    """
    Load criticality results and clip the flood hazard raster to the target country's extent.

    Parameters
    ----------
    config : NetworkConfig
        Must provide:
        - data_path : Directory containing the countries shapefile and flood raster
        - intermediate_results_path : Directory containing 'criticality_results.parquet'

    Returns
    -------
    gpd.GeoDataFrame
        Criticality results loaded from the parquet file.
    xr.Dataset
        Flood hazard raster clipped to the selected country's bounding box.
    gpd.GeoDataFrame
        Geometry of the country for plotting

    Notes
    -----
    - Countries shapefile expected at: 'ne_10m_admin_0_countries.shp'
    - Flood raster expected at: 'Europe_RP100_filled_depth.tif'
    - Country selection currently filters by ISO A3 code 'SRB'. Make this a parameter
      if you want to reuse for other countries.
    """
    # Load results
    gdf_results = gpd.read_parquet(
        config.intermediate_results_path / "criticality_results.parquet"
    )

    # Load countries and select target country
    countries = gpd.read_file(config.data_path / "ne_10m_admin_0_countries.shp")
    country = countries.loc[countries.SOV_A3 == "SRB"]
    country_plot = countries.loc[countries.SOV_A3 == 'SRB']

    # Compute bounding box [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = country.total_bounds

    # Load flood raster and clip to country bbox
    hazard = xr.open_dataset(
        config.data_path / "Europe_RP100_filled_depth.tif",
        engine="rasterio"
    )
    hazard_clipped = hazard.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    ).load()

    return gdf_results, hazard_clipped, country_plot


def flagged_exposed_segments(row: pd.Series):
    """
    Return True if any flood depth on a road segment exceeds 0.25 meters.

    Parameters
    ----------
    row : pandas.Series
        Must contain:
        - 'values' : iterable of flood depths (floats, in meters)

    Returns
    -------
    bool
        True if any depth > 0.25 m, meaning the road is considered impassable.
    """
    return any(val > 0.25 for val in row["values"])


def max_depth(row):
    """
    Return the maximum flood depth for a given road segment.
    """
    return np.max(row["values"])


def calculate_flood_impact(gdf_results: gpd.GeoDataFrame, flood_data: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Calculate flood impact on roads and return results for segments where
    vehicle-hours-lost (VHL) occur due to impassable flooding.

    Parameters
    ----------
    gdf_results: gpd.GeoDataFrame
        Contains the results of the criticality analysis
        and the vhl lost for each road section 
    flood_data: xr.Dataset
        Contains flood map for the country of interest

    Returns
    -------
    gdf_vhl_flooded: gpd.GeoDataFrame
        contains road segements that cause vehicle hours
        lost due to flooding

    """
    exposed_roads = DamageScanner(
        flood_data,
        gdf_results,
        curves=pd.DataFrame(),
        maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False)

    exposed_roads["exposed"] = exposed_roads.progress_apply(flagged_exposed_segments, axis=1)
    exposed_roads["max_depth"] = exposed_roads.progress_apply(max_depth, axis=1)

    gdf_vhl_flooded = gdf_results.merge(
        exposed_roads.loc[
            exposed_roads["exposed"], ["coverage", "values", "max_depth"]
        ],
        left_index=True,
        right_index=True
    )

    return gdf_vhl_flooded


def plot_flood_impact_map(gdf_vhl_flooded: gpd.GeoDataFrame, config: NetworkConfig) -> None:
    
    """
    Plot vehicle-hours-lost (VHL) through flood impact on roads and save figure as PNG. 
    """

    # Define bins based on vehicle hours lost distribution - heavily skewed toward 0
    bins = [0, 1000, 5000, 10000, 25000, np.inf]
    labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']

    # Create binned column
    gdf_vhl_flooded['vhl_class'] = pd.cut(
        gdf_vhl_flooded['vhl'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )

    # Define line widths for each class (higher VHL = thicker lines)
    linewidth_map = {
        '0-1K': 0.5,
        '1K-5K': 1.0,
        '5K-10K': 2.0,
        '10K-25K': 3.5,
        '25K+': 5.0
    }

    # Create a linewidth column
    gdf_vhl_flooded['linewidth'] = gdf_vhl_flooded['vhl_class'].map(linewidth_map)

    # Create figure with high DPI 
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='white')

    # Plot each class separately with both width and color variation
    # Using red-orange color progression for impact severity
    colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']

    for i, (class_name, width) in enumerate(linewidth_map.items()):
        subset = gdf_vhl_flooded[gdf_vhl_flooded['vhl_class'] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(
                ax=ax,
                color=colors[i],
                linewidth=width,
                alpha=0.8,
                label=class_name
            )

    # Add basemap with optimal styling
    cx.add_basemap(ax=ax,
        source=cx.providers.CartoDB.Positron,
                    attribution=False)


    # Enhance the plot styling
    ax.set_aspect('equal')
    ax.axis('off')

    # Create custom legend with line samples that show both width and color
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=width, label=f'{class_name} vehicle hours')
                    for i, (class_name, width) in enumerate(linewidth_map.items())]

    legend = ax.legend(handles=legend_elements, 
                    title='Vehicle Hours Lost', 
                    loc='upper right',
                    fontsize=9,
                    title_fontsize=10,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    framealpha=0.9,
                    facecolor='white',
                    edgecolor='#cccccc')

    # Enhance overall plot appearance
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(config.figure_path / 'vehicle_hours_lost_map_flooded.png', dpi=300, bbox_inches='tight')
    if config.show_figures == True:
        plt.show()



def read_snowdrift_data(config: NetworkConfig) -> gpd.GeoDataFrame:   
    """
    Read snow drift data shapefile.

    Parameters
    ----------
    config : NetworkConfig
        Expects:
        - data_path : pathlib.Path to the directory containing 'snezni_nanosi_studije.shp'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with snow drift features.
    """

    snow_drift = gpd.read_file(config.data_path / "snezni_nanosi_studije.shp")

    return snow_drift


def calculate_vhl_snow_drift(gdf_results: gpd.GeoDataFrame, snow_drift: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    
    """
    Spatially join road segments with snow drift polygons and compute
    the Vehicle Hours Lost (VHL) subset for snow drift-affected roads.

    Parameters
    ----------
    gdf_results : GeoDataFrame
        Road network with traffic attributes and geometry.

    snow_drift : GeoDataFrame
        Snow drift features containing drift lengths and polygons/lines.

    Returns
    -------
    GeoDataFrame
        Road segments intersecting snow drift features, including:
        - traffic attributes
        - VHL/PHL/THL/PKL/TKL disruption values
        - drift attributes inherited from `snow_drift`
    """


    gdf_vhl_snowdrift = gdf_results[['from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
        'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
        'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry', 'id',
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'speed', 'fft','edge_no', 'vhl','phl','thl','pkl','tkl']].sjoin(snow_drift)
    
    return gdf_vhl_snowdrift


def read_landslide_data(config: NetworkConfig) -> gpd.GeoDataFrame: 
    """
    Read landslide data shapefile.

    Parameters
    ----------
    config : NetworkConfig
        Expects:
        - data_path : pathlib.Path to the directory containing "Nestabilne_pojave.shp"

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with landslide features.
    """


    landslides = gpd.read_file(config.data_path / "Nestabilne_pojave.shp")
    landslides.geometry = landslides.geometry.buffer(10)

    return landslides


def calculate_vhl_landslides(gdf_results: gpd.GeoDataFrame, landslides: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    
    """
    Spatially join road segments with landslide polygons and compute
    the Vehicle Hours Lost (VHL) subset for landslide-affected roads.

    Parameters
    ----------
    gdf_results : GeoDataFrame
        Road network with traffic attributes and geometry.

    landslides : GeoDataFrame
        Landslide features.

    Returns
    -------
    GeoDataFrame
        Road segments intersecting landslide features, including:
        - traffic attributes
        - VHL/PHL/THL/PKL/TKL disruption values
        - drift attributes inherited from `landslides`
    """


    gdf_vhl_landslides = gdf_results[['from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
        'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
        'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry', 'id',
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'speed', 'fft','edge_no','vhl','phl','thl','pkl','tkl']].sjoin(landslides)

    return gdf_vhl_landslides



def calculate_combined_hazard(gdf_results: gpd.GeoDataFrame, gdf_vhl_flooded: gpd.GeoDataFrame, gdf_vhl_snowdrift: gpd.GeoDataFrame, gdf_vhl_landslides: gpd.GeoDataFrame, config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Combine flood, snowdrift, and landslide hazard attributes into a unified
    hazard exposure GeoDataFrame.

    Parameters
    ----------
    gdf_results : GeoDataFrame
        Base road network with traffic and disruption data.

    gdf_vhl_flooded : GeoDataFrame
        Contains flood-affected road segments with:
        - max_depth : float
        - values    : array-like flood depths
        - coverage  : array-like coverage mask

    gdf_vhl_snowdrift : GeoDataFrame
        Contains snow drift-affected segments with:
        - dužina_sn : float (snow drift length in meters)

    gdf_vhl_landslides : GeoDataFrame
        Contains landslide-affected segments with:
        - datum_evid : datetime64 (event date)

    config : NetworkConfig
        contains directory for saving the results

    Returns
    -------
    GeoDataFrame
        Combined hazard exposure dataset with one row per road segment.
    """

    # pick an aggregation rule for duplicates; here we use 'max' as an example
    s_depth = gdf_vhl_flooded['max_depth'].groupby(level=0).max()
    s_snow  = gdf_vhl_snowdrift['dužina_sn'].groupby(level=0).max()
    s_date  = gdf_vhl_landslides['datum_evid'].groupby(level=0).max()

    # align everything to the base index
    gdf_hazards = pd.concat([gdf_results, s_depth.rename('max_depth'),
                            s_snow.rename('dužina_sn'),
                            s_date.rename('datum_evid')], axis=1)

    # convert date to string dd/mm/yyyy (NaT -> NaN -> optional empty string)
    gdf_hazards['datum_evid'] = gdf_hazards['datum_evid'].dt.strftime('%d/%m/%Y')



    keep_attrs = ['oznaka_deo', 'smer_gdf1', 'kategorija',
        'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
        'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry',
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'average_time_disruption','vhl','phl','thl','pkl','tkl','max_depth', 'dužina_sn', 'datum_evid']
    gdf_hazards = gdf_hazards[keep_attrs]
    gdf_hazards = gdf_hazards.loc[gdf_hazards[['max_depth', 'dužina_sn', 'datum_evid']].any(axis=1)]
    gdf_hazards = gdf_hazards.loc[gdf_hazards['vhl'].notna()]

    mask = (
        (gdf_hazards['max_depth'].fillna(0) > 0) &
        (gdf_hazards['dužina_sn'].fillna(0) > 0) &
        (gdf_hazards['datum_evid'].notna()))

    affected_all = gdf_hazards.loc[mask]


    gdf_hazards.to_parquet(config.intermediate_results_path / "main_network_hazard_exposure.parquet")

    return gdf_hazards

def plot_vehicle_hours_lost_per_hazard(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    config: NetworkConfig
) -> None:
    """
    Create a 2×2 comparison figure showing vehicle-hours-lost (VHL) for
    flooding, snow drift, and landslides. Each hazard is plotted in its own
    subfigure with a title, and the fourth panel contains a shared legend.
    The final figure is saved to `config.figure_path` and optionally shown.

    Parameters
    ----------
    gdf_vhl_flooded : GeoDataFrame
        Road segments affected by flooding, must contain `vhl` and geometry.
    gdf_vhl_snowdrift : GeoDataFrame
        Road segments affected by snow drift, containing `vhl` and geometry.
    gdf_vhl_landslides : GeoDataFrame
        Road segments affected by landslides, containing `vhl` and geometry.
    country_plot : GeoDataFrame
        Country or study area boundary used as background.
    config : NetworkConfig
        Holds figure output paths and plotting/display settings.

    Returns
    -------
    None
    """
    # VHL bins + styles
    bins = [0, 1000, 5000, 10000, 25000, float("inf")]
    labels = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    linewidth_map = {
        "0-1K": 1.0,
        "1K-5K": 1.5,
        "5K-10K": 2.0,
        "10K-25K": 3.5,
        "25K+": 5.0
    }

    # Hazard datasets
    datasets = {
        "A": ("Floods", gdf_vhl_flooded),
        "B": ("Snow Drift", gdf_vhl_snowdrift),
        "C": ("Landslides", gdf_vhl_landslides)
    }

    # Bin VHL values
    for _, (_, gdf) in datasets.items():
        gdf["vhl_class"] = pd.cut(
            gdf["vhl"], bins=bins, labels=labels, include_lowest=True
        )

    # Layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor="white")
    axes = axes.flatten()

    country_mercator = country_plot.to_crs(3857)

    # Plot hazard panels
    for idx, (letter, (title, gdf)) in enumerate(datasets.items()):
        ax = axes[idx]
        gdf_merc = gdf.to_crs(3857)

        # Outline
        country_mercator.plot(
            ax=ax, facecolor="none", edgecolor="#333333",
            linewidth=1.5, zorder=1
        )

        # VHL classes
        for i, (class_name, width) in enumerate(linewidth_map.items()):
            subset = gdf_merc[gdf_merc["vhl_class"] == class_name]
            if not subset.empty:
                subset.plot(
                    ax=ax, color=colors[i], linewidth=width, zorder=2
                )

        # Basemap
        cx.add_basemap(
            ax=ax,
            source=cx.providers.CartoDB.Positron,
            attribution=False
        )

        ax.set_aspect("equal")
        ax.axis("off")

        # Subfigure title
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

        # Subfigure identifier (A/B/C)
        ax.text(
            0.05, 0.95, letter,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.8)
        )

    # Legend panel
    ax_legend = axes[3]
    ax_legend.axis("off")

    legend_elements = [
        Line2D(
            [0], [0], color=colors[i], lw=width * 1.5,
            label=f"{class_name} vehicle hours"
        )
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]

    ax_legend.legend(
        handles=legend_elements,
        title="Vehicle Hours Lost",
        loc="center",
        fontsize=14,
        title_fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc"
    )

    plt.tight_layout()
    plt.savefig(
        config.figure_path / "vhl_hazards_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    if config.show_figures:
        plt.show()



def plot_passenger_hours_lost_per_hazard(gdf_vhl_flooded: gpd.GeoDataFrame, gdf_vhl_snowdrift: gpd.GeoDataFrame, gdf_vhl_landslides: gpd.GeoDataFrame, country_plot: gpd.GeoDataFrame, config: NetworkConfig) -> None:
    """
    Create a comparison figure with three maps showing passenger-hours-lost (PHL) impacts from
    multiple hazard types (floods, snow drift, and landslides), and save the
    resulting figure to disk.
    
    Parameters
    ----------
    gdf_vhl_flooded : GeoDataFrame
        Road segments affected by flooding.
    gdf_vhl_snowdrift : GeoDataFrame
        Road segments affected by snow drift.
    gdf_vhl_landslides : GeoDataFrame
        Road segments affected by landslides.
    country_plot : GeoDataFrame
        Boundary outline of the study region.
    config : NetworkConfig
        Configuration with output paths and display settings.

    Returns
    -------
    None
    """

    # Define bins and styling for PHL
    bins_phl = [0, 1000, 5000, 10000, 25000, np.inf]
    labels_phl = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']
    colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    linewidth_map = {'0-1K': 1, '1K-5K': 1.5, '5K-10K': 2.0, '10K-25K': 2.5, '25K+': 3.0}

    # Prepare all three datasets for PHL
    datasets_phl = {
        'A': ('Floods', gdf_vhl_flooded),
        'B': ('Snow Drift', gdf_vhl_snowdrift),
        'C': ('Landslides', gdf_vhl_landslides)
    }

    # Bin all datasets
    for letter, (title, gdf) in datasets_phl.items():
        gdf['phl_class'] = pd.cut(gdf['phl'], bins=bins_phl, labels=labels_phl, include_lowest=True)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor='white')
    axes = axes.flatten()

    # Convert country boundary once
    serbia_mercator = country_plot.to_crs(3857)

    # Plot the three hazard maps
    for idx, (letter, (title, gdf)) in enumerate(datasets_phl.items()):
        ax = axes[idx]
        gdf_mercator = gdf.to_crs(3857)
        
        # Plot country outline
        serbia_mercator.plot(ax=ax, facecolor='none', edgecolor='#333333', 
                            linewidth=1.5, zorder=1)
        
        # Plot each PHL class
        for i, (class_name, width) in enumerate(linewidth_map.items()):
            subset = gdf_mercator[gdf_mercator['phl_class'] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)
        
        # Add basemap
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, 
                    attribution=False)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title and letter label
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.05, 0.95, f'{letter}', transform=ax.transAxes, fontsize=20, 
                fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Use the 4th panel for the legend
    ax_legend = axes[3]
    ax_legend.axis('off')

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f'{class_name} hours')
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]

    ax_legend.legend(handles=legend_elements, 
                    title='Passenger Hours Lost',
                    loc='center',
                    fontsize=14,
                    title_fontsize=16,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    framealpha=0.9,
                    facecolor='white',
                    edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(config.figure_path / 'phl_hazards_comparison.png', dpi=300, bbox_inches='tight')
    if config.show_figures == True: 
        plt.show()


def plot_tonnage_kilometers_lost_per_hazard(gdf_vhl_flooded: gpd.GeoDataFrame, gdf_vhl_snowdrift: gpd.GeoDataFrame, gdf_vhl_landslides: gpd.GeoDataFrame, country_plot: gpd.GeoDataFrame, config: NetworkConfig) -> None:
    """
    Create a comparison figure with three maps showing tonnage-kilometers-lost (TKL) impacts from
    multiple hazard types (floods, snow drift, and landslides), and save the
    resulting figure to disk.
    
    Parameters
    ----------
    gdf_vhl_flooded : GeoDataFrame
        Road segments affected by flooding.
    gdf_vhl_snowdrift : GeoDataFrame
        Road segments affected by snow drift.
    gdf_vhl_landslides : GeoDataFrame
        Road segments affected by landslides.
    country_plot : GeoDataFrame
        Boundary outline of the study region.
    config : NetworkConfig
        Configuration with output paths and display settings.

    Returns
    -------
    None
    """

    # Convert country boundary
    serbia_mercator = country_plot.to_crs(3857)

    # Define bins and styling for TKL
    colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    bins_tkl = [10000, 25000, 50000, 100000, 250000, np.inf]
    labels_tkl = ['5K-25K', '25K-50K', '50K-100K', '100K-250K', '250K+']
    linewidth_map_tkl = {
        '10-25K': 0.5, '25K-50K': 1.5, '50K-100K': 2.0, 
        '100K-250K': 2.5, '250K+': 3.0
    }
    # Prepare all three datasets for TKL
    datasets_tkl = {
        'A': ('Floods', gdf_vhl_flooded),
        'B': ('Snow Drift', gdf_vhl_snowdrift),
        'C': ('Landslides', gdf_vhl_landslides)
    }

    # Bin all datasets
    for letter, (title, gdf) in datasets_tkl.items():
        gdf['tkl_class'] = pd.cut(gdf['tkl'], bins=bins_tkl, labels=labels_tkl, include_lowest=True)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor='white')
    axes = axes.flatten()

    # Plot the three hazard maps
    for idx, (letter, (title, gdf)) in enumerate(datasets_tkl.items()):
        ax = axes[idx]
        gdf_mercator = gdf.to_crs(3857)
        
        # Plot country outline
        serbia_mercator.plot(ax=ax, facecolor='none', edgecolor='#333333', 
                            linewidth=1.5, zorder=1)
        
        # Plot each TKL class
        for i, (class_name, width) in enumerate(linewidth_map_tkl.items()):
            subset = gdf_mercator[gdf_mercator['tkl_class'] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)
        
        # Add basemap
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, 
                    alpha=0.4, attribution=False)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title and letter label
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.05, 0.95, f'{letter}', transform=ax.transAxes, fontsize=20, 
                fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Use the 4th panel for the legend
    ax_legend = axes[3]
    ax_legend.axis('off')

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f'{class_name} ton-km')
        for i, (class_name, width) in enumerate(linewidth_map_tkl.items())
    ]

    ax_legend.legend(handles=legend_elements, 
                    title='Tonnage Kilometers Lost',
                    loc='center',
                    fontsize=14,
                    title_fontsize=16,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    framealpha=0.9,
                    facecolor='white',
                    edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(config.figure_path / 'tkl_hazards_comparison.png', dpi=300, bbox_inches='tight')
    if config.show_figures:
        plt.show()



def print_analysis_summary(gdf_vhl_flooded: gpd.GeoDataFrame, gdf_vhl_snowdrift: gpd.GeoDataFrame, gdf_vhl_landslides: gpd.GeoDataFrame, gdf_results: gpd.GeoDataFrame) -> None:
    
    """
    This function writes hazard-specific summaries of disruption metrics across the road network 
    to the console. Includes descriptive statistics, category distributions, top critical segments, 
    overlap analysis, and comparisons with national daily transport benchmarks.
    Output is not saved to disk. 

    Parameters
    ----------
    gdf_vhl_flooded : gpd.GeoDataFrame
        Flood-exposed road segments with disruption metrics. Must contain:
        - 'phl' (float): Passenger-hours lost
        - 'thl' (float): Tonnage-hours lost
        - 'pkl' (float): Passenger-kilometers lost
        - 'tkl' (float): Tonnage-kilometers lost
        - Optional: 'kategorija', 'oznaka_deo', 'oznaka_put', 'naziv_poce', 'naziv_zavr', 'total_aadt'
    gdf_vhl_snowdrift : gpd.GeoDataFrame
        Snow-drift-exposed road segments with the same fields as above.
    gdf_vhl_landslides : gpd.GeoDataFrame
        Landslide-exposed road segments with the same fields as above.
    gdf_results : gpd.GeoDataFrame
        Baseline network results over all segments, used for global statistics and totals.

    Outputs
    -------
    None
        Prints to console:
        - Baseline metric summaries (count, mean, std, min, quartiles, max) for PHL/THL/PKL/TKL
        - Per-hazard statistics and totals for PHL and TKL, with binned distributions
        - Additional per-hazard summaries for THL and PKL
        - Top critical segments by PHL and TKL (up to 3), if identifying columns exist
        - Overlap analysis of segments exposed to multiple hazards (exclusive/dual/triple)
        - Comparison of exposed PKL/TKL as % of national daily benchmarks

    Notes
    -----
    - Uses fixed binning schemes for PHL, THL, PKL, and TKL to derive category counts and shares.
    - Expects numeric columns free of non-numeric strings (NaNs are handled by pandas .describe()).
    - No files are saved; this function is purely for console reporting.
    """
    
    
    # ============================================================
    # HAZARD EXPOSURE ANALYSIS - UPDATED FOR PHL, THL, PKL, TKL
    # ============================================================

    print("=" * 70)
    print("BASELINE RESULTS - ALL METRICS")
    print("=" * 70)

    print(f"\nTotal road segments analyzed: {len(gdf_results):,}")

    # Summary of all metrics
    metrics_summary = gdf_results[['phl', 'thl', 'pkl', 'tkl']].describe()
    print("\nAll Metrics Summary:")
    print(metrics_summary.round(2))

    print(f"\nTotals across network:")
    print(f"  Total PHL: {gdf_results['phl'].sum():,.0f} passenger hours")
    print(f"  Total THL: {gdf_results['thl'].sum():,.0f} ton hours")
    print(f"  Total PKL: {gdf_results['pkl'].sum():,.0f} passenger km")
    print(f"  Total TKL: {gdf_results['tkl'].sum():,.0f} ton km")


    # ============================================================
    # HAZARD-SPECIFIC ANALYSIS
    # ============================================================

    hazard_datasets = {
        'Floods': gdf_vhl_flooded,
        'Snow Drift': gdf_vhl_snowdrift,
        'Landslides': gdf_vhl_landslides
    }

    # Define bins for each metric
    bins_phl = [0, 1000, 5000, 10000, 25000, np.inf]
    labels_phl = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']

    bins_thl = [0, 500, 2500, 5000, 10000, np.inf]
    labels_thl = ['0-500', '500-2.5K', '2.5K-5K', '5K-10K', '10K+']

    bins_pkl = [0, 10000, 50000, 100000, 250000, np.inf]
    labels_pkl = ['0-10K', '10K-50K', '50K-100K', '100K-250K', '250K+']

    bins_tkl = [0, 10000, 50000, 100000, 250000, np.inf]
    labels_tkl = ['0-10K', '10K-50K', '50K-100K', '100K-250K', '250K+']


    # ============================================================
    # PASSENGER HOURS LOST (PHL) - FIGURE 19
    # ============================================================

    print("\n" + "=" * 70)
    print("PASSENGER HOURS LOST (PHL) BY HAZARD - FIGURE 19")
    print("=" * 70)

    phl_summary = []

    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}")
        print(f"{hazard_name.upper()}")
        print(f"{'─' * 50}")
        
        print(f"Exposed road segments: {len(gdf):,}")
        print(f"\nPHL Statistics:")
        print(gdf['phl'].describe().round(2))
        
        print(f"\nTotal PHL: {gdf['phl'].sum():,.0f} passenger hours")
        print(f"Mean PHL: {gdf['phl'].mean():,.2f} passenger hours")
        print(f"Median PHL: {gdf['phl'].median():,.2f} passenger hours")
        print(f"Max PHL: {gdf['phl'].max():,.2f} passenger hours")
        
        # By category
        gdf['phl_class'] = pd.cut(gdf['phl'], bins=bins_phl, labels=labels_phl, include_lowest=True)
        print(f"\nSegments by PHL category:")
        phl_counts = gdf['phl_class'].value_counts().sort_index()
        phl_pcts = (gdf['phl_class'].value_counts(normalize=True).sort_index() * 100).round(1)
        for label in labels_phl:
            count = phl_counts.get(label, 0)
            pct = phl_pcts.get(label, 0)
            print(f"  {label}: {count} ({pct}%)")
        
        # By road category
        if 'kategorija' in gdf.columns:
            print(f"\nPHL by road category:")
            road_summary = gdf.groupby('kategorija')['phl'].agg(['count', 'sum', 'mean', 'max']).round(2)
            road_summary.columns = ['Count', 'Total PHL', 'Mean PHL', 'Max PHL']
            print(road_summary.sort_values('Total PHL', ascending=False))
        
        # Top 5 critical segments
        print(f"\nTop 5 most critical segments (PHL):")
        cols = ['oznaka_deo', 'kategorija', 'naziv_poce', 'naziv_zavr', 'total_aadt', 'phl']
        cols = [c for c in cols if c in gdf.columns]
        print(gdf.nlargest(5, 'phl')[cols].to_string())
        
        # Collect summary
        phl_summary.append({
            'Hazard': hazard_name,
            'Exposed Segments': len(gdf),
            'Total PHL': gdf['phl'].sum(),
            'Mean PHL': gdf['phl'].mean(),
            'Median PHL': gdf['phl'].median(),
            'Max PHL': gdf['phl'].max(),
            'Segments >25K': len(gdf[gdf['phl'] >= 25000]),
            'Segments >10K': len(gdf[gdf['phl'] >= 10000])
        })

    print("\n" + "─" * 50)
    print("PHL HAZARD COMPARISON SUMMARY")
    print("─" * 50)
    phl_summary_df = pd.DataFrame(phl_summary)
    print(phl_summary_df.to_string(index=False))


    # ============================================================
    # TONNAGE KILOMETERS LOST (TKL) - FIGURE 20
    # ============================================================

    print("\n" + "=" * 70)
    print("TONNAGE KILOMETERS LOST (TKL) BY HAZARD - FIGURE 20")
    print("=" * 70)

    tkl_summary = []

    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}")
        print(f"{hazard_name.upper()}")
        print(f"{'─' * 50}")
        
        print(f"Exposed road segments: {len(gdf):,}")
        print(f"\nTKL Statistics:")
        print(gdf['tkl'].describe().round(2))
        
        print(f"\nTotal TKL: {gdf['tkl'].sum():,.0f} ton-km")
        print(f"Mean TKL: {gdf['tkl'].mean():,.2f} ton-km")
        print(f"Median TKL: {gdf['tkl'].median():,.2f} ton-km")
        print(f"Max TKL: {gdf['tkl'].max():,.2f} ton-km")
        
        # By category
        gdf['tkl_class'] = pd.cut(gdf['tkl'], bins=bins_tkl, labels=labels_tkl, include_lowest=True)
        print(f"\nSegments by TKL category:")
        tkl_counts = gdf['tkl_class'].value_counts().sort_index()
        tkl_pcts = (gdf['tkl_class'].value_counts(normalize=True).sort_index() * 100).round(1)
        for label in labels_tkl:
            count = tkl_counts.get(label, 0)
            pct = tkl_pcts.get(label, 0)
            print(f"  {label}: {count} ({pct}%)")
        
        # By road category
        if 'kategorija' in gdf.columns:
            print(f"\nTKL by road category:")
            road_summary = gdf.groupby('kategorija')['tkl'].agg(['count', 'sum', 'mean', 'max']).round(2)
            road_summary.columns = ['Count', 'Total TKL', 'Mean TKL', 'Max TKL']
            print(road_summary.sort_values('Total TKL', ascending=False))
        
        # Top 5 critical segments
        print(f"\nTop 5 most critical segments (TKL):")
        cols = ['oznaka_deo', 'kategorija', 'naziv_poce', 'naziv_zavr', 'total_aadt', 'tkl']
        cols = [c for c in cols if c in gdf.columns]
        print(gdf.nlargest(5, 'tkl')[cols].to_string())
        
        # Collect summary
        tkl_summary.append({
            'Hazard': hazard_name,
            'Exposed Segments': len(gdf),
            'Total TKL': gdf['tkl'].sum(),
            'Mean TKL': gdf['tkl'].mean(),
            'Median TKL': gdf['tkl'].median(),
            'Max TKL': gdf['tkl'].max(),
            'Segments >250K': len(gdf[gdf['tkl'] >= 250000]),
            'Segments >100K': len(gdf[gdf['tkl'] >= 100000])
        })

    print("\n" + "─" * 50)
    print("TKL HAZARD COMPARISON SUMMARY")
    print("─" * 50)
    tkl_summary_df = pd.DataFrame(tkl_summary)
    print(tkl_summary_df.to_string(index=False))


    # ============================================================
    # ADDITIONAL METRICS: THL and PKL
    # ============================================================

    print("\n" + "=" * 70)
    print("TONNAGE HOURS LOST (THL) BY HAZARD")
    print("=" * 70)

    thl_summary = []
    for hazard_name, gdf in hazard_datasets.items():
        thl_summary.append({
            'Hazard': hazard_name,
            'Exposed Segments': len(gdf),
            'Total THL': gdf['thl'].sum(),
            'Mean THL': gdf['thl'].mean(),
            'Max THL': gdf['thl'].max(),
            'Segments >10K': len(gdf[gdf['thl'] >= 10000])
        })
    thl_summary_df = pd.DataFrame(thl_summary)
    print(thl_summary_df.to_string(index=False))


    print("\n" + "=" * 70)
    print("PASSENGER KILOMETERS LOST (PKL) BY HAZARD")
    print("=" * 70)

    pkl_summary = []
    for hazard_name, gdf in hazard_datasets.items():
        pkl_summary.append({
            'Hazard': hazard_name,
            'Exposed Segments': len(gdf),
            'Total PKL': gdf['pkl'].sum(),
            'Mean PKL': gdf['pkl'].mean(),
            'Max PKL': gdf['pkl'].max(),
            'Segments >250K': len(gdf[gdf['pkl'] >= 250000]),
            'Segments >500K': len(gdf[gdf['pkl'] >= 500000])
        })
    pkl_summary_df = pd.DataFrame(pkl_summary)
    print(pkl_summary_df.to_string(index=False))


    # ============================================================
    # OVERLAP ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS - MULTI-HAZARD EXPOSURE")
    print("=" * 70)

    if 'oznaka_deo' in gdf_vhl_flooded.columns:
        # Get segment identifiers - handle potential column name differences
        flooded_col = 'oznaka_deo' if 'oznaka_deo' in gdf_vhl_flooded.columns else 'oznaka_deo_left'
        snow_col = 'oznaka_deo' if 'oznaka_deo' in gdf_vhl_snowdrift.columns else 'oznaka_deo_left'
        landslide_col = 'oznaka_deo' if 'oznaka_deo' in gdf_vhl_landslides.columns else 'oznaka_deo_left'
        
        flooded_segments = set(gdf_vhl_flooded[flooded_col].dropna())
        snow_segments = set(gdf_vhl_snowdrift[snow_col].dropna())
        landslide_segments = set(gdf_vhl_landslides[landslide_col].dropna())
        
        print(f"\nTotal unique segments exposed to floods: {len(flooded_segments)}")
        print(f"Total unique segments exposed to snow drift: {len(snow_segments)}")
        print(f"Total unique segments exposed to landslides: {len(landslide_segments)}")
        
        print(f"\nExclusive exposure:")
        print(f"  Floods only: {len(flooded_segments - snow_segments - landslide_segments)}")
        print(f"  Snow drift only: {len(snow_segments - flooded_segments - landslide_segments)}")
        print(f"  Landslides only: {len(landslide_segments - flooded_segments - snow_segments)}")
        
        print(f"\nDual exposure:")
        print(f"  Floods AND snow drift: {len(flooded_segments & snow_segments - landslide_segments)}")
        print(f"  Floods AND landslides: {len(flooded_segments & landslide_segments - snow_segments)}")
        print(f"  Snow drift AND landslides: {len(snow_segments & landslide_segments - flooded_segments)}")
        
        print(f"\nTriple exposure:")
        print(f"  ALL three hazards: {len(flooded_segments & snow_segments & landslide_segments)}")


    # ============================================================
    # COMPARISON WITH NATIONAL STATISTICS
    # ============================================================

    print("\n" + "=" * 70)
    print("HAZARD EXPOSURE AS % OF NATIONAL DAILY TRANSPORT")
    print("=" * 70)

    national_pkm_daily = 2069 * 1e6 / 180  # ~11.5 million
    national_tkm_daily = 4677 * 1e6 / 180  # ~26 million

    print(f"\nNational daily averages (road transport, H1 2025):")
    print(f"  Passenger-km/day: {national_pkm_daily:,.0f}")
    print(f"  Ton-km/day: {national_tkm_daily:,.0f}")

    print(f"\nTotal exposed PKL as % of national daily:")
    for hazard_name, gdf in hazard_datasets.items():
        total_pkl = gdf['pkl'].sum()
        pct = total_pkl / national_pkm_daily * 100
        print(f"  {hazard_name}: {total_pkl:,.0f} PKL ({pct:.1f}% of national daily)")

    print(f"\nTotal exposed TKL as % of national daily:")
    for hazard_name, gdf in hazard_datasets.items():
        total_tkl = gdf['tkl'].sum()
        pct = total_tkl / national_tkm_daily * 100
        print(f"  {hazard_name}: {total_tkl:,.0f} TKL ({pct:.1f}% of national daily)")


    # ============================================================
    # KEY EXAMPLES FOR TEXT
    # ============================================================

    print("\n" + "=" * 70)
    print("KEY EXAMPLES FOR TEXT")
    print("=" * 70)

    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}")
        print(f"{hazard_name.upper()} - TOP CRITICAL SEGMENTS")
        print(f"{'─' * 50}")
        
        # Top by PHL
        print("\nTop 3 by Passenger Hours Lost:")
        cols = ['oznaka_deo', 'kategorija', 'oznaka_put', 'naziv_poce', 'naziv_zavr', 'total_aadt', 'phl', 'pkl']
        cols = [c for c in cols if c in gdf.columns]
        top_phl = gdf.nlargest(3, 'phl')[cols]
        for idx, row in top_phl.iterrows():
            route = f"{row.get('oznaka_put', 'N/A')} ({row.get('kategorija', 'N/A')})"
            section = f"{row.get('naziv_poce', 'N/A')} → {row.get('naziv_zavr', 'N/A')}"
            print(f"  {route}: {section}")
            print(f"    AADT: {row.get('total_aadt', 0):,.0f}, PHL: {row['phl']:,.0f}, PKL: {row.get('pkl', 0):,.0f}")
        
        # Top by TKL
        print("\nTop 3 by Tonnage Kilometers Lost:")
        cols = ['oznaka_deo', 'kategorija', 'oznaka_put', 'naziv_poce', 'naziv_zavr', 'total_aadt', 'thl', 'tkl']
        cols = [c for c in cols if c in gdf.columns]
        top_tkl = gdf.nlargest(3, 'tkl')[cols]
        for idx, row in top_tkl.iterrows():
            route = f"{row.get('oznaka_put', 'N/A')} ({row.get('kategorija', 'N/A')})"
            section = f"{row.get('naziv_poce', 'N/A')} → {row.get('naziv_zavr', 'N/A')}"
            print(f"  {route}: {section}")
            print(f"    AADT: {row.get('total_aadt', 0):,.0f}, THL: {row.get('thl', 0):,.0f}, TKL: {row['tkl']:,.0f}")


    # ============================================================
    # SUMMARY TABLE FOR TEXT
    # ============================================================

    print("\n" + "=" * 70)
    print("MASTER SUMMARY TABLE FOR TEXT")
    print("=" * 70)

    master_summary = []
    for hazard_name, gdf in hazard_datasets.items():
        master_summary.append({
            'Hazard': hazard_name,
            'Segments': len(gdf),
            'Total PHL': f"{gdf['phl'].sum():,.0f}",
            'Total THL': f"{gdf['thl'].sum():,.0f}",
            'Total PKL': f"{gdf['pkl'].sum():,.0f}",
            'Total TKL': f"{gdf['tkl'].sum():,.0f}",
            'Max PHL': f"{gdf['phl'].max():,.0f}",
            'Max TKL': f"{gdf['tkl'].max():,.0f}",
            'PHL >25K': len(gdf[gdf['phl'] >= 25000]),
            'TKL >250K': len(gdf[gdf['tkl'] >= 250000])
        })

    master_df = pd.DataFrame(master_summary)
    print(master_df.to_string(index=False))


def main():
    """
    Run the analysis of the exposure of the road network to multiple hazards (flooding, snowdrift and landslides)
    and calculate key accessibiliy metrics for each affected road section
    - Vehicle hours lost
    - Passenger hours lost
    - tonnage kilometers lost
    The results are visualized in maps showing the affected road sections that are saved as PNGs in the folder
    specified in config.figure_path and optionally displayed if config.show_figures is True. 
    If config.print_statistics is True, a summary of the results is printed to the console after the analysis as finished. 
    """

    #Load configureation from NetworkConfig class including file paths, and flags
    config = NetworkConfig()

    # Load the results of the network criticality analysis, flood hazard data and the geometry of the selected country
    criticality_results, flood_data, country_plot = load_data(config)

    #Analyse which road sections are affected by flooding and extract their values from the criticality analysis
    flood_results = calculate_flood_impact(criticality_results, flood_data)

    #Load snowdrift hazard data, analyse which road sections it affects and extract their criticality values 
    snow_drift_data = read_snowdrift_data(config)
    vhl_snow_drift = calculate_vhl_snow_drift(criticality_results, snow_drift_data)

    #Load landslide hazard data, analyse which road sections it affects and extract their criticality values
    landslide_data = read_landslide_data(config)
    vhl_landslides = calculate_vhl_landslides(criticality_results, landslide_data)

    # 4) Combine multi-hazard attributes onto baseline
    combined_hazard = calculate_combined_hazard(criticality_results, flood_results, vhl_snow_drift, vhl_landslides, config)

    #Visualize the different criticality metrics for each hazard. Figures are saved to config.figure_path and only displayed if config.show_figures is True
    plot_vehicle_hours_lost_per_hazard(flood_results, vhl_snow_drift, vhl_landslides, country_plot, config)
    plot_passenger_hours_lost_per_hazard(flood_results, vhl_snow_drift, vhl_landslides, country_plot, config)
    plot_tonnage_kilometers_lost_per_hazard(flood_results, vhl_snow_drift, vhl_landslides, country_plot, config)

    # Print summary of the analysis to the console if flag is set in the config class
    if config.print_statistics == True:
        print_analysis_summary(flood_results, vhl_snow_drift, vhl_landslides, criticality_results)



if __name__ == "__main__":
    main()

