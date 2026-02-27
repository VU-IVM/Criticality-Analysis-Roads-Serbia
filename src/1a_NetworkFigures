# Converted from 1a_NetworkFigures.ipynb
"""
Network Figures Generation Module

This module generates road network visualizations for Serbian roads and OSM data,
including network comparisons and length statistics.
"""

# Standard library
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Third-party - Data and scientific computing
import contextily as cx
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
# try if arcpy is available
try:
    import arcpy # type: ignore
    ARCPY_AVAILABLE = True
except ImportError:
    ARCPY_AVAILABLE = False

# Shapely-specific imports for spatial analysis
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


# Project root (repository root, one level above this src folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
print(f"Project root set to: {PROJECT_ROOT}")

@dataclass
class LocalConfig:
    """Configuration for network analysis and visualization."""
    
    # Input paths
    data_path = NetworkConfig.data_path
    
    # Output paths
    output_path = NetworkConfig.figure_path
    
    # ArcGIS parameters
    if ARCPY_AVAILABLE:
        arcgis_input_layer: Optional[str] = None
        arcgis_temp_base: Path = field(default_factory=lambda: Path(r"C:\Temp\arcgis_tmp"))
    else:
        # If no arcpyinput
        gis_input_layer = NetworkConfig.Original_road_network
        
    # OSM configuration
    osm_keys: List[str] = field(default_factory=lambda: [
        'highway', 'name', 'maxspeed', 'oneway', 'lanes', 'surface'
    ])
    osm_road_types: List[str] = field(default_factory=lambda: [
        'primary', 'trunk', 'motorway', 'motorway_link', 'trunk_link',
        'primary_link', 'secondary', 'secondary_link', 'tertiary',
        'tertiary_link', 'residential', 'road', 'unclassified', 'track'
    ])
    osm_plot_road_types: List[str] = field(default_factory=lambda: [
        'primary', 'trunk', 'motorway', 'motorway_link', 'trunk_link',
        'primary_link', 'secondary', 'secondary_link', 'tertiary',
        'tertiary_link', 'residential', 'road', 'unclassified'
    ])
    
    # Serbian road categories
    serbian_road_categories: List[str] = field(default_factory=lambda: ['IA', 'IM', 'IB', 'IIA', 'IIB'])
    
    # Road visualization colors
    osm_road_colors: Dict[str, str] = field(default_factory=lambda: {
        'motorway': '#8B0000', 'trunk': '#1E90FF', 'primary': '#A52A2A',
        'secondary': '#FFA500', 'tertiary': '#228B22', 'other': '#ccc5b9'
    })
    serbian_road_colors: Dict[str, str] = field(default_factory=lambda: {
        'IA': '#8B0000', 'IM': '#1E90FF', 'IB': '#A52A2A',
        'IIA': '#FFA500', 'IIB': '#228B22'
    })
    
    # Road visualization widths
    osm_road_widths: Dict[str, float] = field(default_factory=lambda: {
        'motorway': 3.0, 'trunk': 2.5, 'primary': 2.0,
        'secondary': 1.5, 'tertiary': 1.0, 'other': 0.6
    })
    serbian_road_widths: Dict[str, float] = field(default_factory=lambda: {
        'IA': 3.0, 'IM': 2.5, 'IB': 2.0, 'IIA': 1.5, 'IIB': 1.0
    })
    
    # Road hierarchy and labels
    osm_road_hierarchy: List[str] = field(default_factory=lambda: [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'other'
    ])
    osm_road_labels: Dict[str, str] = field(default_factory=lambda: {
        'motorway': 'Motorway (IA)', 'trunk': 'Trunk (IM)', 'primary': 'Primary (IB)',
        'secondary': 'Secondary (IIA)', 'tertiary': 'Tertiary (~IIB)', 'other': 'Other Roads'
    })
    
    # Visualization parameters
    figure_dpi: int = 300
    basemap_alpha: float = 0.4
    road_alpha: float = 0.8
    
    # CRS for length calculation
    length_crs: int = 8682
    
    @property
    def osm_path(self) -> Path:
        """Full path to OSM file."""
        return NetworkConfig.osm_path
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)


def _extract_value(text: str, key: str) -> Optional[str]:
    """
    Parse the value of a specific key from a semi-structured OSM tag string.

    Args:
        text (str): Raw OSM `other_tags` string.
        key (str): Key to extract value for.

    Returns:
        str or None: Extracted value or None.
    """
    pattern = rf'"{key}"=>"([^"]+)"'
    try:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None
    except:
        return None


def map_road_category(highway_type: str) -> str:
    """
    Map OSM highway type to a simplified road category.
    
    Args:
        highway_type: OSM highway tag value
        
    Returns:
        Simplified category name
    """
    if highway_type in ['motorway', 'motorway_link']:
        return 'motorway'
    elif highway_type in ['trunk', 'trunk_link']:
        return 'trunk'
    elif highway_type in ['primary', 'primary_link']:
        return 'primary'
    elif highway_type in ['secondary', 'secondary_link']:
        return 'secondary'
    elif highway_type in ['tertiary', 'tertiary_link']:
        return 'tertiary'
    elif highway_type in ['residential', 'road', 'unclassified', 'track']:
        return 'other'
    else:
        return highway_type


def load_osm_network(config: LocalConfig) -> gpd.GeoDataFrame:
    """
    Load and process OSM road network data.
    
    Args:
        config: Network configuration
        
    Returns:
        Processed GeoDataFrame with road network
    """
    # Load features from OSM
    features = gpd.read_file(config.osm_path, layer="lines")
    features = features[features["highway"].notna()] #In OSM, the highway tag identifies roads (motorway, primary, residential, etc.).
    
    # Extract OSM keys from other_tags field
    for key in config.osm_keys:
        if key not in features.columns:
            features[key] = features["other_tags"].apply(
                lambda x: _extract_value(x, key)
            )
    
    # Filter for road types
    base_network = features[features['highway'].isin(config.osm_road_types)].reset_index(drop=True)
    base_network = base_network[['osm_id', 'highway', 'name', 'maxspeed', 
                                  'oneway', 'lanes', 'surface', 'geometry']]
    
    return base_network


def prepare_osm_network_for_plotting(base_network: gpd.GeoDataFrame, 
                                      config: LocalConfig) -> gpd.GeoDataFrame:
    """
    Filter and categorize OSM network for visualization.
    
    Args:
        base_network: Raw OSM network
        config: Network configuration
        
    Returns:
        Filtered and categorized network
    """
    # Filter for plot-appropriate road types
    base_network_filtered = base_network[
        base_network['highway'].isin(config.osm_plot_road_types)
    ].copy()
    
    # Apply road category mapping
    base_network_filtered['road_category'] = base_network_filtered['highway'].apply(
        map_road_category
    )
    
    return base_network_filtered


def plot_osm_network(base_network_filtered: gpd.GeoDataFrame, 
                     config: LocalConfig) -> None:
    """
    Create and save OSM road network visualization.
    
    Args:
        base_network_filtered: Categorized OSM network
        config: Network configuration
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='white')
    
    # Convert to Web Mercator for basemap
    network_mercator = base_network_filtered.to_crs(3857)
    
    # Plot each road category (reverse order so higher priority roads on top)
    for category in reversed(config.osm_road_hierarchy):
        if category in base_network_filtered['road_category'].values:
            category_roads = network_mercator[network_mercator['road_category'] == category]
            category_roads.plot(
                ax=ax,
                color=config.osm_road_colors[category],
                linewidth=config.osm_road_widths[category],
                alpha=config.road_alpha,
                zorder=config.osm_road_widths[category]
            )
    
    # Add basemap
    cx.add_basemap(
        ax=ax, 
        source=cx.providers.CartoDB.Positron, 
        alpha=config.basemap_alpha,
        attribution=False
    )
    
    # Styling
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Create legend
    legend_elements = [
        Patch(facecolor=config.osm_road_colors[category], 
              label=config.osm_road_labels[category], 
              edgecolor='none')
        for category in config.osm_road_hierarchy
        if category in base_network_filtered['road_category'].values
    ]
    
    legend = ax.legend(
        handles=legend_elements, 
        title='OpenStreetMap', 
        loc='upper right',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#cccccc'
    )
    
    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(config.output_path / 'osm_road_network.png', 
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()

def load_serbian_network_no_arcpy(config: LocalConfig) -> gpd.GeoDataFrame:
    """
    Load Serbian road network from a GIS file path using GeoPandas.
    
    The file path is specified in ``config.gis_input_layer`` and is read using
    :func:`geopandas.read_file`.
    
    Args:
        config: Network configuration
    """    
    
    # Read into GeoDataFrame
    gdf = gpd.read_file(config.gis_input_layer)
    print(f"Successfully loaded feature layer.")
    
    return gdf

def load_serbian_network_arcpy(config: LocalConfig) -> gpd.GeoDataFrame:
    """
    Load Serbian road network from ArcGIS layer.
    
    Args:
        config: Network configuration
        
    Returns:
        GeoDataFrame with Serbian road network
    """
    print("Successfully loaded road network from file.")
    input_layer = config.arcgis_input_layer or arcpy.GetParameterAsText(0)
    
    # Setup temporary GDB
    config.arcgis_temp_base.mkdir(parents=True, exist_ok=True)
    gdb_path = config.arcgis_temp_base / "temp.gdb"
    
    if not arcpy.Exists(str(gdb_path)):
        arcpy.management.CreateFileGDB(str(config.arcgis_temp_base), "temp.gdb")
    
    out_fc_name = "roads"
    out_fc = gdb_path / out_fc_name
    
    # Copy input layer to GDB
    arcpy.management.CopyFeatures(input_layer, str(out_fc))
    
    # Read into GeoDataFrame
    gdf = gpd.read_file(str(gdb_path), layer=out_fc_name)
    arcpy.AddMessage(f"Successfully loaded feature layer.")
    
    return gdf


def plot_serbian_network(gdf: gpd.GeoDataFrame, config: LocalConfig) -> None:
    """
    Create and save Serbian road network visualization.
    
    Args:
        gdf: Serbian road network
        config: Network configuration
    """
    # Filter for road categories
    gdf_filtered = gdf[gdf['kategorija'].isin(config.serbian_road_categories)].copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='white')
    
    # Convert to Web Mercator
    gdf_mercator = gdf_filtered.to_crs(3857)
    
    # Plot each road category
    for category in config.serbian_road_categories:
        if category in gdf_filtered['kategorija'].values:
            category_roads = gdf_mercator[gdf_mercator['kategorija'] == category]
            category_roads.plot(
                ax=ax,
                color=config.serbian_road_colors[category],
                linewidth=config.serbian_road_widths[category],
                alpha=config.road_alpha,
                zorder=config.serbian_road_widths[category]
            )
    
    # Add basemap
    cx.add_basemap(
        ax=ax, 
        source=cx.providers.CartoDB.Positron, 
        alpha=config.basemap_alpha,
        attribution=False
    )
    
    # Styling
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Create legend
    legend_elements = [
        Patch(facecolor=config.serbian_road_colors[cat], 
              label=f'{cat}', 
              edgecolor='none')
        for cat in config.serbian_road_categories
        if cat in gdf_filtered['kategorija'].values
    ]
    
    legend = ax.legend(
        handles=legend_elements, 
        title='Road categories', 
        loc='upper right',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#cccccc'
    )
    
    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(config.output_path / 'road_categories.png', 
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def plot_network_comparison(gdf: gpd.GeoDataFrame, 
                            base_network: gpd.GeoDataFrame,
                            config: LocalConfig) -> None:
    """
    Create side-by-side comparison of Serbian and OSM road networks.
    
    Args:
        gdf: Serbian road network
        base_network: OSM road network
        config: Network configuration
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), facecolor='white')
    
    # ============ LEFT MAP (A) - Serbian Road Network ============
    gdf_filtered = gdf[gdf['kategorija'].isin(config.serbian_road_categories)].copy()
    gdf_mercator = gdf_filtered.to_crs(3857)
    
    for category in config.serbian_road_categories:
        if category in gdf_filtered['kategorija'].values:
            category_roads = gdf_mercator[gdf_mercator['kategorija'] == category]
            category_roads.plot(
                ax=ax1, 
                color=config.serbian_road_colors[category],
                linewidth=config.serbian_road_widths[category], 
                alpha=config.road_alpha,
                zorder=config.serbian_road_widths[category]
            )
    
    cx.add_basemap(ax=ax1, source=cx.providers.CartoDB.Positron, 
                   alpha=config.basemap_alpha, attribution=False)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    legend_elements1 = [
        Patch(facecolor=config.serbian_road_colors[cat], label=cat, edgecolor='none')
        for cat in config.serbian_road_categories 
        if cat in gdf_filtered['kategorija'].values
    ]
    ax1.legend(handles=legend_elements1, title='Road Categories', loc='upper right',
               fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                facecolor='white', alpha=0.8))
    
    # ============ RIGHT MAP (B) - OSM Road Network ============
    
    base_network_filtered = prepare_osm_network_for_plotting(base_network, config)
    network_mercator = base_network_filtered.to_crs(3857)
    
    for category in reversed(config.osm_road_hierarchy):
        if category in base_network_filtered['road_category'].values:
            category_roads = network_mercator[network_mercator['road_category'] == category]
            category_roads.plot(
                ax=ax2, 
                color=config.osm_road_colors[category],
                linewidth=config.osm_road_widths[category], 
                alpha=config.road_alpha,
                zorder=config.osm_road_widths[category]
            )
    
    cx.add_basemap(ax=ax2, source=cx.providers.CartoDB.Positron, 
                   alpha=config.basemap_alpha, attribution=False)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    legend_elements2 = [
        Patch(facecolor=config.osm_road_colors[cat], 
              label=config.osm_road_labels[cat], 
              edgecolor='none')
        for cat in config.osm_road_hierarchy 
        if cat in base_network_filtered['road_category'].values
    ]
    ax2.legend(handles=legend_elements2, title='OpenStreetMap', loc='upper right',
               fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                facecolor='white', alpha=0.8))
    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(config.output_path / 'road_network_comparison.png', 
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def plot_road_length_statistics(base_network_filtered: gpd.GeoDataFrame,
                                config: LocalConfig) -> None:
    """
    Create bar chart of road lengths by category.
    
    Args:
        base_network_filtered: Categorized OSM network
        config: Network configuration
    """
    # Convert to CRS for accurate length calculation in km
    network_8682 = base_network_filtered.to_crs(config.length_crs)
    
    # Calculate length in kilometers
    network_8682['length_km'] = network_8682.geometry.length / 1000
    
    # Group by road category and sum the lengths
    length_by_category = network_8682.groupby('road_category')['length_km'].sum().reset_index()
    
    # Filter and sort the data according to hierarchy
    length_by_category = length_by_category[
        length_by_category['road_category'].isin(config.osm_road_hierarchy)
    ]
    length_by_category['category_order'] = length_by_category['road_category'].map(
        {cat: i for i, cat in enumerate(config.osm_road_hierarchy)}
    )
    length_by_category = length_by_category.sort_values('category_order')
    
    # Create the bar plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
    
    # Create bars with consistent colors
    bars = ax.bar(
        range(len(length_by_category)), 
        length_by_category['length_km'],
        color=[config.osm_road_colors[cat] for cat in length_by_category['road_category']],
        alpha=config.road_alpha,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Set x-axis labels
    ax.set_xticks(range(len(length_by_category)))
    ax.set_xticklabels(
        [config.osm_road_labels[cat] for cat in length_by_category['road_category']], 
        fontsize=10, 
        ha='center'
    )
    
    # Check if we need log scale
    max_length = length_by_category['length_km'].max()
    min_length = length_by_category['length_km'].min()
    if max_length / min_length > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Total Length (km) - Log Scale', fontsize=12, fontweight='bold')
        title_suffix = ' (Log Scale)'
    else:
        ax.set_ylabel('Total Length (km)', fontsize=12, fontweight='bold')
        title_suffix = ''
    
    # Add value labels on top of bars
    for i, (bar, length) in enumerate(zip(bars, length_by_category['length_km'])):
        height = bar.get_height()
        if ax.get_yscale() == 'log':
            text_y = height * 1.1
        else:
            text_y = height + max_length * 0.01
        
        # Format the length value
        if length >= 1000:
            label = f'{length:,.0f} km'
        else:
            label = f'{length:.1f} km'
        
        ax.text(bar.get_x() + bar.get_width()/2., text_y, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the plot
    ax.set_title(f'Total Road Network Length by Category{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(config.output_path / 'road_length_by_category.png', 
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to orchestrate all network visualization processes.
    """
    # Initialize configuration
    config = LocalConfig()
    
    # Load OSM network
    print("Loading OSM network...")
    base_network = load_osm_network(config)
    
    # Prepare and plot OSM network
    print("Preparing OSM network for plotting...")
    base_network_filtered = prepare_osm_network_for_plotting(base_network, config)
    
    print("Creating OSM network plot...")
    plot_osm_network(base_network_filtered, config)
    
    # Load and plot Serbian network (requires ArcGIS)
    print("Loading Serbian network...")
    
    gdf = None
    if ARCPY_AVAILABLE:
        try:
            print("ArcGIS detected.")
            gdf = load_serbian_network_arcpy(config)
        except Exception as e:
            print(f"Error loading Serbian network with ArcGIS: {e}")
    
    elif ARCPY_AVAILABLE is False:
        try:
            print("ArcGIS not available, attempting to load Serbian network from file...")
            gdf = load_serbian_network_no_arcpy(config)        
        except Exception as e:
            print(f"Error loading Serbian network from file: {e}")

    if gdf is not None:
        print("Creating Serbian network plot...")
        plot_serbian_network(gdf, config)

        print("Creating network comparison plot...")
        plot_network_comparison(gdf, base_network, config)
    else:
        print("Serbian network processing skipped: no Serbian network data loaded.")

    # Plot road length statistics
    print("Creating road length statistics plot...")
    plot_road_length_statistics(base_network_filtered, config)
    
    print("All visualizations completed successfully!")


if __name__ == "__main__":
    main()
