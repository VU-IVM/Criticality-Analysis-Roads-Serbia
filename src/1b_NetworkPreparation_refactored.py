"""
Network Preparation Module

This module prepares the Serbian road network by:
- Loading and snapping the network
- Splitting edges at nodes
- Merging AADT traffic data
- Creating directed network
- Generating visualizations
"""

# Standard library
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party - Data and scientific computing
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from pyproj import Geod
from tqdm import tqdm
import pyarrow as pa
#from exactextract import exact_extract
try:
    import arcpy # type: ignore
    arcpy.env.overwriteOutput = True
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

# Project root (repository root, one level above this src folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
print(f"Project root set to: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local/Project imports
from src.simplify import *

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

@dataclass
class NetworkPrepConfig:
    """Configuration for network preparation and analysis."""
    
    # Input paths
    data_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'input_files')
    aadt_filename: str = "PGDS_2024.shp"
    world_filename: str = "ne_10m_admin_0_countries.shp"
    
    # Output paths
    output_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'intermediate_results')
    figures_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'figures')
    
    # Input file paths (for both ArcGIS and non-ArcGIS environments)
    network_input_layer: Path = field(default_factory=lambda: PROJECT_ROOT / 'input_files' / "roads_serbia_original_full_AADT.parquet")
    arcgis_input_layer: Optional[str] = None
    arcgis_temp_base: Path = field(default_factory=lambda: Path(r"C:\Temp\arcgis_tmp"))
    
    # Network snapping parameters in meters for topology errors (e.g., small gaps at intersections)
    snap_tolerance: float = 2.0
    snap_search_buffer: float = 30.0 # radius
    snap_max_iterations: int = 20 # maximum number of iterations to prevent infinite loops
    
    # AADT merge parameters for spatial joins between AADT segments and road segments.
    overlap_threshold: float = 0.5 # 50% overlap required to consider a match valid
    endpoint_buffer: float = 1.0 # Buffer distance to consider roads as touching endpoints when filling missing AADT values
    
    # Country filtering
    exclude_country_code: str = 'KOS'  # Kosovo
    
    # Road attributes
    road_attributes: List[str] = field(default_factory=lambda: [
        'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija', 'oznaka_put',
        'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr', 'duzina_deo',
        'pocetna_st', 'zavrsna_st', 'stanje'
    ])
    
    # Traffic types
    traffic_types: List[str] = field(default_factory=lambda: [
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt'
    ])
    
    aadt_original_columns: List[str] = field(default_factory=lambda: [
        'PA', 'BUS', 'LT', 'ST', 'TT', 'AV', 'Ukupno'
    ])
    
    # Speed limits by road category (km/h)
    speed_limits: Dict[str, int] = field(default_factory=lambda: {
        "IM": 100, "IA": 100, "IB": 100, "IIA": 80, "IIB": 80
    })
    default_speed: int = 80
    
    # Visualization parameters
    figure_dpi: int = 300
    traffic_colors: List[str] = field(default_factory=lambda: [
        '#005f73', '#9b2226', '#a53860', '#283618', '#2a9d8f', '#582f0e', '#001219'
    ])
    
    # Traffic visualization breaks and labels
    breaks_labels: Dict[str, Tuple[List, List]] = field(default_factory=lambda: {
        'passenger_cars': ([0, 5000, 10000, 20000, 30000, float('inf')], 
                          ['< 5,000', '5,000-10,000', '10,000-20,000', '20,000-30,000', '> 30,000']),
        'buses': ([0, 50, 100, 200, 400, float('inf')], 
                  ['< 50', '50-100', '100-200', '200-400', '> 400']),
        'light_trucks': ([0, 100, 200, 400, 600, float('inf')], 
                        ['< 100', '100-200', '200-400', '400-600', '> 600']),
        'medium_trucks': ([0, 100, 200, 400, 600, float('inf')], 
                         ['< 100', '100-200', '200-400', '400-600', '> 600']),
        'heavy_trucks': ([0, 50, 100, 200, 300, float('inf')], 
                        ['< 50', '50-100', '100-200', '200-300', '> 300']),
        'articulated_vehicles': ([0, 1000, 2000, 4000, 6000, float('inf')], 
                               ['< 1,000', '1,000-2,000', '2,000-4,000', '4,000-6,000', '> 6,000']),
        'total_aadt': ([0, 5000, 10000, 20000, 40000, float('inf')], 
                      ['< 5,000', '5,000-10,000', '10,000-20,000', '20,000-40,000', '> 40,000'])
    })
    
    @property
    def aadt_path(self) -> Path:
        """Full path to AADT data file."""
        return self.data_path / self.aadt_filename
    
    @property
    def world_path(self) -> Path:
        """Full path to world boundaries file."""
        return self.data_path / self.world_filename
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
    
    def get_width_mapping(self, traffic_type: str) -> Dict[str, float]:
        """Generate width mapping for a traffic type."""
        _, labels = self.breaks_labels[traffic_type]
        return {cat: 0.5 + i * 0.75 for i, cat in enumerate(labels)}
    
    def get_legend_title(self, traffic_type: str) -> str:
        """Get formatted legend title for traffic type."""
        titles = {
            'passenger_cars': 'Passenger Cars\n(vehicles/day)',
            'buses': 'Buses\n(vehicles/day)', 
            'light_trucks': 'Light Trucks\n(vehicles/day)',
            'medium_trucks': 'Medium Trucks\n(vehicles/day)',
            'heavy_trucks': 'Heavy Trucks\n(vehicles/day)',
            'articulated_vehicles': 'Articulated Vehicles\n(vehicles/day)',
            'total_aadt': 'Total AADT\n(vehicles/day)'
        }
        return titles.get(traffic_type, traffic_type)


def get_endpoints(geom) -> Tuple[Optional[Point], Optional[Point]]:
    """
    Get start and end points of a linestring or multilinestring.
    
    Args:
        geom: Shapely geometry (LineString or MultiLineString)
        
    Returns:
        Tuple of (start_point, end_point), or (None, None) if invalid
    """
    if geom is None or geom.is_empty:
        return None, None
    
    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        if len(lines) == 0:
            return None, None
        first_line = lines[0]
        last_line = lines[-1]
        start_coords = list(first_line.coords)[0]
        end_coords = list(last_line.coords)[-1]
        return Point(start_coords), Point(end_coords)
    
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        return Point(coords[0]), Point(coords[-1])
    
    else:
        return None, None

def load_network(config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Load Serbian road network from a GIS file path using GeoPandas.
    
    The file path is specified in ``config.network_input_layer`` and is read using
    :func:`geopandas.read_file`.
    
    Args:
        config: Network configuration
    """    
    
    # Read into GeoDataFrame
    gdf = gpd.read_parquet(config.network_input_layer)
    print(f"Successfully loaded feature layer.")
    
    # Select relevant attributes
    attributes = config.road_attributes + ['geometry']
    gdf = gdf[attributes]
    
    return gdf

def load_network_arcpy(config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Load network from ArcGIS input layer.
    
    Args:
        config: Network configuration
        
    Returns:
        GeoDataFrame with road network
    """
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
    pers_network = gpd.read_file(str(gdb_path), layer=out_fc_name)
    arcpy.AddMessage(f"Loaded file with AADT")
    
    # Select relevant attributes
    attributes = config.road_attributes + ['geometry']
    pers_network = pers_network[attributes]
    
    return pers_network


def snap_network_iteratively(gdf: gpd.GeoDataFrame, 
                             config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Iteratively snap road endpoints to nearby endpoints or road segments.
    Uses spatial index for efficiency.
    
    Args:
        gdf: Road network GeoDataFrame
        config: Network configuration
        
    Returns:
        Snapped GeoDataFrame
    """
    gdf = gdf.copy()
    total_snaps = 0
    indices = list(gdf.index)
    already_snapped_pairs = set()
    
    iteration = 0
    while True:
        iteration += 1
        snaps_this_round = 0
        snapped_this_round = set()
        
        # Build spatial index fresh each iteration
        geometries = gdf.geometry.tolist()
        tree = STRtree(geometries)
        idx_to_pos = {idx: i for i, idx in enumerate(indices)}
        pos_to_idx = {i: idx for i, idx in enumerate(indices)}
        
        for idx1 in tqdm(indices, desc=f"Iteration {iteration}"):
            if idx1 in snapped_this_round:
                continue
            
            geom1 = gdf.loc[idx1, 'geometry']
            if geom1 is None or geom1.is_empty:
                continue
                
            start1, end1 = get_endpoints(geom1)
            if start1 is None:
                continue
            
            # Find candidate roads within buffer
            buffer_geom = geom1.buffer(config.snap_search_buffer)
            candidate_positions = tree.query(buffer_geom)
            candidate_indices = [pos_to_idx[pos] for pos in candidate_positions]
            
            for pt1, pos1 in [(start1, 'start'), (end1, 'end')]:
                if idx1 in snapped_this_round:
                    break
                
                for idx2 in candidate_indices:
                    if idx1 == idx2:
                        continue
                    
                    # Skip if this pair was already snapped
                    pair = tuple(sorted([idx1, idx2]))
                    if pair in already_snapped_pairs:
                        continue
                    
                    geom2 = gdf.loc[idx2, 'geometry']
                    if geom2 is None or geom2.is_empty:
                        continue
                    
                    dist = pt1.distance(geom2)
                    
                    if 0 < dist <= config.snap_tolerance:
                        gdf.loc[idx1, 'geometry'] = snap(
                            gdf.loc[idx1, 'geometry'], 
                            geom2, 
                            config.snap_tolerance
                        )
                        snapped_this_round.add(idx1)
                        already_snapped_pairs.add(pair)
                        snaps_this_round += 1
                        
                        start1, end1 = get_endpoints(gdf.loc[idx1, 'geometry'])
                        break
        
        print(f"Iteration {iteration} complete: {snaps_this_round} snaps")
        total_snaps += snaps_this_round
        
        if snaps_this_round == 0:
            break
        if iteration > config.snap_max_iterations:
            print("Max iterations reached")
            break
        if ARCPY_AVAILABLE:
            arcpy.AddMessage(f"Snapped network iteratively.")
            arcpy.AddMessage(f"\nTotal snaps made: {total_snaps}")
        else:
            print(f"Snapped network iteratively.")
            print(f"\nTotal snaps made: {total_snaps}")
    return gdf


def prepare_network_topology(pers_network: gpd.GeoDataFrame, 
                             config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Prepare network topology by adding endpoints, splitting edges, and adding IDs.
    
    Args:
        pers_network: Road network
        config: Network configuration
        
    Returns:
        Network with proper topology
    """
    # #JDP
    # # Add osm_id column if it doesn't exist (required by simplify.py functions)
    # if 'osm_id' not in pers_network.columns:
    #     pers_network['osm_id'] = range(len(pers_network))
    
    # Create a Network object from the input DataFrame
    net = Network(edges=pers_network)
    net = add_endpoints(net)
    split_attributes = [attr for attr in config.road_attributes if attr != 'geometry']
    net = split_edges_at_nodes(net, attributes=split_attributes)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net) 
    
    pers_network = net.edges.set_crs(pers_network.crs)
    
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"After topology prep, columns: {list(pers_network.columns)}")
    else:
        print(f"After topology prep, columns: {list(pers_network.columns)}")
    
    return pers_network


def load_aadt_data(config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Load AADT (Average Annual Daily Traffic) data.
    
    Args:
        config: Network configuration
        
    Returns:
        GeoDataFrame with AADT data
    """
    aadt_network = gpd.read_file(config.aadt_path)
    
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"Loaded AADT data with columns: {list(aadt_network.columns)}")
    else:
        print(f"Loaded AADT data with columns: {list(aadt_network.columns)}")
    
    # Remove rows with missing AADT data
    aadt_network.dropna(subset=config.aadt_original_columns, inplace=True)
    return aadt_network


def merge_aadt_with_network(pers_network: gpd.GeoDataFrame, 
                            aadt_network: gpd.GeoDataFrame,
                            config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Merge AADT data with road network using attribute and spatial joins.
    
    Args:
        pers_network: Road network
        aadt_network: AADT data
        config: Network configuration
        
    Returns:
        Network with AADT values merged
    """
    aadt_cols = config.aadt_original_columns

    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"Merging AADT data with network on columns: {aadt_cols + ['oznaka_deo']}")
    else:
        print(f"Merging AADT data with network on columns: {aadt_cols + ['oznaka_deo']}")
        print(f"pers_network columns: {list(pers_network.columns)}")
        print(f"aadt_network columns: {list(aadt_network.columns)}")

    if 'oznaka_deo' not in pers_network.columns:
        if 'oznaka_deo_left' in pers_network.columns:
            pers_network = pers_network.rename(columns={'oznaka_deo_left': 'oznaka_deo'})
        elif 'oznaka_deo_right' in pers_network.columns:
            pers_network = pers_network.rename(columns={'oznaka_deo_right': 'oznaka_deo'})
        else:
            raise KeyError(
                "Required column 'oznaka_deo' not found in pers_network after topology preparation. "
                f"Available columns: {list(pers_network.columns)}"
            )

    # First merge on oznaka_deo
    first_merger = pers_network.merge(
        aadt_network[aadt_cols + ['oznaka_deo']], 
        how='left', 
        left_on='oznaka_deo', 
        right_on='oznaka_deo'
    )
    
    # Spatial join for unmatched rows
    overlap = first_merger.loc[first_merger.PA.isna()][pers_network.columns].sjoin(
        aadt_network[aadt_cols + ['oznaka_deo', 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    
    # Get the AADT geometries for the matched index_right values
    overlap_with_aadt_geom = overlap.dropna(subset=['index_right']).copy()
    overlap_with_aadt_geom['aadt_geometry'] = aadt_network.loc[
        overlap_with_aadt_geom['index_right'].astype(int), 'geometry'
    ].values
    
    # Calculate intersection and overlap ratio
    overlap_with_aadt_geom['intersection_geom'] = overlap_with_aadt_geom.apply(
        lambda row: row['geometry'].intersection(row['aadt_geometry']), axis=1
    )
    overlap_with_aadt_geom['overlap_ratio'] = (
        overlap_with_aadt_geom['intersection_geom'].length / 
        overlap_with_aadt_geom['geometry'].length
    )
    
    # Keep only rows with >= threshold overlap
    overlap_filtered = overlap_with_aadt_geom[
        overlap_with_aadt_geom['overlap_ratio'] >= config.overlap_threshold
    ].copy()
    overlap_filtered = overlap_filtered.drop(
        columns=['aadt_geometry', 'intersection_geom', 'overlap_ratio']
    )
    
    # Aggregate overlapping matches
    first_cols = ['oznaka_deo_left', 'smer_gdf1', 'kategorija', 'oznaka_put', 
                  'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr', 
                  'duzina_deo', 'pocetna_st', 'geometry', 'index_right', 
                  'oznaka_deo_right']
    
    agg_dict = {col: 'first' for col in first_cols}
    agg_dict.update({col: 'max' for col in config.aadt_original_columns})
    
    result = overlap_filtered.dropna(subset=config.aadt_original_columns).groupby(level=0).agg(agg_dict)
    
    # Concatenate results
    AADT_connected = pd.concat([
        first_merger.loc[first_merger.dropna(subset=aadt_cols).index], 
        result
    ])
    AADT_connected = gpd.GeoDataFrame(pd.concat([
        AADT_connected,
        pers_network.loc[~pers_network.index.isin(AADT_connected.index)]
    ]))
    
    # Rename columns to standard names
    column_mapping = {
        'PA': 'passenger_cars', 
        'BUS': 'buses', 
        'LT': 'light_trucks', 
        'ST': 'medium_trucks',  
        'TT': 'heavy_trucks', 
        'AV': 'articulated_vehicles', 
        'Ukupno': 'total_aadt'
    }
    AADT_connected = AADT_connected.rename(columns=column_mapping)
    
    # Convert to float64 after renaming (matching notebook procedure)
    AADT_connected[config.traffic_types] = AADT_connected[config.traffic_types].astype(np.float64)
    
    # Ensure it's a GeoDataFrame with proper CRS
    AADT_connected = gpd.GeoDataFrame(AADT_connected, geometry='geometry')
    
    return AADT_connected


def find_touching_roads_with_aadt(idx: int, gdf: gpd.GeoDataFrame, 
                                  buffer_dist: float = 1.0) -> List:
    """
    Find roads that touch the endpoints of a given road and have AADT values.
    
    Args:
        idx: Index of road to check
        gdf: GeoDataFrame with roads
        buffer_dist: Buffer distance for intersection check
        
    Returns:
        List of tuples (endpoint_type, road_idx, road_row)
    """
    row = gdf.loc[idx]
    start_pt, end_pt = get_endpoints(row.geometry)
    
    if start_pt is None:
        return []
    
    touching_roads = []
    for other_idx, other_row in gdf.iterrows():
        if other_idx == idx:
            continue
        if pd.isna(other_row['total_aadt']):
            continue
            
        # Check if endpoints touch the other road
        if other_row.geometry is not None and not other_row.geometry.is_empty:
            if start_pt.buffer(buffer_dist).intersects(other_row.geometry):
                touching_roads.append(('start', other_idx, other_row))
            if end_pt.buffer(buffer_dist).intersects(other_row.geometry):
                touching_roads.append(('end', other_idx, other_row))
    
    return touching_roads


def fill_missing_aadt(AADT_connected: gpd.GeoDataFrame, 
                     config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Fill missing AADT values using two-pass approach:
    1. From roads touching both endpoints
    2. From category median, capped by touching roads
    
    Args:
        AADT_connected: Network with partial AADT data
        config: Network configuration
        
    Returns:
        Network with filled AADT values
    """
    traffic_cols = config.traffic_types
    
    # ============================================
    # PASS 1: Fill from both endpoints touching roads with AADT
    # ============================================
    if ARCPY_AVAILABLE:
        arcpy.AddMessage("Pass 1: Filling from roads touching both endpoints...")
    else:
        print("Pass 1: Filling from roads touching both endpoints...")

    missing_aadt = AADT_connected[AADT_connected['total_aadt'].isna()].index.tolist()
    filled_count_pass1 = 0
    
    for idx in tqdm(missing_aadt, total=len(missing_aadt)):
        touching = find_touching_roads_with_aadt(idx, AADT_connected, config.endpoint_buffer)
        
        # Check if we have at least one touch at start and one at end
        start_touches = [t for t in touching if t[0] == 'start']
        end_touches = [t for t in touching if t[0] == 'end']
        
        if len(start_touches) > 0 and len(end_touches) > 0:
            # Get AADT values from touching roads
            start_values = {col: np.mean([t[2][col] for t in start_touches]) 
                          for col in traffic_cols}
            end_values = {col: np.mean([t[2][col] for t in end_touches]) 
                        for col in traffic_cols}
            
            # Take average of start and end
            for col in traffic_cols:
                AADT_connected.loc[idx, col] = (start_values[col] + end_values[col]) / 2
            
            filled_count_pass1 += 1
    
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"Pass 1 filled {filled_count_pass1} roads")
    else:
        print(f"Pass 1 filled {filled_count_pass1} roads")
    
    # ============================================
    # PASS 2: Fill with median by kategorija, then cap by touching roads
    # ============================================
    # JDP: why having arcpy already?
    if ARCPY_AVAILABLE:
        arcpy.AddMessage("Pass 2: Filling with kategorija median...")
    else:
        print("Pass 2: Filling with kategorija median...")

    # Calculate median values per kategorija
    kategoria_medians = AADT_connected.groupby('kategorija')[traffic_cols].median()
    
    missing_aadt = AADT_connected[AADT_connected['total_aadt'].isna()].index.tolist()
    filled_count_pass2 = 0
    
    for idx in tqdm(missing_aadt, total=len(missing_aadt)):
        row = AADT_connected.loc[idx]
        kategorija = row['kategorija']
        
        # Skip if no kategorija
        if pd.isna(kategorija) or kategorija not in kategoria_medians.index:
            continue
        
        # Fill with median values
        median_values = kategoria_medians.loc[kategorija]
        for col in traffic_cols:
            AADT_connected.loc[idx, col] = median_values[col]
        
        # Now check touching roads and cap if our value is higher
        touching = find_touching_roads_with_aadt(idx, AADT_connected, config.endpoint_buffer)
        
        if len(touching) > 0:
            # Get max AADT from any touching road
            max_touching_values = {col: max([t[2][col] for t in touching]) 
                                 for col in traffic_cols}
            
            # Cap our values if they exceed touching roads
            for col in traffic_cols:
                if AADT_connected.loc[idx, col] > max_touching_values[col]:
                    AADT_connected.loc[idx, col] = max_touching_values[col]
        
        filled_count_pass2 += 1
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"Pass 2 filled {filled_count_pass2} roads")
    else:
        print(f"Pass 2 filled {filled_count_pass2} roads")
    
    # Summary
    remaining_missing = AADT_connected['total_aadt'].isna().sum()
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"\nRemaining roads without AADT: {remaining_missing}")
    else:
        print(f"\nRemaining roads without AADT: {remaining_missing}")
    
    return AADT_connected


def filter_by_country(AADT_connected: gpd.GeoDataFrame, 
                     config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Filter roads to exclude specified country (e.g., Kosovo).
    
    Args:
        AADT_connected: Network with AADT data
        config: Network configuration
        
    Returns:
        Filtered network
    """
    # Load country outline
    world = gpd.read_file(config.world_path)
    country = world.loc[world.SOV_A3 == config.exclude_country_code]
    country = country.to_crs(AADT_connected.crs)
    
    # Dissolve in case there are multiple polygons
    kosovo_geom = country.union_all()
    
    # Filter roads that are within Serbia (not in Kosovo)
    AADT_Serbia = AADT_connected[~AADT_connected.geometry.intersects(kosovo_geom)].copy()
    
    return AADT_Serbia


def plot_aadt_categories_combined(gdf_aadt: gpd.GeoDataFrame, 
                                  config: NetworkPrepConfig) -> None:
    """
    Create combined 3x2 subplot visualization of all AADT categories.
    
    Args:
        gdf_aadt: Network with AADT data
        config: Network configuration
    """
    # Exclude 'total_aadt' - only 6 categories
    traffic_types = [t for t in config.traffic_types if t != 'total_aadt']
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Create figure with 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(12, 24), facecolor='white')
    axes = axes.flatten()
    
    for i, traffic_type in enumerate(traffic_types):
        ax = axes[i]
        breaks, labels = config.breaks_labels[traffic_type]
        width_mapping = config.get_width_mapping(traffic_type)
        
        # Create categories for this traffic type
        gdf_aadt[f'{traffic_type}_category'] = pd.cut(
            gdf_aadt[traffic_type], 
            bins=breaks, 
            labels=labels, 
            include_lowest=True
        )
        
        # Plot each category
        for category in labels:
            subset = gdf_aadt[gdf_aadt[f'{traffic_type}_category'] == category]
            if len(subset) > 0:
                width = width_mapping[category]
                subset.plot(ax=ax, color=config.traffic_colors[i], 
                          alpha=0.7, linewidth=width)
        
        # Create legend with line widths
        legend_elements = [
            Line2D([0], [0], color=config.traffic_colors[i], 
                  lw=width_mapping[cat], label=cat, alpha=0.7)
            for cat in labels
        ]
        ax.legend(handles=legend_elements, 
                 title=config.get_legend_title(traffic_type), 
                 loc='upper right', fontsize=13, title_fontsize=15,
                 frameon=True, fancybox=True, shadow=True,
                 framealpha=0.9, facecolor='white', edgecolor='#cccccc')
        
        ax.axis('off')
        
        # Add letter label
        ax.text(0.05, 0.95, letters[i], transform=ax.transAxes, 
               fontsize=20, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(config.figures_path / 'AADT_categories_combined.png', 
               dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def plot_total_aadt_map(aadt_network: gpd.GeoDataFrame, 
                       config: NetworkPrepConfig) -> None:
    """
    Create individual map visualization for total AADT.
    
    Args:
        aadt_network: Original AADT network data
        config: Network configuration
    """
    # Prepare data
    gdf_aadt = aadt_network.copy()
    column_mapping = {
        'PA': 'passenger_cars', 'BUS': 'buses', 'LT': 'light_trucks', 
        'ST': 'medium_trucks', 'TT': 'heavy_trucks', 
        'AV': 'articulated_vehicles', 'Ukupno': 'total_aadt'
    }
    gdf_aadt = gdf_aadt.rename(columns=column_mapping)
    gdf_aadt[config.traffic_types] = gdf_aadt[config.traffic_types].astype(np.float64)
    
    # Plot total AADT
    traffic_type = 'total_aadt'
    breaks, labels = config.breaks_labels[traffic_type]
    width_mapping = config.get_width_mapping(traffic_type)
    
    # Create categories
    gdf_aadt[f'{traffic_type}_category'] = pd.cut(
        gdf_aadt[traffic_type].astype(np.float64), 
        bins=breaks, 
        labels=labels, 
        include_lowest=True
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    for category in labels:
        subset = gdf_aadt[gdf_aadt[f'{traffic_type}_category'] == category]
        if len(subset) > 0:
            width = width_mapping[category]
            subset.plot(ax=ax, color=config.traffic_colors[6], 
                       alpha=0.7, linewidth=width, label=category)
    
    ax.legend(title=config.get_legend_title(traffic_type), loc='upper right')
    ax.axis('off')
    plt.savefig(config.figures_path / f'{traffic_type}_aadt_map_og.png', 
               dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def create_directed_network(AADT_Serbia: gpd.GeoDataFrame, 
                           config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Create directed network with bidirectional edges for non-oneway roads.
    
    Args:
        AADT_Serbia: Network filtered for Serbia
        config: Network configuration
        
    Returns:
        Directed network with speed and travel time attributes
    """
    # Prepare network topology
    net = Network(edges=AADT_Serbia)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)
    base_network = net.edges.set_crs(AADT_Serbia.crs)
    
    # Filter for roads that are not oneway
    non_oneway_roads = base_network[~base_network['smer_gdf1'].isin(['L', 'D'])]
    
    # Create reverse edges
    def reverse_road(row):
        reversed_geometry = shapely.LineString(row['geometry'].coords[::-1])
        new_row = row.copy()
        new_row['from_id'], new_row['to_id'] = row['to_id'], row['from_id']
        new_row['geometry'] = reversed_geometry
        return new_row
    
    # Apply the reverse function
    reversed_edges = non_oneway_roads.apply(reverse_road, axis=1)
    
    # Append reversed edges back to the original GeoDataFrame
    base_network = gpd.GeoDataFrame(
        pd.concat([base_network, reversed_edges])
    ).reset_index(drop=True)
    base_network['id'] = base_network.index
    
    # Calculate speed and travel time
    def fill_speed(x):
        try:
            return config.speed_limits[x.kategorija]
        except:
            return config.default_speed
    
    base_network['road_length'] = base_network.geometry.apply(
        lambda line_string: shapely.length(line_string) / 1e3
    )
    base_network['speed'] = base_network.apply(lambda x: fill_speed(x), axis=1)
    base_network['fft'] = base_network.apply(
        lambda x: (x.road_length / x.speed), axis=1
    )
    
    return base_network


def create_igraph_and_export(base_network: gpd.GeoDataFrame, 
                            AADT_Serbia: gpd.GeoDataFrame,
                            config: NetworkPrepConfig) -> None:
    """
    Create igraph network and export results.
    
    Args:
        base_network: Directed network with all attributes
        AADT_Serbia: Original Serbia network (for CRS)
        config: Network configuration
    """
    # Load into igraph
    edges = base_network.reindex(
        ['from_id', 'to_id'] + [x for x in list(base_network.columns) 
                                if x not in ['from_id', 'to_id']], 
        axis=1
    )
    graph = ig.Graph.TupleList(
        edges.itertuples(index=False), 
        edge_attrs=list(edges.columns)[2:],
        directed=True
    )
    graph = graph.connected_components().giant()
    edges = edges[edges['id'].isin(graph.es['id'])]
    
    # Export to parquet and shapefile
    edges_gdf = edges.reset_index(drop=True).set_crs(AADT_Serbia.crs)
    edges_gdf.to_parquet(config.output_path / 'PERS_directed_final.parquet')
    edges_gdf.to_file(config.output_path / 'PERS_directed_final.shp')
    
    directed_final = (config.output_path / "PERS_directed_final.shp").resolve()
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(f"Directed graph saved to {directed_final}")
    else:
        print(f"Directed graph saved to {directed_final}")
    
    # Add to ArcGIS Pro map
    if ARCPY_AVAILABLE:
        try:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            m = aprx.listMaps()[0]
            layer = m.addDataFromPath(str(directed_final))
            arcpy.AddMessage(f"Directed graph added as layer: {layer.name}")
        except Exception as e:
            arcpy.AddWarning(f"Could not add to map: {e}")
    else:
        print(f"WARNING: Could not add to map, not running in ArcGIS Pro.")


def main():
    """
    Main function to orchestrate the network preparation workflow.
    """
    # Initialize configuration
    config = NetworkPrepConfig()
    
    # Step 1: Load network from ArcGIS
    print("Step 1: Loading network from ArcGIS...")
    if ARCPY_AVAILABLE:
        pers_network = load_network_arcpy(config)
    else:
        pers_network = load_network(config)
    
    # Step 2: Snap network iteratively
    print("Step 2: Snapping network...")
    pers_network = snap_network_iteratively(pers_network, config)
    
    # Step 3: Prepare network topology
    print("Step 3: Preparing network topology...")
    pers_network = prepare_network_topology(pers_network, config)
    
    # Step 4: Load AADT data
    print("Step 4: Loading AADT data...")
    aadt_network = load_aadt_data(config)
    
    # Step 5: Merge AADT with network
    print("Step 5: Merging AADT with network...")
    AADT_connected = merge_aadt_with_network(pers_network, aadt_network, config)
    
    # Step 6: Fill missing AADT values
    print("Step 6: Filling missing AADT values...")
    AADT_connected = fill_missing_aadt(AADT_connected, config)
    
    # Step 7: Filter by country (exclude Kosovo)
    print("Step 7: Filtering by country...")
    AADT_Serbia = filter_by_country(AADT_connected, config)
    
    # Step 8: Create visualizations
    print("Step 8: Creating AADT category visualizations...")
    plot_aadt_categories_combined(AADT_Serbia, config)
    
    print("Step 9: Creating total AADT map...")
    plot_total_aadt_map(aadt_network, config)
    
    # Step 10: Create directed network
    print("Step 10: Creating directed network...")
    base_network = create_directed_network(AADT_Serbia, config)
    
    # Step 11: Create igraph and export
    print("Step 11: Creating igraph and exporting...")
    create_igraph_and_export(base_network, AADT_Serbia, config)
    
    print("Network preparation completed successfully!")


if __name__ == "__main__":
    main()