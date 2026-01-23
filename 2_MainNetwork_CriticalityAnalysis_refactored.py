"""
Main Network Criticality Analysis Module

This module performs Single Point of Failure (SPOF) analysis on the Serbian road network:
- Loads population and network data
- Creates Origin-Destination (OD) matrices
- Calculates demand matrices
- Performs percolation analysis (removing edges)
- Computes criticality metrics (time delays, vehicle hours lost)
- Generates visualizations
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
import contextily as cx
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from pyproj import Geod
from tqdm import tqdm
import seaborn as sns
import arcpy

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


@dataclass
class CriticalityConfig:
    """Configuration for criticality analysis."""
    
    # Input paths
    data_path: Path = field(default_factory=lambda: Path('input_files'))
    intermediate_path: Path = field(default_factory=lambda: Path('intermediate_results'))
    figures_path: Path = field(default_factory=lambda: Path('figures'))
    
    # Network file
    network_filename: str = 'PERS_directed_final.parquet'
    
    # Population/OD parameters
    population_excel: Optional[str] = None
    
    # Demand model parameters
    dist_decay: float = 1.0
    max_trips: int = 25000
    
    # Analysis parameters
    fail_value: int = 12  # Value to represent isolated/failed connections
    weight_attribute: str = 'fft'  # Free-flow travel time
    
    # Visualization parameters
    figure_dpi: int = 300
    delay_colors: List[str] = field(default_factory=lambda: [
        '#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'
    ])
    
    # Delay classification bins and labels
    delay_bins: List[float] = field(default_factory=lambda: [0, 0.01, 0.25, 0.5, 1, np.inf])
    delay_labels: List[str] = field(default_factory=lambda: [
        'No Delay', '1-15 min', '15-30 min', '30-60 min', '60+ min'
    ])
    delay_linewidths: Dict[str, float] = field(default_factory=lambda: {
        'No Delay': 0.5, '1-15 min': 1.0, '15-30 min': 2.0, 
        '30-60 min': 3.5, '60+ min': 5.0
    })
    
    # VHL (Vehicle Hours Lost) classification
    vhl_bins: List[float] = field(default_factory=lambda: [0, 1000, 5000, 10000, 25000, np.inf])
    vhl_labels: List[str] = field(default_factory=lambda: ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+'])
    vhl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-1K': 0.5, '1K-5K': 1.0, '5K-10K': 2.0, 
        '10K-25K': 3.5, '25K+': 5.0
    })
    
    # Heatmap labels for demand visualization
    heatmap_labels: List[str] = field(default_factory=lambda: [
        "Belgrade", "Novi Sad", "", "...", "", "City X"
    ])
    
    @property
    def network_path(self) -> Path:
        """Full path to network file."""
        return self.intermediate_path / self.network_filename
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.intermediate_path.mkdir(parents=True, exist_ok=True)


def load_network_and_graph(config: CriticalityConfig) -> Tuple[gpd.GeoDataFrame, ig.Graph]:
    """
    Load network edges and create igraph object.
    
    Args:
        config: Criticality configuration
        
    Returns:
        Tuple of (edges GeoDataFrame, igraph Graph)
    """
    edges = gpd.read_parquet(config.network_path)
    graph = ig.Graph.TupleList(
        edges.itertuples(index=False), 
        edge_attrs=list(edges.columns)[2:],
        directed=True
    )
    return edges, graph


def load_population_data(config: CriticalityConfig) -> gpd.GeoDataFrame:
    """
    Load and process population data from Excel file.
    
    Args:
        config: Criticality configuration
        
    Returns:
        GeoDataFrame with population points
    """
    # Get Excel file path from ArcGIS or config
    statepop_excel = config.population_excel or arcpy.GetParameterAsText(0)
    
    # Read Excel file
    DataFrame_StatePop = pd.read_excel(statepop_excel)
    arcpy.AddMessage(f"Loaded Excel file: {statepop_excel}")
    
    # Keep only rows with valid coordinates and population
    Clean_DataFrame_StatePop = DataFrame_StatePop.dropna(
        subset=["latitude", "longitude", "Total"]
    )
    
    # Create point geometry
    geometry = [
        Point(xy) for xy in zip(
            Clean_DataFrame_StatePop["longitude"], 
            Clean_DataFrame_StatePop["latitude"]
        )
    ]
    
    # Build GeoDataFrame
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_StatePop[["Total"]].rename(columns={"Total": "population"}),
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    return df_worldpop


def prepare_od_nodes(df_worldpop: gpd.GeoDataFrame, 
                     edges: gpd.GeoDataFrame,
                     graph: ig.Graph) -> gpd.GeoDataFrame:
    """
    Prepare OD nodes by snapping population points to network vertices.
    
    Args:
        df_worldpop: Population points
        edges: Network edges
        graph: Network graph
        
    Returns:
        OD nodes with vertex IDs
    """
    OD_nodes = df_worldpop.reset_index()
    OD_nodes = OD_nodes.set_crs(4326).to_crs(edges.crs)
    OD_nodes.geometry = OD_nodes.geometry.centroid
    
    # Create vertex lookup
    vertex_lookup = dict(zip(
        pd.DataFrame(graph.vs['name'])[0], 
        pd.DataFrame(graph.vs['name']).index
    ))
    
    # Extract node geometries from edges
    tqdm.pandas()
    from_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[0]))
    to_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[-1]))
    
    from_dict = dict(zip(edges['from_id'], from_id_geom))
    to_dict = dict(zip(edges['to_id'], to_id_geom))
    
    nodes = pd.concat([
        pd.DataFrame.from_dict(to_dict, orient='index', columns=['geometry']),
        pd.DataFrame.from_dict(from_dict, orient='index', columns=['geometry'])
    ]).drop_duplicates()
    
    nodes['vertex_id'] = nodes.progress_apply(lambda x: vertex_lookup[x.name], axis=1)
    nodes = nodes.reset_index()
    
    # Snap OD nodes to nearest network nodes
    nodes_sindex = shapely.STRtree(nodes.geometry)
    OD_nodes['vertex_id'] = OD_nodes.geometry.progress_apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values
    
    return OD_nodes


def run_shortest_paths(graph: ig.Graph, 
                       OD_nodes: gpd.GeoDataFrame,
                       config: CriticalityConfig,
                       tqdm_disable: bool = True) -> np.matrix:
    """
    Calculate shortest path distance matrix.
    
    Args:
        graph: Network graph
        OD_nodes: Origin-destination nodes (grouped)
        config: Criticality configuration
        tqdm_disable: Disable progress bar
        
    Returns:
        Distance matrix
    """
    vertex_ids = OD_nodes.vertex_id.values
    distance_matrix = graph.distances(
        source=vertex_ids, 
        target=vertex_ids, 
        weights=config.weight_attribute
    )
    return np.matrix(distance_matrix)


def create_demand(OD_nodes: gpd.GeoDataFrame, 
                  OD_orig: np.matrix,
                  config: CriticalityConfig) -> np.ndarray:
    """
    Create demand matrix using gravity model.
    
    Demand_a,b = Population_a * Population_b * e^[-p * Distance_a,b]
    
    Args:
        OD_nodes: Origin-destination nodes with population
        OD_orig: Shortest path distance matrix
        config: Criticality configuration
        
    Returns:
        Demand matrix
    """
    node_pop = OD_nodes.population.values
    demand = np.zeros((len(OD_nodes), len(OD_nodes)))
    
    normalized_dist = OD_orig / OD_orig.max()
    
    for o in tqdm(range(len(OD_nodes)), total=len(OD_nodes)):
        for d in range(len(OD_nodes)):
            if o == d:
                demand[o][d] = 0
            else:
                demand[o][d] = (
                    (node_pop[o] * node_pop[d]) * 
                    np.exp(-1 * config.dist_decay * normalized_dist[o, d])
                )
    
    demand = ((demand / demand.max()) * config.max_trips)
    demand = np.ceil(demand).astype(int)
    return demand


def summarise_od(OD: np.matrix, 
                 demand: np.ndarray,
                 baseline: np.matrix,
                 frac_counter: int,
                 distance_disruption: float,
                 time_disruption: float,
                 config: CriticalityConfig) -> Tuple:
    """
    Summarize OD matrix changes during percolation analysis.
    
    Calculates percentage of isolated/delayed trips and travel time disruptions.
    
    Args:
        OD: Current OD matrix times (during percolation)
        demand: Demand matrix
        baseline: OD matrix before percolation
        frac_counter: Current edge ID
        distance_disruption: Distance disruption ratio
        time_disruption: Time disruption ratio
        config: Criticality configuration
        
    Returns:
        Tuple of metrics: (frac_counter, pct_isolated, pct_unaffected, pct_delayed,
                          average_time_disruption, distance_disruption, time_disruption,
                          unaffected_percentiles, delayed_percentiles, delayed_trips_time)
    """
    # Adjusted time
    adj_time = OD - baseline
    
    # Total trips
    total_trips = (baseline.shape[0] * baseline.shape[1]) - baseline.shape[0]
    
    # Isolated trips
    isolated_trips_sum = OD[OD == config.fail_value].shape[1]
    pct_isolated = (isolated_trips_sum / total_trips) * 100
    
    # Get travel times for remaining trips
    time_unaffected_trips = OD[OD == baseline]
    
    # Get unaffected trips travel times
    if not (np.isnan(np.array(time_unaffected_trips)).all()):
        unaffected_percentiles = []
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips), 10))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips), 25))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips), 50))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips), 75))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips), 90))
        unaffected_percentiles.append(np.nanmean((time_unaffected_trips)))
    else:
        unaffected_percentiles = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    # Save delayed trips travel times
    delayed_trips_time = adj_time[
        (OD != baseline) & 
        (np.nan_to_num(np.array(OD), nan=config.fail_value) != config.fail_value)
    ]
    
    unaffected_trips = np.array(time_unaffected_trips).shape[1]
    delayed_trips = np.array(delayed_trips_time).shape[1]
    
    # Save percentage unaffected and delayed
    pct_unaffected = (unaffected_trips / total_trips) * 100
    pct_delayed = (delayed_trips / total_trips) * 100
    
    # Get delayed trips travel times
    if not (np.isnan(np.array(delayed_trips_time)).all()):
        delayed_percentiles = []
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time), 10))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time), 25))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time), 50))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time), 75))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time), 90))
        delayed_percentiles.append(np.nanmean(np.array(delayed_trips_time)))
        average_time_disruption = np.nanmean(np.array(delayed_trips_time))
    else:
        delayed_percentiles = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        average_time_disruption = np.nan
    
    return (frac_counter, pct_isolated, pct_unaffected, pct_delayed, 
            average_time_disruption, distance_disruption, time_disruption, 
            unaffected_percentiles, delayed_percentiles, delayed_trips_time)


def plot_demand_heatmap(demand: np.ndarray, config: CriticalityConfig) -> None:
    """
    Create heatmap visualization of demand matrix.
    
    Args:
        demand: Demand matrix
        config: Criticality configuration
    """
    # Calculate row sums and sort
    row_sums = np.sum(demand, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    
    # Reorder and take top 6x6
    restructured_matrix = demand[np.ix_(sorted_indices, sorted_indices)][:6, :6]
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        restructured_matrix, 
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=config.heatmap_labels,
        yticklabels=config.heatmap_labels,
        square=True,
        cbar=False
    )
    
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().xaxis.tick_top()
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.figures_path / 'demand_heatmap.png', 
               dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def perform_percolation_analysis(graph: ig.Graph,
                                 grouped_OD: gpd.GeoDataFrame,
                                 demand: np.ndarray,
                                 OD_orig: np.matrix,
                                 edges: gpd.GeoDataFrame,
                                 config: CriticalityConfig) -> pd.DataFrame:
    """
    Perform edge percolation analysis to identify critical links.
    
    Args:
        graph: Network graph
        grouped_OD: Grouped OD nodes
        demand: Demand matrix
        OD_orig: Baseline OD matrix
        edges: Network edges
        config: Criticality configuration
        
    Returns:
        DataFrame with criticality results
    """
    tot_edge_length = np.sum(graph.es['road_length'])
    tot_edge_time = np.sum(graph.es['fft'])
    exp_edge_no = graph.ecount()
    
    result_df = []
    
    for edge in tqdm(range(exp_edge_no), total=exp_edge_no, desc='percolation'):
        exp_g = graph.copy()
        exp_g.delete_edges(edge)
        
        cur_dis_length = 1 - (np.sum(exp_g.es['road_length']) / tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['fft']) / tot_edge_time)
        
        # Get new matrix
        perc_matrix = run_shortest_paths(exp_g, grouped_OD, config, tqdm_disable=True)
        np.fill_diagonal(perc_matrix, np.nan)
        perc_matrix[perc_matrix == np.inf] = config.fail_value
        
        # Summarize results
        results = summarise_od(
            perc_matrix, demand, OD_orig, 
            graph.es[edge]['id'], cur_dis_length, cur_dis_time, config
        )
        
        result_df.append(results)
    
    result_df = pd.DataFrame(
        result_df, 
        columns=['edge_no', 'pct_isolated', 'pct_unaffected', 'pct_delayed',
                 'average_time_disruption', 'distance_disruption', 'time_disruption',
                 'unaffected_percentiles', 'delayed_percentiles', 'delayed_trips_time']
    )
    
    return result_df


def calculate_vhl(gdf_results: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate Vehicle Hours Lost (VHL) metric.
    
    Args:
        gdf_results: GeoDataFrame with criticality results
        
    Returns:
        GeoDataFrame with VHL column added
    """
    gdf_results['vhl'] = gdf_results.average_time_disruption * gdf_results.total_aadt
    return gdf_results


def save_criticality_results(gdf_results: gpd.GeoDataFrame, 
                             config: CriticalityConfig) -> None:
    """
    Save criticality results to parquet and shapefile.
    
    Args:
        gdf_results: GeoDataFrame with criticality results
        config: Criticality configuration
    """
    # Select relevant columns
    output_cols = [
        'from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
        'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
        'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'id',
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'oznaka_deo_left',
        'index_right', 'oznaka_deo_right', 'road_length', 'speed', 'fft',
        'geometry', 'edge_no', 'pct_isolated', 'pct_unaffected', 'pct_delayed',
        'average_time_disruption', 'distance_disruption', 'time_disruption', 'vhl'
    ]
    
    # Save to parquet
    gdf_results[output_cols].to_parquet(
        config.intermediate_path / "criticality_results.parquet"
    )
    
    # Save to shapefile
    gdf_results[output_cols].to_file(
        config.intermediate_path / "criticality_results.shp"
    )


def plot_criticality_maps(gdf_results: gpd.GeoDataFrame, 
                          config: CriticalityConfig) -> None:
    """
    Create side-by-side maps of average time delay and vehicle hours lost.
    
    Args:
        gdf_results: GeoDataFrame with criticality results
        config: Criticality configuration
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), facecolor='white')
    
    # ============ LEFT MAP (A) - Average Time Delay ============
    gdf_results['disruption_class'] = pd.cut(
        gdf_results['average_time_disruption'], 
        bins=config.delay_bins, 
        labels=config.delay_labels, 
        include_lowest=True
    )
    
    for i, (class_name, width) in enumerate(config.delay_linewidths.items()):
        subset = gdf_results[gdf_results['disruption_class'] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(
                ax=ax1, 
                color=config.delay_colors[i], 
                linewidth=width, 
                alpha=0.8
            )
    
    cx.add_basemap(ax=ax1, source=cx.providers.CartoDB.Positron, 
                   alpha=0.4, attribution=False)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    legend_elements1 = [
        Line2D([0], [0], color=config.delay_colors[i], lw=width, label=class_name)
        for i, (class_name, width) in enumerate(config.delay_linewidths.items())
    ]
    ax1.legend(handles=legend_elements1, title='Average Increased\nTravel Time', 
               loc='upper right', fontsize=12, title_fontsize=14, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9, facecolor='white', 
               edgecolor='#cccccc')
    
    ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes, fontsize=20, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # ============ RIGHT MAP (B) - Vehicle Hours Lost ============
    gdf_results['vhl_class'] = pd.cut(
        gdf_results['vhl'], 
        bins=config.vhl_bins, 
        labels=config.vhl_labels, 
        include_lowest=True
    )
    
    for i, (class_name, width) in enumerate(config.vhl_linewidths.items()):
        subset = gdf_results[gdf_results['vhl_class'] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(
                ax=ax2, 
                color=config.delay_colors[i], 
                linewidth=width, 
                alpha=0.8
            )
    
    cx.add_basemap(ax=ax2, source=cx.providers.CartoDB.Positron, 
                   alpha=0.4, attribution=False)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    legend_elements2 = [
        Line2D([0], [0], color=config.delay_colors[i], lw=width, 
              label=f'{class_name} vehicle hours')
        for i, (class_name, width) in enumerate(config.vhl_linewidths.items())
    ]
    ax2.legend(handles=legend_elements2, title='Vehicle Hours Lost', 
               loc='upper right', fontsize=12, title_fontsize=14, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9, facecolor='white', 
               edgecolor='#cccccc')
    
    ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, fontsize=20, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_results.png', 
               dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def add_to_arcgis_map(config: CriticalityConfig) -> None:
    """
    Add criticality results to ArcGIS Pro map.
    
    Args:
        config: Criticality configuration
    """
    criticality_results = (
        config.intermediate_path / "criticality_results.shp"
    ).resolve()
    
    arcpy.AddMessage(f"Criticality results saved to {criticality_results}")
    
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.listMaps()[0]
        layer = m.addDataFromPath(str(criticality_results))
        arcpy.AddMessage(f"Criticality map added as layer: {layer.name}")
    except Exception as e:
        arcpy.AddMessage(f"Could not add to map: {e}")


def main():
    """
    Main function to orchestrate the criticality analysis workflow.
    """
    # Initialize configuration
    config = CriticalityConfig()
    
    # Step 1: Load network and graph
    print("Step 1: Loading network and graph...")
    edges, graph = load_network_and_graph(config)
    
    # Step 2: Load population data
    print("Step 2: Loading population data...")
    df_worldpop = load_population_data(config)
    
    # Step 3: Prepare OD nodes
    print("Step 3: Preparing OD nodes...")
    OD_nodes = prepare_od_nodes(df_worldpop, edges, graph)
    
    # Step 4: Group OD nodes by vertex
    print("Step 4: Grouping OD nodes...")
    grouped_OD = OD_nodes.groupby('vertex_id').agg({
        'geometry': 'first',
        'population': 'sum'
    }).reset_index()
    
    # Step 5: Calculate baseline shortest paths
    print("Step 5: Calculating baseline shortest paths...")
    OD_orig = run_shortest_paths(graph, grouped_OD, config, tqdm_disable=False)
    
    # Step 6: Create demand matrix
    print("Step 6: Creating demand matrix...")
    demand = create_demand(grouped_OD, OD_orig, config)
    
    # Step 7: Plot demand heatmap
    print("Step 7: Plotting demand heatmap...")
    plot_demand_heatmap(demand, config)
    
    # Step 8: Perform percolation analysis
    print("Step 8: Performing percolation analysis...")
    result_df = perform_percolation_analysis(
        graph, grouped_OD, demand, OD_orig, edges, config
    )
    
    # Step 9: Merge results with edges
    print("Step 9: Merging results with network edges...")
    gdf_results = edges.merge(result_df, left_index=True, right_on='edge_no')
    
    # Step 10: Calculate VHL
    print("Step 10: Calculating Vehicle Hours Lost...")
    gdf_results = calculate_vhl(gdf_results)
    
    # Step 11: Save results
    print("Step 11: Saving criticality results...")
    save_criticality_results(gdf_results, config)
    
    # Step 12: Create visualization maps
    print("Step 12: Creating criticality maps...")
    plot_criticality_maps(gdf_results, config)
    
    # Step 13: Add to ArcGIS map
    print("Step 13: Adding results to ArcGIS map...")
    add_to_arcgis_map(config)
    
    print("Criticality analysis completed successfully!")


if __name__ == "__main__":
    main()