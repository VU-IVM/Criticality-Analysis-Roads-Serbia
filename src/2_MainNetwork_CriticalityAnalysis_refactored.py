# Converted from 2_MainNetwork_CriticalityAnalysis.ipynb
"""
Main Network Criticality Analysis Module

This module performs Single Point of Failure (SPOF) analysis on the Serbian road network:
- Loads network edges and population data
- Builds Origin-Destination (OD) matrices
- Calculates gravity-model demand matrices
- Performs directed-edge percolation analysis
- Computes criticality metrics (time/distance disruption, VHL, PHL, THL, PKL, TKL)
- Generates maps, boxplots, violin plots and scatter plots
- Saves results to parquet
"""

# Standard library
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party – Data and scientific computing
import contextily as cx
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# ArcGIS (optional)
try:
    import arcpy
    arcpy.env.overwriteOutput = True
    ARCPY_AVAILABLE = True
except ImportError:
    ARCPY_AVAILABLE = False

# Shapely
import shapely
from shapely.geometry import Point

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# Project root (repository root, one level above this src folder)
# Project root (repository root, one level above this src folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().resolve()
print(f"Project root set to: {PROJECT_ROOT}")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class CriticalityConfig:
    """Configuration for the main-network criticality analysis."""

    # Paths
    data_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'input_files')
    intermediate_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'intermediate_results')
    figures_path: Path = field(default_factory=lambda: PROJECT_ROOT / 'figures')

    # Input files
    network_filename: str = 'PERS_directed_final.parquet'
    population_filename: str = 'population_NEW_settlement_geocoded.xlsx'

    # Demand-model parameters
    dist_decay: float = 1.0
    max_trips: int = 25_000

    # Percolation parameters
    fail_value: int = 6          # sentinel for isolated / no-path trips
    time_weight: str = 'fft'
    dist_weight: str = 'road_length'

    # Visualization parameters
    figure_dpi: int = 300
    basemap_alpha: float = 0.4

    # Disruption colour ramp (light → dark red)
    disruption_colors: List[str] = field(default_factory=lambda: [
        '#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'
    ])

    # PHL/THL/PKL/TKL colour ramp (pink palette)
    person_colors: List[str] = field(default_factory=lambda: [
        '#fcc5c0', '#fa9fb5', '#f768a1', '#c51b8a', '#7a0177'
    ])

    # Classification bins / labels / linewidths
    time_bins: List[float] = field(default_factory=lambda: [0, 0.08335, 0.25, 0.5, 1, np.inf])
    time_labels: List[str] = field(default_factory=lambda: [
        'No Delay', '5-15 min', '15-30 min', '30-60 min', '60+ min'
    ])
    time_linewidths: Dict[str, float] = field(default_factory=lambda: {
        'No Delay': 0.5, '5-15 min': 1.5, '15-30 min': 2.0,
        '30-60 min': 2.5, '60+ min': 3.0
    })

    vhl_bins: List[float] = field(default_factory=lambda: [0, 1000, 5000, 10_000, 25_000, np.inf])
    vhl_labels: List[str] = field(default_factory=lambda: ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+'])
    vhl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-1K': 0.5, '1K-5K': 1.5, '5K-10K': 2.0, '10K-25K': 2.5, '25K+': 3.0
    })

    dist_bins: List[float] = field(default_factory=lambda: [0, 5, 15, 30, 60, np.inf])
    dist_labels: List[str] = field(default_factory=lambda: [
        '0-5 km', '5-15 km', '15-30 km', '30-60 km', '60+ km'
    ])
    dist_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-5 km': 0.5, '5-15 km': 1.5, '15-30 km': 2.0, '30-60 km': 2.5, '60+ km': 3.0
    })

    vkl_bins: List[float] = field(default_factory=lambda: [
        0, 10_000, 50_000, 100_000, 250_000, np.inf
    ])
    vkl_labels: List[str] = field(default_factory=lambda: [
        '0-10K', '10K-50K', '50K-100K', '100K-250K', '250K+'
    ])
    vkl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-10K': 0.5, '10K-50K': 1.5, '50K-100K': 2.0, '100K-250K': 2.5, '250K+': 3.0
    })

    phl_bins: List[float] = field(default_factory=lambda: [0, 1000, 5000, 10_000, 25_000, np.inf])
    phl_labels: List[str] = field(default_factory=lambda: ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+'])
    phl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-1K': 0.5, '1K-5K': 1.5, '5K-10K': 2.0, '10K-25K': 2.5, '25K+': 3.0
    })

    thl_bins: List[float] = field(default_factory=lambda: [0, 500, 2500, 5000, 10_000, np.inf])
    thl_labels: List[str] = field(default_factory=lambda: ['0-500', '500-2.5K', '2.5K-5K', '5K-10K', '10K+'])
    thl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '0-500': 0.5, '500-2.5K': 1.5, '2.5K-5K': 2.0, '5K-10K': 2.5, '10K+': 3.0
    })

    pkl_bins: List[float] = field(default_factory=lambda: [
        10_000, 50_000, 100_000, 250_000, 500_000, np.inf
    ])
    pkl_labels: List[str] = field(default_factory=lambda: [
        '10K-50K', '50K-100K', '100K-250K', '250K-500K', '500K+'
    ])
    pkl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '10K-50K': 0.5, '50K-100K': 1.5, '100K-250K': 2.0, '250K-500K': 2.5, '500K+': 3.0
    })

    tkl_bins: List[float] = field(default_factory=lambda: [
        10_000, 25_000, 50_000, 100_000, 250_000, np.inf
    ])
    tkl_labels: List[str] = field(default_factory=lambda: [
        '5K-25K', '25K-50K', '50K-100K', '100K-250K', '250K+'
    ])
    tkl_linewidths: Dict[str, float] = field(default_factory=lambda: {
        '10-25K': 0.5, '25K-50K': 1.5, '50K-100K': 2.0, '100K-250K': 2.5, '250K+': 3.0
    })

    # Heatmap display labels (demand visualisation)
    heatmap_labels: List[str] = field(default_factory=lambda: [
        'Belgrade', 'Novi Sad', '', '...', '', 'City X'
    ])

    @property
    def network_path(self) -> Path:
        """Full path to directed network parquet file."""
        return self.intermediate_path / self.network_filename

    @property
    def population_path(self) -> Path:
        """Full path to population Excel file."""
        return self.data_path / self.population_filename

    def __post_init__(self):
        """Ensure output directories exist."""
        self.intermediate_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Helper utilities
# ===========================================================================

def _log(message: str) -> None:
    """Print or ArcGIS-AddMessage depending on environment."""
    if ARCPY_AVAILABLE:
        arcpy.AddMessage(message)
    else:
        print(message)


# ===========================================================================
# Data loading
# ===========================================================================

def load_network_and_graph(
    config: CriticalityConfig,
) -> Tuple[gpd.GeoDataFrame, ig.Graph]:
    """
    Load the directed road network and build an igraph Graph.

    Args:
        config: Criticality configuration.

    Returns:
        Tuple of (edge GeoDataFrame, directed igraph Graph).
    """
    edges = gpd.read_parquet(config.network_path)
    graph = ig.Graph.TupleList(
        edges.itertuples(index=False),
        edge_attrs=list(edges.columns)[2:],
        directed=True,
    )
    return edges, graph


def load_population_data(config: CriticalityConfig) -> gpd.GeoDataFrame:
    """
    Load and clean population points from the Excel file.

    Args:
        config: Criticality configuration.

    Returns:
        GeoDataFrame with 'population' column and WGS-84 point geometry.
    """
    excel_path = str(config.population_path)
    df = pd.read_excel(excel_path)
    _log(f"Loaded population file: {excel_path}")

    df_clean = df.dropna(subset=['latitude', 'longitude', 'Total'])
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df_clean['longitude'], df_clean['latitude'])
    ]
    return gpd.GeoDataFrame(
        df_clean[['Total']].rename(columns={'Total': 'population'}),
        geometry=geometry,
        crs='EPSG:4326',
    )


# ===========================================================================
# OD node preparation
# ===========================================================================

def prepare_od_nodes(
    df_worldpop: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    graph: ig.Graph,
) -> gpd.GeoDataFrame:
    """
    Reproject population points to network CRS and snap each to the
    nearest network vertex, storing the vertex index as 'vertex_id'.

    Args:
        df_worldpop: Population point GeoDataFrame (EPSG:4326).
        edges:        Edge GeoDataFrame (target CRS).
        graph:        igraph Graph built from edges.

    Returns:
        GeoDataFrame with columns ['geometry', 'population', 'vertex_id'].
    """
    od_nodes = df_worldpop.reset_index()
    od_nodes = od_nodes.set_crs(4326).to_crs(edges.crs)
    od_nodes.geometry = od_nodes.geometry.centroid

    # Vertex name → index lookup
    vertex_lookup = dict(
        zip(
            pd.DataFrame(graph.vs['name'])[0],
            pd.DataFrame(graph.vs['name']).index,
        )
    )

    # Extract node geometries from edge endpoints
    tqdm.pandas()
    from_geom = edges.geometry.progress_apply(lambda g: shapely.Point(g.coords[0]))
    to_geom   = edges.geometry.progress_apply(lambda g: shapely.Point(g.coords[-1]))

    nodes = pd.concat([
        pd.DataFrame.from_dict(dict(zip(edges['to_id'],   to_geom)),   orient='index', columns=['geometry']),
        pd.DataFrame.from_dict(dict(zip(edges['from_id'], from_geom)), orient='index', columns=['geometry']),
    ]).drop_duplicates()

    nodes['vertex_id'] = nodes.progress_apply(
        lambda row: vertex_lookup[row.name], axis=1
    )
    nodes = nodes.reset_index()

    sindex = shapely.STRtree(nodes.geometry)
    od_nodes['vertex_id'] = od_nodes.geometry.progress_apply(
        lambda geom: nodes.iloc[sindex.nearest(geom)].vertex_id
    ).values

    return od_nodes


def group_od_nodes(od_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aggregate OD nodes to unique vertices, summing population.

    Args:
        od_nodes: OD nodes with 'vertex_id' and 'population'.

    Returns:
        Grouped GeoDataFrame indexed by vertex_id.
    """
    grouped = (
        od_nodes
        .groupby('vertex_id')
        .agg(geometry=('geometry', 'first'), population=('population', 'sum'))
        .reset_index()
    )
    return grouped


# ===========================================================================
# Shortest-path matrices
# ===========================================================================

def run_shortest_paths(
    graph: ig.Graph,
    grouped_od: gpd.GeoDataFrame,
    weighting: str = 'fft',
) -> np.matrix:
    """
    Compute an all-pairs shortest-path distance matrix.

    Args:
        graph:      igraph Graph.
        grouped_od: Grouped OD nodes containing 'vertex_id'.
        weighting:  Edge attribute to use as weight.

    Returns:
        NumPy matrix of shape (n_od, n_od).
    """
    vertex_ids = grouped_od.vertex_id.values
    distance_matrix = graph.distances(
        source=vertex_ids,
        target=vertex_ids,
        weights=weighting,
    )
    return np.matrix(distance_matrix)


# ===========================================================================
# Demand matrix
# ===========================================================================

def create_demand(
    grouped_od: gpd.GeoDataFrame,
    od_baseline_time: np.matrix,
    config: CriticalityConfig,
) -> np.ndarray:
    """
    Create a demand matrix using a gravity model:

        D_{a,b} = Pop_a × Pop_b × exp(−decay × normalised_dist_{a,b})

    The result is scaled to ``config.max_trips`` and ceiling-rounded to integers.

    Args:
        grouped_od:       Grouped OD nodes with 'population'.
        od_baseline_time: Baseline time-distance matrix (n × n).
        config:           Criticality configuration.

    Returns:
        Integer demand matrix of shape (n, n).
    """
    node_pop = grouped_od.population.values
    n = len(grouped_od)
    demand = np.zeros((n, n))
    normalized_dist = od_baseline_time / np.nanmax(od_baseline_time)

    for origin in tqdm(range(n), total=n, desc='demand matrix'):
        for destination in range(n):
            if origin == destination:
                demand[origin][destination] = 0
            else:
                demand[origin][destination] = (
                    node_pop[origin] * node_pop[destination] *
                    np.exp(-config.dist_decay * normalized_dist[origin, destination])
                )

    demand = (demand / demand.max()) * config.max_trips
    return np.ceil(demand).astype(int)


# ===========================================================================
# Percolation analysis
# ===========================================================================

def _build_sister_pairs(graph: ig.Graph) -> Dict[int, Optional[int]]:
    """
    For each directed edge, find the reverse (sister) edge index if it exists.

    Args:
        graph: Directed igraph Graph.

    Returns:
        Mapping {edge_index: sister_index_or_None}.
    """
    edge_lookup = {(e.source, e.target): e.index for e in graph.es}
    return {
        edge.index: edge_lookup.get((edge.target, edge.source))
        for edge in graph.es
    }


def summarise_od(
    od_time: np.matrix,
    baseline_time: np.matrix,
    speed_ratio: np.ndarray,
    edge_length: float,
    demand: np.ndarray,
    edge_id: int,
    distance_disruption: float,
    time_disruption: float,
    config: CriticalityConfig,
) -> Tuple:
    """
    Compute disruption metrics for a single percolation step.

    Args:
        od_time:              Time matrix after edge removal.
        baseline_time:        Baseline time matrix.
        speed_ratio:          Element-wise speed ratio (km/hr) for all OD pairs.
        edge_length:          Combined physical length of removed edge(s) in km.
        demand:               Demand matrix.
        edge_id:              Identifier of the removed edge.
        distance_disruption:  Fractional reduction in total network length.
        time_disruption:      Fractional reduction in total network travel time.
        config:               Criticality configuration.

    Returns:
        Tuple of (edge_id, pct_isolated, pct_unaffected, pct_delayed,
                  average_time_disruption, average_dist_disruption,
                  distance_disruption, time_disruption).
    """
    adj_time = od_time - baseline_time
    total_trips = (baseline_time.shape[0] * baseline_time.shape[1]) - baseline_time.shape[0]

    isolated_sum = od_time[od_time == config.fail_value].shape[1]
    pct_isolated = (isolated_sum / total_trips) * 100

    time_unaffected = od_time[od_time == baseline_time]
    unaffected_count = np.array(time_unaffected).shape[1]
    pct_unaffected = (unaffected_count / total_trips) * 100

    delayed_mask = (
        (np.array(od_time) != np.array(baseline_time)) &
        (~np.isnan(np.array(od_time)))
    )
    isolated_mask = np.array(od_time) == config.fail_value

    delayed_times   = np.array(adj_time)[delayed_mask]
    delayed_speeds  = np.array(speed_ratio)[delayed_mask]
    delayed_demands = np.array(demand)[delayed_mask]

    isolated_within_delayed = np.array(od_time)[delayed_mask] == config.fail_value
    delayed_dists = delayed_times * delayed_speeds
    delayed_dists[isolated_within_delayed] = edge_length

    delayed_count = delayed_times.shape[0]
    pct_delayed = (delayed_count / total_trips) * 100

    if len(delayed_times) > 0 and delayed_demands.sum() > 0:
        average_time_disruption = float(np.average(delayed_times, weights=delayed_demands))
        average_dist_disruption = float(np.average(delayed_dists, weights=delayed_demands))
    else:
        average_time_disruption = np.nan
        average_dist_disruption = np.nan

    return (
        edge_id,
        pct_isolated,
        pct_unaffected,
        pct_delayed,
        average_time_disruption,
        average_dist_disruption,
        distance_disruption,
        time_disruption,
    )


def perform_percolation_analysis(
    graph: ig.Graph,
    grouped_od: gpd.GeoDataFrame,
    demand: np.ndarray,
    od_baseline_time: np.matrix,
    od_baseline_dist: np.matrix,
    config: CriticalityConfig,
) -> pd.DataFrame:
    """
    Remove each (directed) edge pair and measure network disruption.

    Twin directed edges sharing the same physical road section are removed
    together to reflect a real closure.

    Args:
        graph:             Directed igraph Graph.
        grouped_od:        Grouped OD nodes.
        demand:            Demand matrix.
        od_baseline_time:  Baseline time OD matrix (diagonal = NaN).
        od_baseline_dist:  Baseline distance OD matrix (diagonal = NaN).
        config:            Criticality configuration.

    Returns:
        DataFrame with one row per processed edge pair.
    """
    speed_ratio = np.array(od_baseline_dist) / np.array(od_baseline_time)

    sister_pairs = _build_sister_pairs(graph)
    tot_length = np.sum(graph.es['road_length'])
    tot_time   = np.sum(graph.es['fft'])
    edge_count = graph.ecount()

    results: List[Tuple] = []
    processed: Set[int] = set()

    for edge_idx in tqdm(range(edge_count), total=edge_count, desc='percolation'):
        if edge_idx in processed:
            continue

        sister = sister_pairs.get(edge_idx)

        # Combined physical length
        edge_length = graph.es[edge_idx]['road_length']
        if sister is not None:
            edge_length += graph.es[sister]['road_length']
            processed.add(sister)

        edges_to_remove = [edge_idx] + ([sister] if sister is not None else [])

        exp_g = graph.copy()
        exp_g.delete_edges(edges_to_remove)

        dist_disruption = 1 - (np.sum(exp_g.es['road_length']) / tot_length)
        time_disruption = 1 - (np.sum(exp_g.es['fft'])         / tot_time)

        perc_time = run_shortest_paths(exp_g, grouped_od, weighting=config.time_weight)
        np.fill_diagonal(perc_time, np.nan)
        perc_time[perc_time == np.inf] = config.fail_value

        summary = summarise_od(
            perc_time,
            od_baseline_time,
            speed_ratio,
            edge_length,
            demand,
            graph.es[edge_idx]['id'],
            dist_disruption,
            time_disruption,
            config,
        )
        results.append(summary)

    return pd.DataFrame(
        results,
        columns=[
            'edge_no', 'pct_isolated', 'pct_unaffected', 'pct_delayed',
            'average_time_disruption', 'average_dist_disruption',
            'distance_disruption', 'time_disruption',
        ],
    )


# ===========================================================================
# Derived metrics
# ===========================================================================

def calculate_impact_metrics(gdf_results: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add VHL, VKL, PHL, THL, PKL and TKL columns to the results GeoDataFrame.

    Vehicle metrics use total AADT; person/tonnage metrics use disaggregated
    AADT columns and standard occupancy / loading factors.

    Args:
        gdf_results: Merged results GeoDataFrame.

    Returns:
        Same GeoDataFrame with new metric columns.
    """
    gdf = gdf_results.copy()

    # Vehicle-based metrics
    gdf['vhl'] = gdf['average_time_disruption'] * gdf['total_aadt']
    gdf['vkl'] = gdf['average_dist_disruption'] * gdf['total_aadt']

    # Passenger-based metrics (occupancy factors: cars ×2, buses ×30)
    passenger_volume = gdf['passenger_cars'] * 2 + gdf['buses'] * 30
    gdf['phl'] = gdf['average_time_disruption'] * passenger_volume
    gdf['pkl'] = gdf['average_dist_disruption'] * passenger_volume

    # Tonnage-based metrics (loading factors: LT ×1.2, MT ×3.8, HT ×8.0, AV ×14.5)
    tonnage_volume = (
        gdf['light_trucks'] * 1.2 +
        gdf['medium_trucks'] * 3.8 +
        gdf['heavy_trucks'] * 8.0 +
        gdf['articulated_vehicles'] * 14.5
    )
    gdf['thl'] = gdf['average_time_disruption'] * tonnage_volume
    gdf['tkl'] = gdf['average_dist_disruption'] * tonnage_volume

    return gdf


# ===========================================================================
# Save results
# ===========================================================================

def save_criticality_results(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Save selected columns of the criticality results to a parquet file.

    Args:
        gdf_results: Full results GeoDataFrame.
        config:      Criticality configuration.
    """
    preferred_columns = [
        'from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
        'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
        'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'id',
        'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
        'heavy_trucks', 'articulated_vehicles', 'total_aadt',
        'oznaka_deo_left', 'index_right', 'oznaka_deo_right',
        'road_length', 'speed', 'fft', 'geometry',
        'edge_no', 'pct_isolated', 'pct_unaffected', 'pct_delayed',
        'average_time_disruption', 'average_dist_disruption',
        'distance_disruption', 'time_disruption',
        'vhl', 'phl', 'thl', 'pkl', 'tkl',
    ]
    output_cols = [c for c in preferred_columns if c in gdf_results.columns]
    out_path = config.intermediate_path / 'criticality_results.parquet'
    gdf_results[output_cols].to_parquet(out_path)
    _log(f"Criticality results saved to {out_path.resolve()}")


# ===========================================================================
# Visualisations – demand
# ===========================================================================

def plot_demand_heatmap(
    demand: np.ndarray,
    config: CriticalityConfig,
) -> None:
    """
    Display a heatmap of the top-6 × top-6 demand sub-matrix sorted by
    total demand, with custom city labels.

    Args:
        demand: Full demand matrix.
        config: Criticality configuration.
    """
    row_sums = np.sum(demand, axis=1)
    sorted_idx = np.argsort(row_sums)[::-1]
    sub_matrix = demand[np.ix_(sorted_idx, sorted_idx)][:6, :6]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sub_matrix,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=config.heatmap_labels,
        yticklabels=config.heatmap_labels,
        square=True,
        cbar=False,
    )
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().xaxis.tick_top()
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Visualisations – SPOF maps
# ===========================================================================

def _plot_classified_map(
    ax,
    gdf: gpd.GeoDataFrame,
    column: str,
    bins: List[float],
    labels: List[str],
    linewidths: Dict[str, float],
    colors: List[str],
    panel_label: str,
    legend_title: str,
    legend_suffix: str = '',
) -> None:
    """
    Internal helper: classify a column, plot each class, add basemap and legend.

    Args:
        ax:             Matplotlib axes.
        gdf:            GeoDataFrame to plot (must contain ``column``).
        column:         Column to classify.
        bins:           Classification bin edges.
        labels:         Class label strings.
        linewidths:     Mapping {label: linewidth}.
        colors:         List of colours matching class order.
        panel_label:    Subplot letter (A, B, C, D).
        legend_title:   Title text for the legend.
        legend_suffix:  Suffix appended to each legend label (e.g. ' hours').
    """
    gdf = gdf.copy()
    gdf['_class'] = pd.cut(gdf[column], bins=bins, labels=labels, include_lowest=True)

    for i, (class_name, width) in enumerate(linewidths.items()):
        subset = gdf[gdf['_class'] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(ax=ax, color=colors[i], linewidth=width, alpha=1)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron,
                   attribution=False)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_handles = [
        Line2D([0], [0], color=colors[i], lw=w, label=f'{lbl}{legend_suffix}')
        for i, (lbl, w) in enumerate(linewidths.items())
    ]
    ax.legend(
        handles=legend_handles, title=legend_title, loc='upper right',
        fontsize=10, title_fontsize=12, frameon=True, fancybox=True,
        shadow=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc',
    )
    ax.text(
        0.05, 0.95, panel_label, transform=ax.transAxes,
        fontsize=20, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )


def plot_spof_disruption_maps(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Create 2×2 map grid: (A) time delay, (B) VHL, (C) distance disruption,
    (D) VKL.  Saved as ``SPOF_results_2x2.png``.

    Args:
        gdf_results: Full results GeoDataFrame with computed metrics.
        config:      Criticality configuration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor='white')
    c = config.disruption_colors

    _plot_classified_map(
        axes[0, 0], gdf_results, 'average_time_disruption',
        config.time_bins, config.time_labels, config.time_linewidths, c,
        'A', 'Average Increased\nTravel Time',
    )
    _plot_classified_map(
        axes[0, 1], gdf_results, 'vhl',
        config.vhl_bins, config.vhl_labels, config.vhl_linewidths, c,
        'B', 'Vehicle Hours Lost', legend_suffix=' hours',
    )
    _plot_classified_map(
        axes[1, 0], gdf_results, 'average_dist_disruption',
        config.dist_bins, config.dist_labels, config.dist_linewidths, c,
        'C', 'Average Increased\nTravel Distance',
    )
    _plot_classified_map(
        axes[1, 1], gdf_results, 'vkl',
        config.vkl_bins, config.vkl_labels, config.vkl_linewidths, c,
        'D', 'Vehicle Kilometers Lost', legend_suffix=' km',
    )

    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_results_2x2.png',
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def plot_spof_person_maps(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Create 2×2 map grid: (A) PHL, (B) THL, (C) PKL, (D) TKL.
    Saved as ``SPOF_PHL_THL_PKL_TKL.png``.

    Args:
        gdf_results: Full results GeoDataFrame with computed metrics.
        config:      Criticality configuration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 14), facecolor='white')
    c = config.person_colors

    _plot_classified_map(
        axes[0, 0], gdf_results, 'phl',
        config.phl_bins, config.phl_labels, config.phl_linewidths, c,
        'A', 'Passenger Hours Lost', legend_suffix=' hours',
    )
    _plot_classified_map(
        axes[0, 1], gdf_results, 'thl',
        config.thl_bins, config.thl_labels, config.thl_linewidths, c,
        'B', 'Tonnage Hours Lost', legend_suffix=' ton-hours',
    )
    _plot_classified_map(
        axes[1, 0], gdf_results, 'pkl',
        config.pkl_bins, config.pkl_labels, config.pkl_linewidths, c,
        'C', 'Passenger Kilometers Lost', legend_suffix=' km',
    )
    _plot_classified_map(
        axes[1, 1], gdf_results, 'tkl',
        config.tkl_bins, config.tkl_labels, config.tkl_linewidths, c,
        'D', 'Tonnage Kilometers Lost', legend_suffix=' ton-km',
    )

    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_PHL_THL_PKL_TKL.png',
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


# ===========================================================================
# Visualisations – road-type distribution
# ===========================================================================

def plot_road_type_boxplots(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Create 2×2 boxplot grid of key metrics by road category.
    Saved as ``SPOF_by_road_type_boxplot.png``.

    Args:
        gdf_results: Full results GeoDataFrame.
        config:      Criticality configuration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    gdf_results.boxplot(column='average_time_disruption', by='kategorija', ax=axes[0, 0])
    axes[0, 0].set_title('Average Time Disruption by Road Type')
    axes[0, 0].set_xlabel('Road Category')
    axes[0, 0].set_ylabel('Hours')

    gdf_results.boxplot(column='phl', by='kategorija', ax=axes[0, 1])
    axes[0, 1].set_title('Passenger Hours Lost by Road Type')
    axes[0, 1].set_xlabel('Road Category')
    axes[0, 1].set_ylabel('Passenger Hours')

    gdf_results.boxplot(column='average_dist_disruption', by='kategorija', ax=axes[1, 0])
    axes[1, 0].set_title('Average Distance Disruption by Road Type')
    axes[1, 0].set_xlabel('Road Category')
    axes[1, 0].set_ylabel('Kilometers')

    gdf_results.boxplot(column='pkl', by='kategorija', ax=axes[1, 1])
    axes[1, 1].set_title('Passenger Kilometers Lost by Road Type')
    axes[1, 1].set_xlabel('Road Category')
    axes[1, 1].set_ylabel('Passenger Kilometers')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_by_road_type_boxplot.png',
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


def plot_road_type_violins(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Create 2×2 violin-plot grid of key metrics by road category.
    Saved as ``SPOF_by_road_type_violin.png``.

    Args:
        gdf_results: Full results GeoDataFrame.
        config:      Criticality configuration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        (axes[0, 0], 'average_time_disruption', 'Hours',
         'A. Average Time Disruption by Road Type'),
        (axes[0, 1], 'phl', 'Passenger Hours',
         'B. Passenger Hours Lost by Road Type'),
        (axes[1, 0], 'average_dist_disruption', 'Kilometers',
         'C. Average Distance Disruption by Road Type'),
        (axes[1, 1], 'pkl', 'Passenger Kilometers',
         'D. Passenger Kilometers Lost by Road Type'),
    ]
    for ax, col, ylabel, title in pairs:
        sns.violinplot(data=gdf_results, x='kategorija', y=col, ax=ax, cut=0)
        ax.set_title(title)
        ax.set_xlabel('Road Category')
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_by_road_type_violin.png',
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


# ===========================================================================
# Visualisations – AADT vs criticality
# ===========================================================================

def plot_aadt_vs_criticality(
    gdf_results: gpd.GeoDataFrame,
    config: CriticalityConfig,
) -> None:
    """
    Scatter plot of AADT versus PHL and PKL, coloured by time/distance
    disruption.  Saved as ``SPOF_AADT_vs_criticality.png``.

    Args:
        gdf_results: Full results GeoDataFrame.
        config:      Criticality configuration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(
        gdf_results['total_aadt'], gdf_results['phl'],
        c=gdf_results['average_time_disruption'], cmap='Reds', alpha=0.6, s=20,
    )
    axes[0].set_xlabel('AADT (vehicles/day)')
    axes[0].set_ylabel('Passenger Hours Lost')
    axes[0].set_title('A. AADT vs Passenger Hours Lost')
    plt.colorbar(sc1, ax=axes[0], label='Avg Time Disruption (hours)')
    max_aadt = gdf_results['total_aadt'].max()
    axes[0].plot([0, max_aadt], [0, max_aadt * 2 * 0.5], 'k--', alpha=0.3,
                 label='Reference (2 pax, 30 min)')
    axes[0].legend()

    sc2 = axes[1].scatter(
        gdf_results['total_aadt'], gdf_results['pkl'],
        c=gdf_results['average_dist_disruption'], cmap='Reds', alpha=0.6, s=20,
    )
    axes[1].set_xlabel('AADT (vehicles/day)')
    axes[1].set_ylabel('Passenger Kilometers Lost')
    axes[1].set_title('B. AADT vs Passenger Kilometers Lost')
    plt.colorbar(sc2, ax=axes[1], label='Avg Dist Disruption (km)')

    plt.tight_layout()
    plt.savefig(config.figures_path / 'SPOF_AADT_vs_criticality.png',
                dpi=config.figure_dpi, bbox_inches='tight')
    plt.show()


# ===========================================================================
# Summary statistics
# ===========================================================================

def print_summary_statistics(gdf_results: gpd.GeoDataFrame) -> None:
    """
    Print descriptive statistics, road-type breakdown, top sections and
    AADT–PHL regression results to stdout.

    Args:
        gdf_results: Full results GeoDataFrame with all metric columns.
    """
    metrics = [
        'average_time_disruption', 'average_dist_disruption',
        'phl', 'thl', 'pkl', 'tkl',
    ]
    total = len(gdf_results)

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(gdf_results[metrics].describe())

    # --- Time disruption breakdown ---
    no_delay   = (gdf_results['average_time_disruption'] == 0).sum()
    d_5_15     = gdf_results['average_time_disruption'].between(0,     0.25,  inclusive='right').sum()
    d_15_30    = gdf_results['average_time_disruption'].between(0.25,  0.5,   inclusive='right').sum()
    d_30_60    = gdf_results['average_time_disruption'].between(0.5,   1.0,   inclusive='right').sum()
    d_60_plus  = (gdf_results['average_time_disruption'] > 1).sum()

    print(f"\nTime Disruption (n={total}):")
    for label, count in [
        ('No delay',  no_delay),
        ('5-15 min',  d_5_15),
        ('15-30 min', d_15_30),
        ('30-60 min', d_30_60),
        ('60+ min',   d_60_plus),
    ]:
        print(f"  {label}: {count} ({count / total * 100:.1f}%)")

    # --- Top 10 sections ---
    id_cols = [
        'edge_no', 'oznaka_deo', 'kategorija', 'oznaka_put',
        'naziv_poce', 'naziv_zavr', 'total_aadt',
        'average_time_disruption', 'average_dist_disruption',
        'phl', 'pkl',
    ]
    id_cols = [c for c in id_cols if c in gdf_results.columns]

    print("\n" + "=" * 60)
    print("TOP 10 MOST CRITICAL SECTIONS – by PHL")
    print("=" * 60)
    print(gdf_results.nlargest(10, 'phl')[id_cols].to_string())

    print("\n" + "=" * 60)
    print("TOP 10 MOST CRITICAL SECTIONS – by PKL")
    print("=" * 60)
    print(gdf_results.nlargest(10, 'pkl')[id_cols].to_string())

    # --- Road type summary ---
    print("\n" + "=" * 60)
    print("DISTRIBUTION BY ROAD TYPE")
    print("=" * 60)
    road_summary = gdf_results.groupby('kategorija').agg(
        count=('edge_no', 'count'),
        time_mean=('average_time_disruption', 'mean'),
        time_max=('average_time_disruption', 'max'),
        dist_mean=('average_dist_disruption', 'mean'),
        phl_sum=('phl', 'sum'),
        pkl_sum=('pkl', 'sum'),
        aadt_mean=('total_aadt', 'mean'),
    ).round(2)
    print(road_summary)

    # --- AADT vs PHL regression ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        gdf_results['total_aadt'].fillna(0),
        gdf_results['phl'].fillna(0),
    )
    print(f"\nLinear Regression (AADT → PHL):")
    print(f"  R²={r_value**2:.3f}  slope={slope:.4f}  intercept={intercept:.2f}")


# ===========================================================================
# Main orchestration
# ===========================================================================

def main() -> None:
    """Orchestrate the full criticality analysis workflow."""
    config = CriticalityConfig()

    # ---------- 1. Load data ----------
    _log('Step 1: Loading network and graph...')
    edges, graph = load_network_and_graph(config)

    _log('Step 2: Loading population data...')
    df_worldpop = load_population_data(config)

    # ---------- 2. Build OD structure ----------
    _log('Step 3: Preparing OD nodes...')
    od_nodes = prepare_od_nodes(df_worldpop, edges, graph)

    _log('Step 4: Grouping OD nodes...')
    grouped_od = group_od_nodes(od_nodes)

    # ---------- 3. Baseline matrices ----------
    _log('Step 5: Computing baseline shortest-path matrices...')
    od_baseline_time = run_shortest_paths(graph, grouped_od, config.time_weight)
    od_baseline_dist = run_shortest_paths(graph, grouped_od, config.dist_weight)

    # ---------- 4. Demand matrix (must be built before diagonal is filled) ----------
    _log('Step 6: Creating demand matrix...')
    demand = create_demand(grouped_od, od_baseline_time, config)
    plot_demand_heatmap(demand, config)

    # Fill diagonals with NaN now that demand is built (matches notebook order)
    np.fill_diagonal(od_baseline_time, np.nan)
    np.fill_diagonal(od_baseline_dist, np.nan)

    # ---------- 5. Percolation ----------
    _log('Step 7: Performing percolation analysis...')
    result_df = perform_percolation_analysis(
        graph, grouped_od, demand,
        od_baseline_time, od_baseline_dist,
        config,
    )

    # ---------- 6. Merge & derive metrics ----------
    _log('Step 8: Merging results with network edges...')
    gdf_results = edges.merge(result_df, left_index=True, right_on='edge_no')

    _log('Step 9: Calculating impact metrics (VHL, PHL, THL, PKL, TKL)...')
    gdf_results = calculate_impact_metrics(gdf_results)

    # ---------- 7. Save ----------
    _log('Step 10: Saving criticality results...')
    save_criticality_results(gdf_results, config)

    # ---------- 8. Visualise ----------
    _log('Step 11: Creating SPOF disruption maps (time / VHL / distance / VKL)...')
    plot_spof_disruption_maps(gdf_results, config)

    _log('Step 12: Creating SPOF person/tonnage maps (PHL / THL / PKL / TKL)...')
    plot_spof_person_maps(gdf_results, config)

    _log('Step 13: Creating road-type distribution plots...')
    plot_road_type_boxplots(gdf_results, config)
    plot_road_type_violins(gdf_results, config)

    _log('Step 14: Creating AADT vs criticality scatter plots...')
    plot_aadt_vs_criticality(gdf_results, config)

    # ---------- 9. Statistics ----------
    _log('Step 15: Printing summary statistics...')
    print_summary_statistics(gdf_results)

    _log('Criticality analysis completed successfully!')


if __name__ == '__main__':
    main()
