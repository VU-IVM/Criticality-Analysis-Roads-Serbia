"""
Shared functions for the baseline accessibility analysis (step 3).

Single source of truth for:
- loading the road network, building the igraph graph and snapping points to nodes
- computing origin-destination / nearest-facility access times for factories,
  agricultural areas and emergency services (fire, hospitals, police)
- persisting results (Parquet + ArcGIS File Geodatabase, reprojected to the
  project output CRS)
- all figures and console summaries

Both the scripts (src/3a, src/3b) and the notebooks (notebooks/3a-3e) import from
this module so that they produce identical numbers, files and figures.

This module is self-contained: it does NOT depend on ``NetworkConfig``. Callers
pass plain values - input paths to the loaders, and output *directories* plus a
``show`` flag to the save/plot functions. Output filenames, the per-facility
travel-time column rename and the default output CRS live here, so scripts and
notebooks stay consistent. The scripts derive the directories from
``NetworkConfig``; the notebooks hard-code them.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import Point
import igraph as ig
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # For custom legend
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(
    action="ignore", category=RuntimeWarning
)  # exactextract gives a warning that is invalid


# =============================================================================
# Network loading, graph construction and spatial matching
# =============================================================================
def load_road_network(path: Path) -> gpd.GeoDataFrame:
    """
    Load Serbian road network parquet file.

    Args:
        path: path to the road network parquet file

    Returns:
        GeoDataFrame with Serbian road network
    """

    return gpd.read_parquet(path)


def create_graph_for_spatial_matching(
    base_network: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, ig.Graph]:
    """
    Create graph from road network and create nodes for spatial matching.

    Args:
        base_network: Serbian road network

    Returns:
        Pandas DataFrame with the nodes of the road network graph and the graph
    """

    # Create graph from road network
    edges = base_network.reindex(
        ["from_id", "to_id"]
        + [x for x in list(base_network.columns) if x not in ["from_id", "to_id"]],
        axis=1,
    )
    graph = ig.Graph.TupleList(
        edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:], directed=True
    )
    graph = graph.connected_components().giant()
    edges = edges[edges["id"].isin(graph.es["id"])]

    # Create nodes from edges for spatial matching
    vertex_lookup = dict(
        zip(pd.DataFrame(graph.vs["name"])[0], pd.DataFrame(graph.vs["name"]).index)
    )

    tqdm.pandas()
    from_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[0]))
    to_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[-1]))

    from_dict = dict(zip(edges["from_id"], from_id_geom))
    to_dict = dict(zip(edges["to_id"], to_id_geom))

    nodes = pd.concat(
        [
            pd.DataFrame.from_dict(to_dict, orient="index", columns=["geometry"]),
            pd.DataFrame.from_dict(from_dict, orient="index", columns=["geometry"]),
        ]
    ).drop_duplicates()

    nodes["vertex_id"] = nodes.apply(lambda x: vertex_lookup[x.name], axis=1)
    nodes = nodes.reset_index()

    return nodes, graph


def nearest_network_nodes(
    gdf_locations: gpd.GeoDataFrame, nodes: pd.DataFrame
) -> pd.Series:
    """
    Assign the nearest network node to each input point geometry (e.g., factories,
    agricultural areas or any other locations) using a spatial index.

    Parameters
    ----------
    gdf_locations : gpd.GeoDataFrame
        GeoDataFrame of point locations to snap to the network. Must contain a
        'geometry' (Point) column. Side effect: a 'vertex_id' column is
        created/overwritten with the nearest node identifier.
    nodes : pd.DataFrame
        Table of network nodes with 'geometry' (Point) and 'vertex_id' columns.

    Returns
    -------
    pd.Series
        Series of nearest node identifiers (vertex_id), index-aligned with
        gdf_locations, and also written to gdf_locations['vertex_id'].
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)
    tqdm.pandas()
    gdf_locations["vertex_id"] = gdf_locations.geometry.progress_apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return gdf_locations["vertex_id"]


def map_settlements_to_nodes_in_road_network(
    df_worldpop: gpd.GeoDataFrame, nodes: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Map each settlement to the nearest node in the road network.

    Args:
        df_worldpop: GeoDataFrame containing the settlements and their location,
        nodes: DataFrame containing the nodes of the road network

    Returns:
        GeoDataFrame of settlements with a new 'vertex_id' column
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)
    tqdm.pandas()
    df_worldpop["vertex_id"] = df_worldpop.geometry.progress_apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return df_worldpop


# =============================================================================
# Input data loaders (take an explicit input path)
# =============================================================================
def load_factory_data(path: Path) -> gpd.GeoDataFrame:
    """
    Load location of factories in Serbia from Excel file.

    Args:
        path: path to the factory Excel file

    Returns:
        GeoDataFrame with Serbian factories
    """

    DataFrame_Factory = pd.read_excel(path)

    Clean_DataFrame_Factory = DataFrame_Factory.dropna(
        subset=["Latitude", "Longitude", "Factory"]
    )

    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_Factory["Longitude"], Clean_DataFrame_Factory["Latitude"]
        )
    ]

    df_factories = gpd.GeoDataFrame(
        Clean_DataFrame_Factory[["Number"]].copy(), geometry=geometry, crs="EPSG:4326"
    )

    return df_factories


def load_border_crossings(path: Path, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Load border crossings (Sinks) from Excel file and snap them to network nodes.

    Args:
        path: path to the borders Excel file, nodes: nodes of the road network graph

    Returns:
        DataFrame with border crossings and their nearest 'vertex_id'
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)

    Sink = pd.read_excel(path)
    Sink = Sink.rename(columns={"LON": "Longitude", "LAT": "Latitude"})
    Sink["geometry"] = Sink.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    Sink["vertex_id"] = Sink.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return Sink


def load_agricultural_data(path: Path) -> gpd.GeoDataFrame:
    """
    Load location data of agricultural areas from xlsm file.

    Args:
        path: path to the agriculture Excel file

    Returns:
        GeoDataFrame with agricultural areas
    """
    DataFrame_Agri = pd.read_excel(path)

    Clean_DataFrame_Agri = DataFrame_Agri.dropna(
        subset=["latitude", "longitude", "Utilized agricultural land (UAL)"]
    )

    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_Agri["longitude"], Clean_DataFrame_Agri["latitude"]
        )
    ]

    df_agri = gpd.GeoDataFrame(
        Clean_DataFrame_Agri[["Utilized agricultural land (UAL)"]].rename(
            columns={"Utilized agricultural land (UAL)": "UAL"}
        ),
        geometry=geometry,
        crs="EPSG:4326",
    )

    return df_agri


def load_sinks(
    path: Path, nodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load location of border crossings, ports and rail terminals.

    Args:
        path: path to the combined sinks Excel file, nodes: nodes of the road network graph

    Returns:
        Border crossings, ports, rail terminals, and all sinks combined (DataFrames)
    """

    Sinks = pd.read_excel(path)
    Sinks = Sinks.rename(
        columns={"LON": "Longitude", "LAT": "Latitude", "TYPE OF\nTRAFFIC": "type"}
    )
    Sinks["geometry"] = Sinks.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    nodes_sindex = shapely.STRtree(nodes.geometry)
    Sinks["vertex_id"] = Sinks.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    # Split by type
    Sinks_road = Sinks[Sinks["type"] == "road"]
    Sinks_port = Sinks[Sinks["type"] == "port"]
    Sinks_rail = Sinks[Sinks["type"] == "rail"]

    print(f"Road border crossings: {len(Sinks_road)}")
    print(f"Ports: {len(Sinks_port)}")
    print(f"Rail terminals: {len(Sinks_rail)}")

    return Sinks_road, Sinks_port, Sinks_rail, Sinks


def load_population_data(path: Path) -> gpd.GeoDataFrame:
    """
    Load settlement location data.

    Args:
        path: path to the settlement population Excel file

    Returns:
        GeoDataFrame with location and population of settlements
    """

    # reading the Excel file
    DataFrame_StatePop = pd.read_excel(path)

    # to keep only rows with valid coordinates and population
    Clean_DataFrame_StatePop = DataFrame_StatePop.dropna(
        subset=["latitude", "longitude", "Total"]
    )

    # to make point geometry
    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_StatePop["longitude"], Clean_DataFrame_StatePop["latitude"]
        )
    ]

    # build GeoDataFrame matching df_worldpop structure
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_StatePop[["Total"]].rename(columns={"Total": "population"}),
        geometry=geometry,
        crs="EPSG:4326",  # longitude/latitude WGS84
    )

    return df_worldpop


def load_and_map_sinks(path: Path, nodes: pd.DataFrame, sink_type: str) -> pd.DataFrame:
    """
    Load sink data (firefighters, hospitals or police stations) and map them to
    the nearest node in the road network.

    Args:
        path: path to the sink Excel file, nodes: nodes of the road network,
        sink_type: identifier for the sink category ('firefighters', 'hospitals' or 'police')

    Returns:
        DataFrame with location of sinks and their nearest network node
    """
    nodes_sindex = shapely.STRtree(nodes.geometry)

    if sink_type == "firefighters":
        Sink = pd.read_excel(path)
        Sink = Sink.rename(columns={"lon": "Longitude", "lat": "Latitude"})
    elif sink_type == "hospitals":
        Sink = pd.read_excel(path)
    elif sink_type == "police":
        Sink = pd.read_excel(path)
        Sink = Sink.rename(columns={"lon": "Longitude", "lat": "Latitude"})
    else:
        raise ValueError(
            f"Invalid sink_type '{sink_type}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    Sink["geometry"] = Sink.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    Sink["vertex_id"] = Sink.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return Sink


# =============================================================================
# Accessibility calculations
# =============================================================================
def calculate_average_access_time(
    df_factories: gpd.GeoDataFrame, Sink: pd.DataFrame, graph: ig.Graph
) -> tuple[pd.Series, np.array]:
    """
    Calculate the origin-destination matrix and the average access times for
    factories to reach all border crossings.

    Args:
        df_factories: industrial centers in Serbia, Sink: border crossings,
        graph: graph representation of Serbia's road network

    Returns:
        Series with average access time of factories to road borders and the
        baseline origin-destination matrix
    """

    factory_vertices = df_factories["vertex_id"].unique()
    sink_vertices = Sink["vertex_id"].unique()

    OD_baseline = np.array(
        graph.distances(source=factory_vertices, target=sink_vertices, weights="fft")
    )
    OD_baseline[np.isinf(OD_baseline)] = 12

    avg_time_per_factory = np.mean(OD_baseline, axis=1)
    vertex_to_avg_time = dict(zip(factory_vertices, avg_time_per_factory))
    df_factories["avg_access_time"] = df_factories["vertex_id"].map(vertex_to_avg_time)

    return df_factories["avg_access_time"], OD_baseline


def calculate_access_times(graph, origin_vertices, sink_df, sink_name):
    """
    Calculate average access time from origin vertices to a set of sinks.

    Args:
        graph: graph representation of the road network, origin_vertices: origin
        vertices, sink_df: sink DataFrame, sink_name: label for printing

    Returns:
        dict mapping origin vertices to average time, and the OD matrix
    """
    sink_vertices = sink_df["vertex_id"].unique()

    OD_matrix = np.array(
        graph.distances(source=origin_vertices, target=sink_vertices, weights="fft")
    )
    OD_matrix[np.isinf(OD_matrix)] = 12

    avg_time = np.mean(OD_matrix, axis=1)
    vertex_to_avg = dict(zip(origin_vertices, avg_time))

    print(
        f"\n{sink_name}: {len(sink_vertices)} destinations, global avg = {np.mean(OD_matrix):.2f} hours"
    )

    return vertex_to_avg, OD_matrix


def get_distance_to_nearest_facility(
    df_population: gpd.GeoDataFrame,
    Sink: pd.DataFrame,
    graph: ig.Graph,
    id_col: str = "i.d.",
) -> gpd.GeoDataFrame:
    """
    Calculate the distance to the nearest facility for each origin (settlement or
    agricultural area), using a bulk distance matrix over unique vertices.

    Args:
        df_population: GeoDataFrame of origins with a 'vertex_id' column,
        Sink: DataFrame of facilities with 'vertex_id' and an identifier column,
        graph: graph of the road network,
        id_col: name of the facility identifier column ('i.d.' for emergency
        services, 'N°' for agricultural sinks)

    Returns:
        GeoDataFrame with the distance to the nearest facility for each origin
    """
    # Initialize new columns
    df_population = df_population.copy()
    df_population["closest_sink_vertex_id"] = None
    df_population["closest_sink_osm_id"] = None
    df_population["closest_sink_total_fft"] = None

    # Get unique vertex IDs for both population and sinks
    unique_pop_vertex_ids = df_population["vertex_id"].unique()
    unique_sink_vertex_ids = Sink["vertex_id"].unique()

    # Create mapping from unique sink vertex_ids back to original sink data
    sink_lookup = {}
    for _, row in Sink.iterrows():
        sink_lookup[row["vertex_id"]] = row[id_col]

    # Calculate distance matrix once for unique vertices only
    distance_matrix = np.array(
        graph.distances(
            source=unique_pop_vertex_ids, target=unique_sink_vertex_ids, weights="fft"
        )
    )

    # Create lookup dictionary: vertex_id -> (closest_sink_vertex_id, closest_sink_osm_id, min_distance)
    vertex_to_closest_sink = {}

    for i, pop_vertex_id in enumerate(unique_pop_vertex_ids):
        # Get distances from this population point to all unique sinks
        distances_to_sinks = distance_matrix[i, :]

        # Find the index of the minimum distance
        min_sink_idx = np.argmin(distances_to_sinks)
        min_distance = distances_to_sinks[min_sink_idx]

        # Handle infinite distances (no path found)
        if np.isinf(min_distance):
            vertex_to_closest_sink[pop_vertex_id] = (None, None, float("inf"))
        else:
            closest_sink_vertex_id = unique_sink_vertex_ids[min_sink_idx]
            closest_sink_osm_id = sink_lookup[closest_sink_vertex_id]
            vertex_to_closest_sink[pop_vertex_id] = (
                closest_sink_vertex_id,
                closest_sink_osm_id,
                min_distance,
            )

    # Map results back to all population points (including duplicates)
    for idx, row in df_population.iterrows():
        vertex_id = row["vertex_id"]
        closest_sink_vertex_id, closest_sink_osm_id, closest_sink_total_fft = (
            vertex_to_closest_sink[vertex_id]
        )

        df_population.at[idx, "closest_sink_vertex_id"] = closest_sink_vertex_id
        df_population.at[idx, "closest_sink_osm_id"] = closest_sink_osm_id
        df_population.at[idx, "closest_sink_total_fft"] = closest_sink_total_fft

    return df_population


def calculate_OD_matrix(
    df_agri: pd.DataFrame,
    graph: ig.Graph,
    Sinks_road: pd.DataFrame,
    Sinks_port: pd.DataFrame,
    Sinks_rail: pd.DataFrame,
    Sinks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average and nearest access time from agricultural areas to border
    crossings, ports and rail terminals.

    Args:
        df_agri: agricultural areas, graph: road network graph, Sinks_road: border
        crossings, Sinks_port: ports, Sinks_rail: rail terminals, Sinks: all sinks combined

    Returns:
        DataFrame with all average and nearest access times
    """
    agri_vertices = df_agri["vertex_id"].unique()
    # Calculate for each sink type
    road_access, OD_road = calculate_access_times(
        graph, agri_vertices, Sinks_road, "Road borders"
    )
    port_access, OD_port = calculate_access_times(
        graph, agri_vertices, Sinks_port, "Ports"
    )
    rail_access, OD_rail = calculate_access_times(
        graph, agri_vertices, Sinks_rail, "Rail terminals"
    )

    # Also calculate combined (all sinks)
    all_access, OD_all = calculate_access_times(
        graph, agri_vertices, Sinks, "All sinks combined"
    )

    # Map back to df_agri
    df_agri["avg_access_road"] = df_agri["vertex_id"].map(road_access)
    df_agri["avg_access_port"] = df_agri["vertex_id"].map(port_access)
    df_agri["avg_access_rail"] = df_agri["vertex_id"].map(rail_access)
    df_agri["avg_access_all"] = df_agri["vertex_id"].map(all_access)

    df_agri["nearest_access_road"] = get_distance_to_nearest_facility(
        df_agri, Sinks_road, graph, id_col="N°"
    )["closest_sink_total_fft"]
    df_agri["nearest_access_port"] = get_distance_to_nearest_facility(
        df_agri, Sinks_port, graph, id_col="N°"
    )["closest_sink_total_fft"]
    df_agri["nearest_access_rail"] = get_distance_to_nearest_facility(
        df_agri, Sinks_rail, graph, id_col="N°"
    )["closest_sink_total_fft"]

    return df_agri


# =============================================================================
# Persisting results (Parquet + ArcGIS File Geodatabase, in the output CRS)
# =============================================================================
# facility_type -> (results file/layer stem, sink file/layer stem,
#                   travel-time column name used in the saved results file)
# These stems are the single source of truth for the output filenames; they match
# the corresponding ``Path_*`` attributes in NetworkConfig.
_FACILITY_SPEC = {
    "firefighters": (
        "firefighter_accessibility_results",
        "firefighters",
        "travel_time_ff",
    ),
    "hospitals": (
        "hospital_accessibility_results",
        "hospitals",
        "travel_time_hosp",
    ),
    "police": ("police_accessibility_results", "police", "travel_time_pol"),
    "factories": ("factory_accessibility", "factories_sinks", None),
    "agriculture": ("agriculture_accessibility", "agriculture_sinks", None),
}

# Fallback output CRS, used only when a caller does not pass ``output_crs``.
# Scripts pass ``config.output_crs``; notebooks set ``output_crs`` at the top.
# (MGI 1901 / Balkans zone 7)
DEFAULT_OUTPUT_CRS = "EPSG:6316"


def _as_geodataframe(obj, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Wrap a DataFrame/GeoDataFrame as a GeoDataFrame with a defined CRS.

    Inputs may come from disk (already projected to the output CRS) or be freshly
    computed lon/lat data without a CRS; in the latter case ``crs`` (WGS84) is set.
    """
    gdf = (
        obj
        if isinstance(obj, gpd.GeoDataFrame)
        else gpd.GeoDataFrame(obj, geometry="geometry")
    )
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    return gdf


def _to_web_mercator(obj) -> gpd.GeoDataFrame:
    """Return a Web-Mercator (EPSG:3857) copy for plotting on a basemap.

    Works regardless of the input CRS: in-memory lon/lat results (no CRS) are
    assumed WGS84, while results loaded from disk already carry the output CRS.
    """
    return _as_geodataframe(obj).to_crs(3857)


def _savefig(figure_dir: Path, filename: str, **kwargs) -> None:
    """Save the current figure to ``figure_dir/filename``, creating the dir."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / filename, **kwargs)


def _save_vector(
    obj, parquet_dir: Path, gdb_path: Path, stem: str, output_crs: str
) -> None:
    """
    Save a (Geo)DataFrame to both ``parquet_dir/<stem>.parquet`` and the File
    Geodatabase ``gdb_path`` (layer ``stem``), reprojected to ``output_crs``.
    """
    gdf = _as_geodataframe(obj).to_crs(output_crs)

    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{stem}.parquet"
    gdf.to_parquet(parquet_path)

    gdb_path = Path(gdb_path)
    gdb_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gdb_path, driver="OpenFileGDB", layer=stem)

    print(f"Saved {stem} -> {parquet_path}")
    print(f"Saved {stem} -> {gdb_path} (layer '{stem}')")


def save_accessibility_results(
    df_results,
    Sink,
    facility_type: str,
    parquet_dir: Path,
    gdb_path: Path,
    output_crs: str = DEFAULT_OUTPUT_CRS,
) -> None:
    """
    Save accessibility results and sink geometries for a facility type to both
    Parquet and the ArcGIS File Geodatabase, reprojected to ``output_crs``.

    For the emergency services the generic 'closest_sink_total_fft' column is
    renamed to a facility-specific, descriptive name in the saved results file
    (and renamed back by :func:`load_accessibility_results`).

    Args:
        df_results: accessibility results, Sink: facility locations,
        facility_type: one of {'firefighters', 'hospitals', 'police',
        'factories', 'agriculture'}, parquet_dir: directory for the .parquet files,
        gdb_path: path to the .gdb that receives the layers,
        output_crs: CRS to reproject outputs to (default ``DEFAULT_OUTPUT_CRS``)
    """
    if facility_type not in _FACILITY_SPEC:
        raise ValueError(
            f"Invalid facility_type '{facility_type}'. Expected one of: "
            f"{', '.join(_FACILITY_SPEC)}."
        )

    results_stem, sink_stem, time_col = _FACILITY_SPEC[facility_type]

    df_out = df_results
    if time_col is not None:
        df_out = df_results.rename(columns={"closest_sink_total_fft": time_col})

    _save_vector(df_out, parquet_dir, gdb_path, results_stem, output_crs)
    _save_vector(Sink, parquet_dir, gdb_path, sink_stem, output_crs)


def load_accessibility_results(
    facility_type: str, parquet_dir: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load accessibility results and facility locations for a facility type from
    ``parquet_dir``.

    The facility-specific travel-time column written by
    :func:`save_accessibility_results` is renamed back to 'closest_sink_total_fft'
    so downstream plotting is facility-agnostic.

    Args:
        facility_type: one of {'firefighters', 'hospitals', 'police',
        'factories', 'agriculture'}, parquet_dir: directory holding the .parquet files

    Returns:
        (results GeoDataFrame, sink GeoDataFrame)
    """
    if facility_type not in _FACILITY_SPEC:
        raise ValueError(
            f"Invalid facility_type '{facility_type}'. Expected one of: "
            f"{', '.join(_FACILITY_SPEC)}."
        )

    results_stem, sink_stem, time_col = _FACILITY_SPEC[facility_type]
    parquet_dir = Path(parquet_dir)

    gdf = gpd.read_parquet(parquet_dir / f"{results_stem}.parquet")
    if time_col is not None:
        gdf = gdf.rename(columns={time_col: "closest_sink_total_fft"})
    Sink = gpd.read_parquet(parquet_dir / f"{sink_stem}.parquet")

    return gdf, Sink


# =============================================================================
# Console summaries
# =============================================================================
def print_statistics(
    df_factories: pd.DataFrame, Sink: pd.DataFrame, OD_baseline: np.array
) -> None:
    """Print baseline accessibility statistics for factories to the console."""

    print("=" * 60)
    print("BASELINE ACCESSIBILITY SUMMARY")
    print("=" * 60)

    # Basic stats
    print(f"\nNumber of factories: {len(df_factories)}")
    print(f"Number of border crossings: {len(Sink)}")
    print(f"Number of OD pairs: {OD_baseline.size}")

    print("\n--- Access Time Statistics (hours) ---")
    print(f"Mean:   {df_factories['avg_access_time'].mean():.2f}")
    print(f"Median: {df_factories['avg_access_time'].median():.2f}")
    print(f"Std:    {df_factories['avg_access_time'].std():.2f}")
    print(f"Min:    {df_factories['avg_access_time'].min():.2f}")
    print(f"Max:    {df_factories['avg_access_time'].max():.2f}")

    # Percentiles
    print("\n--- Percentiles (hours) ---")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"P{p}: {df_factories['avg_access_time'].quantile(p / 100):.2f}")

    # Category distribution
    print("\n--- Factories by Access Time Category ---")
    bins = [1, 2, 3, 4, 5, float("inf")]
    labels = ["1-2", "2-3", "3-4", "4-5", "5+"]
    df_factories["category"] = pd.cut(
        df_factories["avg_access_time"], bins=bins, labels=labels, right=False
    )

    category_counts = df_factories["category"].value_counts().sort_index()
    for cat, count in category_counts.items():
        pct = count / len(df_factories) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Unreachable pairs in OD matrix
    n_unreachable = np.sum(OD_baseline == 12)
    pct_unreachable = n_unreachable / OD_baseline.size * 100
    print("\n--- Connectivity ---")
    print(f"Unreachable OD pairs: {n_unreachable} ({pct_unreachable:.1f}%)")


def print_statistics_agriculture(df_agri: pd.DataFrame) -> None:
    """Print average access time statistics for agricultural areas to the console."""
    print("\n" + "=" * 60)
    print("AGRICULTURAL ACCESSIBILITY SUMMARY")
    print("=" * 60)

    print(f"\nNumber of agricultural locations: {len(df_agri)}")
    print(f"Total UAL (ha): {df_agri['UAL'].sum():,.0f}")

    for col, label in [
        ("avg_access_road", "Road Borders"),
        ("avg_access_port", "Ports"),
        ("avg_access_rail", "Rail Terminals"),
        ("avg_access_all", "All Combined"),
    ]:
        print(f"\n--- {label} ---")
        print(f"  Mean:   {df_agri[col].mean():.2f} hours")
        print(f"  Median: {df_agri[col].median():.2f} hours")
        print(f"  Std:    {df_agri[col].std():.2f} hours")
        print(f"  Min:    {df_agri[col].min():.2f} hours")
        print(f"  Max:    {df_agri[col].max():.2f} hours")
        print(f"  P10:    {df_agri[col].quantile(0.10):.2f} hours")
        print(f"  P90:    {df_agri[col].quantile(0.90):.2f} hours")

    # Category distribution for each type
    bins = [0, 0.5, 1, 2, 5, float("inf")]
    labels_cat = ["0-0.5h", "0.5-1h", "1-2h", "2-5h", "5h+"]

    print("\n--- Distribution by Access Time Category ---")
    for col, label in [
        ("avg_access_road", "Road"),
        ("avg_access_port", "Port"),
        ("avg_access_rail", "Rail"),
    ]:
        print(f"\n{label}:")
        cat = pd.cut(df_agri[col], bins=bins, labels=labels_cat, right=False)
        for c in labels_cat:
            count = (cat == c).sum()
            pct = count / len(df_agri) * 100
            print(f"  {c}: {count} ({pct:.1f}%)")


def print_statistics_emergency_accessibility(
    df_worldpop_fire=None,
    Sink_fire=None,
    df_worldpop_hospital=None,
    Sink_hospitals=None,
    df_worldpop_police=None,
    Sink_police=None,
):
    """
    Print accessibility statistics for the emergency services (fire, hospitals,
    police). All inputs are optional; only services with both a results table and
    corresponding sinks are analyzed. Expects 'closest_sink_total_fft' in hours.
    """

    # Define bins and labels
    bins = [0, 0.25, 0.5, 1, 1.5, 2, float("inf")]
    labels = ["0-15", "15-30", "30-60", "60-90", "90-120", ">120"]

    # Prepare datasets
    datasets = {}

    if df_worldpop_fire is not None and Sink_fire is not None:
        datasets["Fire Departments"] = (df_worldpop_fire, Sink_fire)

    if df_worldpop_hospital is not None and Sink_hospitals is not None:
        datasets["Hospitals"] = (df_worldpop_hospital, Sink_hospitals)

    if df_worldpop_police is not None and Sink_police is not None:
        datasets["Police Stations"] = (df_worldpop_police, Sink_police)

    print("=" * 70)
    print("EMERGENCY SERVICES ACCESSIBILITY ANALYSIS")
    print("=" * 70)

    summary_data = []
    pop_col = None

    for service_name, (df_worldpop, Sink) in datasets.items():
        print(f"\n{'─' * 50}")
        print(f"{service_name.upper()}")
        print(f"{'─' * 50}")

        # Number of service locations
        n_facilities = len(Sink)
        print(f"\nNumber of {service_name}: {n_facilities:,}")

        # Number of settlements analyzed
        n_settlements = len(df_worldpop)
        print(f"Number of settlements analyzed: {n_settlements:,}")

        # Access time statistics (convert to minutes)
        access_time_minutes = df_worldpop["closest_sink_total_fft"] * 60

        print("\nAccess Time Statistics (minutes):")
        print(f"  Mean: {access_time_minutes.mean():.2f} minutes")
        print(f"  Median: {access_time_minutes.median():.2f} minutes")
        print(f"  Std Dev: {access_time_minutes.std():.2f} minutes")
        print(f"  Min: {access_time_minutes.min():.2f} minutes")
        print(f"  Max: {access_time_minutes.max():.2f} minutes")

        # Create categories
        df_worldpop["category"] = pd.cut(
            df_worldpop["closest_sink_total_fft"], bins=bins, labels=labels, right=False
        )
        df_worldpop["category"] = df_worldpop["category"].astype("object")
        df_worldpop.loc[df_worldpop["category"].isna(), "category"] = "Not Accessible"

        # Distribution by category
        print("\nSettlements by Access Time Category:")
        cat_counts = df_worldpop["category"].value_counts()
        cat_order = labels + ["Not Accessible"]
        for cat in cat_order:
            if cat in cat_counts.index:
                count = cat_counts[cat]
                pct = (count / n_settlements) * 100
                print(f"  {cat:>12} min: {count:>6,} settlements ({pct:>5.1f}%)")

        # Population-weighted analysis (if population column exists)
        pop_col = None
        for col in ["population", "pop", "worldpop", "pop_sum", "population_sum"]:
            if col in df_worldpop.columns:
                pop_col = col
                break

        if pop_col:
            total_pop = df_worldpop[pop_col].sum()
            print(f"\nTotal Population Covered: {total_pop:,.0f}")

            print("\nPopulation by Access Time Category:")
            for cat in cat_order:
                subset = df_worldpop[df_worldpop["category"] == cat]
                if len(subset) > 0:
                    pop = subset[pop_col].sum()
                    pct = (pop / total_pop) * 100
                    print(f"  {cat:>12} min: {pop:>12,.0f} people ({pct:>5.1f}%)")

            # Population-weighted mean access time
            valid_data = df_worldpop[df_worldpop["closest_sink_total_fft"].notna()]
            if len(valid_data) > 0 and valid_data[pop_col].sum() > 0:
                weighted_mean = np.average(
                    valid_data["closest_sink_total_fft"] * 60,
                    weights=valid_data[pop_col],
                )
                print(
                    f"\nPopulation-Weighted Mean Access Time: {weighted_mean:.2f} minutes"
                )

        # Key thresholds
        print("\nKey Coverage Statistics:")
        within_15 = len(df_worldpop[df_worldpop["closest_sink_total_fft"] < 0.25])
        within_30 = len(df_worldpop[df_worldpop["closest_sink_total_fft"] < 0.5])
        within_60 = len(df_worldpop[df_worldpop["closest_sink_total_fft"] < 1])
        beyond_60 = len(df_worldpop[df_worldpop["closest_sink_total_fft"] >= 1])
        not_accessible = len(df_worldpop[df_worldpop["category"] == "Not Accessible"])

        print(
            f"  Within 15 minutes: {within_15:,} ({within_15 / n_settlements * 100:.1f}%)"
        )
        print(
            f"  Within 30 minutes: {within_30:,} ({within_30 / n_settlements * 100:.1f}%)"
        )
        print(
            f"  Within 60 minutes: {within_60:,} ({within_60 / n_settlements * 100:.1f}%)"
        )
        print(
            f"  Beyond 60 minutes: {beyond_60:,} ({beyond_60 / n_settlements * 100:.1f}%)"
        )
        print(
            f"  Not Accessible: {not_accessible:,} ({not_accessible / n_settlements * 100:.1f}%)"
        )

        # Collect summary for comparison table
        summary_data.append(
            {
                "Service": service_name,
                "Facilities": n_facilities,
                "Settlements": n_settlements,
                "Mean Access (min)": round(access_time_minutes.mean(), 1),
                "Median Access (min)": round(access_time_minutes.median(), 1),
                "Within 15 min (%)": round(within_15 / n_settlements * 100, 1),
                "Within 30 min (%)": round(within_30 / n_settlements * 100, 1),
                "Within 60 min (%)": round(within_60 / n_settlements * 100, 1),
                "Not Accessible (%)": round(not_accessible / n_settlements * 100, 1),
            }
        )

    # Comparison summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # If population data exists, add population comparison
    if pop_col:
        print("\n" + "=" * 70)
        print("POPULATION COVERAGE COMPARISON")
        print("=" * 70)

        pop_summary = []
        for service_name, (df_worldpop, Sink) in datasets.items():
            df_worldpop["category"] = pd.cut(
                df_worldpop["closest_sink_total_fft"],
                bins=bins,
                labels=labels,
                right=False,
            )
            df_worldpop["category"] = df_worldpop["category"].astype("object")
            df_worldpop.loc[df_worldpop["category"].isna(), "category"] = (
                "Not Accessible"
            )

            total_pop = df_worldpop[pop_col].sum()
            pop_within_30 = df_worldpop[df_worldpop["closest_sink_total_fft"] < 0.5][
                pop_col
            ].sum()
            pop_within_60 = df_worldpop[df_worldpop["closest_sink_total_fft"] < 1][
                pop_col
            ].sum()
            pop_not_accessible = df_worldpop[
                df_worldpop["category"] == "Not Accessible"
            ][pop_col].sum()

            pop_summary.append(
                {
                    "Service": service_name,
                    "Total Population": f"{total_pop:,.0f}",
                    "Pop Within 30 min": f"{pop_within_30:,.0f} ({pop_within_30 / total_pop * 100:.1f}%)",
                    "Pop Within 60 min": f"{pop_within_60:,.0f} ({pop_within_60 / total_pop * 100:.1f}%)",
                    "Pop Not Accessible": f"{pop_not_accessible:,.0f} ({pop_not_accessible / total_pop * 100:.1f}%)",
                }
            )

        pop_df = pd.DataFrame(pop_summary)
        print(pop_df.to_string(index=False))


# =============================================================================
# Figures
# =============================================================================
def plot_access_times_factories(
    df_factories: pd.DataFrame, Sink: pd.DataFrame, figure_dir: Path, show: bool = True
) -> None:
    """Plot average access times from factories to border crossings and save the map."""

    df_factories_plot = _to_web_mercator(df_factories)
    Sink_plot = _to_web_mercator(Sink)

    bins = [1, 2, 3, 4, 5, float("inf")]
    labels = ["1-2", "2-3", "3-4", "4-5", "5+"]
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]

    df_factories_plot["category"] = pd.cut(
        df_factories_plot["avg_access_time"], bins=bins, labels=labels, right=False
    )
    df_factories_plot["category"] = df_factories_plot["category"].astype("object")
    df_factories_plot.loc[df_factories_plot["category"].isna(), "category"] = (
        "Not Accessible"
    )

    color_map = dict(zip(labels, colors))
    color_map["Not Accessible"] = "#bdbdbd"

    fig, ax = plt.subplots(figsize=(24, 14))

    for category, color in color_map.items():
        data = df_factories_plot[df_factories_plot["category"] == category]
        if not data.empty:
            data.plot(
                ax=ax,
                color=color,
                legend=False,
                linewidth=0.1,
                edgecolor="grey",
                markersize=200,
            )

    Sink_plot.plot(ax=ax, color="black", markersize=200, marker="^")
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    ax.set_aspect("equal")
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=color, label=f"{label} hours")
        for label, color in zip(labels, colors)
    ]
    legend_patches.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            lw=0,
            label="Border Crossings",
            markersize=15,
        )
    )

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=12,
        title="Average Travel Time",
        title_fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
    )

    _savefig(figure_dir, "factory_access_avg.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_accessibility_curves_agriculture(
    df_agri: pd.DataFrame, figure_dir: Path, show: bool = True
) -> None:
    """Plot 3x2 accessibility curves for agricultural areas (road/port/rail; nearest vs. avg)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey="row")

    # Define thresholds (shared)
    thresholds = np.arange(0, 8.1, 0.5)

    # Total agricultural land
    total_ual = df_agri["UAL"].sum()

    sink_configs = [
        # Top row - Nearest
        {
            "ax": axes[0, 0],
            "col": "nearest_access_road",
            "label": "A",
            "title": "road border crossings",
            "metric": "Nearest",
            "show_ylabel": True,
            "text_offset": (0.1, 94),
        },
        {
            "ax": axes[0, 1],
            "col": "nearest_access_port",
            "label": "B",
            "title": "ports",
            "metric": "Nearest",
            "show_ylabel": False,
            "text_offset": (0.1, 94),
        },
        {
            "ax": axes[0, 2],
            "col": "nearest_access_rail",
            "label": "C",
            "title": "rail terminals",
            "metric": "Nearest",
            "show_ylabel": False,
            "text_offset": (0.1, 94),
        },
        # Bottom row - Average
        {
            "ax": axes[1, 0],
            "col": "avg_access_road",
            "label": "D",
            "title": "road border crossings",
            "metric": "Avg.",
            "show_ylabel": True,
            "text_offset": (-1, 92),
        },
        {
            "ax": axes[1, 1],
            "col": "avg_access_port",
            "label": "E",
            "title": "ports",
            "metric": "Avg.",
            "show_ylabel": False,
            "text_offset": (-0.9, 94),
        },
        {
            "ax": axes[1, 2],
            "col": "avg_access_rail",
            "label": "F",
            "title": "rail terminals",
            "metric": "Avg.",
            "show_ylabel": False,
            "text_offset": (0.1, 94),
        },
    ]

    for sink_config in sink_configs:
        ax = sink_config["ax"]
        col = sink_config["col"]

        # Calculate percentage of UAL within each threshold
        percentage_ual = []
        for threshold in thresholds:
            ual_sum = df_agri.loc[df_agri[col] <= threshold, "UAL"].sum()
            ual_percentage = (ual_sum / total_ual) * 100
            percentage_ual.append(ual_percentage)

        # Find 100% threshold
        threshold_100 = next(
            (
                threshold
                for i, threshold in enumerate(thresholds)
                if percentage_ual[i] >= 99.9
            ),
            None,
        )

        # Plot
        ax.plot(
            thresholds,
            percentage_ual,
            linestyle="-",
            color="#003049",
            linewidth=2,
            label="Normal condition",
        )
        ax.set_xlabel(
            f"{sink_config['metric']} access time to \n {sink_config['title']} (hours)",
            fontsize=11,
        )

        # Only show y-axis label on first column
        if sink_config["show_ylabel"]:
            ax.set_ylabel("Agricultural land with access (%)", fontsize=11)

        ax.minorticks_on()
        ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_aspect("auto", adjustable="box")
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(thresholds))

        # Add panel label
        ax.text(
            0.05,
            0.95,
            sink_config["label"],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        if threshold_100 is not None:
            x_offset, y_pos = sink_config["text_offset"]
            ax.axvline(
                x=threshold_100, color="#003049", linestyle="--", linewidth=1, alpha=0.7
            )
            ax.plot(threshold_100, 100, "o", color="#003049", markersize=6)
            ax.text(
                threshold_100 + x_offset,
                y_pos,
                f"{threshold_100:.1f}h",
                color="black",
                ha="left",
                fontsize=14,
            )

    # Add shared legend to the last panel (bottom right)
    axes[1, 2].legend(fontsize=12, loc="lower right")

    # Final layout
    plt.tight_layout()
    _savefig(
        figure_dir,
        "baseline_accessibility_agri_road_port_rail_3x2.png",
        dpi=150,
        transparent=True,
    )
    if show:
        plt.show()


def plot_access_time_agriculture_map(
    df_agri: pd.DataFrame, Sinks: pd.DataFrame, figure_dir: Path, show: bool = True
) -> None:
    """Plot three maps of average access time from agricultural areas to road/port/rail sinks."""
    df_agri_plot = _to_web_mercator(df_agri)
    Sinks_plot = _to_web_mercator(Sinks)

    bins = [1, 2, 3, 4, 5, float("inf")]
    labels_cat = ["1-2", "2-3", "3-4", "4-5", "5+"]
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    color_map = dict(zip(labels_cat, colors))

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for ax, col, title in zip(
        axes, ["avg_access_road", "avg_access_port", "avg_access_rail"], ["A", "B", "C"]
    ):
        df_plot = df_agri_plot.copy()
        df_plot["category"] = pd.cut(
            df_plot[col], bins=bins, labels=labels_cat, right=False
        )
        df_plot["category"] = df_plot["category"].astype("object")

        for category, color in color_map.items():
            data = df_plot[df_plot["category"] == category]
            if not data.empty:
                data.plot(
                    ax=ax,
                    color=color,
                    legend=False,
                    linewidth=0.1,
                    edgecolor="grey",
                    markersize=50,
                )

        # Plot relevant sinks
        if "road" in col:
            sink_subset = Sinks_plot[Sinks_plot["type"] == "road"]
            marker = "^"
        elif "port" in col:
            sink_subset = Sinks_plot[Sinks_plot["type"] == "port"]
            marker = "s"
        else:
            sink_subset = Sinks_plot[Sinks_plot["type"] == "rail"]
            marker = "o"

        sink_subset.plot(ax=ax, color="black", markersize=100, marker=marker)

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

        # Add letter label
        ax.text(
            0.05,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_aspect("equal")
        ax.axis("off")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=color, label=f"{label} hours")
        for label, color in zip(labels_cat, colors)
    ]
    legend_patches.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                color="black",
                lw=0,
                label="Road Borders",
                markersize=12,
            ),
            Line2D(
                [0], [0], marker="s", color="black", lw=0, label="Ports", markersize=12
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                lw=0,
                label="Rail Terminals",
                markersize=12,
            ),
        ]
    )

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=8,
        fontsize=12,
        title="Average Access Time",
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    _savefig(figure_dir, "agriculture_access_by_type.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_access_curve(
    df_worldpop: gpd.GeoDataFrame,
    emergency_service: str,
    figure_dir: Path,
    show: bool = True,
) -> None:
    """Plot cumulative population access curve to the nearest emergency service."""

    fig, ax_curve = plt.subplots(1, 1, figsize=(5, 5))

    total_population = df_worldpop["population"].sum()
    thresholds = np.arange(0, 3.1, 1 / 3)

    percentage_population_within_threshold_2 = []

    for threshold in thresholds:
        population_sum_2 = df_worldpop.loc[
            df_worldpop["closest_sink_total_fft"] <= threshold, "population"
        ].sum()
        population_percentage_2 = (population_sum_2 / total_population) * 100
        percentage_population_within_threshold_2.append(population_percentage_2)

    # Find 100% thresholds
    threshold_100_2 = next(
        (
            threshold
            for i, threshold in enumerate(thresholds)
            if percentage_population_within_threshold_2[i] == 100
        ),
        None,
    )

    # Plot access curves
    ax_curve.plot(
        thresholds,
        percentage_population_within_threshold_2,
        linestyle="-",
        color="#003049",
        linewidth=2,
        label="Normal condition",
    )

    if emergency_service == "firefighters":
        ax_curve.set_xlabel("Access time to closest fire station (hours)", fontsize=12)
        file_name = "baseline_accessibility_fire_stations.png"
    elif emergency_service == "hospitals":
        ax_curve.set_xlabel(
            "Access time to closest health care facility (hours)", fontsize=12
        )
        file_name = "baseline_accessibility_hospitals.png"
    elif emergency_service == "police":
        ax_curve.set_xlabel(
            "Access time to closest police station (hours)", fontsize=12
        )
        file_name = "baseline_accessibility_police_stations.png"
    else:
        raise ValueError(
            f"Invalid emergency_service '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    ax_curve.set_ylabel("Population with access (%)", fontsize=12)
    ax_curve.legend(fontsize=12)
    ax_curve.minorticks_on()
    ax_curve.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax_curve.set_aspect("auto", adjustable="box")

    if threshold_100_2 is not None:
        ax_curve.axvline(
            x=threshold_100_2, color="#003049", linestyle="--", linewidth=1, alpha=0.7
        )
        ax_curve.plot(threshold_100_2, 100, "o", color="#003049", markersize=6)
        ax_curve.text(
            threshold_100_2 + 0.05,
            94,
            f"{threshold_100_2:.1f}h",
            color="black",
            ha="left",
            fontsize=12,
        )

    _savefig(figure_dir, file_name, dpi=150, transparent=True)

    if show:
        plt.show()


def plot_access_time_map(
    df_worldpop: gpd.GeoDataFrame,
    Sink: pd.DataFrame,
    emergency_service: str,
    figure_dir: Path,
    show: bool = True,
) -> gpd.GeoDataFrame:
    """Plot an access-time map for an emergency service and save the figure."""

    # Prepare data
    df_worldpop_plot = _to_web_mercator(df_worldpop)
    Sink_fire = _to_web_mercator(Sink)

    # Create bins for categories (1-hour intervals)
    bins = [0, 0.25, 0.5, 1, float("inf")]
    labels = ["0-15", "15-30", "30-60", ">60"]
    colors = ["#4cc9f0", "#4895ef", "#4361ee", "#3f37c9", "#3a0ca3"]

    # Assign categories
    df_worldpop_plot["category"] = pd.cut(
        df_worldpop_plot["closest_sink_total_fft"],
        bins=bins,
        labels=labels,
        right=False,
    )

    # Convert to object type to allow mixed values
    df_worldpop_plot["category"] = df_worldpop_plot["category"].astype("object")

    # Handle NaN values as "Not Accessible"
    df_worldpop_plot.loc[df_worldpop_plot["category"].isna(), "category"] = (
        "Not Accessible"
    )

    # Add "Not Accessible" to color mapping (using gray)
    color_map = dict(zip(labels, colors))
    color_map["Not Accessible"] = "#bdbdbd"  # Gray color

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot by category
    for category, color in color_map.items():
        data = df_worldpop_plot[df_worldpop_plot["category"] == category]
        if not data.empty:
            data.plot(ax=ax, color=color, legend=False, linewidth=0.1, edgecolor="grey")

    Sink_fire.plot(ax=ax, color="red", markersize=100, marker="+")
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    # Enhance the plot styling
    ax.set_aspect("equal")
    ax.axis("off")  # Remove axis for cleaner look

    # Create legend patches (add "Not Accessible" at the end)
    legend_patches = [
        mpatches.Patch(color=color, label=f"{label} minutes")
        for label, color in zip(labels, colors)
    ]
    legend_patches.append(mpatches.Patch(color="#bdbdbd", label="Not Accessible"))

    if emergency_service == "firefighters":
        legend_patches.append(
            Line2D(
                [0],
                [0],
                marker="+",
                color="red",
                lw=0,
                label="Fire departments",
                markersize=15,
            )
        )
        file_name = "firefighter_access.png"

    elif emergency_service == "hospitals":
        legend_patches.append(
            Line2D(
                [0],
                [0],
                marker="+",
                color="red",
                lw=0,
                label="hospitals",
                markersize=15,
            )
        )
        file_name = "hospital_access.png"

    elif emergency_service == "police":
        legend_patches.append(
            Line2D(
                [0],
                [0],
                marker="+",
                color="red",
                lw=0,
                label="police stations",
                markersize=15,
            )
        )
        file_name = "police_station_access.png"

    else:
        raise ValueError(
            f"Invalid emergency_service '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    # Add legend
    legend = ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=12,
        title="Access Time",
        title_fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
    )

    legend.get_title().set_fontweight("bold")

    _savefig(figure_dir, file_name, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return df_worldpop_plot


def plot_accessibility_chart(
    df_worldpop_plot: gpd.GeoDataFrame,
    emergency_service: str,
    figure_dir: Path,
    show: bool = True,
) -> None:
    """Plot population distribution by access-time category (bar chart + inset pie)."""

    # Calculate total population per category
    pop_by_category = df_worldpop_plot.groupby("category")["population"].sum() / 1e6

    # Define the order of categories and their colors
    category_order = ["0-15", "15-30", "30-60", "60-90", "90-120", ">120"]
    category_colors = [
        "#fff7f3",
        "#fde0dd",
        "#fcc5c0",
        "#fa9fb5",
        "#f768a1",
        "#c51b8a",
        "#bdbdbd",
    ]
    color_dict = dict(zip(category_order, category_colors))

    # Reindex to ensure all categories are present and in correct order
    pop_by_category = pop_by_category.reindex(category_order, fill_value=0)

    # Reverse order for horizontal plot (so 0-0.5 is at top)
    pop_by_category_reversed = pop_by_category[::-1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(4.5, 7))

    # Create horizontal bar chart with narrower bars
    ax.barh(
        range(len(pop_by_category_reversed)),
        pop_by_category_reversed.values,
        height=0.5,
        color=[color_dict[cat] for cat in pop_by_category_reversed.index],
        edgecolor="black",
        linewidth=1.5,
    )

    # Customize the plot
    if emergency_service == "firefighters":
        ax.set_ylabel(
            "Access Time to closest fire station (hours)",
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )
        file_name = "firefighter_access_chart.png"

    elif emergency_service == "hospitals":
        ax.set_ylabel(
            "Access Time to closest health care facility (hours)",
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )
        file_name = "hospital_access_chart.png"

    elif emergency_service == "police":
        ax.set_ylabel(
            "Access Time to closest police station (hours)",
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )
        file_name = "police_station_access_chart.png"

    else:
        raise ValueError(
            f"Invalid emergency_service '{emergency_service}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    ax.set_xlabel("Population (in millions)", fontsize=14, fontweight="bold")

    # Set y-axis labels
    ax.set_yticks(range(len(pop_by_category_reversed)))
    ax.set_yticklabels(pop_by_category_reversed.index, fontsize=12)

    # Format x-axis with thousands separator
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    ax.tick_params(axis="x", labelsize=12)

    # Add grid for better readability
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Create inset pie chart
    ax_inset = fig.add_axes([0.3, 0.15, 0.6, 0.6])  # [left, bottom, width, height]

    # Pie chart with same order as original (not reversed) and matching colors
    pie_colors = [color_dict[cat] for cat in pop_by_category.index]
    wedges, texts = ax_inset.pie(
        pop_by_category.values,
        colors=pie_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )

    _savefig(figure_dir, file_name, dpi=200, bbox_inches="tight")

    if show:
        plt.show()


def plot_emergency_curves_combined(
    df_worldpop_fire: gpd.GeoDataFrame,
    df_worldpop_police: gpd.GeoDataFrame,
    figure_dir: Path,
    show: bool = True,
) -> None:
    """Plot combined (fire | police) cumulative population access curves (2x1)."""

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    thresholds = np.arange(0, 3.1, 1 / 3)

    panels = [
        (axes[0], df_worldpop_fire, "Access time to fire stations (hours)", "A", False),
        (
            axes[1],
            df_worldpop_police,
            "Access time to police stations (hours)",
            "B",
            True,
        ),
    ]

    for ax, df_worldpop, xlabel, letter, show_legend in panels:
        total_population = df_worldpop["population"].sum()
        percentage_population = []
        for threshold in thresholds:
            population_sum = df_worldpop.loc[
                df_worldpop["closest_sink_total_fft"] <= threshold, "population"
            ].sum()
            percentage_population.append((population_sum / total_population) * 100)

        threshold_100 = next(
            (
                threshold
                for i, threshold in enumerate(thresholds)
                if percentage_population[i] == 100
            ),
            None,
        )

        ax.plot(
            thresholds,
            percentage_population,
            linestyle="-",
            color="#003049",
            linewidth=2,
            label="Normal condition",
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(
            "Population with access (%)" if letter == "A" else "", fontsize=12
        )
        if show_legend:
            ax.legend(fontsize=12)
        ax.minorticks_on()
        ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_aspect("auto", adjustable="box")

        ax.text(
            0.05,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        if threshold_100 is not None:
            x_text = threshold_100 - 0.35 if letter == "A" else threshold_100 + 0.05
            ax.axvline(
                x=threshold_100, color="#003049", linestyle="--", linewidth=1, alpha=0.7
            )
            ax.plot(threshold_100, 100, "o", color="#003049", markersize=6)
            ax.text(
                x_text,
                94,
                f"{threshold_100:.1f}h",
                color="black",
                ha="left",
                fontsize=12,
            )

    plt.tight_layout()
    _savefig(
        figure_dir,
        "baseline_accessibility_fire_police.png",
        dpi=150,
        transparent=True,
    )
    if show:
        plt.show()
    plt.close()


def plot_emergency_map_combined(
    df_worldpop_fire: gpd.GeoDataFrame,
    Sink_fire: pd.DataFrame,
    df_worldpop_police: gpd.GeoDataFrame,
    Sink_police: pd.DataFrame,
    figure_dir: Path,
    show: bool = True,
) -> None:
    """Plot combined (fire | police) access-time maps (1x2) with a shared legend."""

    bins = [0, 0.25, 0.5, 1, float("inf")]
    labels = ["0-15", "15-30", "30-60", ">60"]
    colors = ["#4cc9f0", "#4895ef", "#4361ee", "#3f37c9", "#3a0ca3"]
    color_map = dict(zip(labels, colors))
    color_map["Not Accessible"] = "#bdbdbd"

    datasets = {
        "A": (_to_web_mercator(df_worldpop_fire), _to_web_mercator(Sink_fire)),
        "B": (_to_web_mercator(df_worldpop_police), _to_web_mercator(Sink_police)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), facecolor="white")

    for idx, (letter, (df_worldpop_plot, Sink_plot)) in enumerate(datasets.items()):
        ax = axes[idx]
        df_worldpop_plot = df_worldpop_plot.copy()

        df_worldpop_plot["category"] = pd.cut(
            df_worldpop_plot["closest_sink_total_fft"],
            bins=bins,
            labels=labels,
            right=False,
        )
        df_worldpop_plot["category"] = df_worldpop_plot["category"].astype("object")
        df_worldpop_plot.loc[df_worldpop_plot["category"].isna(), "category"] = (
            "Not Accessible"
        )

        for category, color in color_map.items():
            data = df_worldpop_plot[df_worldpop_plot["category"] == category]
            if not data.empty:
                data.plot(
                    ax=ax, color=color, legend=False, linewidth=0.1, edgecolor="grey"
                )

        Sink_plot.plot(ax=ax, color="red", markersize=80, marker="+", zorder=5)
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.5)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(
            0.05,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    legend_patches = [
        mpatches.Patch(color=color, label=f"{label} minutes")
        for label, color in zip(labels, colors)
    ]
    legend_patches.append(mpatches.Patch(color="#bdbdbd", label="Not Accessible"))
    legend_patches.append(
        Line2D(
            [0],
            [0],
            marker="+",
            color="red",
            lw=0,
            label="Service Locations",
            markersize=12,
        )
    )

    legend = fig.legend(
        handles=legend_patches,
        loc="center right",
        bbox_to_anchor=(0.43, 0.85),
        fontsize=10,
        title="Access Time",
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(right=0.90)
    _savefig(
        figure_dir,
        "emergency_services_access.png",
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
