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
from typing import Tuple
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



def load_road_network(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Load Serbian road network parquet file.
    
    Args:
        config: Network configuration
        
    Returns:
        GeoDataFrame with Serbian road network
    """

    gdf = gpd.read_parquet(config.Path_RoadNetwork)
    return gdf


def load_factory_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Load location of factories in Serbia from Excel file.
    
    Args:
        config: Network configuration
        
    Returns:
        GeoDataFrame with Serbian factories
    """

    DataFrame_Factory = pd.read_excel(config.Path_FactoryFile)

    Clean_DataFrame_Factory = DataFrame_Factory.dropna(subset=["Latitude", "Longitude", "Factory"])

    geometry = [Point(xy) for xy in zip(Clean_DataFrame_Factory["Longitude"], Clean_DataFrame_Factory["Latitude"])]

    df_factories = gpd.GeoDataFrame(
        Clean_DataFrame_Factory[["Number"]].copy(),
        geometry=geometry,
        crs="EPSG:4326"
    )

    return df_factories




def create_graph_for_spatial_matching(base_network: gpd.GeoDataFrame) -> tuple[pd.DataFrame, ig.Graph]:

    """
    Create graph from road network and create nodes for spatial matching.
    
    Args:
        base_network: Serbian road network
        
    Returns:
        Pandas DataFrame with the nodes of the road network graph
    """

    #Create graph from road network
    edges = base_network.reindex(['from_id','to_id'] + [x for x in list(base_network.columns) if x not in ['from_id','to_id']], axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:], directed=True)
    graph = graph.connected_components().giant()
    edges = edges[edges['id'].isin(graph.es['id'])]

    #Create nodes from edges for spatial matching
    vertex_lookup = dict(zip(pd.DataFrame(graph.vs['name'])[0], pd.DataFrame(graph.vs['name']).index))

    tqdm.pandas()
    from_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[0]))
    to_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[-1]))

    from_dict = dict(zip(edges['from_id'], from_id_geom))
    to_dict = dict(zip(edges['to_id'], to_id_geom))

    nodes = pd.concat([
        pd.DataFrame.from_dict(to_dict, orient='index', columns=['geometry']),
        pd.DataFrame.from_dict(from_dict, orient='index', columns=['geometry'])
    ]).drop_duplicates()

    nodes['vertex_id'] = nodes.apply(lambda x: vertex_lookup[x.name], axis=1)
    nodes = nodes.reset_index()

    return nodes, graph

def nearest_network_nodes(df_factories: gpd.GeoDataFrame, nodes: pd.DataFrame) -> pd.Series:

    nodes_sindex = shapely.STRtree(nodes.geometry)
    df_factories['vertex_id'] = df_factories.geometry.progress_apply(
    lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    return df_factories['vertex_id']


def load_border_crossings(config: NetworkConfig, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Load border crossings (Sinks) from Excel file.
    
    Args:
        config: Network configuration
        
    Returns:
        Pandas DataFrame with border crossings
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)

    Sink = pd.read_excel(config.path_to_Borders)
    Sink = Sink.rename(columns={"LON": "Longitude", "LAT": "Latitude"})
    Sink['geometry'] = Sink.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    Sink['vertex_id'] = Sink.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    return Sink


def calculate_average_access_time(df_factories: gpd.GeoDataFrame, Sink: pd.DataFrame, graph: ig.Graph) -> tuple[pd.Series,np.array]:
    """
    Calculate the origin-destination matrix and the average access times for factories to reach all border crossings.
    
    Args:
        df_factories: data frame with industrial centers in Serbia, Sink: border crossings, graph: graph representation of Serbia's road network
        
    Returns:
        pd.Series with average access time of factories to road borders and np.array with baseline origin destinatino maxtrix
    """
     
    factory_vertices = df_factories['vertex_id'].unique()
    sink_vertices = Sink['vertex_id'].unique()

    OD_baseline = np.array(graph.distances(
        source=factory_vertices,
        target=sink_vertices,
        weights='fft'
    ))
    OD_baseline[np.isinf(OD_baseline)] = 12

    avg_time_per_factory = np.mean(OD_baseline, axis=1)
    vertex_to_avg_time = dict(zip(factory_vertices, avg_time_per_factory))
    df_factories['avg_access_time'] = df_factories['vertex_id'].map(vertex_to_avg_time)

    return df_factories['avg_access_time'], OD_baseline


#Move this to the plotting script
def plot_access_times(df_factories: pd.DataFrame, Sink: pd.DataFrame, config: NetworkConfig) -> None:
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


def print_statistics(df_factories: pd.DataFrame, Sink: pd.DataFrame, OD_baseline: np.array) -> None:
    """
    Print statistics of the access time calculations to the console.
    
    Args:
        df_factories: data frame with industrial centers in Serbia, Sink: border crossings, np.array: origin-destination matrix
        
    Returns:
        None
    """

    print("="*60)
    print("BASELINE ACCESSIBILITY SUMMARY")
    print("="*60)

    # Basic stats
    print(f"\nNumber of factories: {len(df_factories)}")
    print(f"Number of border crossings: {len(Sink)}")
    print(f"Number of OD pairs: {OD_baseline.size}")

    print(f"\n--- Access Time Statistics (hours) ---")
    print(f"Mean:   {df_factories['avg_access_time'].mean():.2f}")
    print(f"Median: {df_factories['avg_access_time'].median():.2f}")
    print(f"Std:    {df_factories['avg_access_time'].std():.2f}")
    print(f"Min:    {df_factories['avg_access_time'].min():.2f}")
    print(f"Max:    {df_factories['avg_access_time'].max():.2f}")

    # Percentiles
    print(f"\n--- Percentiles (hours) ---")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"P{p}: {df_factories['avg_access_time'].quantile(p/100):.2f}")

    # Category distribution
    print(f"\n--- Factories by Access Time Category ---")
    bins = [1, 2, 3, 4, 5, float('inf')]
    labels = ['1-2', '2-3', '3-4', '4-5', '5+']
    df_factories['category'] = pd.cut(df_factories['avg_access_time'], bins=bins, labels=labels, right=False)

    category_counts = df_factories['category'].value_counts().sort_index()
    for cat, count in category_counts.items():
        pct = count / len(df_factories) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Unreachable pairs in OD matrix
    n_unreachable = np.sum(OD_baseline == 12)
    pct_unreachable = n_unreachable / OD_baseline.size * 100
    print(f"\n--- Connectivity ---")
    print(f"Unreachable OD pairs: {n_unreachable} ({pct_unreachable:.1f}%)")


def load_agricultural_data(config: NetworkConfig) -> pd.DataFrame:
    """
    Load location data of agricultural areas from xslm file.
    
    Args:
        config: Network configuration
        
    Returns:
        Pandas DataFrame with agricultural areas
    """


    Path_AgriFile = config.data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    DataFrame_Agri = pd.read_excel(Path_AgriFile)

    Clean_DataFrame_Agri = DataFrame_Agri.dropna(subset=["latitude", "longitude", "Utilized agricultural land (UAL)"])

    geometry = [Point(xy) for xy in zip(Clean_DataFrame_Agri["longitude"], Clean_DataFrame_Agri["latitude"])]

    df_agri = gpd.GeoDataFrame(
        Clean_DataFrame_Agri[["Utilized agricultural land (UAL)"]].rename(columns={"Utilized agricultural land (UAL)": "UAL"}),
        geometry=geometry,
        crs="EPSG:4326"
    )

    return df_agri


def load_sinks(config: NetworkConfig, nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load location of border crossings, ports and rail terminals.
    
    Args:
        config: Network configuration, nodes: nodes of the road network graph
        
    Returns:
        Pandas DataFrame with location of border crossings, Pandas DataFrame with location of ports, Pandas DataFrame with location of rail terminals, Pandas DataFrame with location of all sinks combined
    """

    Sinks = pd.read_excel(config.path_to_Sinks)
    Sinks = Sinks.rename(columns={"LON": "Longitude", "LAT": "Latitude", "TYPE OF\nTRAFFIC": "type"})
    Sinks['geometry'] = Sinks.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    nodes_sindex = shapely.STRtree(nodes.geometry)
    Sinks['vertex_id'] = Sinks.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    # Split by type
    Sinks_road = Sinks[Sinks['type'] == 'road']
    Sinks_port = Sinks[Sinks['type'] == 'port']
    Sinks_rail = Sinks[Sinks['type'] == 'rail']

    print(f"Road border crossings: {len(Sinks_road)}")
    print(f"Ports: {len(Sinks_port)}")
    print(f"Rail terminals: {len(Sinks_rail)}")

    return Sinks_road, Sinks_port, Sinks_rail, Sinks


def calculate_access_times(graph, origin_vertices, sink_df, sink_name):
    """
    Calculate average access time from agricultural areas to border crossings, ports and rail terminals.
    
    Args:
        graph: graph represenation of road network, NDarray with the origin vertices
        
    Returns:
        dict mapping origin nodes to an average time, NDarray with the origin destination matrix
    """
    sink_vertices = sink_df['vertex_id'].unique()
    
    OD_matrix = np.array(graph.distances(
        source=origin_vertices,
        target=sink_vertices,
        weights='fft'
    ))
    OD_matrix[np.isinf(OD_matrix)] = 12
    
    avg_time = np.mean(OD_matrix, axis=1)
    vertex_to_avg = dict(zip(origin_vertices, avg_time))
    
    print(f"\n{sink_name}: {len(sink_vertices)} destinations, global avg = {np.mean(OD_matrix):.2f} hours")
    
    return vertex_to_avg, OD_matrix



def calculate_OD_matrix(df_agri: pd.DataFrame, graph: ig.Graph, Sinks_road: pd.DataFrame, Sinks_port: pd.DataFrame, Sinks_rail: pd.DataFrame, Sinks: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average access time from agricultural areas to border crossings, ports and rail terminals.
    
    Args:
        Pandas DataFrame with agricultural areas, graph representing the road network, Pandas Data Frame with the border crossings, Pandas DataFrame with the port locations, Pandas DataFrame with rail terminals,
        Pandas DataFrame combing all three sink types. 
        
    Returns:
        Pandas DataFrame containing all the average access times
    """
    agri_vertices = df_agri['vertex_id'].unique()
    # Calculate for each sink type
    road_access, OD_road = calculate_access_times(graph, agri_vertices, Sinks_road, "Road borders")
    port_access, OD_port = calculate_access_times(graph, agri_vertices, Sinks_port, "Ports")
    rail_access, OD_rail = calculate_access_times(graph, agri_vertices, Sinks_rail, "Rail terminals")

    # Also calculate combined (all sinks)
    all_access, OD_all = calculate_access_times(graph, agri_vertices, Sinks, "All sinks combined")

    # Map back to df_agri
    df_agri['avg_access_road'] = df_agri['vertex_id'].map(road_access)
    df_agri['avg_access_port'] = df_agri['vertex_id'].map(port_access)
    df_agri['avg_access_rail'] = df_agri['vertex_id'].map(rail_access)
    df_agri['avg_access_all'] = df_agri['vertex_id'].map(all_access)

    df_agri['nearest_access_road'] = get_distance_to_nearest_facility_agriculture(df_agri, Sinks_road, graph)['closest_sink_total_fft']
    df_agri['nearest_access_port'] = get_distance_to_nearest_facility_agriculture(df_agri, Sinks_port, graph)['closest_sink_total_fft']
    df_agri['nearest_access_rail'] = get_distance_to_nearest_facility_agriculture(df_agri, Sinks_rail, graph)['closest_sink_total_fft']

    return df_agri

def get_distance_to_nearest_facility_agriculture(df_population: gpd.GeoDataFrame, Sink: pd.DataFrame, graph: ig.Graph) -> gpd.GeoDataFrame:
    """
    Calculate the distance to the nearest facility (border, rail terminal, port) for each agricultural area 
    
    Args:
        df_population: GeoDataFrame containing the agricultural areas and their location, Sink: DataFrame with the sinks (firefighters, police stations, hospitals) graph: graph of the road network         

    Returns:
        GeoPandas DataFrame the distance to the nearest facility assinged to each agricultural area
    """

    # Initialize new columns
    df_population = df_population.copy()
    df_population['closest_sink_vertex_id'] = None
    df_population['closest_sink_osm_id'] = None
    df_population['closest_sink_total_fft'] = None
    
    # Get unique vertex IDs for both population and sinks
    unique_pop_vertex_ids = df_population['vertex_id'].unique()
    unique_sink_vertex_ids = Sink['vertex_id'].unique()
    
    # Create mapping from unique sink vertex_ids back to original sink data
    sink_lookup = {}
    for _, row in Sink.iterrows():
        sink_lookup[row['vertex_id']] = row['N°']
    
    # Calculate distance matrix once for unique vertices only
    distance_matrix = np.array(graph.distances(
        source=unique_pop_vertex_ids,
        target=unique_sink_vertex_ids, 
        weights='fft'
    ))
    
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
            vertex_to_closest_sink[pop_vertex_id] = (None, None, float('inf'))
        else:
            closest_sink_vertex_id = unique_sink_vertex_ids[min_sink_idx]
            closest_sink_osm_id = sink_lookup[closest_sink_vertex_id]
            vertex_to_closest_sink[pop_vertex_id] = (
                closest_sink_vertex_id,
                closest_sink_osm_id, 
                min_distance
            )
    
    # Map results back to all population points (including duplicates)
    for idx, row in df_population.iterrows():
        vertex_id = row['vertex_id']
        closest_sink_vertex_id, closest_sink_osm_id, closest_sink_total_fft = vertex_to_closest_sink[vertex_id]
        
        df_population.at[idx, 'closest_sink_vertex_id'] = closest_sink_vertex_id
        df_population.at[idx, 'closest_sink_osm_id'] = closest_sink_osm_id
        df_population.at[idx, 'closest_sink_total_fft'] = closest_sink_total_fft
    
    return df_population



def print_statistics_agriculture(df_agri: pd.DataFrame) -> None:
    """
    Print all the statistics of the average access times from agricultural areas to borders, ports and rail 
    
    Args:
        Pandas DataFrame with agricultural areas 
        
    Returns:
        Nothing
    """
    print("\n" + "="*60)
    print("AGRICULTURAL ACCESSIBILITY SUMMARY")
    print("="*60)

    print(f"\nNumber of agricultural locations: {len(df_agri)}")
    print(f"Total UAL (ha): {df_agri['UAL'].sum():,.0f}")

    for col, label in [('avg_access_road', 'Road Borders'), 
                    ('avg_access_port', 'Ports'), 
                    ('avg_access_rail', 'Rail Terminals'),
                    ('avg_access_all', 'All Combined')]:
        print(f"\n--- {label} ---")
        print(f"  Mean:   {df_agri[col].mean():.2f} hours")
        print(f"  Median: {df_agri[col].median():.2f} hours")
        print(f"  Std:    {df_agri[col].std():.2f} hours")
        print(f"  Min:    {df_agri[col].min():.2f} hours")
        print(f"  Max:    {df_agri[col].max():.2f} hours")
        print(f"  P10:    {df_agri[col].quantile(0.10):.2f} hours")
        print(f"  P90:    {df_agri[col].quantile(0.90):.2f} hours")

    # Category distribution for each type
    bins = [0, 0.5, 1, 2, 5, float('inf')]
    labels_cat = ['0-0.5h', '0.5-1h', '1-2h', '2-5h', '5h+']

    print("\n--- Distribution by Access Time Category ---")
    for col, label in [('avg_access_road', 'Road'), 
                    ('avg_access_port', 'Port'), 
                    ('avg_access_rail', 'Rail')]:
        print(f"\n{label}:")
        cat = pd.cut(df_agri[col], bins=bins, labels=labels_cat, right=False)
        for c in labels_cat:
            count = (cat == c).sum()
            pct = count / len(df_agri) * 100
            print(f"  {c}: {count} ({pct:.1f}%)")




def load_population_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Load settlement location data
    
    Args:
        config: Network configuration
        
    Returns:
        GeoPandas DataFrame with location of settlements
    """

    # reading the Excel file
    DataFrame_StatePop = pd.read_excel(config.Path_SettlementData_Excel)

    # to keep only rows with valid coordinates and population
    Clean_DataFrame_StatePop = DataFrame_StatePop.dropna(subset=["latitude", "longitude", "Total"])

    # to make point geometry
    geometry = [Point(xy) for xy in zip(Clean_DataFrame_StatePop["longitude"], Clean_DataFrame_StatePop["latitude"])]

    # build GeoDataFrame matching df_worldpop structure
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_StatePop[["Total"]].rename(columns={"Total": "population"}),
        geometry=geometry,
        crs="EPSG:4326"  # longitude/latitude WGS84    
        )
    
    return df_worldpop

def map_settlements_to_nodes_in_road_network(df_worldpop: gpd.GeoDataFrame, nodes: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Map each settlement to the nearest node in the road network
    
    Args:
        df_worldpop: GeoDataFrame containing the settlements and their location, nodes: DataFrame containing the nodes of the road network         

    Returns:
        GeoPandas DataFrame with location of settlements and a new column that contains the closest network node
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)

    df_worldpop['vertex_id'] = df_worldpop.geometry.progress_apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    return df_worldpop

def load_and_map_sinks(config: NetworkConfig, nodes: pd.DataFrame, sink_type) -> pd.DataFrame:
    """
    Load sink data (firefighters, hospitals or policestations) and map them to the nearest node in the road network
    
    Args:
        config: network configuration, nodes: nodes of the road network, sink_type: identifier for the sink category (firefighters, hospitals or policestations)         

    Returns:
        DataFrame location of sinks and their nearest network node
    """
    nodes_sindex = shapely.STRtree(nodes.geometry)

    if sink_type == "firefighters":
        Sink = pd.read_excel(config.firefighters)
        Sink = Sink.rename(columns={"lon": "Longitude", "lat": "Latitude"})
    elif sink_type == "hospitals":
        Sink = pd.read_excel(config.hospitals)
    elif sink_type == "police":
        Sink = pd.read_excel(config.police_stations)
        Sink = Sink.rename(columns={"lon": "Longitude", "lat": "Latitude"})
    else:
        raise ValueError(
            f"Invalid sink_type '{sink_type}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police'."
        )

    Sink['geometry'] = Sink.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    Sink['vertex_id'] = Sink.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values   

    return Sink


def get_distance_to_nearest_facility(df_population: gpd.GeoDataFrame, Sink: pd.DataFrame, graph: ig.Graph) -> gpd.GeoDataFrame:
    """
    Calculate the distance to the nearest facility for each settlement 
    
    Args:
        df_population: GeoDataFrame containing the settlements and their location, Sink: DataFrame with the sinks (firefighters, police stations, hospitals) graph: graph of the road network         

    Returns:
        GeoPandas DataFrame the distance to the nearest facility assinged to each settlement
    """
    # Initialize new columns
    df_population = df_population.copy()
    df_population['closest_sink_vertex_id'] = None
    df_population['closest_sink_osm_id'] = None
    df_population['closest_sink_total_fft'] = None
    
    # Get unique vertex IDs for both population and sinks
    unique_pop_vertex_ids = df_population['vertex_id'].unique()
    unique_sink_vertex_ids = Sink['vertex_id'].unique()
    
    # Create mapping from unique sink vertex_ids back to original sink data
    sink_lookup = {}
    for _, row in Sink.iterrows():
        sink_lookup[row['vertex_id']] = row['i.d.']
    
    # Calculate distance matrix once for unique vertices only
    distance_matrix = np.array(graph.distances(
        source=unique_pop_vertex_ids,
        target=unique_sink_vertex_ids, 
        weights='fft'
    ))
    
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
            vertex_to_closest_sink[pop_vertex_id] = (None, None, float('inf'))
        else:
            closest_sink_vertex_id = unique_sink_vertex_ids[min_sink_idx]
            closest_sink_osm_id = sink_lookup[closest_sink_vertex_id]
            vertex_to_closest_sink[pop_vertex_id] = (
                closest_sink_vertex_id,
                closest_sink_osm_id, 
                min_distance
            )
    
    # Map results back to all population points (including duplicates)
    for idx, row in df_population.iterrows():
        vertex_id = row['vertex_id']
        closest_sink_vertex_id, closest_sink_osm_id, closest_sink_total_fft = vertex_to_closest_sink[vertex_id]
        
        df_population.at[idx, 'closest_sink_vertex_id'] = closest_sink_vertex_id
        df_population.at[idx, 'closest_sink_osm_id'] = closest_sink_osm_id
        df_population.at[idx, 'closest_sink_total_fft'] = closest_sink_total_fft
    
    return df_population


def save_accessibilty_results(config: NetworkConfig, df_worldpop, Sink, facility_type) -> None:

    if facility_type == "firefighters":
        df_worldpop.to_parquet(config.Path_firefighter_accessibilty)
        gpd.GeoDataFrame(Sink).to_parquet(config.Path_firefighters_sink)

    elif facility_type == "hospitals":
        df_worldpop.to_parquet(config.Path_hospital_accessibilty)
        gpd.GeoDataFrame(Sink).to_parquet(config.Path_hospital_sink)

    elif facility_type == "police":
        df_worldpop.to_parquet(config.Path_police_accessibilty)
        gpd.GeoDataFrame(Sink).to_parquet(config.Path_police_sink)
    
    elif facility_type == "factories":
        df_worldpop.to_parquet(config.Path_factory_accessibility)
        gpd.GeoDataFrame(Sink).to_parquet(config.Path_factory_sink)
    
    elif facility_type == "agriculture":
        df_worldpop.to_parquet(config.Path_agriculture_accessibility)
        gpd.GeoDataFrame(Sink).to_parquet(config.Path_agriculture_sink)

    else:
        raise ValueError(
            f"Invalid sink_type '{facility_type}'. "
            "Expected one of: 'firefighters', 'hospitals', 'police', 'factories', 'agriculture'."
        )



def print_statistics_emergency_accessibility(df_worldpop_fire=None, Sink_fire=None, df_worldpop_hospital=None, Sink_hospitals=None, df_worldpop_police=None, Sink_police=None):

        
    # Define bins and labels
    bins = [0, 0.25, 0.5, 1, 1.5, 2, float('inf')]
    labels = ['0-15', '15-30', '30-60', '60-90', '90-120', '>120']

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
        access_time_minutes = df_worldpop['closest_sink_total_fft'] * 60
        
        print(f"\nAccess Time Statistics (minutes):")
        print(f"  Mean: {access_time_minutes.mean():.2f} minutes")
        print(f"  Median: {access_time_minutes.median():.2f} minutes")
        print(f"  Std Dev: {access_time_minutes.std():.2f} minutes")
        print(f"  Min: {access_time_minutes.min():.2f} minutes")
        print(f"  Max: {access_time_minutes.max():.2f} minutes")
        
        # Create categories
        df_worldpop['category'] = pd.cut(df_worldpop['closest_sink_total_fft'], 
                                        bins=bins, labels=labels, right=False)
        df_worldpop['category'] = df_worldpop['category'].astype('object')
        df_worldpop.loc[df_worldpop['category'].isna(), 'category'] = 'Not Accessible'
        
        # Distribution by category
        print(f"\nSettlements by Access Time Category:")
        cat_counts = df_worldpop['category'].value_counts()
        cat_order = labels + ['Not Accessible']
        for cat in cat_order:
            if cat in cat_counts.index:
                count = cat_counts[cat]
                pct = (count / n_settlements) * 100
                print(f"  {cat:>12} min: {count:>6,} settlements ({pct:>5.1f}%)")
        
        # Population-weighted analysis (if population column exists)
        pop_col = None
        for col in ['population', 'pop', 'worldpop', 'pop_sum', 'population_sum']:
            if col in df_worldpop.columns:
                pop_col = col
                break
        
        if pop_col:
            total_pop = df_worldpop[pop_col].sum()
            print(f"\nTotal Population Covered: {total_pop:,.0f}")
            
            print(f"\nPopulation by Access Time Category:")
            for cat in cat_order:
                subset = df_worldpop[df_worldpop['category'] == cat]
                if len(subset) > 0:
                    pop = subset[pop_col].sum()
                    pct = (pop / total_pop) * 100
                    print(f"  {cat:>12} min: {pop:>12,.0f} people ({pct:>5.1f}%)")
            
            # Population-weighted mean access time
            valid_data = df_worldpop[df_worldpop['closest_sink_total_fft'].notna()]
            if len(valid_data) > 0 and valid_data[pop_col].sum() > 0:
                weighted_mean = np.average(valid_data['closest_sink_total_fft'] * 60, 
                                        weights=valid_data[pop_col])
                print(f"\nPopulation-Weighted Mean Access Time: {weighted_mean:.2f} minutes")
        
        # Key thresholds
        print(f"\nKey Coverage Statistics:")
        within_15 = len(df_worldpop[df_worldpop['closest_sink_total_fft'] < 0.25])
        within_30 = len(df_worldpop[df_worldpop['closest_sink_total_fft'] < 0.5])
        within_60 = len(df_worldpop[df_worldpop['closest_sink_total_fft'] < 1])
        beyond_60 = len(df_worldpop[df_worldpop['closest_sink_total_fft'] >= 1])
        not_accessible = len(df_worldpop[df_worldpop['category'] == 'Not Accessible'])
        
        print(f"  Within 15 minutes: {within_15:,} ({within_15/n_settlements*100:.1f}%)")
        print(f"  Within 30 minutes: {within_30:,} ({within_30/n_settlements*100:.1f}%)")
        print(f"  Within 60 minutes: {within_60:,} ({within_60/n_settlements*100:.1f}%)")
        print(f"  Beyond 60 minutes: {beyond_60:,} ({beyond_60/n_settlements*100:.1f}%)")
        print(f"  Not Accessible: {not_accessible:,} ({not_accessible/n_settlements*100:.1f}%)")
        
        # Collect summary for comparison table
        summary_data.append({
            'Service': service_name,
            'Facilities': n_facilities,
            'Settlements': n_settlements,
            'Mean Access (min)': round(access_time_minutes.mean(), 1),
            'Median Access (min)': round(access_time_minutes.median(), 1),
            'Within 15 min (%)': round(within_15/n_settlements*100, 1),
            'Within 30 min (%)': round(within_30/n_settlements*100, 1),
            'Within 60 min (%)': round(within_60/n_settlements*100, 1),
            'Not Accessible (%)': round(not_accessible/n_settlements*100, 1)
        })

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
            df_worldpop['category'] = pd.cut(df_worldpop['closest_sink_total_fft'], 
                                            bins=bins, labels=labels, right=False)
            df_worldpop['category'] = df_worldpop['category'].astype('object')
            df_worldpop.loc[df_worldpop['category'].isna(), 'category'] = 'Not Accessible'
            
            total_pop = df_worldpop[pop_col].sum()
            pop_within_30 = df_worldpop[df_worldpop['closest_sink_total_fft'] < 0.5][pop_col].sum()
            pop_within_60 = df_worldpop[df_worldpop['closest_sink_total_fft'] < 1][pop_col].sum()
            pop_not_accessible = df_worldpop[df_worldpop['category'] == 'Not Accessible'][pop_col].sum()
            
            pop_summary.append({
                'Service': service_name,
                'Total Population': f"{total_pop:,.0f}",
                'Pop Within 30 min': f"{pop_within_30:,.0f} ({pop_within_30/total_pop*100:.1f}%)",
                'Pop Within 60 min': f"{pop_within_60:,.0f} ({pop_within_60/total_pop*100:.1f}%)",
                'Pop Not Accessible': f"{pop_not_accessible:,.0f} ({pop_not_accessible/total_pop*100:.1f}%)"
            })
        
        pop_df = pd.DataFrame(pop_summary)
        print(pop_df.to_string(index=False))



def main():
    """
    Main function to orchestrate all accessibilty calculations.
    """
    # Initialize configuration
    config = NetworkConfig()
    
    # Load OSM network
    print("Loading OSM network...")
    base_network = load_road_network(config)

    # =============================================================================
    # 1. Accessibility Calculations for factories
    # =============================================================================

    #Load factory location data
    print("Loading factory location data...")
    df_factories = load_factory_data(config)

    #Create graph for spatial matching
    print("Creating graph representation of the road network...")
    nodes, graph = create_graph_for_spatial_matching(base_network)
    
    #Map factories to nearest network nodes
    print("Mapping factories to nearest network nodes...")
    df_factories['vertex_id'] = nearest_network_nodes(df_factories, nodes)

    #Load border crossing location data
    print("Loading border crossing locations...")
    border_crossings = load_border_crossings(config, nodes)

    #Calculate origin-destination (OD) matrix and average access time
    df_factories['avg_access_time'], OD_baseline = calculate_average_access_time(df_factories, border_crossings, graph)
    print(f"Baseline average access time: {np.mean(OD_baseline):.2f} hours")

    #summarize analysis if flag is set
    if config.print_statistics == True:
        print_statistics(df_factories, border_crossings, OD_baseline)

    #save results of analysis as .parquet files
    save_accessibilty_results(config, df_factories, border_crossings, "factories")
    print(f"Saved results to {config.Path_factory_accessibility}")
    print(f"Saved results to {config.Path_factory_sink}")

    print(f"\n--- Accessibility analysis for factories complete. ---")

    # =============================================================================
    # 2. Accessibility calculations for agricultural areas
    # =============================================================================
    
    print(f"\n--- Starting accessibility analysis for agricultural areas ---")

    #Load location data of agricultural areas
    print("Loading location data of agricultural areas...")
    df_agri = load_agricultural_data(config)

    #Map agricultural locations to nearest network nodes
    print("Mapping agricultural locations to nearest network nodes...")
    df_agri['vertex_id'] = nearest_network_nodes(df_agri, nodes)

    #Load sinks (borders, ports, rail)
    print("Loading location data of border crossings, ports and rail cargo terminals...")
    Sinks_road, Sinks_port, Sinks_rail, all_sinks = load_sinks(config, nodes)

    #Calculate OD matrices by sink type
    print("Calculating OD matrices for agricultural areas to border crossings, ports and rail cargo stations...")
    df_agri = calculate_OD_matrix(df_agri, graph, Sinks_road, Sinks_port, Sinks_rail, all_sinks)

    #summarize analysis if flag is set
    if config.print_statistics == True:
        print_statistics_agriculture(df_agri)

    #save results of analysis as .parquet files
    save_accessibilty_results(config, df_agri, all_sinks, "agriculture")
    print(f"Saved results to {config.Path_agriculture_accessibility}")
    print(f"Saved results to {config.Path_agriculture_sink}")

    print(f"\n--- Accessibility analysis for agricultural areas complete. ---")
    
    
    # =============================================================================
    # 3. Accessibility calculations for firefighters
    # =============================================================================

    print(f"\n--- Starting accessibility analysis for firefighters ---")

    #Load population data
    print("Loading population data...")
    df_settlements = load_population_data(config)

    #Map each settlement to the closest node in the road network
    print("Mapping each settlement to the closest node in the road network...")
    df_settlements = map_settlements_to_nodes_in_road_network(df_settlements, nodes)
    
    #Load location of firefighters and map them to the nearest road network node
    print("Loading firefighter locations...")
    sink_firefighters = load_and_map_sinks(config, nodes, "firefighters")

    #Calculate the distance to the nearest firefighters
    print("Calculating distance to the nearest fire station for each settlement...")
    acessibility_firefighters = get_distance_to_nearest_facility(df_settlements,sink_firefighters,graph)

    #Save results of accessibility analysis and firefighter locations to parquet
    save_accessibilty_results(config, acessibility_firefighters, sink_firefighters, "firefighters")
    print(f"Saved results to {config.Path_firefighter_accessibilty}")
    print(f"Saved results to {config.Path_firefighters_sink}")

    print(f"\n--- Accessibility analysis for firefighters complete. ---")
    

    # =============================================================================
    # 4. Accessibility calculations for hospitals
    # =============================================================================
    
    print(f"\n--- Starting accessibility analysis for hospitals ---")

    #Load location of hospitals and map them to the nearest road network node
    print("Loading hospital locations...")
    sink_hospitals = load_and_map_sinks(config, nodes, "hospitals")

    #Calculate the distance to the nearest hospital for each settlement
    print("Calculating distance to the nearest hospital...")
    acessibility_hospitals = get_distance_to_nearest_facility(df_settlements,sink_hospitals,graph)

    save_accessibilty_results(config, acessibility_hospitals, sink_hospitals, "hospitals")
    print(f"Saved results to {config.Path_hospital_accessibilty}")
    print(f"Saved results to {config.Path_hospital_sink}")

    print(f"\n--- Accessibility analysis for hospitals complete. ---")
    
    # =============================================================================
    # 5. Accessibility calculations for police stations
    # =============================================================================

    print(f"\n--- Starting accessibility analysis for police stations ---")

    #Load location of police stations and map them to the nearest road network node
    print("Loading location data of police stations...")
    police_stations = load_and_map_sinks(config, nodes, "police")

    #Calculate the distance to the nearest police station for each settlement
    print("Calculating distance to the nearest police station...")
    acessibility_police_stations = get_distance_to_nearest_facility(df_settlements,police_stations,graph)

    save_accessibilty_results(config, acessibility_police_stations, police_stations, "police")
    print(f"Saved results to {config.Path_police_accessibilty}")
    print(f"Saved results to {config.Path_police_sink}")

    print(f"\n--- Accessibility analysis for police stations complete. ---\n")

    #print combined accessibility summary for firefighters, hospitals and police stations
    if config.print_statistics == True:
        print_statistics_emergency_accessibility(acessibility_firefighters, sink_firefighters, acessibility_hospitals, sink_hospitals, acessibility_police_stations, police_stations)
    

if __name__ == "__main__":
    main()
