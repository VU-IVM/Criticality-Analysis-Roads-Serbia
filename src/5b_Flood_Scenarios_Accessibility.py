# ### Step1: Importing the required packages

import os,sys
import pickle
import xarray as xr
import igraph as ig
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
from shapely.geometry import Point
import cftime
from pathlib import Path
from tqdm import tqdm
from rasterio.enums import Resampling
from exactextract import exact_extract

from simplify import *
from config.network_config import NetworkConfig 

# load functions
def get_average_access_time(df_population, Sink, graph):
    """Calculate average travel time from each origin to ALL sinks"""
    df_population = df_population.copy()
    
    unique_pop_vertex_ids = df_population['vertex_id'].unique()
    unique_sink_vertex_ids = Sink['vertex_id'].unique()
    
    # Full OD matrix
    OD_matrix = np.array(graph.distances(
        source=unique_pop_vertex_ids,
        target=unique_sink_vertex_ids,
        weights='fft'
    ))
    OD_matrix[np.isinf(OD_matrix)] = 12  # 12 hour penalty
    
    # Average across all sinks (axis=1)
    avg_time_per_origin = np.mean(OD_matrix, axis=1)
    vertex_to_avg_time = dict(zip(unique_pop_vertex_ids, avg_time_per_origin))
    
    df_population['avg_access_time'] = df_population['vertex_id'].map(vertex_to_avg_time)
    
    return df_population

def create_grid(bbox,height):
    xmin, ymin = shapely.total_bounds(bbox)[0],shapely.total_bounds(bbox)[1]
    xmax, ymax = shapely.total_bounds(bbox)[2],shapely.total_bounds(bbox)[3]
    
    rows = int(np.ceil((ymax-ymin) / height))
    cols = int(np.ceil((xmax-xmin) / height))

    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append((
                ((x_left_origin, y_top), (x_right_origin, y_top),
                (x_right_origin, y_bottom), (x_left_origin, y_bottom)
                )))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height

    return shapely.polygons(res_geoms)

def get_exposure_values(country_iso3,base_network,hazard_map,Threshold,Stru_Threshold):
    world = gpd.read_file(Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Shirazian, S. (Shadi)'s files - NewCodes\Tajikistan\ne_10m_admin_0_countries.shp"))
    country_bounds = world.loc[world.ADM0_A3 == country_iso3].bounds
    country_geom = world.loc[world.ADM0_A3 == country_iso3].geometry
    
    hazard_country = hazard_map.rio.clip_box(minx=country_bounds.minx.values[0],
                         miny=country_bounds.miny.values[0],
                         maxx=country_bounds.maxx.values[0],
                         maxy=country_bounds.maxy.values[0]
                        )
    grid_cell_size = 1
    
    gridded = create_grid(shapely.box(hazard_country.rio.bounds()[0],hazard_country.rio.bounds()[1],hazard_country.rio.bounds()[2],
                                          hazard_country.rio.bounds()[3]),grid_cell_size)
        
    all_bounds = gpd.GeoDataFrame(gridded,columns=['geometry']).bounds
    
    
    features_to_clip = base_network.to_crs(4326)
    
    collect_overlay = []
    
    for bounds in tqdm(all_bounds.itertuples(),total=len(all_bounds)):
        try:
            subset_hazard = hazard_country.rio.clip_box(
            minx=bounds.minx,
            miny=bounds.miny,
            maxx=bounds.maxx,
            maxy=bounds.maxy,
            )
    
            subset_hazard['band_data'] = subset_hazard.band_data.rio.write_nodata(np.nan, inplace=True)
            
            subset_features = gpd.clip(features_to_clip, list(bounds)[1:]).to_crs(3857)
    
            if len(subset_features) == 0:
                continue
    
            subset_hazard = subset_hazard.rio.reproject("EPSG:3857")
    
            values_and_coverage_per_object = exact_extract(
                subset_hazard, subset_features, ["coverage", "values"], output="pandas"
            )
    
            values_and_coverage_per_object.index = subset_features.index
            collect_overlay.append(values_and_coverage_per_object)
            
        except:
            continue 
    
    if not collect_overlay:
        print("⚠ No hazard data was extracted. Returning unmodified base_network.")
        base_network['exposed'] = False
        base_network['exposed_values_depth'] = [[] for _ in range(len(base_network))]
        return base_network
    
    base_network = base_network.merge(pd.concat(collect_overlay),left_index=True,right_index=True)
    
    def Flagged_exposed_segments(row):
        if pd.isna(row['bridge']) or row['bridge'] == 'no':
            return any(val > Threshold for val in row['values'])
        else:
            return any(val > Stru_Threshold for val in row['values'])

    base_network['exposed'] = base_network.progress_apply(Flagged_exposed_segments, axis=1)
    base_network['exposed_values_depth'] = base_network['values']
    
    return base_network

def _get_river_basin(road_segment,basins):
    try:
        return basins.loc[road_segment.geometry.intersects(basins.geometry)].HYBAS_ID.values[0]
    except:
        return None
    

def read_factory_data(config):

    DataFrame_Factory = pd.read_excel(config.Path_FactoryFile)

    Clean_DataFrame_Factory = DataFrame_Factory.dropna(subset=["Latitude", "Longitude", "Factory"])

    geometry = [Point(xy) for xy in zip(Clean_DataFrame_Factory["Longitude"], Clean_DataFrame_Factory["Latitude"])]

    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_Factory[["Number"]].rename(columns={"Number": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326"
    )

    return df_worldpop

def read_road_border_data(config):

    Sink = pd.read_excel(config.path_to_Borders)
    Sink = Sink.rename(columns={"LON": "Longitude", "LAT": "Latitude"})

    return Sink


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

def nearest_network_nodes(gdf_locations: gpd.GeoDataFrame, nodes: pd.DataFrame) -> pd.Series:
    """
    Assign the nearest network node to each input point geometry (e.g., factories,
    agricultural areas or any other locations) using a spatial index.

    Parameters
    ----------
    gdf_locations : gpd.GeoDataFrame
        GeoDataFrame of point locations to snap to the network. Must contain:
        - 'geometry' (Point): location of each feature.
        Side effect: a new column 'vertex_id' is created/overwritten with the
        nearest node identifier.
    nodes : pd.DataFrame
        Table of network nodes. Must contain:
        - 'geometry' (Point): node coordinates (preferably as a GeoSeries/GeoDataFrame column).
        - 'vertex_id' (hashable/int/str): unique node identifier.

    Returns
    -------
    pd.Series
        Series of nearest node identifiers (vertex_id), index-aligned with
        df_factories, and also written to df_factories['vertex_id'].

    Notes
    -----
    - Uses a shapely STRtree for efficient nearest-neighbor lookup.
    - Ensure both inputs use the same coordinate reference system (CRS) before calling.
    - If performance is critical for very large inputs, consider batching or pre-filtering.
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)
    gdf_locations['vertex_id'] = gdf_locations.geometry.progress_apply(
    lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    return gdf_locations['vertex_id']

def map_sinks_to_nearest_network_node(Sink):

    nodes_sindex = shapely.STRtree(nodes.geometry)

    Sink['geometry'] = Sink.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    Sink['vertex_id'] = Sink.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    return Sink

def flood_exposure_factory_accessibility(base_network, df_worldpop, Sink, Factory_criticality_folder, basins_data):

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc='get basin')    
    exposed_roads['subregion'] = exposed_roads.progress_apply(lambda road_segment: _get_river_basin(road_segment,basins_data),axis=1)

    unique_scenarios = {}
    for subregion,subregion_exposed in tqdm(exposed_roads.groupby('subregion'),total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f'unique_scenarios_{country_iso3}_{Subregion}.pkl'
    file_path = os.path.join(Factory_criticality_folder, filename_unique_scenarios)
    with open(file_path, 'wb') as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(exposed_roads.groupby('subregion'), total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove2 = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]
        
        non_empty_values = flooded_edges[flooded_edges['values'].apply(len) > 0]['values']
        
        Merged_Values_Column = [item for sublist in non_empty_values for item in sublist]
        
        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan
        
        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,  
            "max_value": max_value    
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append({
            "basinID": subregion,
            "min water depth (m)": values["min_value"],
            "mean water depth (m)": values["mean_value"],
            "max water depth (m)": values["max_value"]
        })

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(Factory_criticality_folder, f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv")
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # ### Step 12: Run the new access times (in post-event condition) from factories to all border crossings

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)

    for BasinID in tqdm(unique_scenarios,total=len(unique_scenarios)):
        save_new_results[BasinID] = {}
        
        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [x.index for x in graph.es if x['id'] in edges_to_remove]
            
            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)
            
            buffer_zone = basins_data.loc[basins_data.HYBAS_ID == BasinID].to_crs(3857).buffer(50000).to_crs(4326)
            df_population = df_worldpop.iloc[sindex_pop.query(buffer_zone,predicate='intersects')[1]].copy()
            df_population_backup = df_population.copy()
            InitialTotalPopulationPerBasin = df_population_backup['band_data'].sum()

            # Calculate post-flood average access time
            df_population = get_average_access_time(df_population, Sink, damaged_graph)
            
            # Merge with baseline average access time
            scenario_outcome = df_population.merge(df_worldpop['avg_access_time'], left_index=True, right_index=True)
            scenario_outcome = scenario_outcome.rename(columns={
                'avg_access_time_x': 'new_tt',
                'avg_access_time_y': 'old_tt'
            })
            scenario_outcome['Delta'] = scenario_outcome.new_tt - scenario_outcome.old_tt

            scenario_outcome_numeric = scenario_outcome.copy()
            scenario_outcome_numeric['Delta'] = pd.to_numeric(scenario_outcome_numeric['Delta'], errors='coerce')
            scenario_outcome_backup = scenario_outcome_numeric[
                ~(scenario_outcome_numeric['Delta'] == 0) & 
                ~scenario_outcome_numeric['Delta'].isnull() & 
                ~np.isinf(scenario_outcome_numeric['Delta'])
            ].copy()
        
            AffectedPopulation = scenario_outcome_backup['band_data'].sum()
            AffectedPopRatio = AffectedPopulation / InitialTotalPopulationPerBasin
            
            # Lost connections: factories where avg access time hits 12 hours (all crossings unreachable)
            Lost_Connections = scenario_outcome[scenario_outcome['new_tt'] == 12].copy()
            TotalAffectedPopulation = AffectedPopulation + (Lost_Connections['band_data'].sum())
            filtered_scenario_outcome_NotNoneInf = scenario_outcome[
                ~(scenario_outcome['Delta'].isnull()) & 
                ~(scenario_outcome['new_tt'] == 12)
            ]

            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [x['osm_id'] for x in graph.es if x['id'] in edges_to_remove],
                "scenario_outcome": scenario_outcome,
                "Lost_Connections": Lost_Connections,
                "AffectedPopulation": AffectedPopulation,
                "TotalAffectedPopulation": TotalAffectedPopulation,
                "AffectedPopRatio": AffectedPopRatio,
                "filtered_scenario_outcome_NotNoneInf": filtered_scenario_outcome_NotNoneInf,
            }

        except Exception as e:
            save_new_results[BasinID] = {
                "status": "Error",
                "reason": str(e),
                "df_population_backup": None,
                "df_population": None,
                "real_edges_to_remove": None,
                "scenario_outcome": None,
                "Lost_Connections": None,
                "AffectedPopulation": None,
                "TotalAffectedPopulation": None,
                "AffectedPopRatio": None,
                "filtered_scenario_outcome_NotNoneInf": None,
            }

    # ##### -12-1: Save the resulted nested dictionary, save_new_results, as a pickle file:

    filename_nested_Dictionary = f'save_new_results_{country_iso3}_{Subregion}.pkl'
    file_path = os.path.join(Factory_criticality_folder, filename_nested_Dictionary)

    with open(file_path, 'wb') as file:
        pickle.dump(save_new_results, file)

    # ### Step 13: Analysis of the results

    print("Analysis completed successfully!")


def read_population_data(config):

    DataFrame_StatePop = pd.read_excel(config.Path_SettlementData_Excel)

    Clean_DataFrame_StatePop = DataFrame_StatePop.dropna(subset=["latitude", "longitude", "Total"])

    geometry = [Point(xy) for xy in zip(Clean_DataFrame_StatePop["longitude"], Clean_DataFrame_StatePop["latitude"])]

    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_StatePop[["Total"]].rename(columns={"Total": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326"
    )

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


def get_distance_to_nearest_facility(df_population, Sink, graph):
    df_population = df_population.copy()
    df_population['closest_sink_vertex_id'] = None
    df_population['closest_sink_osm_id'] = None
    df_population['closest_sink_total_fft'] = None
    
    unique_pop_vertex_ids = df_population['vertex_id'].unique()
    unique_sink_vertex_ids = Sink['vertex_id'].unique()
    
    sink_lookup = {}
    for _, row in Sink.iterrows():
        sink_lookup[row['vertex_id']] = row['i.d.']

    distance_matrix = np.array(graph.distances(
        source=unique_pop_vertex_ids,
        target=unique_sink_vertex_ids, 
        weights='fft'
    ))
    
    vertex_to_closest_sink = {}
    
    for i, pop_vertex_id in enumerate(unique_pop_vertex_ids):
        distances_to_sinks = distance_matrix[i, :]
        
        min_sink_idx = np.argmin(distances_to_sinks)
        min_distance = distances_to_sinks[min_sink_idx]
        
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
    
    for idx, row in df_population.iterrows():
        vertex_id = row['vertex_id']
        closest_sink_vertex_id, closest_sink_osm_id, closest_sink_total_fft = vertex_to_closest_sink[vertex_id]
        
        df_population.at[idx, 'closest_sink_vertex_id'] = closest_sink_vertex_id
        df_population.at[idx, 'closest_sink_osm_id'] = closest_sink_osm_id
        df_population.at[idx, 'closest_sink_total_fft'] = closest_sink_total_fft
    
    return df_population

def flood_exposure_emergency_service_accessibility(df_worldpop, Sink, TheFolder, basins_data):

    
    # ## Step 10: Identify exposed assets

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc='get basin')    
    exposed_roads['subregion'] = exposed_roads.progress_apply(lambda road_segment: _get_river_basin(road_segment,basins_data),axis=1)

    unique_scenarios = {}
    for subregion,subregion_exposed in tqdm(exposed_roads.groupby('subregion'),total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f'unique_scenarios_{country_iso3}_{Subregion}.pkl'
    file_path = os.path.join(TheFolder, filename_unique_scenarios)
    with open(file_path, 'wb') as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(exposed_roads.groupby('subregion'), total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove2 = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]
        
        non_empty_values = flooded_edges[flooded_edges['values'].apply(len) > 0]['values']
        
        Merged_Values_Column = [item for sublist in non_empty_values for item in sublist]
        
        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan
        
        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,  
            "max_value": max_value    
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append({
            "basinID": subregion,
            "min water depth (m)": values["min_value"],
            "mean water depth (m)": values["mean_value"],
            "max water depth (m)": values["max_value"]
        })

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(TheFolder, f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv")
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # ### Step 12: Run the new access times (in post-event condition) from populaiton points to the nearest health facilities

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)
    for BasinID in tqdm(unique_scenarios,total=len(unique_scenarios)):
        save_new_results[BasinID] = {}
        
        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [x.index  for x in graph.es if x['id'] in edges_to_remove]
            
            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)
            
            buffer_zone = basins_data.loc[basins_data.HYBAS_ID == BasinID].to_crs(3857).buffer(50000).to_crs(4326)
            df_population = df_worldpop.iloc[sindex_pop.query(buffer_zone,predicate='intersects')[1]].copy()
            df_population_backup=df_population.copy()
            InitialTotalPopulationPerBasin=df_population_backup['band_data'].sum()

            df_population=get_distance_to_nearest_facility(df_population,Sink,damaged_graph)
            
            scenario_outcome = df_population.merge(df_worldpop['closest_sink_total_fft'],left_index=True,right_index=True)
            scenario_outcome = scenario_outcome.rename(columns={'closest_sink_total_fft_x': 'new_tt' ,
                                                'closest_sink_total_fft_y': 'old_tt' })
            scenario_outcome['Delta']  = scenario_outcome.new_tt - scenario_outcome.old_tt

            scenario_outcome_numeric = scenario_outcome.copy()
            scenario_outcome_numeric['Delta'] = pd.to_numeric(scenario_outcome_numeric['Delta'], errors='coerce')
            scenario_outcome_backup = scenario_outcome_numeric[~(scenario_outcome_numeric['Delta'] == 0) & ~scenario_outcome_numeric['Delta'].isnull() & 
                                                               ~np.isinf(scenario_outcome_numeric['Delta'])].copy()
        
            AffectedPopulation = scenario_outcome_backup['band_data'].sum()
            AffectedPopRatio=AffectedPopulation/InitialTotalPopulationPerBasin
            
            Lost_Connections = scenario_outcome[scenario_outcome['Delta'] == np.inf].copy()
            TotalAffectedPopulation= AffectedPopulation+ (Lost_Connections['band_data'].sum())
            filtered_scenario_outcome_NotNoneInf = scenario_outcome[~(scenario_outcome['Delta'].isnull() | (scenario_outcome['Delta'] == np.inf))]

            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [x['osm_id']  for x in graph.es if x['id'] in edges_to_remove],
                "scenario_outcome":scenario_outcome,
                "Lost_Connections":Lost_Connections,
                "AffectedPopulation":AffectedPopulation,
                "TotalAffectedPopulation":TotalAffectedPopulation,
                "AffectedPopRatio":AffectedPopRatio,
                "filtered_scenario_outcome_NotNoneInf":filtered_scenario_outcome_NotNoneInf,
            }

        except Exception as e:
            save_new_results[BasinID] = {
                "status": "Error",
                "reason": str(e),
                "df_population_backup": None,
                "df_population": None,
                "real_edges_to_remove": None,
                "scenario_outcome": None,
                "Lost_Connections":None,
                "AffectedPopulation":None,
                "TotalAffectedPopulation":None,
                "AffectedPopRatio":None,
                "filtered_scenario_outcome_NotNoneInf":None,
            }

    # ##### -12-1: Save the resulted nested dictionary, save_new_results, as a pickle file:

    filename_nested_Dictionary = f'save_new_results_{country_iso3}_{Subregion}.pkl'

    file_path = os.path.join(TheFolder, filename_nested_Dictionary)

    with open(file_path, 'wb') as file:
        pickle.dump(save_new_results, file)

    # ### Step 13: Analysis of the resutls

    # #### 13-1: To rank the scenarios and get the top 3 with the highest imapct:

    print("Analysis completed successfully!")

def read_agri_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    # ### Step 4: Read world population data

    # reading the Excel file
    DataFrame_StatePop = pd.read_excel(config.Path_AgriFile)

    # to keep only rows with valid coordinates and Number of AgriLands
    Clean_DataFrame_Agri_Statistics = DataFrame_StatePop.dropna(subset=["latitude", "longitude","Utilized agricultural land (UAL)"])
 
    # to make point geometry
    geometry = [Point(xy) for xy in zip(Clean_DataFrame_Agri_Statistics["longitude"], Clean_DataFrame_Agri_Statistics["latitude"])]

    # build GeoDataFrame matching df_worldpop structure
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_Agri_Statistics[["Utilized agricultural land (UAL)"]].rename(columns={"Utilized agricultural land (UAL)": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326"  # longitude/latitude WGS84
    )

    return df_worldpop

def load_sinks(config: NetworkConfig, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Load border crossings (Sinks) from Excel file.
    
    Args:
        config: Network configuration
        
    Returns:
        Pandas DataFrame with border crossings
    """

    nodes_sindex = shapely.STRtree(nodes.geometry)

    Sinks = pd.read_excel(config.path_to_Sinks)
    Sinks = Sinks.rename(columns={"LON": "Longitude", "LAT": "Latitude", "TYPE OF\nTRAFFIC": "type"})
    Sinks['geometry'] = Sinks.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    Sinks['vertex_id'] = Sinks.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    # Ensure 'name' column exists for compatibility with get_distance_to_nearest_facility
    if 'name' not in Sinks.columns:
        Sinks['name'] = Sinks.index.astype(str)  # or use another identifier column

    # Split by type
    Sinks_road = Sinks[Sinks['type'] == 'road'].copy()
    Sinks_port = Sinks[Sinks['type'] == 'port'].copy()
    Sinks_rail = Sinks[Sinks['type'] == 'rail'].copy()

    # Create sinks dictionary for easy iteration
    sinks_dict = {
        'road': Sinks_road,
        'port': Sinks_port,
        'rail': Sinks_rail,
        'all': Sinks
    }

    return sinks_dict


def calculate_accessibility_by_sink_type(df_population, sinks_dict, graph, inf_replacement=12):
    """
    Calculate both nearest and average access times for each sink type.
    
    Parameters:
    -----------
    df_population : GeoDataFrame
        Population/agricultural points with vertex_id
    sinks_dict : dict
        Dictionary with keys 'road', 'port', 'rail', 'all' and GeoDataFrame values
    graph : igraph.Graph
        Network graph
    inf_replacement : float
        Value to replace inf with for average calculations
        
    Returns:
    --------
    df_population : GeoDataFrame
        With columns for nearest_* and avg_* for each sink type
    """
    df_population = df_population.copy()
    
    for sink_type, sink_df in sinks_dict.items():
        if len(sink_df) == 0:
            print(f"Warning: No sinks of type '{sink_type}', skipping...")
            df_population[f'nearest_{sink_type}'] = np.nan
            df_population[f'avg_{sink_type}'] = np.nan
            continue
            
        unique_pop_vertex_ids = df_population['vertex_id'].unique()
        unique_sink_vertex_ids = sink_df['vertex_id'].unique()
        
        # Calculate distance matrix
        distance_matrix = np.array(graph.distances(
            source=unique_pop_vertex_ids,
            target=unique_sink_vertex_ids,
            weights='fft'
        ))
        
        # NEAREST: minimum distance to any sink
        min_distances = np.min(distance_matrix, axis=1)
        vertex_to_nearest = dict(zip(unique_pop_vertex_ids, min_distances))
        df_population[f'nearest_{sink_type}'] = df_population['vertex_id'].map(vertex_to_nearest)
        
        # AVERAGE: mean distance to all sinks (with inf replacement)
        distance_matrix_for_avg = distance_matrix.copy()
        distance_matrix_for_avg[np.isinf(distance_matrix_for_avg)] = inf_replacement
        avg_distances = np.mean(distance_matrix_for_avg, axis=1)
        vertex_to_avg = dict(zip(unique_pop_vertex_ids, avg_distances))
        df_population[f'avg_{sink_type}'] = df_population['vertex_id'].map(vertex_to_avg)
        
        print(f"{sink_type}: {len(unique_sink_vertex_ids)} sinks | "
              f"Nearest avg: {np.mean(min_distances[~np.isinf(min_distances)]):.2f}h | "
              f"Avg to all: {np.mean(avg_distances):.2f}h")
    
    return df_population


def update_column_names(df_agri):

    # Rename columns to indicate baseline
    baseline_cols = {}
    for sink_type in sinks_dict.keys():
        baseline_cols[f'nearest_{sink_type}'] = f'baseline_nearest_{sink_type}'
        baseline_cols[f'avg_{sink_type}'] = f'baseline_avg_{sink_type}'

    df_agri = df_agri.rename(columns=baseline_cols)

    return df_agri


def flood_exposure_analysis_agriculture(base_network, df_worldpop, TheFolder, basins_data):

     # ## Step 10: Identify exposed assets

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc='get basin')    
    exposed_roads['subregion'] = exposed_roads.progress_apply(lambda road_segment: _get_river_basin(road_segment,basins_data),axis=1)

    unique_scenarios = {}
    for subregion,subregion_exposed in tqdm(exposed_roads.groupby('subregion'),total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f'unique_scenarios_{country_iso3}_{Subregion}.pkl'
    file_path = os.path.join(TheFolder, filename_unique_scenarios)
    with open(file_path, 'wb') as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(exposed_roads.groupby('subregion'), total=len(exposed_roads.groupby('subregion'))):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove2 = base_network.loc[base_network.id.isin(EdgeExposedList)].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]
        
        non_empty_values = flooded_edges[flooded_edges['values'].apply(len) > 0]['values']
        
        Merged_Values_Column = [item for sublist in non_empty_values for item in sublist]
        
        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan
        
        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,  
            "max_value": max_value    
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append({
            "basinID": subregion,
            "min water depth (m)": values["min_value"],
            "mean water depth (m)": values["mean_value"],
            "max water depth (m)": values["max_value"]
        })

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(TheFolder, f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv")
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # =============================================================================
    # UPDATED STEP 12: Run post-flood accessibility for ALL sink types
    # =============================================================================

    print("\n" + "="*60)
    print("FLOOD SCENARIO ACCESSIBILITY CALCULATIONS")
    print("="*60)

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)
    C = 1

    for BasinID in tqdm(unique_scenarios, desc="Processing basins"):
        save_new_results[BasinID] = {}
        
        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [x.index for x in graph.es if x['id'] in edges_to_remove]
            
            # Create damaged graph
            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)
            
            # Get population in buffer zone around basin
            buffer_zone = basins_data.loc[basins_data.HYBAS_ID == BasinID].to_crs(3857).buffer(50000).to_crs(4326)
            df_population = df_worldpop.iloc[sindex_pop.query(buffer_zone, predicate='intersects')[1]].copy()
            df_population_backup = df_population.copy()
            InitialTotalPopulationPerBasin = df_population_backup['band_data'].sum()
            
            # Calculate post-flood accessibility for ALL sink types
            df_population = calculate_accessibility_by_sink_type(df_population, sinks_dict, damaged_graph)
            
            # Rename to post-flood columns
            postflood_cols = {}
            for sink_type in sinks_dict.keys():
                postflood_cols[f'nearest_{sink_type}'] = f'postflood_nearest_{sink_type}'
                postflood_cols[f'avg_{sink_type}'] = f'postflood_avg_{sink_type}'
            df_population = df_population.rename(columns=postflood_cols)
            
            # Calculate deltas for each sink type and metric
            for sink_type in sinks_dict.keys():
                # Delta for nearest
                df_population[f'delta_nearest_{sink_type}'] = (
                    df_population[f'postflood_nearest_{sink_type}'] - 
                    df_population[f'baseline_nearest_{sink_type}']
                )
                # Delta for average
                df_population[f'delta_avg_{sink_type}'] = (
                    df_population[f'postflood_avg_{sink_type}'] - 
                    df_population[f'baseline_avg_{sink_type}']
                )
            
            # Calculate summary statistics per sink type
            results_by_sink_type = {}
            
            for sink_type in sinks_dict.keys():
                # Nearest sink analysis
                delta_nearest = df_population[f'delta_nearest_{sink_type}']
                affected_nearest = df_population[
                    (~delta_nearest.isnull()) & 
                    (delta_nearest != 0) & 
                    (~np.isinf(delta_nearest))
                ]
                lost_nearest = df_population[delta_nearest == np.inf]
                
                # Average sink analysis
                delta_avg = df_population[f'delta_avg_{sink_type}']
                affected_avg = df_population[
                    (~delta_avg.isnull()) & 
                    (delta_avg != 0) & 
                    (~np.isinf(delta_avg))
                ]
                
                results_by_sink_type[sink_type] = {
                    # Nearest metrics
                    'affected_pop_nearest': affected_nearest['band_data'].sum(),
                    'lost_connections_nearest': lost_nearest['band_data'].sum(),
                    'mean_delta_nearest': delta_nearest[~np.isinf(delta_nearest)].mean(),
                    'max_delta_nearest': delta_nearest[~np.isinf(delta_nearest)].max(),
                    
                    # Average metrics
                    'affected_pop_avg': affected_avg['band_data'].sum(),
                    'mean_delta_avg': delta_avg.mean(),
                    'max_delta_avg': delta_avg.max(),
                }
            
            # Store results
            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [x['osm_id'] for x in graph.es if x['id'] in edges_to_remove],
                "InitialTotalPopulation": InitialTotalPopulationPerBasin,
                "results_by_sink_type": results_by_sink_type,
            }
            
        except Exception as e:
            save_new_results[BasinID] = {
                "status": "Error",
                "reason": str(e),
                "df_population_backup": None,
                "df_population": None,
                "real_edges_to_remove": None,
                "InitialTotalPopulation": None,
                "results_by_sink_type": None,
            }
        
        C += 1

    # ##### -12-1: Save the resulted nested dictionary, save_new_results, as a pickle file:

    filename_nested_Dictionary = f'save_new_results_{country_iso3}_{Subregion}.pkl'
    file_path = os.path.join(TheFolder, filename_nested_Dictionary)
    with open(file_path, 'wb') as file:
        pickle.dump(save_new_results, file)

    # =============================================================================
    # UPDATED STEP 13: Create summary DataFrames by sink type
    # =============================================================================

    print("\n" + "="*60)
    print("CREATING SUMMARY TABLES")
    print("="*60)

    summary_rows = []

    for BasinID, results in save_new_results.items():
        if results.get('status') == 'Error':
            continue
        
        row = {'BasinID': BasinID, 'InitialTotalPopulation': results['InitialTotalPopulation']}
        
        for sink_type, metrics in results['results_by_sink_type'].items():
            for metric_name, value in metrics.items():
                row[f'{sink_type}_{metric_name}'] = value
        
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save summary CSV
    output_csv_file = os.path.join(TheFolder, f"{country_iso3}_accessibility_impact_summary_{Subregion}.csv")
    summary_df.to_csv(output_csv_file, index=False)

    print(f"\nSummary saved to: {output_csv_file}")
    print(f"Columns: {list(summary_df.columns)}")
    print(summary_df.head())

    print("\nAnalysis completed successfully!")



def main():
    """
    Run the end-to-end flood-scenario accessibility and criticality analysis.
    Processes industrial areas, agricultural areas, and emergency services
    (firefighters, hospitals, police) by mapping each location to the road
    network, computing baseline accessibility, and then evaluating how flooding
    disrupts access under basin-specific scenarios. All outputs are saved into
    sector-specific criticality folders for further analysis.

    
    WARNING: This script performs large-scale network routing and basin-level
    flood disruption simulations and may take SEVERAL HOURS to run depending on
    hardware and data size.

    """

    # ------------------------------------------------------------
    # Load configuration with file paths and settings
    # ------------------------------------------------------------
    config = NetworkConfig()

    # Analysis parameters (only used indirectly in sub-functions)
    upscale_factor = 10
    country_iso3 = 'SRB'
    PostEvent_speed = 20
    Subregion = 'basins'
    basins = True
    subnational = False
    Threshold = 0.25
    Stru_Threshold = 4

    # ------------------------------------------------------------
    # 1. Load road network and build routing graph
    # ------------------------------------------------------------
    base_network = gpd.read_parquet(config.Path_RoadNetwork)
    print("Creating graph representation of the road network...")
    nodes, graph = create_graph_for_spatial_matching(base_network)

    # Load flood maps + basins for clipping/aggregation
    hazard_map = xr.open_dataset(config.flood_map_RP100, engine="rasterio")
    basins_data = gpd.read_file(config.basins_shapefile)

    # ------------------------------------------------------------
    # 2. Load and prepare settlements (baseline demand points)
    # ------------------------------------------------------------
    df_settlements = read_population_data(config)
    df_settlements['vertex_id'] = nearest_network_nodes(df_settlements, nodes)

    # ============================================================
    # 3. INDUSTRIAL AREAS (Factories)
    # ============================================================
    Factory_criticality_folder = "factory_criticality_results"
    os.makedirs(Factory_criticality_folder, exist_ok=True)

    # Load factories + map to nearest road node
    df_factories = read_factory_data(config)
    print("Mapping industrial areas to nearest network nodes...")
    df_factories['vertex_id'] = nearest_network_nodes(df_factories, nodes)

    # Load border crossings (sinks) and map them to nodes
    Sink = read_road_border_data(config)
    Sink = map_sinks_to_nearest_network_node(Sink)

    # Compute baseline average access times (normal conditions)
    print("Calculating average access times from factories to road border crossings...")
    df_factories = get_average_access_time(df_factories, Sink, graph)

    # Assess access disruption under flooding
    print("Calculating factory access times under flooding scenarios...")
    flood_exposure_factory_accessibility(
        base_network, df_factories, Sink, Factory_criticality_folder, basins_data
    )

    # ============================================================
    # 4. AGRICULTURAL AREAS
    # ============================================================
    Agriculture_criticality_folder = "allagri_criticality_results"
    os.makedirs(Agriculture_criticality_folder, exist_ok=True)

    df_agri = read_agri_data(config)
    print("Mapping agricultural areas to nearest network nodes...")
    df_agri['vertex_id'] = nearest_network_nodes(df_agri, nodes)

    # Load sinks: borders, ports, rail terminals
    print("Loading sink location data (borders, ports, rail terminals)...")
    sinks_dict = load_sinks(config, nodes)

    print("\n" + "=" * 60)
    print("BASELINE ACCESSIBILITY CALCULATIONS")
    print("=" * 60)

    # Baseline access to borders/ports/terminals
    print("Calculating baseline agricultural accessibility to all sink types...")
    df_agri = calculate_accessibility_by_sink_type(df_agri, sinks_dict, graph)
    df_agri = update_column_names(df_agri)

    # Flood impact on agricultural accessibility
    print("Calculating agricultural access under flooding scenarios...")
    flood_exposure_analysis_agriculture(
        base_network, df_agri, Agriculture_criticality_folder, basins_data
    )

    # ============================================================
    # 5. FIREFIGHTERS
    # ============================================================
    Fire_criticality_folder = "fire_criticality_results"
    os.makedirs(Fire_criticality_folder, exist_ok=True)

    print("Loading firefighter locations...")
    sink_firefighters = load_and_map_sinks(config, nodes, "firefighters")

    print("Calculating distance to the nearest fire station...")
    accessibility_firefighters = get_distance_to_nearest_facility(
        df_settlements, sink_firefighters, graph
    )

    flood_exposure_emergency_service_accessibility(
        accessibility_firefighters, sink_firefighters,
        Fire_criticality_folder, basins_data
    )

    # ============================================================
    # 6. HEALTHCARE FACILITIES
    # ============================================================
    Health_care_criticality_folder = "healthcare_criticality_results"
    os.makedirs(Health_care_criticality_folder, exist_ok=True)

    print("Loading healthcare locations...")
    sink_hospitals = load_and_map_sinks(config, nodes, "hospitals")

    print("Calculating access to the nearest hospital...")
    accessibility_hospitals = get_distance_to_nearest_facility(
        df_settlements, sink_hospitals, graph
    )

    flood_exposure_emergency_service_accessibility(
        accessibility_hospitals, sink_hospitals,
        Health_care_criticality_folder, basins_data
    )

    # ============================================================
    # 7. POLICE
    # ============================================================
    Police_criticality_folder = "police_criticality_results"
    os.makedirs(Police_criticality_folder, exist_ok=True)

    print("Loading police station locations...")
    sink_police = load_and_map_sinks(config, nodes, "police")

    print("Calculating access to the nearest police station...")
    accessibility_police = get_distance_to_nearest_facility(
        df_settlements, sink_police, graph
    )

    flood_exposure_emergency_service_accessibility(
        accessibility_police, sink_police,
        Police_criticality_folder, basins_data
    )




if __name__ == "__main__":
    main()