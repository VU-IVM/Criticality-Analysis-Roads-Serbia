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

sys.path.append(os.path.join('..','src'))

from simplify import *

# load functions
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


if __name__ == "__main__":
    
    # ### File Paths Configuration
    # Base directories
    base_onedrive = r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam"
    shadi_base = os.path.join(base_onedrive, "Shirazian, S. (Shadi)'s files - AFD")
    
    # Input file paths
    file_paths = {
        # World boundaries shapefile
        'world_boundaries': os.path.join(base_onedrive, "Shirazian, S. (Shadi)'s files - NewCodes", "Tajikistan", "ne_10m_admin_0_countries.shp"),
        
        # Population data
        'agriculture_excel': os.path.join(shadi_base, "data", "Jelena", "SocioEconomicPack_Received on july24", "agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"),
        
        # Health facilities data
        'agriports': r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\2_Projects\AFD_Serbia\Data\socioeconomic\Ports.xlsx",
        
        # Hazard data
        'hazard_map': os.path.join(shadi_base, "data", "Hazard", "Flood", "JRC", "Europe_RP100_filled_depth.tif"),
        
        # Basins data
        'basins_shapefile': os.path.join(base_onedrive, "2_Projects", "WorldBank_Projects", "accesibility", "hybas_eu_lev09_v1c.shp")
    }
    
    # ### Step 3: Provide the project-specific information

    upscale_factor=10
    country_iso3='SRB'
    PostEvent_speed=20
    Subregion='basins'
    basins = True
    subnational = False

    speed_d = {
            'motorway': 80,
            'motorway_link': 65,
            'trunk': 60,
            'trunk_link': 50,
            'primary': 50,
            'primary_link': 40,
            'secondary': 40,
            'secondary_link': 30,
            'tertiary': 30,
            'tertiary_link': 20,
            'unclassified': 20,
            'service': 20,
            'residential': 20,
             'road':30,
             'path' : 30,
             'track':30,
        }

    lanes_d = {'motorway': '4','motorway_link': '2','trunk': '4','trunk_link': '2','primary': '2','primary_link': '1','secondary': '2','secondary_link': '1','tertiary': '2','tertiary_link': '1','unclassified': '2','service': '1','residential': '1'}

    Threshold=0.25
    Stru_Threshold=4
    TheFolder = "perishables_criticality_results"
    os.makedirs(TheFolder, exist_ok=True)

    # ### Step 4: Read world population data

    # reading the Excel file
    Path_StatementFile_Excel = file_paths['agriculture_excel']

    DataFrame_StatePop = pd.read_excel(Path_StatementFile_Excel)

    # to keep only rows with valid coordinates and Number of AgriLands
    Clean_DataFrame_Agri_Statistics = DataFrame_StatePop.dropna(subset=["latitude", "longitude"])
    cols = [
        "Vegetables, melons and strawberries total",
        "Flowers and ornamental plants",
        "Orchards"
    ]

    Clean_DataFrame_Agri_Statistics["perishable"] = Clean_DataFrame_Agri_Statistics[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # to make point geometry
    geometry = [Point(xy) for xy in zip(Clean_DataFrame_Agri_Statistics["longitude"], Clean_DataFrame_Agri_Statistics["latitude"])]

    # build GeoDataFrame matching df_worldpop structure
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_Agri_Statistics[["perishable"]].rename(columns={"perishable": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326"  # longitude/latitude WGS84
    )

    # ### Step 5: Input files
    Path_to_AgriPorts= file_paths['agriports']
     
    Sink = pd.read_excel(
        Path_to_AgriPorts,
        index_col=None,  # keep all columns, no index from Excel
    )
    
    hazard_map = xr.open_dataset(file_paths['hazard_map'], engine="rasterio")
    basins_data = gpd.read_file(file_paths['basins_shapefile'])

    # ### Step 6: Prepare the base network then create the graph

    filename_parquet = f'base_network_{country_iso3}_{Subregion}.parquet'
    file_path = os.path.join(filename_parquet)

    base_network = gpd.read_parquet(file_path)

    # #### - 6-1: Load the baseline into igraph:

    edges = base_network.reindex(['from_id','to_id'] + [x for x in list(base_network.columns) if x not in ['from_id','to_id']],axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=True)
    graph.vs['id'] = graph.vs['name']

    graph = graph.connected_components().giant()
    edges = edges[edges['id'].isin(graph.es['id'])]

    edge_attributes = graph.es.attributes()

    # ### Step 7: Get the nearest node of the graph to each population point in df_worldpop
    vertex_lookup = dict(zip(pd.DataFrame(graph.vs['name'])[0], pd.DataFrame(graph.vs['name']).index))

    tqdm.pandas()
    from_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[0]))
    to_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[-1]))

    from_dict = dict(zip(edges['from_id'], from_id_geom))
    to_dict = dict(zip(edges['to_id'], to_id_geom))

    nodes = pd.concat([pd.DataFrame.from_dict(to_dict, orient='index', columns=['geometry']),
                       pd.DataFrame.from_dict(from_dict, orient='index', columns=['geometry'])]).drop_duplicates()

    nodes['vertex_id'] = nodes.progress_apply(lambda x: vertex_lookup[x.name], axis=1)
    nodes = nodes.reset_index()

    nodes_sindex = shapely.STRtree(nodes.geometry)

    df_worldpop['vertex_id'] = df_worldpop.geometry.progress_apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    # ### Step 8: Get the nearest node of the graph to each health facility point

    Sink['geometry'] = Sink.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    Sink['vertex_id'] = Sink.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

    # ### Step 9: Get the access time IN NORMAL CONDITION from each population node to the nearest Health Facility
    df_worldpop = get_distance_to_nearest_facility(df_worldpop,Sink,graph)

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