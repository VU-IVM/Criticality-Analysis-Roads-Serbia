# ### Step1: Importing the required packages

import os
import pickle
import xarray as xr
import igraph as ig
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from tqdm import tqdm
from exactextract import exact_extract
from concurrent.futures import ProcessPoolExecutor, as_completed

from simplify import *
from config.network_config import NetworkConfig

# Add near your other config constants
OSM_TO_SERBIAN_CATEGORY = {
    "motorway": "IA",
    "trunk":    "IM",
    "primary":  "IB",
    "secondary":"IIA",
    "tertiary": "IIB",
}

FINAL_BIAS = {
    "IA":  4.601,
    "IM":  3.659,
    "IB":  0.82,
    "IIA": 0.82,
    "IIB": 0.82,
}

# load functions
def get_average_access_time(df_population, Sink, graph):
    """Calculate average travel time from each origin to ALL sinks"""
    df_population = df_population.copy()

    unique_pop_vertex_ids = df_population["vertex_id"].unique()
    unique_sink_vertex_ids = Sink["vertex_id"].unique()

    # Full OD matrix
    OD_matrix = np.array(
        graph.distances(
            source=unique_pop_vertex_ids, target=unique_sink_vertex_ids, weights="fft"
        )
    )
    OD_matrix[np.isinf(OD_matrix)] = 12  # 12 hour penalty

    # Average across all sinks (axis=1)
    avg_time_per_origin = np.mean(OD_matrix, axis=1)
    vertex_to_avg_time = dict(zip(unique_pop_vertex_ids, avg_time_per_origin))

    df_population["avg_access_time"] = df_population["vertex_id"].map(
        vertex_to_avg_time
    )

    return df_population


def create_grid(bbox, height):
    xmin, ymin = shapely.total_bounds(bbox)[0], shapely.total_bounds(bbox)[1]
    xmax, ymax = shapely.total_bounds(bbox)[2], shapely.total_bounds(bbox)[3]

    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / height))

    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append(
                (
                    (
                        (x_left_origin, y_top),
                        (x_right_origin, y_top),
                        (x_right_origin, y_bottom),
                        (x_left_origin, y_bottom),
                    )
                )
            )
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height

    return shapely.polygons(res_geoms)


def apply_road_elevation_correction(
    base_network: gpd.GeoDataFrame, Threshold: float, Stru_Threshold: float
) -> gpd.GeoDataFrame:
    """
    Apply road-category elevation bias correction to a pre-loaded base_network
    that already has raw sampled depth values in 'exposed_values_depth'.

    This corrects for the fact that some road categories sit higher relative to
    the surrounding terrain, so raw flood depths overestimate inundation.
    The corrected depths are clamped to 0 and used to re-evaluate exposure.

    Parameters
    ----------
    base_network : gpd.GeoDataFrame
        Road network with 'exposed_values_depth', 'highway', and 'bridge' columns.
    Threshold : float
        Depth threshold (m) above which a normal road is considered exposed.
    Stru_Threshold : float
        Depth threshold (m) above which a bridge is considered exposed.

    Returns
    -------
    gpd.GeoDataFrame
        base_network with updated 'exposed' and 'exposed_values_depth' columns.
    """
    def adjust_depths(row):
        """Subtract road-category bias from raw sampled depths, clamp to 0."""
        serbian_cat = OSM_TO_SERBIAN_CATEGORY.get(row.get("highway"))
        bias = FINAL_BIAS.get(serbian_cat, 0.0)
        if bias == 0.0:
            return row["exposed_values_depth"]
        return [max(v - bias, 0.0) for v in row["exposed_values_depth"]]

    base_network = base_network.copy()
    tqdm.pandas(desc="elevation bias correction")
    base_network["adjusted_values"] = base_network.progress_apply(adjust_depths, axis=1)

    def flagged_exposed_segments(row):
        if pd.isna(row["bridge"]) or row["bridge"] == "no":
            return any(val > Threshold for val in row["adjusted_values"])
        else:
            return any(val > Stru_Threshold for val in row["adjusted_values"])

    tqdm.pandas(desc="re-evaluating exposure")
    base_network["exposed"] = base_network.progress_apply(
        flagged_exposed_segments, axis=1
    )
    base_network["exposed_values_depth"] = base_network["adjusted_values"]
    base_network.drop(columns=["adjusted_values"], inplace=True)

    return base_network


def get_exposure_values(
    country_iso3, base_network, hazard_map, Threshold, Stru_Threshold
):
    world = gpd.read_file(NetworkConfig.world_boundaries)
    country_bounds = world.loc[world.ADM0_A3 == country_iso3].bounds
    world.loc[world.ADM0_A3 == country_iso3].geometry
    hazard_country = hazard_map.rio.clip_box(
        minx=country_bounds.minx.values[0],
        miny=country_bounds.miny.values[0],
        maxx=country_bounds.maxx.values[0],
        maxy=country_bounds.maxy.values[0],
    )
    grid_cell_size = 1
    gridded = create_grid(
        shapely.box(
            hazard_country.rio.bounds()[0],
            hazard_country.rio.bounds()[1],
            hazard_country.rio.bounds()[2],
            hazard_country.rio.bounds()[3],
        ),
        grid_cell_size,
    )
    all_bounds = gpd.GeoDataFrame(gridded, columns=["geometry"]).bounds
    features_to_clip = base_network.to_crs(4326)
    collect_overlay = []
    for bounds in tqdm(all_bounds.itertuples(), total=len(all_bounds)):
        try:
            subset_hazard = hazard_country.rio.clip_box(
                minx=bounds.minx,
                miny=bounds.miny,
                maxx=bounds.maxx,
                maxy=bounds.maxy,
            )
            subset_hazard["band_data"] = subset_hazard.band_data.rio.write_nodata(
                np.nan, inplace=True
            )
            subset_features = gpd.clip(features_to_clip, list(bounds)[1:]).to_crs(3857)
            if len(subset_features) == 0:
                continue
            subset_hazard = subset_hazard.rio.reproject("EPSG:3857")
            values_and_coverage_per_object = exact_extract(
                subset_hazard, subset_features, ["coverage", "values"], output="pandas"
            )
            values_and_coverage_per_object.index = subset_features.index
            collect_overlay.append(values_and_coverage_per_object)
        except Exception:
            continue

    if not collect_overlay:
        print("⚠ No hazard data was extracted. Returning unmodified base_network.")
        base_network["exposed"] = False
        base_network["exposed_values_depth"] = [[] for _ in range(len(base_network))]
        return base_network

    base_network = base_network.merge(
        pd.concat(collect_overlay), left_index=True, right_index=True
    )

    # Store raw sampled values before bias correction
    base_network["exposed_values_depth"] = base_network["values"]

    # Apply elevation bias correction and re-evaluate exposure
    base_network = apply_road_elevation_correction(
        base_network, Threshold, Stru_Threshold
    )

    return base_network


def _get_river_basin(road_segment, basins):
    try:
        return basins.loc[
            road_segment.geometry.intersects(basins.geometry)
        ].HYBAS_ID.values[0]
    except Exception:
        return None


def read_factory_data(config):

    DataFrame_Factory = pd.read_excel(config.Path_FactoryFile)

    Clean_DataFrame_Factory = DataFrame_Factory.dropna(
        subset=["Latitude", "Longitude", "Factory"]
    )

    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_Factory["Longitude"], Clean_DataFrame_Factory["Latitude"]
        )
    ]

    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_Factory[["Number"]].rename(columns={"Number": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326",
    )

    return df_worldpop


def read_road_border_data(config):

    Sink = pd.read_excel(config.path_to_Borders)
    Sink = Sink.rename(columns={"LON": "Longitude", "LAT": "Latitude"})

    return Sink


def create_graph_for_spatial_matching(
    base_network: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, ig.Graph]:
    """
    Create graph from road network and create nodes for spatial matching.

    Args:
        base_network: Serbian road network

    Returns:
        Pandas DataFrame with the nodes of the road network graph
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
    gdf_locations["vertex_id"] = gdf_locations.geometry.progress_apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return gdf_locations["vertex_id"]


def map_sinks_to_nearest_network_node(
    Sink: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:

    nodes_sindex = shapely.STRtree(nodes.geometry)

    Sink["geometry"] = Sink.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    Sink["vertex_id"] = Sink.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return Sink


def flood_exposure_factory_accessibility(
    base_network,
    df_worldpop,
    Sink,
    Factory_criticality_folder,
    basins_data,
    graph,
    country_iso3,
    Subregion,
):

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc="get basin")
    exposed_roads["subregion"] = exposed_roads.progress_apply(
        lambda road_segment: _get_river_basin(road_segment, basins_data), axis=1
    )

    unique_scenarios = {}
    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f"unique_scenarios_{country_iso3}_{Subregion}.pkl"
    file_path = os.path.join(Factory_criticality_folder, filename_unique_scenarios)
    with open(file_path, "wb") as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]

        non_empty_values = flooded_edges[flooded_edges["values"].apply(len) > 0][
            "values"
        ]

        Merged_Values_Column = [
            item for sublist in non_empty_values for item in sublist
        ]

        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan

        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,
            "max_value": max_value,
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append(
            {
                "basinID": subregion,
                "min water depth (m)": values["min_value"],
                "mean water depth (m)": values["mean_value"],
                "max water depth (m)": values["max_value"],
            }
        )

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(
        Factory_criticality_folder,
        f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv",
    )
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # ### Step 12: Run the new access times (in post-event condition) from factories to all border crossings

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)

    for BasinID in tqdm(unique_scenarios, total=len(unique_scenarios)):
        save_new_results[BasinID] = {}

        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [
                x.index for x in graph.es if x["id"] in edges_to_remove
            ]

            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)

            buffer_zone = (
                basins_data.loc[basins_data.HYBAS_ID == BasinID]
                .to_crs(3857)
                .buffer(50000)
                .to_crs(4326)
            )
            df_population = df_worldpop.iloc[
                sindex_pop.query(buffer_zone, predicate="intersects")[1]
            ].copy()
            df_population_backup = df_population.copy()
            InitialTotalPopulationPerBasin = df_population_backup["band_data"].sum()

            # Calculate post-flood average access time
            df_population = get_average_access_time(df_population, Sink, damaged_graph)

            # Merge with baseline average access time
            scenario_outcome = df_population.merge(
                df_worldpop["avg_access_time"], left_index=True, right_index=True
            )
            scenario_outcome = scenario_outcome.rename(
                columns={"avg_access_time_x": "new_tt", "avg_access_time_y": "old_tt"}
            )
            scenario_outcome["Delta"] = (
                scenario_outcome.new_tt - scenario_outcome.old_tt
            )

            scenario_outcome_numeric = scenario_outcome.copy()
            scenario_outcome_numeric["Delta"] = pd.to_numeric(
                scenario_outcome_numeric["Delta"], errors="coerce"
            )
            scenario_outcome_backup = scenario_outcome_numeric[
                ~(scenario_outcome_numeric["Delta"] == 0)
                & ~scenario_outcome_numeric["Delta"].isnull()
                & ~np.isinf(scenario_outcome_numeric["Delta"])
            ].copy()

            AffectedPopulation = scenario_outcome_backup["band_data"].sum()
            AffectedPopRatio = AffectedPopulation / InitialTotalPopulationPerBasin

            # Lost connections: factories where avg access time hits 12 hours (all crossings unreachable)
            Lost_Connections = scenario_outcome[scenario_outcome["new_tt"] == 12].copy()
            TotalAffectedPopulation = AffectedPopulation + (
                Lost_Connections["band_data"].sum()
            )
            filtered_scenario_outcome_NotNoneInf = scenario_outcome[
                ~(scenario_outcome["Delta"].isnull())
                & ~(scenario_outcome["new_tt"] == 12)
            ]

            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [
                    x["osm_id"] for x in graph.es if x["id"] in edges_to_remove
                ],
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

    filename_nested_Dictionary = f"save_new_results_{country_iso3}_{Subregion}.pkl"
    file_path = os.path.join(Factory_criticality_folder, filename_nested_Dictionary)

    with open(file_path, "wb") as file:
        pickle.dump(save_new_results, file)

    # ### Step 13: Analysis of the results

    print("Analysis completed successfully!")


def read_population_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Load settlement population data from an Excel file, filter out rows with missing
    coordinates or population values, and convert the result into a GeoDataFrame of
    settlement points for use in accessibility and network analyses.

    Parameters
    ----------
    config : NetworkConfig
        Provides the path to the settlement Excel file containing 'latitude',
        'longitude', and 'Total' population fields.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with one point per settlement, including a 'band_data'
        population column and geometries defined in EPSG:4326.
    """

    DataFrame_StatePop = pd.read_excel(config.Path_SettlementData_Excel)

    Clean_DataFrame_StatePop = DataFrame_StatePop.dropna(
        subset=["latitude", "longitude", "Total"]
    )

    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_StatePop["longitude"], Clean_DataFrame_StatePop["latitude"]
        )
    ]

    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_StatePop[["Total"]].rename(columns={"Total": "band_data"}),
        geometry=geometry,
        crs="EPSG:4326",
    )

    return df_worldpop


def load_and_map_sinks(
    config: NetworkConfig, nodes: pd.DataFrame, sink_type
) -> pd.DataFrame:
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

    Sink["geometry"] = Sink.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    Sink["vertex_id"] = Sink.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    return Sink


def get_distance_to_nearest_facility(df_population, Sink, graph) -> gpd.GeoDataFrame:
    df_population = df_population.copy()
    df_population["closest_sink_vertex_id"] = None
    df_population["closest_sink_osm_id"] = None
    df_population["closest_sink_total_fft"] = None

    unique_pop_vertex_ids = df_population["vertex_id"].unique()
    unique_sink_vertex_ids = Sink["vertex_id"].unique()

    sink_lookup = {}
    for _, row in Sink.iterrows():
        sink_lookup[row["vertex_id"]] = row["i.d."]

    distance_matrix = np.array(
        graph.distances(
            source=unique_pop_vertex_ids, target=unique_sink_vertex_ids, weights="fft"
        )
    )

    vertex_to_closest_sink = {}

    for i, pop_vertex_id in enumerate(unique_pop_vertex_ids):
        distances_to_sinks = distance_matrix[i, :]

        min_sink_idx = np.argmin(distances_to_sinks)
        min_distance = distances_to_sinks[min_sink_idx]

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

    for idx, row in df_population.iterrows():
        vertex_id = row["vertex_id"]
        closest_sink_vertex_id, closest_sink_osm_id, closest_sink_total_fft = (
            vertex_to_closest_sink[vertex_id]
        )

        df_population.at[idx, "closest_sink_vertex_id"] = closest_sink_vertex_id
        df_population.at[idx, "closest_sink_osm_id"] = closest_sink_osm_id
        df_population.at[idx, "closest_sink_total_fft"] = closest_sink_total_fft

    return df_population


def flood_exposure_emergency_service_accessibility(
    df_worldpop,
    Sink,
    TheFolder,
    basins_data,
    base_network,
    graph,
    country_iso3,
    Subregion,
):

    # ## Step 10: Identify exposed assets

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc="get basin")
    exposed_roads["subregion"] = exposed_roads.progress_apply(
        lambda road_segment: _get_river_basin(road_segment, basins_data), axis=1
    )

    unique_scenarios = {}
    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f"unique_scenarios_{country_iso3}_{Subregion}.pkl"
    file_path = os.path.join(TheFolder, filename_unique_scenarios)
    with open(file_path, "wb") as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]

        non_empty_values = flooded_edges[flooded_edges["values"].apply(len) > 0][
            "values"
        ]

        Merged_Values_Column = [
            item for sublist in non_empty_values for item in sublist
        ]

        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan

        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,
            "max_value": max_value,
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append(
            {
                "basinID": subregion,
                "min water depth (m)": values["min_value"],
                "mean water depth (m)": values["mean_value"],
                "max water depth (m)": values["max_value"],
            }
        )

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(
        TheFolder, f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv"
    )
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # ### Step 12: Run the new access times (in post-event condition) from populaiton points to the nearest health facilities

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)
    for BasinID in tqdm(unique_scenarios, total=len(unique_scenarios)):
        save_new_results[BasinID] = {}

        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [
                x.index for x in graph.es if x["id"] in edges_to_remove
            ]

            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)

            buffer_zone = (
                basins_data.loc[basins_data.HYBAS_ID == BasinID]
                .to_crs(3857)
                .buffer(50000)
                .to_crs(4326)
            )
            df_population = df_worldpop.iloc[
                sindex_pop.query(buffer_zone, predicate="intersects")[1]
            ].copy()
            df_population_backup = df_population.copy()
            InitialTotalPopulationPerBasin = df_population_backup["band_data"].sum()

            df_population = get_distance_to_nearest_facility(
                df_population, Sink, damaged_graph
            )

            scenario_outcome = df_population.merge(
                df_worldpop["closest_sink_total_fft"], left_index=True, right_index=True
            )
            scenario_outcome = scenario_outcome.rename(
                columns={
                    "closest_sink_total_fft_x": "new_tt",
                    "closest_sink_total_fft_y": "old_tt",
                }
            )
            scenario_outcome["Delta"] = (
                scenario_outcome.new_tt - scenario_outcome.old_tt
            )

            scenario_outcome_numeric = scenario_outcome.copy()
            scenario_outcome_numeric["Delta"] = pd.to_numeric(
                scenario_outcome_numeric["Delta"], errors="coerce"
            )
            scenario_outcome_backup = scenario_outcome_numeric[
                ~(scenario_outcome_numeric["Delta"] == 0)
                & ~scenario_outcome_numeric["Delta"].isnull()
                & ~np.isinf(scenario_outcome_numeric["Delta"])
            ].copy()

            AffectedPopulation = scenario_outcome_backup["band_data"].sum()
            AffectedPopRatio = AffectedPopulation / InitialTotalPopulationPerBasin

            Lost_Connections = scenario_outcome[
                scenario_outcome["Delta"] == np.inf
            ].copy()
            TotalAffectedPopulation = AffectedPopulation + (
                Lost_Connections["band_data"].sum()
            )
            filtered_scenario_outcome_NotNoneInf = scenario_outcome[
                ~(
                    scenario_outcome["Delta"].isnull()
                    | (scenario_outcome["Delta"] == np.inf)
                )
            ]

            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [
                    x["osm_id"] for x in graph.es if x["id"] in edges_to_remove
                ],
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

    filename_nested_Dictionary = f"save_new_results_{country_iso3}_{Subregion}.pkl"

    file_path = os.path.join(TheFolder, filename_nested_Dictionary)

    with open(file_path, "wb") as file:
        pickle.dump(save_new_results, file)

    # ### Step 13: Analysis of the resutls

    # #### 13-1: To rank the scenarios and get the top 3 with the highest imapct:

    print("Analysis completed successfully!")


def read_agri_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    # ### Step 4: Read world population data

    # reading the Excel file
    DataFrame_StatePop = pd.read_excel(config.Path_AgriFile)

    # to keep only rows with valid coordinates and Number of AgriLands
    Clean_DataFrame_Agri_Statistics = DataFrame_StatePop.dropna(
        subset=["latitude", "longitude", "Utilized agricultural land (UAL)"]
    )

    # to make point geometry
    geometry = [
        Point(xy)
        for xy in zip(
            Clean_DataFrame_Agri_Statistics["longitude"],
            Clean_DataFrame_Agri_Statistics["latitude"],
        )
    ]

    # build GeoDataFrame matching df_worldpop structure
    df_worldpop = gpd.GeoDataFrame(
        Clean_DataFrame_Agri_Statistics[["Utilized agricultural land (UAL)"]].rename(
            columns={"Utilized agricultural land (UAL)": "band_data"}
        ),
        geometry=geometry,
        crs="EPSG:4326",  # longitude/latitude WGS84
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
    Sinks = Sinks.rename(
        columns={"LON": "Longitude", "LAT": "Latitude", "TYPE OF\nTRAFFIC": "type"}
    )
    Sinks["geometry"] = Sinks.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    Sinks["vertex_id"] = Sinks.geometry.apply(
        lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
    ).values

    # Ensure 'name' column exists for compatibility with get_distance_to_nearest_facility
    if "name" not in Sinks.columns:
        Sinks["name"] = Sinks.index.astype(str)  # or use another identifier column

    # Split by type
    Sinks_road = Sinks[Sinks["type"] == "road"].copy()
    Sinks_port = Sinks[Sinks["type"] == "port"].copy()
    Sinks_rail = Sinks[Sinks["type"] == "rail"].copy()

    # Create sinks dictionary for easy iteration
    sinks_dict = {
        "road": Sinks_road,
        "port": Sinks_port,
        "rail": Sinks_rail,
        "all": Sinks,
    }

    return sinks_dict


def calculate_accessibility_by_sink_type(
    df_population, sinks_dict, graph, inf_replacement=12
):
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
            df_population[f"nearest_{sink_type}"] = np.nan
            df_population[f"avg_{sink_type}"] = np.nan
            continue

        unique_pop_vertex_ids = df_population["vertex_id"].unique()
        unique_sink_vertex_ids = sink_df["vertex_id"].unique()

        # Calculate distance matrix
        distance_matrix = np.array(
            graph.distances(
                source=unique_pop_vertex_ids,
                target=unique_sink_vertex_ids,
                weights="fft",
            )
        )

        # NEAREST: minimum distance to any sink
        min_distances = np.min(distance_matrix, axis=1)
        vertex_to_nearest = dict(zip(unique_pop_vertex_ids, min_distances))
        df_population[f"nearest_{sink_type}"] = df_population["vertex_id"].map(
            vertex_to_nearest
        )

        # AVERAGE: mean distance to all sinks (with inf replacement)
        distance_matrix_for_avg = distance_matrix.copy()
        distance_matrix_for_avg[np.isinf(distance_matrix_for_avg)] = inf_replacement
        avg_distances = np.mean(distance_matrix_for_avg, axis=1)
        vertex_to_avg = dict(zip(unique_pop_vertex_ids, avg_distances))
        df_population[f"avg_{sink_type}"] = df_population["vertex_id"].map(
            vertex_to_avg
        )

        print(
            f"{sink_type}: {len(unique_sink_vertex_ids)} sinks | "
            f"Nearest avg: {np.mean(min_distances[~np.isinf(min_distances)]):.2f}h | "
            f"Avg to all: {np.mean(avg_distances):.2f}h"
        )

    return df_population


def update_column_names(df_agri, sinks_dict):

    # Rename columns to indicate baseline
    baseline_cols = {}
    for sink_type in sinks_dict.keys():
        baseline_cols[f"nearest_{sink_type}"] = f"baseline_nearest_{sink_type}"
        baseline_cols[f"avg_{sink_type}"] = f"baseline_avg_{sink_type}"

    df_agri = df_agri.rename(columns=baseline_cols)

    return df_agri


def flood_exposure_analysis_agriculture(
    base_network,
    df_worldpop,
    TheFolder,
    basins_data,
    graph,
    sinks_dict,
    country_iso3,
    Subregion,
):

    # ## Step 10: Identify exposed assets

    exposed_roads = base_network[base_network.exposed].reset_index(drop=True)

    tqdm.pandas(desc="get basin")
    exposed_roads["subregion"] = exposed_roads.progress_apply(
        lambda road_segment: _get_river_basin(road_segment, basins_data), axis=1
    )

    unique_scenarios = {}
    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        edges_to_remove = base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        unique_scenarios[subregion] = edges_to_remove

    # ###### -10-1: Save the uniqe_scenarios dictionary as a pickle file:

    filename_unique_scenarios = f"unique_scenarios_{country_iso3}_{Subregion}.pkl"
    file_path = os.path.join(TheFolder, filename_unique_scenarios)
    with open(file_path, "wb") as file:
        pickle.dump(unique_scenarios, file)

    # ### Step 11: Calculate the flood_statistics_per_scenario and save the results as a csv file

    unique_scenarios_Second = {}

    for subregion, subregion_exposed in tqdm(
        exposed_roads.groupby("subregion"),
        total=len(exposed_roads.groupby("subregion")),
    ):
        EdgeExposedList = subregion_exposed.id.values
        base_network.loc[
            base_network.id.isin(EdgeExposedList)
        ].index.values
        flooded_edges = base_network.loc[base_network.id.isin(EdgeExposedList)]

        non_empty_values = flooded_edges[flooded_edges["values"].apply(len) > 0][
            "values"
        ]

        Merged_Values_Column = [
            item for sublist in non_empty_values for item in sublist
        ]

        min_value = np.min(Merged_Values_Column) if Merged_Values_Column else np.nan
        mean_value = np.mean(Merged_Values_Column) if Merged_Values_Column else np.nan
        max_value = np.max(Merged_Values_Column) if Merged_Values_Column else np.nan

        unique_scenarios_Second[subregion] = {
            "min_value": min_value,
            "mean_value": mean_value,
            "max_value": max_value,
        }

    flood_statistics_per_scenario = []

    for subregion, values in unique_scenarios_Second.items():
        flood_statistics_per_scenario.append(
            {
                "basinID": subregion,
                "min water depth (m)": values["min_value"],
                "mean water depth (m)": values["mean_value"],
                "max water depth (m)": values["max_value"],
            }
        )

    Flood_Statistics_Per_Scenario = pd.DataFrame(flood_statistics_per_scenario)

    output_csv_file = os.path.join(
        TheFolder, f"{country_iso3}_flood_statistics_per_Basin_{Subregion}_scenario.csv"
    )
    Flood_Statistics_Per_Scenario.to_csv(output_csv_file, index=False)

    print(Flood_Statistics_Per_Scenario)

    # =============================================================================
    # UPDATED STEP 12: Run post-flood accessibility for ALL sink types
    # =============================================================================

    print("\n" + "=" * 60)
    print("FLOOD SCENARIO ACCESSIBILITY CALCULATIONS")
    print("=" * 60)

    save_new_results = {}
    sindex_pop = shapely.STRtree(df_worldpop.geometry)
    C = 1

    for BasinID in tqdm(unique_scenarios, desc="Processing basins"):
        save_new_results[BasinID] = {}

        try:
            edges_to_remove = unique_scenarios[BasinID]
            real_edges_to_remove = [
                x.index for x in graph.es if x["id"] in edges_to_remove
            ]

            # Create damaged graph
            damaged_graph = graph.copy()
            damaged_graph.delete_edges(real_edges_to_remove)

            # Get population in buffer zone around basin
            buffer_zone = (
                basins_data.loc[basins_data.HYBAS_ID == BasinID]
                .to_crs(3857)
                .buffer(50000)
                .to_crs(4326)
            )
            df_population = df_worldpop.iloc[
                sindex_pop.query(buffer_zone, predicate="intersects")[1]
            ].copy()
            df_population_backup = df_population.copy()
            InitialTotalPopulationPerBasin = df_population_backup["band_data"].sum()

            # Calculate post-flood accessibility for ALL sink types
            df_population = calculate_accessibility_by_sink_type(
                df_population, sinks_dict, damaged_graph
            )

            # Rename to post-flood columns
            postflood_cols = {}
            for sink_type in sinks_dict.keys():
                postflood_cols[f"nearest_{sink_type}"] = (
                    f"postflood_nearest_{sink_type}"
                )
                postflood_cols[f"avg_{sink_type}"] = f"postflood_avg_{sink_type}"
            df_population = df_population.rename(columns=postflood_cols)

            # Calculate deltas for each sink type and metric
            for sink_type in sinks_dict.keys():
                # Delta for nearest
                df_population[f"delta_nearest_{sink_type}"] = (
                    df_population[f"postflood_nearest_{sink_type}"]
                    - df_population[f"baseline_nearest_{sink_type}"]
                )
                # Delta for average
                df_population[f"delta_avg_{sink_type}"] = (
                    df_population[f"postflood_avg_{sink_type}"]
                    - df_population[f"baseline_avg_{sink_type}"]
                )

            # Calculate summary statistics per sink type
            results_by_sink_type = {}

            for sink_type in sinks_dict.keys():
                # Nearest sink analysis
                delta_nearest = df_population[f"delta_nearest_{sink_type}"]
                affected_nearest = df_population[
                    (~delta_nearest.isnull())
                    & (delta_nearest != 0)
                    & (~np.isinf(delta_nearest))
                ]
                lost_nearest = df_population[delta_nearest == np.inf]

                # Average sink analysis
                delta_avg = df_population[f"delta_avg_{sink_type}"]
                affected_avg = df_population[
                    (~delta_avg.isnull()) & (delta_avg != 0) & (~np.isinf(delta_avg))
                ]

                results_by_sink_type[sink_type] = {
                    # Nearest metrics
                    "affected_pop_nearest": affected_nearest["band_data"].sum(),
                    "lost_connections_nearest": lost_nearest["band_data"].sum(),
                    "mean_delta_nearest": delta_nearest[
                        ~np.isinf(delta_nearest)
                    ].mean(),
                    "max_delta_nearest": delta_nearest[~np.isinf(delta_nearest)].max(),
                    # Average metrics
                    "affected_pop_avg": affected_avg["band_data"].sum(),
                    "mean_delta_avg": delta_avg.mean(),
                    "max_delta_avg": delta_avg.max(),
                }

            # Store results
            save_new_results[BasinID] = {
                "df_population_backup": df_population_backup,
                "df_population": df_population,
                "real_edges_to_remove": [
                    x["osm_id"] for x in graph.es if x["id"] in edges_to_remove
                ],
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

    filename_nested_Dictionary = f"save_new_results_{country_iso3}_{Subregion}.pkl"
    file_path = os.path.join(TheFolder, filename_nested_Dictionary)
    with open(file_path, "wb") as file:
        pickle.dump(save_new_results, file)

    # =============================================================================
    # UPDATED STEP 13: Create summary DataFrames by sink type
    # =============================================================================

    print("\n" + "=" * 60)
    print("CREATING SUMMARY TABLES")
    print("=" * 60)

    summary_rows = []

    for BasinID, results in save_new_results.items():
        if results.get("status") == "Error":
            continue

        row = {
            "BasinID": BasinID,
            "InitialTotalPopulation": results["InitialTotalPopulation"],
        }

        for sink_type, metrics in results["results_by_sink_type"].items():
            for metric_name, value in metrics.items():
                row[f"{sink_type}_{metric_name}"] = value

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save summary CSV
    output_csv_file = os.path.join(
        TheFolder, f"{country_iso3}_accessibility_impact_summary_{Subregion}.csv"
    )
    summary_df.to_csv(output_csv_file, index=False)

    print(f"\nSummary saved to: {output_csv_file}")
    print(f"Columns: {list(summary_df.columns)}")
    print(summary_df.head())

    print("\nAnalysis completed successfully!")

def normalise_exposed_column(network: gpd.GeoDataFrame, col: str = "exposed") -> gpd.GeoDataFrame:
    """
    Ensure the exposed column is a clean boolean regardless of whether the
    source raster was binary (0/1 int) or continuous float depth values.
    NaN → False (no data = not exposed).
    """
    network[col] = network[col].fillna(0).astype(float) > 0
    return network

def _run_factory_analysis(
    *, config, base_network, nodes, basins_data, graph, country_iso3, Subregion
):
    tqdm.pandas()
    print("[factory] Loading data and computing baseline...")
    Factory_criticality_folder = (
        config.accessibility_analysis_path / "factory_criticality_results"
    )
    os.makedirs(Factory_criticality_folder, exist_ok=True)

    df_factories = read_factory_data(config)
    df_factories["vertex_id"] = nearest_network_nodes(df_factories, nodes)

    Sink = read_road_border_data(config)
    Sink = map_sinks_to_nearest_network_node(Sink, nodes)

    df_factories = get_average_access_time(df_factories, Sink, graph)

    print("[factory] Running flood scenario analysis...")
    flood_exposure_factory_accessibility(
        base_network, df_factories, Sink, Factory_criticality_folder,
        basins_data, graph, country_iso3, Subregion,
    )
    print("[factory] Done.")


def _run_agriculture_analysis(
    *, config, base_network, nodes, basins_data, graph, country_iso3, Subregion
):
    tqdm.pandas()
    print("[agriculture] Loading data and computing baseline...")
    Agriculture_criticality_folder = (
        config.accessibility_analysis_path / "allagri_criticality_results"
    )
    os.makedirs(Agriculture_criticality_folder, exist_ok=True)

    df_agri = read_agri_data(config)
    df_agri["vertex_id"] = nearest_network_nodes(df_agri, nodes)

    sinks_dict = load_sinks(config, nodes)
    df_agri = calculate_accessibility_by_sink_type(df_agri, sinks_dict, graph)
    df_agri = update_column_names(df_agri, sinks_dict)

    print("[agriculture] Running flood scenario analysis...")
    flood_exposure_analysis_agriculture(
        base_network, df_agri, Agriculture_criticality_folder,
        basins_data, graph, sinks_dict, country_iso3, Subregion,
    )
    print("[agriculture] Done.")


def _run_emergency_analysis(
    *, config, base_network, nodes, basins_data, df_settlements,
    graph, country_iso3, Subregion, label
):
    tqdm.pandas()
    print(f"[{label}] Loading data and computing baseline...")

    criticality_folder = (
        config.accessibility_analysis_path / f"{label}_criticality_results"
    )
    os.makedirs(criticality_folder, exist_ok=True)

    sink_df = load_and_map_sinks(config, nodes, label)
    accessibility_df = get_distance_to_nearest_facility(df_settlements, sink_df, graph)

    print(f"[{label}] Running flood scenario analysis...")
    flood_exposure_emergency_service_accessibility(
        accessibility_df, sink_df, criticality_folder,
        basins_data, base_network, graph, country_iso3, Subregion,
    )
    print(f"[{label}] Done.")


def main():
    """
    Run the end-to-end flood-scenario accessibility and criticality analysis.
    Processes industrial areas, agricultural areas, and emergency services
    (firefighters, hospitals, police) by mapping each location to the road
    network, computing baseline accessibility, and then evaluating how flooding
    disrupts access under basin-specific scenarios. All outputs are saved into
    sector-specific criticality folders for further analysis.

    All five sector analyses (including their data loading and baseline
    calculations) are run in parallel using ProcessPoolExecutor.

    WARNING: This script performs large-scale network routing and basin-level
    flood disruption simulations and may take SEVERAL HOURS to run depending on
    hardware and data size.

    """

    # ------------------------------------------------------------
    # Load configuration with file paths and settings
    # ------------------------------------------------------------
    config = NetworkConfig()

    country_iso3 = "SRB"
    Subregion = "basins"

    # Flood-depth thresholds for exposure classification
    Threshold = 0.1        # normal roads (m)
    Stru_Threshold = 0.5   # bridges (m)

    # ------------------------------------------------------------
    # 1. Shared setup: network, graph, basins, settlements
    #    (all five sectors depend on these, so kept serial)
    # ------------------------------------------------------------
    base_network = gpd.read_parquet(config.Path_RoadNetwork)

    print("Applying road-category elevation bias correction...")
    base_network = apply_road_elevation_correction(base_network, Threshold, Stru_Threshold)

    print("Normalising exposed column...")
    base_network = normalise_exposed_column(base_network)

    print("Creating graph representation of the road network...")
    nodes, graph = create_graph_for_spatial_matching(base_network)

    basins_data = gpd.read_file(config.basins_shapefile)

    df_settlements = read_population_data(config)
    df_settlements["vertex_id"] = nearest_network_nodes(df_settlements, nodes)

    # ------------------------------------------------------------
    # 2. Run all five sector analyses in parallel
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LAUNCHING PARALLEL SECTOR ANALYSES")
    print("=" * 60)

    shared = dict(
        config=config,
        base_network=base_network,
        nodes=nodes,
        basins_data=basins_data,
        graph=graph,
        country_iso3=country_iso3,
        Subregion=Subregion,
    )

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_run_factory_analysis, **shared): "factory",
            executor.submit(_run_agriculture_analysis, **shared): "agriculture",
            executor.submit(_run_emergency_analysis, **shared, df_settlements=df_settlements, label="firefighters"): "firefighters",
            executor.submit(_run_emergency_analysis, **shared, df_settlements=df_settlements, label="hospitals"): "hospitals",
            executor.submit(_run_emergency_analysis, **shared, df_settlements=df_settlements, label="police"): "police",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
                print(f"✓ [{label}] completed successfully.")
            except Exception as exc:
                print(f"✗ [{label}] raised an exception: {exc}")

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
