"""
Step 3a - Baseline accessibility analysis.

Computes origin-destination / nearest-facility access times for factories,
agricultural areas and emergency services (fire, hospitals, police), prints
summary statistics and persists all results (Parquet + ArcGIS File Geodatabase,
in the project output CRS). Figures are produced in 3b_Plot_Baseline_Accessibility_Analysis.py.

All analysis functions live in utils/accessibility_functions.py so that this
script and the notebooks (notebooks/3a-3e) produce identical results.
"""

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so the shared utils package resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.accessibility_functions import (
    load_road_network,
    create_graph_for_spatial_matching,
    nearest_network_nodes,
    map_settlements_to_nodes_in_road_network,
    load_factory_data,
    load_border_crossings,
    load_agricultural_data,
    load_sinks,
    load_population_data,
    load_and_map_sinks,
    calculate_average_access_time,
    calculate_OD_matrix,
    get_distance_to_nearest_facility,
    save_accessibility_results,
    print_statistics,
    print_statistics_agriculture,
    print_statistics_emergency_accessibility,
)
from config.network_config import NetworkConfig


def main():
    """Run the full multi-sector baseline accessibility analysis and save results."""

    # Initialize configuration and resolve the output locations once
    config = NetworkConfig()
    parquet_dir = config.local_accessibility_parquet
    gdb_path = config.local_accessibility_gdb
    output_crs = config.output_crs

    # Load OSM network and build the graph (shared across all sectors)
    print("Loading OSM network...")
    base_network = load_road_network(config.Path_RoadNetwork)

    print("Creating graph representation of the road network...")
    nodes, graph = create_graph_for_spatial_matching(base_network)

    # =============================================================================
    # 1. Accessibility calculations for factories
    # =============================================================================
    print("Loading factory location data...")
    df_factories = load_factory_data(config.Path_FactoryFile)

    print("Mapping factories to nearest network nodes...")
    df_factories["vertex_id"] = nearest_network_nodes(df_factories, nodes)

    print("Loading border crossing locations...")
    border_crossings = load_border_crossings(config.path_to_Borders, nodes)

    df_factories["avg_access_time"], OD_baseline = calculate_average_access_time(
        df_factories, border_crossings, graph
    )
    print(f"Baseline average access time: {np.mean(OD_baseline):.2f} hours")

    if config.print_statistics:
        print_statistics(df_factories, border_crossings, OD_baseline)

    save_accessibility_results(
        df_factories, border_crossings, "factories", parquet_dir, gdb_path, output_crs
    )
    print("\n--- Accessibility analysis for factories complete. ---")

    # =============================================================================
    # 2. Accessibility calculations for agricultural areas
    # =============================================================================
    print("\n--- Starting accessibility analysis for agricultural areas ---")

    print("Loading location data of agricultural areas...")
    df_agri = load_agricultural_data(config.Path_agriculture_input)

    print("Mapping agricultural locations to nearest network nodes...")
    df_agri["vertex_id"] = nearest_network_nodes(df_agri, nodes)

    print("Loading border crossings, ports and rail cargo terminals...")
    Sinks_road, Sinks_port, Sinks_rail, all_sinks = load_sinks(config.path_to_Sinks, nodes)

    print("Calculating OD matrices for agricultural areas...")
    df_agri = calculate_OD_matrix(
        df_agri, graph, Sinks_road, Sinks_port, Sinks_rail, all_sinks
    )

    if config.print_statistics:
        print_statistics_agriculture(df_agri)

    save_accessibility_results(
        df_agri, all_sinks, "agriculture", parquet_dir, gdb_path, output_crs
    )
    print("\n--- Accessibility analysis for agricultural areas complete. ---")

    # =============================================================================
    # 3. Settlements -> emergency services (fire, hospitals, police)
    # =============================================================================
    print("\n--- Starting accessibility analysis for emergency services ---")

    print("Loading population data...")
    df_settlements = load_population_data(config.Path_SettlementData_Excel)

    print("Mapping each settlement to the closest node in the road network...")
    df_settlements = map_settlements_to_nodes_in_road_network(df_settlements, nodes)

    # 3a. Firefighters
    print("Loading firefighter locations...")
    sink_firefighters = load_and_map_sinks(config.firefighters, nodes, "firefighters")
    print("Calculating distance to the nearest fire station for each settlement...")
    acessibility_firefighters = get_distance_to_nearest_facility(
        df_settlements, sink_firefighters, graph
    )
    save_accessibility_results(
        acessibility_firefighters,
        sink_firefighters,
        "firefighters",
        parquet_dir,
        gdb_path,
        output_crs,
    )
    print("\n--- Accessibility analysis for firefighters complete. ---")

    # 3b. Hospitals
    print("Loading hospital locations...")
    sink_hospitals = load_and_map_sinks(config.hospitals, nodes, "hospitals")
    print("Calculating distance to the nearest hospital...")
    acessibility_hospitals = get_distance_to_nearest_facility(
        df_settlements, sink_hospitals, graph
    )
    save_accessibility_results(
        acessibility_hospitals,
        sink_hospitals,
        "hospitals",
        parquet_dir,
        gdb_path,
        output_crs,
    )
    print("\n--- Accessibility analysis for hospitals complete. ---")

    # 3c. Police stations
    print("Loading location data of police stations...")
    police_stations = load_and_map_sinks(config.police_stations, nodes, "police")
    print("Calculating distance to the nearest police station...")
    acessibility_police_stations = get_distance_to_nearest_facility(
        df_settlements, police_stations, graph
    )
    save_accessibility_results(
        acessibility_police_stations,
        police_stations,
        "police",
        parquet_dir,
        gdb_path,
        output_crs,
    )
    print("\n--- Accessibility analysis for police stations complete. ---\n")

    # Combined emergency-services summary
    if config.print_statistics:
        print_statistics_emergency_accessibility(
            acessibility_firefighters,
            sink_firefighters,
            acessibility_hospitals,
            sink_hospitals,
            acessibility_police_stations,
            police_stations,
        )


if __name__ == "__main__":
    main()
