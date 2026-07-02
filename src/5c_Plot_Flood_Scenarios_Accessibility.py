"""
5c — Plot Flood Scenarios Accessibility.

Visualises the basin-scenario accessibility results computed by script 5b:
increased travel times from settlements to emergency services (hospitals,
police, fire stations), from industrial areas to border crossings, and from
agricultural areas to road borders / ports / rail terminals. The exposed-edge
impact layers are saved to parquet + ArcGIS File Geodatabase + Excel in
EPSG:6316.
"""

import sys
import warnings

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.criticality_functions import (
    calculate_agri_criticality,
    calculate_service_criticality,
    load_base_network,
    load_basins,
    plot_agri_criticality_3x1,
    plot_basin_water_depths,
    plot_criticality_2x2,
    plot_service_criticality,
    print_delay_category_counts,
    save_impact_layers,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    """Run the flood-scenario accessibility visualisation workflow."""
    config = NetworkConfig()

    # --- Basin water depths ---
    basins = load_basins(
        config.Path_flood_statistics_per_basin_scenarios, config.basins_shapefile
    )
    plot_basin_water_depths(basins, config.figure_path, show_figures=config.show_figures)

    # --- Base network (giant component, EPSG:3857) ---
    base_network = load_base_network(config.Path_RoadNetwork)

    # --- Emergency services and factories: criticality per service ---
    services = {
        "hospital_impacts": (
            config.accessibility_analysis_path / "healthcare_criticality_results" / "save_new_results_SRB_basins.pkl",
            "hospital_criticality.png",
        ),
        "police_impacts": (
            config.accessibility_analysis_path / "police_criticality_results" / "save_new_results_SRB_basins.pkl",
            "police_criticality.png",
        ),
        "fire_impacts": (
            config.accessibility_analysis_path / "fire_criticality_results" / "save_new_results_SRB_basins.pkl",
            "fire_criticality.png",
        ),
        "factory_impacts": (
            config.accessibility_analysis_path / "factory_criticality_results" / "save_new_results_SRB_basins.pkl",
            "factory_criticality.png",
        ),
    }

    service_edges = {}
    for layer_name, (results_path, figure_name) in services.items():
        print(f"\n=== {layer_name} ===")
        exposed_edges = calculate_service_criticality(results_path, base_network)
        print_delay_category_counts(exposed_edges, layer_name.replace("_impacts", ""))
        plot_service_criticality(exposed_edges, base_network, config.figure_path, figure_name, show_figures=config.show_figures)
        service_edges[layer_name] = exposed_edges

    # --- Agriculture: average and nearest-sink criticality per sink type ---
    agri_results_path = (
        config.accessibility_analysis_path / "allagri_criticality_results" / "save_new_results_SRB_basins.pkl"
    )

    agri_avg = calculate_agri_criticality(agri_results_path, base_network, delta_prefix="delta_avg")
    for sink_type, edges in agri_avg.items():
        print_delay_category_counts(edges, f"agriculture {sink_type} (average sink)")
    plot_agri_criticality_3x1(
        agri_avg, base_network, config.figure_path,
        file_name="SRB_agri_criticality_avg_3x1.png",
        legend_title="Average Increased Travel Time",
        show_figures=config.show_figures,
    )

    agri_nearest = calculate_agri_criticality(agri_results_path, base_network, delta_prefix="delta_nearest")
    for sink_type, edges in agri_nearest.items():
        print_delay_category_counts(edges, f"agriculture {sink_type} (nearest sink)")
    plot_agri_criticality_3x1(
        agri_nearest, base_network, config.figure_path,
        file_name="SRB_agri_criticality_nearest_3x1.png",
        legend_title="Increased Travel Time To Nearest",
        show_figures=config.show_figures,
    )

    # --- Combined 2x2 figure (hospitals, factories, police, fire) ---
    plot_criticality_2x2(
        service_edges["hospital_impacts"], service_edges["factory_impacts"],
        service_edges["police_impacts"], service_edges["fire_impacts"],
        base_network, config.figure_path,
        show_figures=config.show_figures,
    )

    # --- Save all impact layers (parquet + GDB + Excel, EPSG:6316) ---
    impact_layers = {
        **service_edges,
        "road_impacts": agri_nearest["road"],
        "rail_impacts": agri_nearest["rail"],
        "port_impacts": agri_nearest["port"],
    }
    save_impact_layers(
        impact_layers,
        parquet_dir=config.local_accessibility_parquet,
        gdb_path=config.local_accessibility_gdb,
        output_crs=config.output_crs,
    )


if __name__ == "__main__":
    main()
