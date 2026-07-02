"""
5a — Main Network Hazard Criticality.

Combines the single-point-of-failure criticality results with five hazard
overlays (floods with DEM-based depth bias correction, snow drift, landslides,
wildfires, pavement heat) into one main-network hazard exposure dataset.
Outputs are written to parquet + ArcGIS File Geodatabase + Excel in EPSG:6316.
"""

import sys
import warnings

import geopandas as gpd

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.criticality_functions import (
    apply_bias_correction,
    build_hazard_count,
    calculate_flood_exposure,
    calculate_heat_exposure,
    calculate_vhl_landslides,
    calculate_vhl_snowdrift,
    calculate_wildfire_exposure,
    combine_hazard_exposure,
    compute_bias_confidence_intervals,
    compute_bias_statistics,
    extract_profiles,
    fill_missing_categories,
    flag_future_precipitation,
    load_vertical_coordinates,
    merge_baseline_roads,
    plot_all_hazard_comparisons,
    plot_elevation_bias,
    plot_elevation_profiles,
    plot_flood_exposure_correction,
    plot_hazard_count_map,
    plot_vhl_flooded_map,
    print_hazard_analysis_summary,
    save_criticality_vector,
    summarize_hazard_counts,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    """Run the full main-network multi-hazard exposure analysis."""
    config = NetworkConfig()

    # --- Load criticality results and merge with baseline road sections ---
    gdf_results = gpd.read_parquet(config.Path_criticality_results)
    gdf_results = merge_baseline_roads(
        gdf_results, config.Path_baseline_road_network, config.world_boundaries
    )

    # --- Flood exposure (DamageScanner) ---
    exposed_roads, country_plot = calculate_flood_exposure(
        gdf_results, config.flood_map_RP100, config.world_boundaries
    )
    exposed_roads = fill_missing_categories(exposed_roads)

    # --- Elevation bias analysis (road Z-profiles vs DEM) ---
    vertical_coordinates, vertical_coordinates_4326 = load_vertical_coordinates(
        config.Path_vertical_coordinates
    )
    vertical_coordinates = extract_profiles(
        vertical_coordinates, vertical_coordinates_4326, config.dem_serbia
    )
    plot_elevation_profiles(
        vertical_coordinates, config.figure_path, show_figures=config.show_figures
    )

    clean = compute_bias_statistics(vertical_coordinates)
    ci_df = compute_bias_confidence_intervals(clean)
    plot_elevation_bias(ci_df, config.figure_path, show_figures=config.show_figures)

    # --- Apply bias correction to flood depths ---
    exposed_roads = apply_bias_correction(exposed_roads)
    plot_flood_exposure_correction(
        exposed_roads,
        country_plot,
        config.figure_path,
        show_figures=config.show_figures,
    )

    # --- Flood-exposed roads with VHL ---
    corrected_roads = exposed_roads.loc[exposed_roads.corrected_max_depth > 0]
    gdf_vhl_flooded = gdf_results.merge(
        corrected_roads.loc[corrected_roads.exposed][
            ["coverage", "values", "corrected_max_depth"]
        ],
        left_index=True,
        right_index=True,
    )
    plot_vhl_flooded_map(
        gdf_vhl_flooded, config.figure_path, show_figures=config.show_figures
    )

    # --- Other hazard overlays ---
    gdf_vhl_snowdrift = calculate_vhl_snowdrift(
        gdf_results, config.Path_snow_drift_data
    )
    gdf_vhl_landslides = calculate_vhl_landslides(
        gdf_results, config.Path_landslide_data
    )
    gdf_vhl_wildfire = calculate_wildfire_exposure(
        gdf_results, config.wildfire_risk, config.world_boundaries
    )
    gdf_vhl_heat = calculate_heat_exposure(
        gdf_results, config.Future_pavement_temperatures
    )

    # --- Combine and save (parquet + GDB + Excel, EPSG:6316) ---
    gdf_hazards = combine_hazard_exposure(
        gdf_results,
        gdf_vhl_flooded,
        gdf_vhl_snowdrift,
        gdf_vhl_landslides,
        gdf_vhl_wildfire,
        gdf_vhl_heat,
    )
    save_criticality_vector(
        gdf_hazards,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        layer_name="main_network_hazard_exposure",
        output_crs=config.output_crs,
    )

    # --- Hazard comparison figures (VHL / PHL / TKL) ---
    plot_all_hazard_comparisons(
        gdf_vhl_flooded,
        gdf_vhl_snowdrift,
        gdf_vhl_landslides,
        gdf_vhl_wildfire,
        country_plot,
        config.figure_path,
        show_figures=config.show_figures,
    )

    # --- Multi-hazard exposure count (binary yes/no across 6 hazards) ---
    hazard_specs = [
        ("hazard_flood", "Flood", gdf_results.index.isin(gdf_vhl_flooded.index)),
        (
            "hazard_snow_drift",
            "Snow drift",
            gdf_results.index.isin(gdf_vhl_snowdrift.index),
        ),
        (
            "hazard_landslide",
            "Landslide",
            gdf_results.index.isin(gdf_vhl_landslides.index),
        ),
        ("hazard_wildfire", "Wildfire", gdf_results.index.isin(gdf_vhl_wildfire.index)),
        (
            "hazard_heat",
            "Heat (pavement > 60°C)",
            (gdf_vhl_heat["max_pavement_temp"] > 60)
            .reindex(gdf_results.index, fill_value=False)
            .to_numpy(),
        ),
        (
            "hazard_future_precipitation",
            "Future precipitation (≥ 10% change)",
            flag_future_precipitation(
                gdf_results,
                config.Path_precipitation_change_rcp_8_5_far_future,
                threshold=10.0,
            ),
        ),
    ]
    gdf_hazard_count = build_hazard_count(gdf_results, hazard_specs)
    plot_hazard_count_map(
        gdf_hazard_count,
        config.figure_path,
        n_hazards=len(hazard_specs),
        show_figures=config.show_figures,
    )

    # --- Summary statistics ---
    if config.print_statistics:
        print_hazard_analysis_summary(
            gdf_results,
            gdf_vhl_flooded,
            gdf_vhl_snowdrift,
            gdf_vhl_landslides,
            gdf_vhl_wildfire,
        )
        summarize_hazard_counts(gdf_hazard_count, hazard_specs)


if __name__ == "__main__":
    main()
