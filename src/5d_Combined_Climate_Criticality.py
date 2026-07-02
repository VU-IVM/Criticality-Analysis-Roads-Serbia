"""
5d — Combined Climate Criticality.

Final step of the workflow: assemble hazard, national-disruption and
local-accessibility results into a single per-road climate-criticality index.

Pipeline
--------
1. load_and_preprocess  — join hazard exposure (4a/5a), criticality results (2),
   accessibility impacts (5c) and climate-change layers (4b) onto the network.
2. clean_data           — drop invalid / duplicate / negligibly short segments.
3. prepare_metrics      — standardise column names and derive hazard indicators.
4. deduplicate_by_section — collapse to one representative row per ``oznaka_deo``.
5. score_climate_criticality — log -> normalise -> quintile -> convex score,
   sum into sub-indices H / T / A and combine multiplicatively into the index.
6. plots, statistics, multi-sheet Excel and geospatial export.

Two behaviours are read from NetworkConfig:
  * ``climate_hazards_only`` — climate-only vs all hazards for the H sub-index.
  * ``normalize_subindices`` — CC_norm (normalised sub-indices) vs CC_raw.
ArcGIS outputs are feature classes in a results File GDB (reprojected to
EPSG:6316) plus matching .lyrx files; no GeoPackage is written.
"""

import sys
import warnings

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.criticality_functions import (
    clean_data,
    deduplicate_by_section,
    export_climate_criticality_excel,
    load_and_preprocess_criticality_data,
    plot_climate_criticality_components,
    plot_climate_criticality_components_4panel,
    plot_combined_climate_criticality,
    prepare_metrics,
    print_climate_criticality_statistics,
    save_climate_criticality_geospatial,
    score_climate_criticality,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    """Run the combined climate-criticality workflow."""
    config = NetworkConfig()

    # 1. Load + spatially join every per-topic metric onto the network
    gdf = load_and_preprocess_criticality_data(
        hazard_exposure_path=config.Path_main_network_hazard_exposure,
        criticality_results_path=config.Path_criticality_results,
        hospital_impacts_path=config.Path_hospital_impacts,
        factory_impacts_path=config.Path_factory_impacts,
        police_impacts_path=config.Path_police_impacts,
        fire_impacts_path=config.Path_fire_fighter_impacts,
        border_impacts_path=config.Path_road_border_impacts,
        port_impacts_path=config.Path_port_impacts,
        railway_impacts_path=config.Path_railway_impacts,
        future_floods_change_rp_path=config.Path_future_floods_change_RP,
        future_rainfall_change_path=config.Path_precipitation_change_rcp_8_5_far_future,
    )

    # 2. Clean, 3. standardise metric columns, 4. deduplicate per road section
    gdf = clean_data(gdf)
    gdf = prepare_metrics(gdf)
    gdf = deduplicate_by_section(gdf)

    # 5. Score the combined climate-criticality index
    gdf = score_climate_criticality(
        gdf,
        climate_hazards_only=config.climate_hazards_only,
        normalize_subindices=config.normalize_subindices,
    )

    # 6. Maps (sub-indices + combined); ArcGIS layers -> results GDB + .lyrx
    plot_climate_criticality_components(
        gdf,
        config.figure_path,
        config.results_gdb,
        config.lyrx_results,
        show_figures=config.show_figures,
    )
    plot_climate_criticality_components_4panel(
        gdf,
        config.figure_path,
        show_figures=config.show_figures,
    )
    plot_combined_climate_criticality(
        gdf,
        config.figure_path,
        config.results_gdb,
        config.lyrx_results,
        show_figures=config.show_figures,
    )

    # Statistics
    if config.print_statistics:
        print_climate_criticality_statistics(gdf)

    # Excel (formatted, multi-sheet) + geospatial outputs
    export_climate_criticality_excel(
        gdf,
        config.Path_climate_criticality_results,
        climate_hazards_only=config.climate_hazards_only,
        normalize_subindices=config.normalize_subindices,
    )
    save_climate_criticality_geospatial(
        gdf,
        parquet_path=config.intermediate_results_path
        / config.Path_climate_criticality_results.with_suffix(".parquet").name,
    )


if __name__ == "__main__":
    main()
