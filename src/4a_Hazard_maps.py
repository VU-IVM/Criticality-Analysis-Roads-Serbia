import sys
import warnings

import geopandas as gpd

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.hazard_functions import (
    assign_wildfire_risk_to_roads,
    clip_roads_by_country,
    load_and_clip_flood_raster,
    load_country_boundaries,
    plot_flood_depth_map,
    plot_flood_depth_roads,
    plot_landslide_susceptibility_map,
    plot_landslides_map,
    plot_snowdrift_map,
    plot_wildfire_raster_map,
    plot_wildfire_roads_AB,
)
from utils.utils import assign_flood_depth_to_roads, read_road_network

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    config = NetworkConfig()

    # --- Country boundaries and road layers ---
    world, serbia, kosovo = load_country_boundaries(config.world_boundaries)
    baseline_roads = gpd.read_file(config.AADT_data.parent / "Deonice_Februar_2025.shp")
    serbia_roads, kosovo_roads = clip_roads_by_country(baseline_roads, serbia, kosovo)
    serbia_roads_mercator = serbia_roads.to_crs(3857)
    kosovo_roads_mercator = kosovo_roads.to_crs(3857)

    # --- Flood depth map ---
    hazard_country = load_and_clip_flood_raster(config.flood_map_RP100, serbia)
    plot_flood_depth_map(hazard_country, config.figure_path, dpi=600, show_figures=config.show_figures)

    # --- Flood depth on roads ---
    roads = read_road_network(config.Path_processed_road_network)
    roads = assign_flood_depth_to_roads(
        roads=roads,
        raster_data=hazard_country.band_data.values.squeeze(),
        raster_bounds=hazard_country.rio.bounds(),
        raster_crs=hazard_country.rio.crs,
    )
    roads["flood_class"] = roads["flood_class"].apply(
        lambda x: "No flooding" if x == "No flooding" else "Flooded"
    )
    plot_flood_depth_roads(
        roads=roads,
        figure_path=config.figure_path,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        arcgis_gpgk=config.arcgis_gpgk,
        arcgis_results=config.arcgis_results,
        output_crs=config.output_crs,
        dpi=600,
        show_figures=config.show_figures,
    )

    # --- Snow drift map ---
    plot_snowdrift_map(config.Path_snow_drift_data, serbia_roads_mercator, kosovo_roads_mercator, config.figure_path, dpi=600, show_figures=config.show_figures)

    # --- Landslides map (filtered to Klizište) ---
    plot_landslides_map(config.Path_landslide_data, serbia_roads_mercator, kosovo_roads_mercator, config.figure_path, dpi=600, show_figures=config.show_figures)

    # --- Wildfire raster and roads ---
    plot_wildfire_raster_map(config.wildfire_risk, serbia_roads_mercator, kosovo_roads_mercator, config.figure_path, dpi=600, show_figures=config.show_figures)
    roads_with_risk = assign_wildfire_risk_to_roads(config.wildfire_risk, baseline_roads)
    plot_wildfire_roads_AB(
        wildfire_path=config.wildfire_risk,
        roads_with_risk=roads_with_risk,
        serbia_roads_mercator=serbia_roads_mercator,
        kosovo_roads_mercator=kosovo_roads_mercator,
        figure_path=config.figure_path,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        arcgis_gpgk=config.arcgis_gpgk,
        arcgis_results=config.arcgis_results,
        output_crs=config.output_crs,
        dpi=600,
        show_figures=config.show_figures,
    )

    # --- Landslide susceptibility map ---
    plot_landslide_susceptibility_map(
        config.landslide_susceptibility, serbia_roads_mercator, kosovo_roads_mercator, config.figure_path, dpi=600,
        show_figures=config.show_figures,
    )


if __name__ == "__main__":
    main()
