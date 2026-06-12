import sys
import warnings

import geopandas as gpd

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.hazard_functions import (
    apply_urban_heat_island,
    assign_and_plot_road_temperatures,
    clip_roads_by_country,
    load_country_boundaries,
    mask_to_serbia,
    plot_pavement_temperature,
    plot_pavement_temperature_roads_AB,
    plot_temperature_difference,
    read_tif,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    config = NetworkConfig()

    # --- Load Serbia outline ---
    world = gpd.read_file(config.world_boundaries)
    country_plot = world.loc[world.SOV_A3 == "SRB"]
    serbia_3857 = country_plot.to_crs(3857)

    # --- Kosovo road network (grey on the A/B road panel) ---
    _, serbia, kosovo = load_country_boundaries(config.world_boundaries)
    baseline_roads = gpd.read_file(config.data_path / "Deonice_Februar_2025.shp")
    _, kosovo_roads = clip_roads_by_country(baseline_roads, serbia, kosovo)
    kosovo_roads_mercator = kosovo_roads.to_crs(3857)

    # --- Temperature difference plot (historic vs current air temp) ---
    plot_temperature_difference(
        config.historic_temperature, config.current_temperature, serbia_3857,
        config.temperature_figures_folder, dpi=300,
    )

    # --- Current pavement temperature ---
    data, bounds, crs = read_tif(config.current_max_pavement_temperature)
    data_serbia = mask_to_serbia(data, bounds, crs, serbia_3857)

    import numpy as np
    valid = data_serbia[~np.isnan(data_serbia)]
    print(f"Pavement temp range: {valid.min():.2f} – {valid.max():.2f} °C")

    plot_pavement_temperature(data, "Pavement Temperature", config.temperature_figures_folder, bounds, crs, serbia_3857)
    plot_pavement_temperature(data + 2, "Pavement Temperature (+ 2°C)", config.temperature_figures_folder, bounds, crs, serbia_3857)

    # --- Urban heat island adjustment ---
    data_uhi = apply_urban_heat_island(
        bounds=bounds,
        degree_of_urbanization_path=config.degree_of_urbanization,
        current_max_pavement_temperature_path=config.current_max_pavement_temperature,
        output_tif_path=config.Future_pavement_temperatures,
    )
    plot_pavement_temperature(
        data_uhi,
        "Pavement Temperature with urban heat islands under climate change",
        config.temperature_figures_folder,
        bounds, crs, serbia_3857,
    )

    # --- Assign temperatures to road segments and save ---
    assign_and_plot_road_temperatures(
        raster_data=data,
        raster_bounds=bounds,
        raster_crs=crs,
        input_parquet=config.Path_processed_road_network,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        output_folder=config.temperature_figures_folder,
        title="Road Network — Pavement Temperature",
        output_crs=config.output_crs,
    )

    assign_and_plot_road_temperatures(
        raster_data=data_uhi,
        raster_bounds=bounds,
        raster_crs=crs,
        input_parquet=config.Path_processed_road_network,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        output_folder=config.temperature_figures_folder,
        title="Road Network — Pavement Temperature with UHI",
        output_crs=config.output_crs,
    )

    # --- Combined A/B figure: UHI raster + road network at UHI temperatures ---
    plot_pavement_temperature_roads_AB(
        raster_data=data_uhi,
        raster_bounds=bounds,
        raster_crs=crs,
        input_parquet=config.Path_processed_road_network,
        serbia_3857=serbia_3857,
        kosovo_roads_mercator=kosovo_roads_mercator,
        output_folder=config.temperature_figures_folder,
        dpi=300,
    )


if __name__ == "__main__":
    main()
