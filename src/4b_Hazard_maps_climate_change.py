import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.hazard_functions import (
    calculate_future_flood_return_periods,
    calculate_future_max_precipitation,
    clip_roads_by_country,
    load_country_boundaries,
    plot_future_flood_basins,
    plot_future_flood_roads,
    plot_precipitation_change,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

SCENARIOS = ["15", "20", "30", "40"]
SCENARIO_LABELS = {"15": "1.5", "20": "2.0", "30": "3.0", "40": "4.0"}


def main():
    config = NetworkConfig()

    # --- Load base data ---
    basins_csv = pd.read_csv(config.Path_flood_statistics_per_basin)
    all_basins = gpd.read_file(config.basins_shapefile)
    basins = gpd.GeoDataFrame(basins_csv.merge(all_basins, left_on="basinID", right_on="HYBAS_ID"))

    ds = xr.open_dataset(config.Path_flooding_climate_change)
    ds = ds.rio.write_crs("EPSG:3035", inplace=False)

    # Serbia only (no Kosovo) for analysis
    world, serbia, kosovo = load_country_boundaries(config.world_boundaries)
    roads_raw = gpd.read_file(config.data_path / "Deonice_Februar_2025.shp")
    _, kosovo_roads = clip_roads_by_country(roads_raw, serbia, kosovo)
    kosovo_roads_mercator = kosovo_roads.to_crs(3857)

    # Filter roads to Serbia only (no Kosovo)
    serbia_only = gpd.overlay(serbia, kosovo, how="difference").to_crs(roads_raw.crs)
    roads = gpd.sjoin(roads_raw, serbia_only[["geometry"]], how="inner", predicate="within").drop(
        columns=["index_right"], errors="ignore"
    )
    roads = roads.to_crs("EPSG:3035")
    basins_3035 = basins.to_crs("EPSG:3035")

    # --- Future flood return periods ---
    basins_3035, roads_rp = calculate_future_flood_return_periods(ds, basins_3035, roads, SCENARIOS)

    plot_future_flood_basins(
        basins_3035=basins_3035,
        scenarios=SCENARIOS,
        scenario_labels=SCENARIO_LABELS,
        figure_path=config.figure_path,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        output_crs=config.output_crs,
        dpi=300,
        show_figures=config.show_figures,
    )

    plot_future_flood_roads(
        roads_rp_3035=roads_rp,
        kosovo_roads_mercator=kosovo_roads_mercator,
        scenarios=SCENARIOS,
        scenario_labels=SCENARIO_LABELS,
        figure_path=config.figure_path,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        output_crs=config.output_crs,
        dpi=300,
        show_figures=config.show_figures,
    )

    # --- Future precipitation change ---
    results = calculate_future_max_precipitation(
        data_path=config.data_path,
        roads_path=config.data_path / "Deonice_Februar_2025.shp",
        world_path=config.world_boundaries,
        parquet_dir=config.hazard_exposure_parquet,
        gdb_path=config.hazard_exposure_gdb,
        output_crs=config.output_crs,
    )

    plot_precipitation_change(results, config.figure_path, dpi=300, show_figures=config.show_figures)


if __name__ == "__main__":
    main()
