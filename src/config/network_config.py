
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NetworkConfig:
    """Configuration for accesibility analysis and visualization."""

    # folder paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    data_path = BASE_DIR / "input_files"
    intermediate_results_path = BASE_DIR / 'intermediate_results'
    accessibility_analysis_path = BASE_DIR / 'accessibility_analysis'
    figure_path = BASE_DIR / "figures"
    climate_change_precipitation_folder = data_path / "Climate Change Precipitation\results"

    ##########################################
    # Input files
    ##########################################
    Path_RoadNetwork = data_path / "base_network_SRB_basins.parquet"
    osm_path = data_path / "SRB.osm.pbf"
    AADT_data = data_path / "PGDS_2024.shp"
    Network_original_full_AADT = data_path / "roads_serbia_original_full_AADT.parquet"
    Original_road_network = roads_path = data_path / 'DeoniceRSDP-Jul2025..shp'
    Path_FactoryFile = data_path / "2_Factory_Company_geolocations.xlsx"
    path_to_Borders = data_path / "Borders_geocoded.xlsx"
    Path_AgriFile = data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    path_to_Sinks = data_path / "Borders_Ports_Rail_geocoded.xlsx"
    Path_SettlementData_Excel = data_path / "population_NEW_settlement_geocoded.xlsx"
    firefighters = data_path / "6_Firefighters_geocoded.xlsx"
    hospitals = data_path / "4_Hospitals_healthcenters_geocoded.xlsx"
    police_stations = data_path / "6_Police_geocoded.xlsx"
    basins_shapefile = data_path / "hybas_eu_lev09_v1c.shp"
    flood_map_RP100 = data_path / "Europe_RP100_filled_depth.tif"
    world_boundaries = data_path / "ne_10m_admin_0_countries.shp"
    Path_agriculture_input = data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    Path_snow_drift_data = data_path / "snezni_nanosi_studije.shp"
    Path_landslide_data = data_path / "Nestabilne_pojave.shp"
    Path_flooding_climate_change = data_path / "disEnsemble_highExtremes.nc"
    

    #Paths for intermediate results
    Path_processed_road_network = intermediate_results_path / "PERS_directed_final.parquet"
    Path_processed_road_network_shp = intermediate_results_path / "PERS_directed_final.shp"
    Path_firefighter_accessibilty = intermediate_results_path / 'firefighter_settle_results.parquet'
    Path_firefighters_sink = intermediate_results_path / 'firefighters.parquet'
    Path_hospital_accessibilty = intermediate_results_path / 'hospital_accessibility_results.parquet'
    Path_hospital_sink = intermediate_results_path / 'hospitals.parquet'
    Path_police_accessibilty = intermediate_results_path / 'police_accessibility_results.parquet'
    Path_police_sink = intermediate_results_path / 'police.parquet'
    Path_factory_accessibility = intermediate_results_path / 'factory_accessibility.parquet'
    Path_factory_sink = intermediate_results_path / 'factories_sinks.parquet'
    Path_agriculture_accessibility = intermediate_results_path / 'agriculture_accessibility.parquet'
    Path_agriculture_sink = intermediate_results_path / 'agriculture_sinks.parquet'
    Path_flood_statistics_per_basin = intermediate_results_path / "SRB_flood_statistics_per_Basin_basins_scenario.csv"
    Path_future_floods_change_RP = intermediate_results_path / "Future Floods change in RP.parquet"
    Path_future_flooding_roads = intermediate_results_path / "Future Floods change in RP experienced by roads.parquet"

    #Path for figures that are created by the scripts
    Path_factory_acces_avg = figure_path / 'factory_access_avg.png'
    Path_baseline_accessibility_fire_stations = figure_path / 'baseline_accessibility_fire_stations.png'
    Path_baseline_accessibility_hospitals = figure_path / 'baseline_accessibility_hospitals.png'
    Path_baseline_accessibility_police_stations = figure_path / 'baseline_accessibility_police_stations.png'
    Path_agriculture_access_by_type = figure_path / 'agriculture_access_by_type.png'
    Path_firefighter_access_map = figure_path / 'firefighter_access.png'
    Path_hospital_access_map = figure_path / 'hospital_access.png'
    Path_police_station_access_map = figure_path / 'police_station_access.png'
    

    #Flags to activate/ deactivate outputs
    show_figures = True #Flag to set whether plots will be shown in a pop up window or not
    print_statistics = True #prints summary of the analysis to the console