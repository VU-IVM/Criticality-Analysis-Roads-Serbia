from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetworkConfig:
    """Configuration for accesibility analysis and visualization."""

    # folder paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    data_path = BASE_DIR / "input_files"
    intermediate_results_path = BASE_DIR / "intermediate_results"
    # Intermediate results are stored side by side as Parquet and ArcGIS File
    # Geodatabase, each under a dedicated subfolder. Step-3 (local accessibility)
    # outputs go to the 'local_accessibility' subfolder of each.
    parquet_path = intermediate_results_path / "parquet"
    database_path = intermediate_results_path / "database"
    local_accessibility_parquet = parquet_path / "local_accessibility"
    local_accessibility_database = database_path / "local_accessibility"
    local_accessibility_gdb = local_accessibility_database / "local_accessibility.gdb"
    # Step-4 (hazard exposure) outputs go to the 'hazard_exposure' subfolder of each.
    hazard_exposure_parquet = parquet_path / "hazard_exposure"
    hazard_exposure_database = database_path / "hazard_exposure"
    hazard_exposure_gdb = hazard_exposure_database / "hazard_exposure.gdb"
    results_path = BASE_DIR / "results"
    arcgis_results = results_path / "ArcGIS layers"
    arcgis_gpgk = arcgis_results / "Geopackages"

    accessibility_analysis_path = BASE_DIR / "accessibility_analysis"
    figure_path = BASE_DIR / "figures"
    climate_change_precipitation_folder = (
        data_path / "Climate Change Precipitation" / "results"
    )
    temperature_input_folder = (data_path / "Temperatures climate change")
    temperature_figures_folder = BASE_DIR / "figures" / "temperature"

    ##########################################
    # Input files
    ##########################################
    # TODO rename var path files to be more consistent and clear
    Path_RoadNetwork = data_path / "base_network_SRB_basins.parquet"
    osm_path = data_path / "SRB.osm.pbf"
    AADT_data = data_path / "PGDS_2024.shp"
    Network_PERS_Corr = data_path / "DeoniceRSDP-Jul2025_corrected_topology.parquet"
    Original_road_network = roads_path = data_path / "DeoniceRSDP-Jul2025..shp"
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
    Path_agriculture_input = (
        data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    )
    Path_snow_drift_data = data_path / "snezni_nanosi_studije.shp"
    Path_landslide_data = data_path / "Nestabilne_pojave.shp"
    Path_flooding_climate_change = data_path / "disEnsemble_highExtremes.nc"
    landslide_susceptibility = data_path / "landslide_susceptibility.tif"
    Path_flood_statistics_per_basin = data_path / "SRB_flood_statistics_per_Basin_basins_scenario.csv"
    wildfire_risk = data_path / "wildfire risk" / "stepen ugrozenosti od pozara Srbijasume.tif"
    historic_temperature = temperature_input_folder / "TX7D_1961-1990.tif"
    current_temperature = temperature_input_folder / "TX7D_1991-2020.tif"
    current_max_pavement_temperature = temperature_input_folder / "TPAV_2021-2025.tif"
    degree_of_urbanization = data_path / "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0_R4_C20.tif"
    # baseline road network sections and elevation data (used in 5a)
    Path_baseline_road_network = data_path / "Deonice_Februar_2025.shp"
    dem_serbia = data_path / "dem_serbia.tif"
    Path_vertical_coordinates = (
        data_path / "Vertical coordinates" / "RSDP_Feb_2026" / "RSDP_Feb_2026"
        / "Deonice" / "RSDP_Deonice_Feb_2026.shp"
    )

    
    #####################################################
    # Paths for intermediate results
    #####################################################

    # processed road network (produced in 1b)
    Path_processed_road_network = intermediate_results_path / "PERS_directed_final.parquet"
    Path_processed_road_network_shp = intermediate_results_path / "PERS_directed_final.shp"

    #network criticality (single point of failure analysis) (produced in 2). Also saved to results folder.
    Path_criticality_results = intermediate_results_path / "criticality_results.parquet"

    # baseline accessibility results (Parquet; matching .gdb layers live in
    # local_accessibility_gdb with the same stem as the parquet file)
    Path_firefighter_accessibilty = local_accessibility_parquet / 'firefighter_accessibility_results.parquet'
    Path_firefighters_sink = local_accessibility_parquet / 'firefighters.parquet'
    Path_hospital_accessibilty = local_accessibility_parquet / 'hospital_accessibility_results.parquet'
    Path_hospital_sink = local_accessibility_parquet / 'hospitals.parquet'
    Path_police_accessibilty = local_accessibility_parquet / 'police_accessibility_results.parquet'
    Path_police_sink = local_accessibility_parquet / 'police.parquet'
    Path_factory_accessibility = local_accessibility_parquet / 'factory_accessibility.parquet'
    Path_factory_sink = local_accessibility_parquet / 'factories_sinks.parquet'
    Path_agriculture_accessibility = local_accessibility_parquet / 'agriculture_accessibility.parquet'
    Path_agriculture_sink = local_accessibility_parquet / 'agriculture_sinks.parquet'

    # hazard point-in-time outputs (produced in 4a)
    flood_depth_roads = hazard_exposure_parquet / "flood_depth_roads.parquet"
    wildfire_risk_roads = hazard_exposure_parquet / "wildfire_risk_roads.parquet"

    # hazards under climate change (produced in 4b)
    Path_future_floods_change_RP = hazard_exposure_parquet / "Future Floods change in RP.parquet"
    Path_future_flooding_roads = hazard_exposure_parquet / "Future Floods change in RP experienced by roads.parquet"
    Path_precipitation_change_rcp_45_period_1 = hazard_exposure_parquet / "change in maximum daily precipitation rcp 45 period 1.parquet"
    Path_precipitation_change_rcp_45_period_2 = hazard_exposure_parquet / "change in maximum daily precipitation rcp 45 period 2.parquet"
    Path_precipitation_change_rcp_85_period_1 = hazard_exposure_parquet / "change in maximum daily precipitation rcp 85 period 1.parquet"
    Path_precipitation_change_rcp_8_5_far_future = hazard_exposure_parquet / "change in maximum daily precipitation rcp 85 period 2.parquet"

    # heat data and impacts (produced in 4c)
    Future_pavement_temperatures = intermediate_results_path / "future_pavement_temperatures.tif"
    roads_current_max_pavement_temperature = hazard_exposure_parquet / "roads_current_max_pavement_temperatures.parquet"
    roads_future_max_pavement_temperature = hazard_exposure_parquet / "roads_future_max_pavement_temperatures.parquet"
   
    # flood statistics per basin computed in 5b (input to 5c)
    Path_flood_statistics_per_basin_scenarios = (
        intermediate_results_path / "SRB_flood_statistics_per_Basin_basins_scenario.csv"
    )

    # combined list of all roads that are exposed to at least one hazard, its intensity and the criticality
    # of each road (produced in 5a; parquet + GDB layer + Excel)
    Path_main_network_hazard_exposure = hazard_exposure_parquet / "main_network_hazard_exposure.parquet"

    # impacts of flooding on accessibility (produced in 5c; parquet + GDB layer + Excel each)
    Path_hospital_impacts = local_accessibility_parquet / 'hospital_impacts.parquet'
    Path_factory_impacts = local_accessibility_parquet / 'factory_impacts.parquet'
    Path_police_impacts = local_accessibility_parquet / 'police_impacts.parquet'
    Path_fire_fighter_impacts = local_accessibility_parquet / 'fire_impacts.parquet'
    Path_road_border_impacts = local_accessibility_parquet / 'road_impacts.parquet'
    Path_port_impacts = local_accessibility_parquet / 'port_impacts.parquet'
    Path_railway_impacts = local_accessibility_parquet / 'rail_impacts.parquet'

    #final climate criticality ranking (produced in 5d, also saved as parquet and gpkg)
    Path_climate_criticality_results = results_path / 'Climate_Criticality_PuteviSrbije.xlsx'




    #Path for figures that are created by the scripts
    Path_factory_acces_avg = figure_path / 'factory_access_avg.png'
    Path_baseline_accessibility_fire_stations = figure_path / 'baseline_accessibility_fire_stations.png'
    Path_baseline_accessibility_hospitals = figure_path / 'baseline_accessibility_hospitals.png'
    Path_baseline_accessibility_police_stations = figure_path / 'baseline_accessibility_police_stations.png'
    Path_agriculture_access_by_type = figure_path / 'agriculture_access_by_type.png'
    Path_firefighter_access_map = figure_path / 'firefighter_access.png'
    Path_hospital_access_map = figure_path / 'hospital_access.png'
    Path_police_station_access_map = figure_path / 'police_station_access.png'
    
    ######################################################
    # Flags to activate/ deactivate outputs
    ######################################################
    show_figures = True #Flag to set whether plots will be shown in a pop up window or not
    print_statistics = True #prints summary of the analysis to the console

    # CRS in which all vector outputs are written (MGI 1901 / Balkans zone 7)
    output_crs = "EPSG:6316"


    #####################################################
    # Make sure all folders exist
    #####################################################

    def __post_init__(self):
        for path in [
            self.data_path,
            self.intermediate_results_path,
            self.local_accessibility_parquet,
            self.local_accessibility_database,
            self.hazard_exposure_parquet,
            self.hazard_exposure_database,
            self.results_path,
            self.arcgis_results,
            self.arcgis_gpgk,
            self.accessibility_analysis_path,
            self.figure_path,
            self.climate_change_precipitation_folder,
            self.temperature_input_folder,
            self.temperature_figures_folder,
        ]:
            path.mkdir(parents=True, exist_ok=True)