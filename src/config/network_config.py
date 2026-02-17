
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NetworkConfig:
    """Configuration for accesibility analysis and visualization."""

    # Input paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    data_path = BASE_DIR / "input_files"
    intermediate_results_path = BASE_DIR / 'intermediate_results'
    accessibility_analysis_path = BASE_DIR / 'accessibility_analysis'
    Path_RoadNetwork = data_path / "base_network_SRB_basins.parquet"
    Path_FactoryFile = data_path / "2_Factory_Company_geolocations.xlsx"
    path_to_Borders = data_path / "Borders_geocoded.xlsx"
    Path_AgriFile = data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
    path_to_Sinks = data_path / "Borders_Ports_Rail_geocoded.xlsx"
    Path_SettlementData_Excel = data_path / "population_NEW_settlement_geocoded.xlsx"
    firefighters = data_path / "6_Firefighters_geocoded.xlsx"
    hospitals = data_path / "4_Hospitals_healthcenters_geocoded.xlsx"
    police_stations = data_path / "6_Police_geocoded.xlsx"
    

    #Output paths
    figure_path = BASE_DIR / "figures"
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

    #Flags to activate/ deactivate outputs
    show_figures = True #Flag to set whether plots will be shown in a pop up window or not
    print_statistics = True #prints summary of the analysis to the console