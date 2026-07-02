# Road Network Criticality & Accessibility Analysis

A collection of Python scripts and Jupyter Notebooks designed to assess the **criticality** of road network segments and evaluate **population accessibility** to emergency services.  
Originally developed for the **Republic of Serbia**, the workflow is transferable to other national road networks with slight adjustments.

---

## Overview

This repository provides a comprehensive workflow to:

1. **Prepare and process road networks**  
2. **Assess the criticality** of each road segment by quantifying the impact of its disruption:  
   - Vehicle hours lost, vehicle kilometres lost
   - Tonnage hours lost, tonnage kilometres lost 
3. **Analyse accessibility** of population clusters to:  
   - Fire stations  
   - Hospitals  
   - Police stations  
   And the access times of industrial and agricultural areas to road borders, ports and rail terminals
4. **Evaluate hazard exposure**, including:  
   - Flooding, landslides and snowdrifts under current climatic conditions
   - Flooding and heavy precipitation under climate change  
5. **Calculate combined climate–criticality metric** that considers the previously evaluated hazard exposure, national-scale travel disruptions and local accessibiliy

---

## Workflow Structure

The analysis is implemented through a series of Jupyter Notebooks and corresponding Python scripts.  
The workflow follows this approximate sequence:

| Step | Description | Notebooks | Scripts |
| ------ | ------------- | ----------- | --------- |
| **1. Network Preparation** | Load, simplify, and preprocess the national road network | `1a_NetworkFigures.ipynb`<br>`1b_NetworkPreparation.ipynb` | `1a_NetworkFigures.py`<br>`1b_NetworkPreparation.py` |
| **2. Criticality Analysis** | Compute disruption impact of each road segment | `2_MainNetwork_CriticalityAnalysis.ipynb` | `2_MainNetwork_CriticalityAnalysis.py` |
| **3. Accessibility Analysis** | Assess travel time of population clusters to facilities |`3a_Baseline_Accesibility_Analysis-factories.ipynb`<br>`3b_Baseline_Accesibility_Analysis-farms.ipynb`<br>`3c_Baseline_Accesibility_Analysis-firefighters.ipynb`<br>`3d_Baseline_Accesibility_Analysis-hospital.ipynb`<br>`3e_Baseline_Accesibility_Analysis-policestations.ipynb` | `3a_Baseline_Accesibility_Analysis.py`<br>`3b_plot_figures.py` |
| **4. Hazard Mapping** | Generate hazard layers (baseline + climate change) | `4a_Hazard_maps.ipynb`<br>`4b_Hazard_Maps_Climate_Change.ipynb` | `4a_Hazard_maps.py`<br>`4b_Hazard_maps_climate_change.py` |
| **5. Combined Risk Analysis** | Hazard‑informed network criticality and accessibility | `5a_MainNetwork_Hazard_Criticality.ipynb`<br>`5b_Flood_Scenarios_Accessibility.ipynb`<br>`5c_CombinedClimateCriticality.ipynb` | `5a_MainNetwork_Hazard_Criticality.py`<br>`5b_Flood_Scenarios_Accessibility.py`<br>`5c_CombinedClimateCriticality.py` |

---

## Repository Structure

```plaintext
criticality-analysis/
├── notebooks/                     # Interactive Jupyter notebooks showing the full workflow step‑by‑step.
└── src/                           # Python scripts for easier execution and slightly extended functionality
    └── config/                    
        └── network_config.py      # Main configuration file (paths & settings)

```
---

## Installation

```bash
git clone https://github.com/VU-IVM/Criticality-Analysis-Roads-Serbia.git
cd <repository-folder>
```

## Create the environment from environment.yml

### 1. Install `mamba` in your base conda environment:

```bash
conda install -n base -c conda-forge mamba
```

### 2. Create the project environment from `environment.yml`:

```bash
mamba env create -f environment.yml
```

### 3. Activate the new environment:

```bash
conda activate criticality_env
```

If the environment name changes in `environment.yml`, activate with that name instead.

---

## Data

### Input Files

The input data (hazard maps, road networks, exposure layers, etc.) is **not included** in this repository; the table below lists the files it expects. Unless a sub-folder is shown in the filename, every file is read from `input_files/`. To use different filenames, rename them in the config file (`network_config.py`) when running the scripts, or directly in the notebooks when running those.


| Category | File (relative to `input_files/`) | Type | Contents | Used in |
|---|---|---|---|---|
| Road network & traffic | `base_network_SRB_basins.parquet` | Vector — line (Parquet) | OpenStreetMap-derived routing network covering all road classes including local roads for the basin flood-accessibility analysis; includes flood-exposure flags with sampled depths. Built outside the tracked pipeline. | 3, 5b, 5c |
| Road network & traffic | `Deonice_Februar_2025.shp` | Vector — line (Shapefile) | Baseline PERS main-road sections (Feb 2025); sections missing from the criticality results are appended (core Serbia only, Kosovo excluded). | 5a |
| Road network & traffic | `DeoniceRSDP-Jul2025..shp` | Vector — line (Shapefile) | Original PERS main-road network sections (Deonice), July 2025 — the raw road geometry input. | 1b |
| Road network & traffic | `DeoniceRSDP-Jul2025_`<br>`corrected_topology.parquet` | Vector — line (Parquet) | PERS main-road sections with slightly adjusted (snapped/connected) topology; used to build the directed network. | 1b |
| Road network & traffic | `PGDS_2024.shp` | Vector — line (Shapefile) | AADT (annual average daily traffic, *Prosečan godišnji dnevni saobraćaj*) counts per segment, 2024, broken down by vehicle class; merged onto the network. | 1b |
| Road network & traffic | `SRB.osm.pbf` | Vector — OSM Protobuf | OpenStreetMap road extract for Serbia; source for the OSM-derived accessibility network. | 1a / network build |
| Boundaries & basins | `hybas_eu_lev09_v1c.shp` | Vector — polygon (Shapefile) | HydroBASINS level-9 sub-basin polygons (Europe); define the flood-scenario units (one flood scenario per basin). | 5b, 5c |
| Boundaries & basins | `ne_10m_admin_0_countries.shp` | Vector — polygon (Shapefile) | Natural Earth 1:10 m country boundaries; used to clip rasters and filter to Serbia. | 1b, 4, 5a, 5b |
| Flood hazard | `disEnsemble_highExtremes.nc` | Gridded (netCDF) | Climate-change fluvial-flood ensemble (high discharge extremes); used to derive the projected change in flood return period. | 4 |
| Flood hazard | `Europe_RP100_filled_depth.tif` | Raster (GeoTIFF) | European fluvial flood depth for the 100-year return period (void/depth-filled); used to sample the flood depth onto roads. | 5a, 5b |
| Flood hazard | `SRB_flood_statistics_`<br>`per_Basin_basins_scenario.csv` | Tabular (CSV) | Per-basin flood statistics (min/mean/max water depth) for the basin scenarios; produced by 5b, plotted by 5c. | 5c |
| Other hazards | `landslide_susceptibility.tif` | Raster (GeoTIFF) | Landslide susceptibility classes. | 4 |
| Other hazards | `Nestabilne_pojave.shp` | Vector — point/polygon (Shapefile) | Recorded "unstable phenomena"; filtered to landslides (`tip == "Klizište"`) and buffered for the road overlay. | 5a |
| Other hazards | `snezni_nanosi_studije.shp` | Vector — line/point (Shapefile) | Historic snow-drift locations/segments; spatially joined to flag snow-drift-exposed roads. | 5a |
| Other hazards | `wildfire risk/`<br>`stepen ugrozenosti od `<br>`pozara Srbijasume.tif` | Raster (GeoTIFF) | Wildfire susceptibility / degree of fire endangerment raster (Srbijašume forest service). | 5a, 4 |
| Climate-change precipitation | `Climate Change `<br>`Precipitation/results/`<br>`rcp45_rx1d_change1.nc`<br>`rcp45_rx1d_change2.nc`<br>`rcp85_rx1d_change1.nc`<br>`rcp85_rx1d_change2.nc`<br>`rcp45_rx1d_change1_ensmed.nc`<br>`rcp45_rx1d_change2_ensmed.nc`<br>`rcp85_rx1d_change1_ensmed.nc`<br>`rcp85_rx1d_change2_ensmed.nc` | Gridded (netCDF) | Projected change in **Rx1day** (annual maximum 1-day precipitation) over Serbia — one grid per scenario, period and ensemble type. `rcp45` / `rcp85` = RCP 4.5 (moderate) / RCP 8.5 (high) emission scenario; `change1` / `change2` = first (near/mid-future) / second (far-future) projection period; `_ensmed` = ensemble median across the climate models (files without the suffix carry the full multi-model ensemble). | 4 |
| Temperature&nbsp;& urbanisation | `GHS_SMOD_E2025_`<br>`GLOBE_R2023A_54009_`<br>`1000_V2_0_R4_C20.tif` | Raster (GeoTIFF) | GHS Settlement Model (GHSL), 2025 epoch, 1 km, World Mollweide (ESRI:54009), tile R4/C20 — degree-of-urbanisation grid used for the urban-heat-island (UHI) adjustment of pavement temperatures. | 4 |
| Temperature & urbanisation | `Temperatures climate change/`<br>`TPAV_2021-2025.tif` | Raster (GeoTIFF) | Current maximum pavement temperature at 20mm depth, 2021–2025. | 4 |
| Temperature & urbanisation | `Temperatures climate change/`<br>`TX7D_1961-1990.tif`<br>`TX7D_1991-2020.tif` | Raster (GeoTIFF) | Annual maximum 7-day temperature index (TX7D) for two reference periods — `1961-1990` (historic baseline) and `1991-2020` (recent); their difference gives the observed warming applied in the pavement-temperature adjustment. | 4 |
| Accessibility (service points) | `4_Hospitals_healthcenters_`<br>`geocoded.xlsx` | Tabular (Excel) — points | Hospitals & health centres with lat/long (geocoded); destinations for healthcare accessibility. | 3, 5b |
| Accessibility (service points) | `6_Firefighters_geocoded.xlsx` | Tabular (Excel) — points | Fire stations with lat/long (geocoded); destinations for fire-service accessibility. | 3, 5b |
| Accessibility (service points) | `6_Police_geocoded.xlsx` | Tabular (Excel) — points | Police stations with lat/long (geocoded); destinations for police accessibility. | 3, 5b |
| Accessibility (economic activities) | `1_agriculture_2023_`<br>`serbia_NEW_FINAL_26092025.xlsm` | Tabular (Excel) — points | Agricultural production 2023 — locations and volumes; origins for the agriculture → border/port/rail accessibility. | 3, 5b |
| Accessibility (economic activities) | `2_Factory_Company_`<br>`geolocations.xlsx` | Tabular (Excel) — points | Industrial areas / company geolocations; origins for the industry → border-crossing accessibility. | 3, 5b |
| Accessibility (economic activities) | `Borders_geocoded.xlsx` | Tabular (Excel) — points | Border-crossing locations (geocoded). | 3 |
| Accessibility (economic activities) | `Borders_Ports_Rail_`<br>`geocoded.xlsx` | Tabular (Excel) — points | Combined sink locations: border crossings, ports, and rail terminals (geocoded); destinations for the agriculture accessibility. | 3, 5b |
| Population | `population_NEW_`<br>`settlement_`<br>`geocoded.xlsx` | Tabular (Excel) — points | Settlement population points (geocoded) with population counts; the demand origins for accessibility routing. | 2, 3, 5b |
| Elevation | `dem_serbia.tif` | Raster (GeoTIFF) | Digital elevation model used by the JRC flood map, cropped to Serbia; used to estimate per-category road elevation bias for flood-depth correction. | 5a |
| Elevation | `Vertical coordinates/`<br>`RSDP_Feb_2026/`<br>`RSDP_Feb_2026/Deonice/`<br>`RSDP_Deonice_Feb_2026.shp` | Vector — line Z (Shapefile) | Road sections with measured Z (vertical) coordinates (RSDP, Feb 2026); compared against the DEM to derive the elevation bias. | 5a |


## Output data structure

Executing the full workflow produces the following files.

### Intermediate results (network)

| File name | Description |
|---|---|
| main_network_directed | Directed national road network used for routing purposes |

### Intermediate results (travel_disruptions)

| File name | Description |
|---|---|
| criticality_results | Vehicle hours lost, person kilometers lost and tonnage kilometers lost that the disruption of each road section causes (single point of failure analysis) |

### Intermediate results (local_accessibility)

| File name | Description |
|---|---|
| agriculture_accessibility | Contains all origins of agricultural production (points) and the access times to ports, borders and rail (nearest and average) |
| agriculture_sinks | Agriculture origin points and their sink destinations |
| port_impacts | All road sections that cause delays in the accessibility of agricultural areas to ports in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| rail_impacts | All road sections that cause delays in the accessibility of agricultural areas to rail cargo stations in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| road_impacts | All road sections that cause delays in the accessibility of agricultural areas to road border crossings in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| factory_accessibility | Baseline average access time of factories to all road border crossings |
| factory_impacts | All road sections that cause delays in the accessibility of factories to border crossings in the flood scenario analysis (OpenStreetMap network) |
| factories_sinks | Industrial areas and road border export points |
| fire_impacts | All road sections that cause delays in the accessibility of firefighters in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| firefighter_accessibility_results | Baseline access time of all settlements to the closest firefighter (attribute “travel_time_ff”) |
| firefighters | Firefighter location points and the population clusters they serve |
| hospital_accessibility_results | Baseline access time of all settlements to the closest hospital in hours (attribute “travel_time_hosp”) |
| hospital_impacts | All road sections that cause delays in the accessibility of hospitals in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| hospitals | Hospital location points and the population clusters they serve |
| main_network_hazard_exposure | Travel and hazard metrics for each road section (flood_depth for flood exposure, dužina_sn for snow drift, datum_evid for the date of landslide occurrence, wildfire_susceptibility indicating whether the wildfire susceptibility is a yes or no, and max_pavement_temp in °C) |
| police_accessibility_results | Baseline access time of all settlements to the closest police station in hours (attribute “travel_time_pol”) |
| police_impacts | All road sections that cause delays in the accessibility to police station in the flood scenario analysis (OpenStreetMap network, attributes “travel_times_impact” and “impact_class”) |
| police | Police station locations and their associated population clusters |

### Intermediate results (hazard_exposure)

| File name | Description |
|---|---|
| flood_depth_roads | Road segments exposed to present-day RP100 flood depth and their flood depth in m |
| wildfire_risk_roads | Road segments running through an area susceptible to wildfire (1/0 for yes/ no) |
| Future Floods change in RP | Basin-level change in flood return period under climate change |
| Future Floods change in RP experienced by roads | The basin level return-period change applied to each road in the affected basins |
| change in maximum daily precipitation rcp 45 period 1 | Max daily precipitation change per road under RCP 4.5 in the medium future (2031 – 2060) in % |
| change in maximum daily precipitation rcp 45 period 2 | Max daily precipitation change per road under RCP 4.5 in the far future (2071 – 2100) in % |
| change in maximum daily precipitation rcp 85 period 1 | Max daily precipitation change per road under RCP 8.5 in the medium future (2031 – 2060) in % |
| change in maximum daily precipitation rcp 85 period 2 | Max daily precipitation change per road under RCP 8.5 in the far future (2071 – 2100) in % |
| roads_current_max_pavement_temperatures | Present-day maximum pavement temperatures (in °C) during the seven hottest consecutive days of the years 2021-2025 sampled onto each road . |
| roads_future_max_pavement_temperatures | Future maximum pavement temperature sampled onto each road, accounting for a temperature increase of 2°C with respect to the present temperatures and the heat island effect. |
| main_network_hazard_exposure | Travel and hazard metrics for each road section (flood_depth for flood exposure, dužina_sn for snow drift, datum_evid for the date of landslide occurrence, wildfire_susceptibility indicating whether the wildfire susceptibility is a yes or no, and max_pavement_temp in °C) |

### Final Results

| File name | Description |
|---|---|
| Climate_Criticality_PuteviSrbije | Final file that summarises all previous analysis on a road section level for the national road network. Attributes include the final criticality score (“CC_climate_criticality”), the sub indices for hazard exposure, transport disruptions and local accessibility (“H_class”, “T_class”, “A_class”) as well as all previously investigated metrics. |
| climate_criticality_index.gdb | ArcGIS feature class for the combined climate-criticality index. |
| hazard_exposure.gdb | ArcGIS feature class for the hazard-exposure sub-index (H). |
| travel_disruption.gdb | ArcGIS feature class for the national travel-disruption sub-index (T). |
| local_accessibility.gdb | ArcGIS feature class for the local-accessibility sub-index (A). |

---

## Authors

- **Elco Koks**
- **Joël De Plaen**
- **Valentin Weiwad**

---

## License

This project is licensed under the terms of the [License](LICENSE) file.
