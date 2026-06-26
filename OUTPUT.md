## Data dictionary

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
