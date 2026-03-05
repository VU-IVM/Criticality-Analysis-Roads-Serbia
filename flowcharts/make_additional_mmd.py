#!/usr/bin/env python3
"""Generate Mermaid flowchart .mmd files for additional pipeline scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FlowSpec:
    filename: str
    source_script: str
    inputs: list[tuple[str, str]]
    process: list[tuple[str, str]]
    outputs: list[tuple[str, str]]
    edges: list[str]


def render_mmd(spec: FlowSpec) -> str:
    lines: list[str] = [
        "flowchart TB",
        f"  %% {spec.source_script}",
        '  subgraph IN["Inputs"]',
    ]
    for node_id, label in spec.inputs:
        lines.append(f'    {node_id}["{label}"]')
    lines.append("  end")
    lines.append("")
    lines.append(f'  subgraph PROC["Process ({spec.source_script})"]')
    for node_id, label in spec.process:
        lines.append(f'    {node_id}["{label}"]')
    lines.append("  end")
    lines.append("")
    lines.append('  subgraph OUT["Outputs"]')
    for node_id, label in spec.outputs:
        lines.append(f'    {node_id}["{label}"]')
    lines.append("  end")
    lines.append("")
    for edge in spec.edges:
        lines.append(f"  {edge}")
    lines.append("")
    return "\n".join(lines)


def build_specs() -> list[FlowSpec]:
    return [
        FlowSpec(
            filename="3b_plot_figures.mmd",
            source_script="src/3b_plot_figures.py",
            inputs=[
                ("IN_ACC", "intermediate_results/*_accessibility*.parquet\\n(fire/hospital/police/factory/agri)"),
                ("IN_SINK", "intermediate_results/*_sink*.parquet\\n(facility/sink locations)"),
                ("IN_BASEMAP", "Basemap tiles (contextily)"),
            ],
            process=[
                ("P1", "Load accessibility outputs by facility type"),
                ("P2", "Create baseline accessibility curves"),
                ("P3", "Plot map figures (factories/agriculture/emergency services)"),
                ("P4", "Plot summary bar+pie accessibility charts"),
                ("P5", "Save all figures"),
            ],
            outputs=[
                ("O1", "figures/factory_access_avg.png"),
                ("O2", "figures/baseline_accessibility_agri_road_port_rail_3x2.png"),
                ("O3", "figures/agriculture_access_by_type.png"),
                ("O4", "figures/baseline_accessibility_fire_stations.png"),
                ("O5", "figures/baseline_accessibility_hospitals.png"),
                ("O6", "figures/baseline_accessibility_police_stations.png"),
                ("O7", "figures/firefighter_access.png"),
                ("O8", "figures/hospital_access.png"),
                ("O9", "figures/police_station_access.png"),
            ],
            edges=[
                "IN_ACC --> P1",
                "IN_SINK --> P1",
                "P1 --> P2",
                "P1 --> P3",
                "P1 --> P4",
                "IN_BASEMAP --> P3",
                "P2 --> P5",
                "P3 --> P5",
                "P4 --> P5",
                "P5 --> O1",
                "P5 --> O2",
                "P5 --> O3",
                "P5 --> O4",
                "P5 --> O5",
                "P5 --> O6",
                "P5 --> O7",
                "P5 --> O8",
                "P5 --> O9",
            ],
        ),
        FlowSpec(
            filename="4a_hazard_maps.mmd",
            source_script="src/4a_Hazard_maps.py",
            inputs=[
                ("IN_WORLD", "input_files/ne_10m_admin_0_countries.shp"),
                ("IN_FLOOD", "input_files/Europe_RP100_filled_depth.tif"),
                ("IN_SNOW", "input_files/snezni_nanosi_studije.shp"),
                ("IN_LAND", "input_files/Nestabilne_pojave.shp"),
                ("IN_ROADS", "intermediate_results/PERS_directed_final.parquet"),
                ("IN_BASEMAP", "Basemap tiles (contextily)"),
            ],
            process=[
                ("P1", "Load Serbia boundary + flood raster (clip to country)"),
                ("P2", "Plot flood depth map"),
                ("P3", "Load snow drift + roads and plot snow drift map"),
                ("P4", "Load landslide inventory and plot landslide map"),
            ],
            outputs=[
                ("O1", "figures/flood_depth_map.png"),
                ("O2", "figures/snow_drift_map.png"),
                ("O3", "figures/landslides_map_by_year.png"),
            ],
            edges=[
                "IN_WORLD --> P1",
                "IN_FLOOD --> P1 --> P2 --> O1",
                "IN_SNOW --> P3 --> O2",
                "IN_ROADS --> P3",
                "IN_LAND --> P4 --> O3",
                "IN_BASEMAP --> P2",
                "IN_BASEMAP --> P3",
                "IN_BASEMAP --> P4",
            ],
        ),
        FlowSpec(
            filename="4b_hazard_maps_climate_change.mmd",
            source_script="src/4b_Hazard_maps_climate_change.py",
            inputs=[
                ("IN_BASIN_STATS", "intermediate_results/SRB_flood_statistics_per_Basin_basins_scenario.csv"),
                ("IN_BASINS", "input_files/hybas_eu_lev09_v1c.shp"),
                ("IN_CC_FLOOD", "input_files/disEnsemble_highExtremes.nc"),
                ("IN_ROADS", "intermediate_results/PERS_directed_final.parquet"),
                ("IN_RX1D", "climate_change_precipitation_folder/*.nc"),
                ("IN_BASEMAP", "Basemap tiles (contextily)"),
            ],
            process=[
                ("P1", "Load climate-change flood data + basins + roads"),
                ("P2", "Compute future return-period shift by basin"),
                ("P3", "Join basin changes to roads"),
                ("P4", "Compute future heavy-precipitation (Rx1d) change"),
                ("P5", "Plot flood RP and precipitation change maps"),
                ("P6", "Export climate-change layers"),
            ],
            outputs=[
                ("O1", "figures/Change in return period.png"),
                ("O2", "figures/Change in return period experienced by roads.png"),
                ("O3", "figures/change in Rx1d.png"),
                ("O4", "intermediate_results/Future Floods change in RP.parquet"),
                ("O5", "intermediate_results/Future Floods change in RP experienced by roads.parquet"),
                ("O6", "intermediate_results/change in maximum daily precipitation rcp 85 period 2.paquet"),
            ],
            edges=[
                "IN_BASIN_STATS --> P1",
                "IN_BASINS --> P1",
                "IN_CC_FLOOD --> P1 --> P2 --> P3",
                "IN_ROADS --> P1",
                "IN_RX1D --> P4",
                "P2 --> P5 --> O1",
                "P3 --> P5 --> O2",
                "P4 --> P5 --> O3",
                "P2 --> P6 --> O4",
                "P3 --> P6 --> O5",
                "P4 --> P6 --> O6",
                "IN_BASEMAP --> P5",
            ],
        ),
        FlowSpec(
            filename="5a_mainnetwork_hazard_criticality.mmd",
            source_script="src/5a_MainNetwork_Hazard_Criticality.py",
            inputs=[
                ("IN_CRIT", "intermediate_results/criticality_results.parquet"),
                ("IN_WORLD", "input_files/ne_10m_admin_0_countries.shp"),
                ("IN_FLOOD", "input_files/Europe_RP100_filled_depth.tif"),
                ("IN_SNOW", "input_files/snezni_nanosi_studije.shp"),
                ("IN_LAND", "input_files/Nestabilne_pojave.shp"),
                ("IN_BASEMAP", "Basemap tiles (contextily)"),
            ],
            process=[
                ("P1", "Load criticality + hazard layers"),
                ("P2", "Compute flood-exposed critical segments"),
                ("P3", "Compute snowdrift-exposed critical segments"),
                ("P4", "Compute landslide-exposed critical segments"),
                ("P5", "Combine hazards into one exposure dataset"),
                ("P6", "Plot multi-hazard criticality figures"),
            ],
            outputs=[
                ("O1", "figures/vehicle_hours_lost_map_flooded.png"),
                ("O2", "figures/vhl_hazards_comparison.png"),
                ("O3", "figures/phl_hazards_comparison.png"),
                ("O4", "figures/tkl_hazards_comparison.png"),
                ("O5", "intermediate_results/main_network_hazard_exposure.parquet"),
            ],
            edges=[
                "IN_CRIT --> P1",
                "IN_WORLD --> P1",
                "IN_FLOOD --> P1 --> P2",
                "IN_SNOW --> P3",
                "IN_LAND --> P4",
                "P2 --> P5",
                "P3 --> P5",
                "P4 --> P5 --> O5",
                "P2 --> P6 --> O1",
                "P5 --> P6 --> O2",
                "P5 --> P6 --> O3",
                "P5 --> P6 --> O4",
                "IN_BASEMAP --> P6",
            ],
        ),
        FlowSpec(
            filename="5b_flood_scenarios_accessibility.mmd",
            source_script="src/5b_Flood_Scenarios_Accessibility.py",
            inputs=[
                ("IN_NET", "input_files/base_network_SRB_basins.parquet"),
                ("IN_FLOOD", "input_files/Europe_RP100_filled_depth.tif"),
                ("IN_BASINS", "input_files/hybas_eu_lev09_v1c.shp"),
                ("IN_FACT", "input_files/2_Factory_Company_geolocations.xlsx"),
                ("IN_AGRI", "input_files/1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"),
                ("IN_POP", "input_files/population_NEW_settlement_geocoded.xlsx"),
                ("IN_SINKS", "input_files/Borders*.xlsx + Ports/Rail + emergency services"),
            ],
            process=[
                ("P1", "Build routing graph from road network"),
                ("P2", "Compute flood exposure on road segments"),
                ("P3", "Factory flood-scenario accessibility"),
                ("P4", "Agriculture flood-scenario accessibility"),
                ("P5", "Emergency services flood-scenario accessibility"),
                ("P6", "Aggregate basin statistics + summaries"),
            ],
            outputs=[
                ("O1", "accessibility_analysis/factory_criticality_results/*_scenario.csv"),
                ("O2", "accessibility_analysis/allagri_criticality_results/*_scenario.csv"),
                ("O3", "accessibility_analysis/fire_criticality_results/*_scenario.csv"),
                ("O4", "accessibility_analysis/healthcare_criticality_results/*_scenario.csv"),
                ("O5", "accessibility_analysis/police_criticality_results/*_scenario.csv"),
                ("O6", "accessibility_analysis/*_criticality_results/*_impact_summary_*.csv"),
            ],
            edges=[
                "IN_NET --> P1",
                "IN_FLOOD --> P2",
                "IN_BASINS --> P2",
                "P1 --> P3",
                "P1 --> P4",
                "P1 --> P5",
                "P2 --> P3",
                "P2 --> P4",
                "P2 --> P5",
                "IN_FACT --> P3 --> P6 --> O1",
                "IN_AGRI --> P4 --> P6 --> O2",
                "IN_POP --> P5 --> P6",
                "IN_SINKS --> P3",
                "IN_SINKS --> P4",
                "IN_SINKS --> P5",
                "P6 --> O3",
                "P6 --> O4",
                "P6 --> O5",
                "P6 --> O6",
            ],
        ),
        FlowSpec(
            filename="5c_combined_climate_criticality.mmd",
            source_script="src/5c_CombinedClimateCriticality.py",
            inputs=[
                ("IN_HAZ", "intermediate_results/main_network_hazard_exposure.parquet"),
                ("IN_NET", "intermediate_results/PERS_directed_final.parquet"),
                ("IN_SECTOR", "intermediate_results/*_impacts.parquet\\n(hospital/factory/police/fire/road/port/rail)"),
                ("IN_CC", "intermediate_results/Future Floods change in RP.parquet\\n+ precipitation change parquet"),
                ("IN_BASEMAP", "Basemap tiles (contextily)"),
            ],
            process=[
                ("P1", "Load and harmonize hazard + impact + climate layers"),
                ("P2", "Spatially attach impact metrics to baseline roads"),
                ("P3", "Normalize metrics and compute H, T, A sub-indices"),
                ("P4", "Compute combined climate-criticality score/classes"),
                ("P5", "Plot 3-panel and combined criticality maps"),
                ("P6", "Export climate criticality table"),
            ],
            outputs=[
                ("O1", "figures/criticality_analysis_3panel.png"),
                ("O2", "figures/climate_criticality_mean.png"),
                ("O3", "intermediate_results/VUA_Climate_Criticality_PERS.xlsx"),
            ],
            edges=[
                "IN_HAZ --> P1",
                "IN_NET --> P1",
                "IN_SECTOR --> P1",
                "IN_CC --> P1 --> P2 --> P3 --> P4",
                "P4 --> P5 --> O1",
                "P4 --> P5 --> O2",
                "P4 --> P6 --> O3",
                "IN_BASEMAP --> P5",
            ],
        ),
    ]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    specs = build_specs()

    for spec in specs:
        content = render_mmd(spec)
        out_path = out_dir / spec.filename
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
