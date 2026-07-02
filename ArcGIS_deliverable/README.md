# ArcGIS Pro deliverable — Serbia road criticality

This folder contains 26 layer pairs (`.gpkg` + `.lyrx`) ready to open in ArcGIS Pro, plus a one-click master atlas.

## Quickest start — master atlas
Drag **`serbia_criticality_atlas.lyrx`** (at the bundle root) into ArcGIS Pro.
The Contents pane shows every layer in the hierarchy from `layer_inventory.md`:
```
Serbia Road Criticality Atlas
├── Final results
│   ├── Short — total_climate_criticality_short    (visible at start)
│   └── Long  — total_climate_criticality_long
├── Sub-indices
│   ├── H — Hazard exposure
│   ├── T — Travel disruption
│   └── A — Local accessibility
├── Intermediate results
│   ├── Hazard exposure (7)
│   │   └── Network — combined hazard exposure (group of 4 sub-renderers)
│   ├── National travel disruption (1)
│   └── Local accessibility (11)
└── Input
    ├── OSM Road Network — Serbia
    └── PERS Road Network — Serbia
```
All sub-groups start collapsed; only `total_climate_criticality_short` is visible —
toggle the rest on as needed. `network_hazard_exposure` itself is a sub-group
exposing flood depth, snow drift, pavement temperature and wildfire susceptibility
as four separate renderers over the same `.gpkg`.

## Individual layers
Each layer also has its own `.lyrx` next to its `.gpkg`. Drag any of those for a
single-layer load. Data connections are by **relative path**, so the whole
`ArcGIS_deliverable/` folder is portable.

## Folder layout
```
ArcGIS_deliverable/
├── serbia_criticality_atlas.lyrx      # master atlas — open this first
├── 6_Build_ArcGIS_Deliverable.ipynb   # regenerate everything
├── layer_inventory.md                 # full description of every layer
├── input/                   # 2 layers  — OSM & PERS road network (from 1a)
├── hazard_exposure/         # 7 layers — per-hazard exposure on the road network
├── travel_disruption/       # 1 layer  — national-scale SPOF criticality
├── local_accessibility/     # 11 layers — baseline accessibility + TT impacts
└── final_results/           # 2 layers — final composite climate criticality
```

## Layer inventory
| sub-folder | layer | geom | classified field | features |
|---|---|---|---|---|
| `input` | `osm_road_network` | line | `road_category` | 375,455 |
| `input` | `pers_road_network` | line | `kategorija` | 2,023 |
| `hazard_exposure` | `network_hazard_exposure` | line | `flood_class, snow_class, pavement_class, wildfire_class` | 2,286 |
| `hazard_exposure` | `future_floods_network` | line | `future_floods_network_class` | 7,438 |
| `hazard_exposure` | `future_floods` | polygon | `future_floods_class` | 301 |
| `hazard_exposure` | `future_precipitation_rcp45_near_future` | line | `future_precipitation_rcp45_near_future_class` | 4,788 |
| `hazard_exposure` | `future_precipitation_rcp45_far_future` | line | `future_precipitation_rcp45_far_future_class` | 4,815 |
| `hazard_exposure` | `future_precipitation_rcp85_near_future` | line | `future_precipitation_rcp85_near_future_class` | 4,839 |
| `hazard_exposure` | `future_precipitation_rcp85_far_future` | line | `future_precipitation_rcp85_far_future_class` | 5,214 |
| `travel_disruption` | `network_criticality` | line | `network_criticality_class` | 3,135 |
| `local_accessibility` | `port_tt_impact` | line | `port_tt_impact_class` | 59,974 |
| `local_accessibility` | `agriculture_tt_impact` | line | `agriculture_tt_impact_class` | 56,701 |
| `local_accessibility` | `police_tt_impact` | line | `police_tt_impact_class` | 78,057 |
| `local_accessibility` | `hospital_tt_impact` | line | `hospital_tt_impact_class` | 78,544 |
| `local_accessibility` | `firefighter_tt_impact` | line | `firefighter_tt_impact_class` | 78,316 |
| `local_accessibility` | `factory_tt_impact` | line | `factory_tt_impact_class` | 36,020 |
| `local_accessibility` | `police_bl_accessibility` | point | `police_bl_accessibility_class` | 4,673 |
| `local_accessibility` | `hospital_bl_accessibility` | point | `hospital_bl_accessibility_class` | 4,673 |
| `local_accessibility` | `firefighter_bl_accessibility` | point | `firefighter_bl_accessibility_class` | 4,673 |
| `local_accessibility` | `factory_bl_accessibility` | point | `factory_bl_accessibility_class` | 108 |
| `local_accessibility` | `agriculture_bl_accessibility` | point | `agriculture_bl_accessibility_class` | 4,621 |
| `final_results` | `total_climate_criticality_long` | line | `CC_class` | 2,158 |
| `final_results` | `total_climate_criticality_short` | line | `mean_class` | 2,158 |
| `final_results` | `hazard_exposure_index` | line | `H_class` | 2,158 |
| `final_results` | `travel_disruption_index` | line | `T_class` | 2,158 |
| `final_results` | `local_accessibility_index` | line | `A_class` | 2,158 |

See `layer_inventory.md` for descriptions, attribute columns and source-file
renames.

## Classification methods
- **Composite criticality** (`total_climate_criticality_long/short`) — pre-computed
  quintile labels `CC_class` / `mean_class` from `5d_Combined_Climate_Criticality.ipynb`.
- **`network_hazard_exposure`** — 4 sub-renderers in one `.lyrx`:
  flood depth (log-quintile, zero-aware), snow drift (log-quintile, zero-aware),
  max pavement temperature (quintile), wildfire susceptibility (binary).
- **Travel-time impact** & **accessibility** — 5-class quantile (Very Low … Very
  High) computed on-the-fly.
- **Future precipitation** — fixed bins on `max_rx1day_pct`:
  `[-∞, -2, 0, 5, 10, 20, ∞] %`.
- **Future flood RP** — fixed bins on `rp30_mean`: `[0, 25, 50, 100, 150, ∞]`
  years. Red = lower RP = higher risk.
- **Network criticality** — log-quantile on `phl` (long-tailed distribution).
- **Sub-indices** (`hazard_exposure_index`, `travel_disruption_index`,
  `local_accessibility_index`) — reuse `total_climate_criticality_long.gpkg`
  and render its pre-computed `H_class` / `T_class` / `A_class` quintile fields
  with the long-composite purple palette.

## Regenerate
Open `6_Build_ArcGIS_Deliverable.ipynb` and run all cells. The bundle is
overwritten in place — the master atlas, README and all layer pairs.
