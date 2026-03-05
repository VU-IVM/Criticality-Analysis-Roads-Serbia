"""Generate CSV metadata tables for pipeline IO files.

Supported scripts:
- src/1a_NetworkFigures
- src/1b_NetworkPreparation.py
- src/2_MainNetwork_CriticalityAnalysis.py
- src/3a_Baseline_Accesibility_Analysis.py
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import fiona
except ImportError:  # pragma: no cover
    fiona = None

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover
    CRS = None
    Transformer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.network_config import NetworkConfig


@dataclass(frozen=True)
class IOMetadata:
    stage: str  # input | output
    name: str
    path: str
    file_type: str  # vector | raster | image | table
    geometry: str
    feature_count: str
    resolution: str
    crs: str
    coverage: str
    required: str  # yes | no
    produced_by: str
    consumed_by: str
    description: str


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.1f} {unit} ({num_bytes} bytes)"


def _dataset_size_bytes(path: Path) -> int:
    if path.suffix.lower() == ".shp":
        return sum(p.stat().st_size for p in path.parent.glob(f"{path.stem}.*") if p.is_file())
    return path.stat().st_size


def _png_resolution(path: Path) -> str:
    with path.open("rb") as f:
        signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            return "n/a"
        _chunk_len = f.read(4)
        chunk_type = f.read(4)
        if chunk_type != b"IHDR":
            return "n/a"
        width = struct.unpack(">I", f.read(4))[0]
        height = struct.unpack(">I", f.read(4))[0]
    return f"{width}x{height} px"


def _format_coverage_from_bounds(minx: float, miny: float, maxx: float, maxy: float, crs_value: str | None) -> str:
    if crs_value is None:
        return "n/a"
    if Transformer is None:
        return "n/a"
    try:
        transformer = Transformer.from_crs(crs_value, "EPSG:4326", always_xy=True)
        lon_min, lat_min, lon_max, lat_max = transformer.transform_bounds(minx, miny, maxx, maxy)
        return (
            f"min_lat={lat_min:.6f}, min_lon={lon_min:.6f}, "
            f"max_lat={lat_max:.6f}, max_lon={lon_max:.6f}"
        )
    except Exception:
        return "n/a"


def _parse_epsg_from_prj(prj_path: Path) -> str:
    if not prj_path.exists():
        return "n/a"
    try:
        content = prj_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', content)
        if match:
            return f"EPSG:{match.group(1)}"
        if CRS is not None:
            parsed = CRS.from_wkt(content)
            epsg = parsed.to_epsg()
            if epsg is not None:
                return f"EPSG:{epsg}"
    except Exception:
        return "n/a"
    return "n/a"


def _shapefile_metadata(path: Path) -> dict[str, str]:
    meta = {"geometry": "n/a", "feature_count": "n/a", "crs": "n/a", "coverage": "n/a"}
    shx_path = path.with_suffix(".shx")
    prj_path = path.with_suffix(".prj")
    try:
        with path.open("rb") as f:
            header = f.read(100)
            if len(header) < 100:
                return meta

            shape_type = struct.unpack("<i", header[32:36])[0]
            minx, miny, maxx, maxy = struct.unpack("<4d", header[36:68])
            geometry_map = {
                1: "point", 3: "line", 5: "polygon", 8: "point",
                11: "point", 13: "line", 15: "polygon", 18: "point",
                21: "point", 23: "line", 25: "polygon", 28: "point",
            }
            geometry = geometry_map.get(shape_type, "n/a")

            feature_count = "n/a"
            if shx_path.exists() and shx_path.stat().st_size >= 100:
                feature_count = str((shx_path.stat().st_size - 100) // 8)

            crs_text = _parse_epsg_from_prj(prj_path)
            coverage = _format_coverage_from_bounds(minx, miny, maxx, maxy, crs_text if crs_text != "n/a" else None)

            return {
                "geometry": geometry,
                "feature_count": feature_count,
                "crs": crs_text,
                "coverage": coverage,
            }
    except Exception:
        return meta


def _vector_metadata(path: Path, layer: str | None = None) -> dict[str, str]:
    meta = {"geometry": "n/a", "feature_count": "n/a", "crs": "n/a", "coverage": "n/a"}

    if fiona is not None:
        try:
            open_kwargs = {"layer": layer} if layer else {}
            with fiona.open(path, **open_kwargs) as src:
                schema_geom = (src.schema.get("geometry") or "n/a").lower()
                if "line" in schema_geom:
                    geometry = "line"
                elif "point" in schema_geom:
                    geometry = "point"
                elif "polygon" in schema_geom:
                    geometry = "polygon"
                else:
                    geometry = schema_geom

                crs_epsg = src.crs.to_epsg() if src.crs else None
                crs_text = f"EPSG:{crs_epsg}" if crs_epsg is not None else "n/a"
                minx, miny, maxx, maxy = src.bounds
                coverage = _format_coverage_from_bounds(minx, miny, maxx, maxy, src.crs.to_string() if src.crs else None)

                return {
                    "geometry": geometry,
                    "feature_count": str(len(src)),
                    "crs": crs_text,
                    "coverage": coverage,
                }
        except Exception:
            pass

    if gpd is not None:
        try:
            read_kwargs = {"layer": layer} if layer else {}
            gdf = gpd.read_file(path, **read_kwargs)
            if gdf.empty:
                return meta
            geom_values = [str(x).lower() for x in gdf.geom_type.dropna().unique()]
            if any("line" in g for g in geom_values):
                geometry = "line"
            elif any("point" in g for g in geom_values):
                geometry = "point"
            elif any("polygon" in g for g in geom_values):
                geometry = "polygon"
            else:
                geometry = geom_values[0] if geom_values else "n/a"

            crs_text = "n/a"
            if gdf.crs and gdf.crs.to_epsg():
                crs_text = f"EPSG:{gdf.crs.to_epsg()}"
            minx, miny, maxx, maxy = gdf.total_bounds
            coverage = _format_coverage_from_bounds(minx, miny, maxx, maxy, gdf.crs.to_string() if gdf.crs else None)
            return {
                "geometry": geometry,
                "feature_count": str(len(gdf)),
                "crs": crs_text,
                "coverage": coverage,
            }
        except Exception:
            pass

    if path.suffix.lower() == ".shp":
        return _shapefile_metadata(path)

    return meta


def _tabular_point_metadata(path: Path) -> dict[str, str]:
    meta = {"geometry": "point", "feature_count": "n/a", "crs": "EPSG:4326", "coverage": "n/a"}
    if pd is None:
        return meta

    try:
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            return meta

        lat_lon_pairs = [
            ("Latitude", "Longitude"),
            ("LAT", "LON"),
            ("latitude", "longitude"),
            ("lat", "lon"),
            ("Y", "X"),
        ]

        for lat_col, lon_col in lat_lon_pairs:
            if lat_col in df.columns and lon_col in df.columns:
                cleaned = df[[lat_col, lon_col]].dropna()
                if cleaned.empty:
                    return meta
                lat_min = float(cleaned[lat_col].min())
                lat_max = float(cleaned[lat_col].max())
                lon_min = float(cleaned[lon_col].min())
                lon_max = float(cleaned[lon_col].max())
                return {
                    "geometry": "point",
                    "feature_count": str(len(cleaned)),
                    "crs": "EPSG:4326",
                    "coverage": (
                        f"min_lat={lat_min:.6f}, min_lon={lon_min:.6f}, "
                        f"max_lat={lat_max:.6f}, max_lon={lon_max:.6f}"
                    ),
                }

        return {"geometry": "point", "feature_count": str(len(df)), "crs": "EPSG:4326", "coverage": "n/a"}
    except Exception:
        return meta


def _build_rows_1a() -> list[IOMetadata]:
    figures_dir = NetworkConfig.figure_path
    return [
        IOMetadata("input", "OSM road network", str(NetworkConfig.osm_path), "vector", "line", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_osm_network", "OSM lines layer."),
        IOMetadata("input", "Serbian road network shapefile", str(NetworkConfig.Original_road_network), "vector", "line", "auto", "n/a", "auto", "auto", "no", "external", "load_serbian_network_no_arcpy", "Used when arcpy is unavailable."),
        IOMetadata("input", "ArcGIS feature layer", "ArcGIS parameter (GetParameterAsText(0) or config.arcgis_input_layer)", "vector", "line (expected)", "n/a", "n/a", "n/a", "Serbia (expected)", "no", "ArcGIS runtime", "load_serbian_network_arcpy", "Alternative Serbian network input."),
        IOMetadata("input", "Basemap tiles", "CartoDB.Positron via contextily provider", "raster", "n/a", "n/a", "XYZ web tiles (zoom-dependent)", "EPSG:3857", "Global", "yes", "CartoDB", "plot_*", "Basemap tiles."),
        IOMetadata("output", "OSM road network figure", str(figures_dir / "osm_road_network.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "yes", "plot_osm_network", "user/report", "Categorized OSM road map."),
        IOMetadata("output", "Serbian road categories figure", str(figures_dir / "road_categories.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_serbian_network", "user/report", "Serbian categories map."),
        IOMetadata("output", "Network comparison figure", str(figures_dir / "road_network_comparison.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_network_comparison", "user/report", "Side-by-side comparison."),
        IOMetadata("output", "Road length by category figure", str(figures_dir / "road_length_by_category.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "yes", "plot_road_length_statistics", "user/report", "Length bar chart."),
    ]


def _build_rows_1b() -> list[IOMetadata]:
    figures_dir = NetworkConfig.figure_path
    interm = NetworkConfig.intermediate_results_path
    return [
        IOMetadata("input", "Road network with AADT", str(NetworkConfig.Network_original_full_AADT), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "external", "load_network", "Base road network parquet."),
        IOMetadata("input", "AADT traffic counts", str(NetworkConfig.AADT_data), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "external", "merge_aadt_data", "AADT shapefile."),
        IOMetadata("input", "World boundaries", str(NetworkConfig.world_boundaries), "vector", "polygon", "auto", "n/a", "auto", "auto", "yes", "external", "filter_to_serbia", "Country boundaries for filtering."),
        IOMetadata("input", "ArcGIS feature layer", "ArcGIS parameter (GetParameterAsText(0) or config.arcgis_input_layer)", "vector", "line (expected)", "n/a", "n/a", "n/a", "Serbia (expected)", "no", "ArcGIS runtime", "load_network_arcpy", "Alternative network input."),
        IOMetadata("input", "Basemap tiles", "CartoDB.Positron via contextily provider", "raster", "n/a", "n/a", "XYZ web tiles (zoom-dependent)", "EPSG:3857", "Global", "no", "CartoDB", "plot_aadt_*", "Basemap tiles."),
        IOMetadata("output", "AADT categories combined figure", str(figures_dir / "AADT_categories_combined.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_aadt_categories", "user/report", "Combined AADT category map."),
        IOMetadata("output", "Total AADT map figure", str(figures_dir / "total_aadt_aadt_map_og.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_total_aadt_map", "user/report", "Total AADT map."),
        IOMetadata("output", "Directed network parquet", str(interm / "PERS_directed_final.parquet"), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "create_igraph_and_export", "downstream scripts", "Prepared directed network."),
        IOMetadata("output", "Directed network shapefile", str(interm / "PERS_directed_final.shp"), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "create_igraph_and_export", "GIS/ArcGIS", "Prepared directed network shapefile."),
    ]


def _build_rows_2() -> list[IOMetadata]:
    figures_dir = NetworkConfig.figure_path
    interm = NetworkConfig.intermediate_results_path
    return [
        IOMetadata("input", "Processed directed road network", str(NetworkConfig.Path_processed_road_network), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "1b_NetworkPreparation", "load_network_and_graph", "Directed network parquet."),
        IOMetadata("input", "Settlement population", str(NetworkConfig.Path_SettlementData_Excel), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_population_data", "Population points in Excel."),
        IOMetadata("input", "Basemap tiles", "CartoDB.Positron via contextily provider", "raster", "n/a", "n/a", "XYZ web tiles (zoom-dependent)", "EPSG:3857", "Global", "no", "CartoDB", "plot_spof_*", "Basemap tiles."),
        IOMetadata("output", "Criticality results parquet", str(interm / "criticality_results.parquet"), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "save_criticality_results", "downstream scripts", "Criticality metrics per edge."),
        IOMetadata("output", "SPOF 2x2 figure", str(figures_dir / "SPOF_results_2x2.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_spof_disruption_maps", "user/report", "Disruption map panels."),
        IOMetadata("output", "SPOF person-loss 2x2 figure", str(figures_dir / "SPOF_PHL_THL_PKL_TKL.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_spof_person_maps", "user/report", "Person and tonnage loss map panels."),
        IOMetadata("output", "SPOF boxplots figure", str(figures_dir / "SPOF_by_road_type_boxplot.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_road_type_boxplots", "user/report", "Road-type boxplots."),
        IOMetadata("output", "SPOF violin figure", str(figures_dir / "SPOF_by_road_type_violin.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_road_type_violins", "user/report", "Road-type violins."),
        IOMetadata("output", "AADT vs criticality figure", str(figures_dir / "SPOF_AADT_vs_criticality.png"), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_aadt_vs_criticality", "user/report", "Scatter plots of AADT vs metrics."),
    ]


def _build_rows_3a() -> list[IOMetadata]:
    return [
        IOMetadata("input", "Road network", str(NetworkConfig.Path_RoadNetwork), "vector", "line", "auto", "n/a", "auto", "auto", "yes", "1b_NetworkPreparation", "load_road_network", "Road network for accessibility analysis."),
        IOMetadata("input", "Factory locations", str(NetworkConfig.Path_FactoryFile), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_factory_data", "Factory points from Excel."),
        IOMetadata("input", "Border crossings", str(NetworkConfig.path_to_Borders), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_border_crossings", "Border crossing points."),
        IOMetadata("input", "Agriculture locations", str(NetworkConfig.Path_agriculture_input), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_agricultural_data", "Agriculture points."),
        IOMetadata("input", "Ports and rail sinks", str(NetworkConfig.path_to_Sinks), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_sinks", "Road/port/rail sink points."),
        IOMetadata("input", "Settlement population", str(NetworkConfig.Path_SettlementData_Excel), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_population_data", "Settlement points."),
        IOMetadata("input", "Firefighters", str(NetworkConfig.firefighters), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_and_map_sinks", "Fire station points."),
        IOMetadata("input", "Hospitals", str(NetworkConfig.hospitals), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_and_map_sinks", "Hospital points."),
        IOMetadata("input", "Police stations", str(NetworkConfig.police_stations), "table", "point", "auto", "n/a", "EPSG:4326", "auto", "yes", "external", "load_and_map_sinks", "Police station points."),
        IOMetadata("output", "Factory accessibility", str(NetworkConfig.Path_factory_accessibility), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Factory accessibility results."),
        IOMetadata("output", "Factory sinks", str(NetworkConfig.Path_factory_sink), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Factory sink points."),
        IOMetadata("output", "Agriculture accessibility", str(NetworkConfig.Path_agriculture_accessibility), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Agriculture accessibility results."),
        IOMetadata("output", "Agriculture sinks", str(NetworkConfig.Path_agriculture_sink), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Agriculture sink points."),
        IOMetadata("output", "Firefighter accessibility", str(NetworkConfig.Path_firefighter_accessibilty), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Firefighter accessibility results."),
        IOMetadata("output", "Firefighter sinks", str(NetworkConfig.Path_firefighters_sink), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Firefighter sink points."),
        IOMetadata("output", "Hospital accessibility", str(NetworkConfig.Path_hospital_accessibilty), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Hospital accessibility results."),
        IOMetadata("output", "Hospital sinks", str(NetworkConfig.Path_hospital_sink), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Hospital sink points."),
        IOMetadata("output", "Police accessibility", str(NetworkConfig.Path_police_accessibilty), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Police accessibility results."),
        IOMetadata("output", "Police sinks", str(NetworkConfig.Path_police_sink), "vector", "point", "auto", "n/a", "auto", "auto", "yes", "save_accessibilty_results", "3b_plot_figures", "Police sink points."),
        IOMetadata("output", "Factory average access figure", str(NetworkConfig.Path_factory_acces_avg), "image", "n/a", "n/a", "n/a", "n/a", "n/a", "no", "plot_access_times", "user/report", "Factory average travel-time figure (if plotting enabled)."),
    ]


MODULE_BUILDERS = {
    "1a": _build_rows_1a,
    "1b": _build_rows_1b,
    "2": _build_rows_2,
    "3a": _build_rows_3a,
}

MODULE_OUTPUTS = {
    "1a": "1a_network_figures_io_metadata.csv",
    "1b": "1b_network_preparation_io_metadata.csv",
    "2": "2_mainnetwork_criticality_io_metadata.csv",
    "3a": "3a_baseline_accessibility_io_metadata.csv",
}


def _with_runtime_columns(rows: Iterable[IOMetadata]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for row in rows:
        path_str = row.path
        path_exists = "n/a"
        volume = "n/a"
        resolution = row.resolution
        geometry = row.geometry
        feature_count = row.feature_count
        crs = row.crs
        coverage = row.coverage

        is_runtime_source = path_str.startswith("ArcGIS parameter") or path_str.startswith("CartoDB")
        if not is_runtime_source:
            path = Path(path_str)
            path_exists = "yes" if path.exists() else "no"
            if path_exists == "yes":
                volume = _format_bytes(_dataset_size_bytes(path))

                if row.file_type == "image" and path.suffix.lower() == ".png":
                    resolution = _png_resolution(path)

                if row.file_type == "vector":
                    layer = "lines" if path.suffix.lower() == ".pbf" else None
                    v_meta = _vector_metadata(path, layer=layer)
                    if v_meta["geometry"] != "n/a":
                        geometry = v_meta["geometry"]
                    if v_meta["feature_count"] != "n/a":
                        feature_count = v_meta["feature_count"]
                    if v_meta["crs"] != "n/a":
                        crs = v_meta["crs"]
                    if v_meta["coverage"] != "n/a":
                        coverage = v_meta["coverage"]

                if row.file_type == "table":
                    t_meta = _tabular_point_metadata(path)
                    if t_meta["geometry"] != "n/a":
                        geometry = t_meta["geometry"]
                    if t_meta["feature_count"] != "n/a":
                        feature_count = t_meta["feature_count"]
                    if t_meta["crs"] != "n/a":
                        crs = t_meta["crs"]
                    if t_meta["coverage"] != "n/a":
                        coverage = t_meta["coverage"]

        rendered.append(
            {
                "stage": row.stage,
                "name": row.name,
                "path": row.path,
                "file_type": row.file_type,
                "geometry": geometry,
                "feature_count": "n/a" if feature_count == "auto" else feature_count,
                "resolution": resolution,
                "crs": "n/a" if crs == "auto" else crs,
                "coverage": "n/a" if coverage == "auto" else coverage,
                "volume": volume,
                "required": row.required,
                "exists_now": path_exists,
                "produced_by": row.produced_by,
                "consumed_by": row.consumed_by,
                "description": row.description,
            }
        )
    return rendered


def write_csv(rows: list[IOMetadata], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = _with_runtime_columns(rows)
    fieldnames = [
        "stage", "name", "path", "file_type", "geometry", "feature_count",
        "resolution", "crs", "coverage", "volume", "required", "exists_now",
        "produced_by", "consumed_by", "description",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Wrote IO metadata CSV: {output_csv}")
    print(f"Rows: {len(data)}")


def _resolve_modules(requested: list[str]) -> list[str]:
    if "all" in requested:
        return ["1a", "1b", "2", "3a"]
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Create IO metadata CSVs for project scripts")
    parser.add_argument(
        "--module",
        nargs="+",
        choices=["1a", "1b", "2", "3a", "all"],
        default=["all"],
        help="Which script IO metadata to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("flowcharts"),
        help="Output directory for generated CSV files",
    )
    args = parser.parse_args()

    modules = _resolve_modules(args.module)
    for module in modules:
        rows = MODULE_BUILDERS[module]()
        output_csv = args.output_dir / MODULE_OUTPUTS[module]
        write_csv(rows, output_csv)


if __name__ == "__main__":
    main()
