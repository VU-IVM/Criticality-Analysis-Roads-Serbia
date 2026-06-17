import json
from pathlib import Path
import geopandas as gpd

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (R, G, B) integers."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def save_lyrx_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    lyrx_path: Path,
    layer_name: str,
    labels: list[str],
    colors: list[str],
    width_mapping: dict[str, float],
    field: str,
    title: str | None = None,
    gdb_path: Path | None = None,
    output_crs: str = "EPSG:6316",
) -> None:
    """
    Save a GeoDataFrame as a vector layer and generate a matching ArcGIS Pro
    layer file (.lyrx) with unique-value symbology (colour + width).

    The .lyrx references its data source by absolute path, so the source must
    remain in place. Two data-source modes:

    * ``gdb_path`` given — the data is written as a feature class inside that
      File GDB (reprojected to *output_crs*) and used as the lyrx source. No
      GeoPackage is written. ``gpkg_path`` is ignored.
    * ``gdb_path`` is None — the data is written to ``gpkg_path`` (GeoPackage)
      and used as the lyrx source (legacy behaviour).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save.
    gpkg_path : Path
        Output path for the .gpkg file (used only when *gdb_path* is None).
    lyrx_path : Path
        Output path for the .lyrx file.
    layer_name : str
        Feature class / table name inside the data source.
    labels, colors, width_mapping, field, title
        Unique-value symbology definition (see above).
    gdb_path : Path, optional
        File GDB to use as the data source instead of a GeoPackage.
    output_crs : str
        CRS the GDB feature class is written in (GDB mode only).
    """

    # arcpy is optional — needed only for .lyrx generation.
    try:
        import arcpy
    except ImportError:
        arcpy = None

    if gdb_path is not None:
        # GDB feature class as the data source (no GeoPackage written).
        from utils.hazard_functions import save_gdb_layer

        # arcpy holds a File GDB schema lock after reading a feature class, which
        # would block save_gdb_layer from rebuilding the GDB for the next layer.
        # Release any lock this process still holds before (re)writing.
        if arcpy is not None:
            try:
                arcpy.management.ClearWorkspaceCache()
            except Exception:
                pass

        save_gdb_layer(gdf, gdb_path, layer_name, output_crs)
        data_source = str(Path(gdb_path).absolute()) + f"/{layer_name}"
    else:
        # Save GeoPackage
        gdf.to_file(gpkg_path, driver="GPKG", layer=layer_name)
        print(f"Saved GeoPackage → {gpkg_path}")
        data_source = str(Path(gpkg_path).absolute()) + f"/{layer_name}"

    Path(lyrx_path).parent.mkdir(parents=True, exist_ok=True)

    if arcpy is None:
        print(
            "Warning: arcpy could not be imported — skipping .lyrx layer generation for ArcGIS. "
            "The data was written; to also generate ArcGIS layer files, run using the "
            r"ArcGIS Pro Python environment: C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
        )
        return

    title = title or layer_name

    # 1. Let arcpy build a valid data connection
    gpkg_layer_path = data_source
    tmp_layer_name = f"tmp_lyr_{layer_name}"
    arcpy.management.MakeFeatureLayer(gpkg_layer_path, tmp_layer_name)
    tmp_lyrx = lyrx_path.with_suffix(".tmp.lyrx")
    arcpy.management.SaveToLayerFile(tmp_layer_name, str(tmp_lyrx), "ABSOLUTE")
    arcpy.management.Delete(tmp_layer_name)

    # 2. Load arcpy-generated lyrx
    with open(tmp_lyrx) as f:
        lyrx = json.load(f)

    # 3. Build renderer
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def make_symbol(hex_color: str, width: float) -> dict:
        r, g, b = hex_to_rgb(hex_color)
        return {
            "type": "CIMSymbolReference",
            "symbol": {
                "type": "CIMLineSymbol",
                "symbolLayers": [{
                    "type": "CIMSolidStroke",
                    "enable": True,
                    "capStyle": "Round",
                    "joinStyle": "Round",
                    "width": width,
                    "color": {"type": "CIMRGBColor", "values": [r, g, b, 100]},
                }],
            },
        }

    renderer = {
        "type": "CIMUniqueValueRenderer",
        "defaultLabel": "<all other values>",
        "defaultSymbolPatch": "Default",
        "defaultSymbol": make_symbol("#cccccc", 0.5),
        "defaultSymbolVisible": False,
        "fields": [field],
        "groups": [{
            "type": "CIMUniqueValueGroup",
            "classes": [
                {
                    "type": "CIMUniqueValueClass",
                    "label": label,
                    "patch": "Default",
                    "symbol": make_symbol(color, width_mapping.get(label, 1.0)),
                    "values": [{"type": "CIMUniqueValue", "fieldValues": [label]}],
                    "visible": True,
                }
                for label, color in zip(labels, colors)
            ],
            "heading": field,
        }],
        "useDefaultSymbol": False,
        "polygonSymbolColorTarget": "Fill",
    }

    # 4. Inject renderer, preserve arcpy's data connection
    lyrx["layerDefinitions"][0]["renderer"] = renderer
    lyrx["layerDefinitions"][0]["name"] = title

    with open(lyrx_path, "w") as f:
        json.dump(lyrx, f, indent=2)

    tmp_lyrx.unlink()
    print(f"Saved layer file → {lyrx_path}")

    # Release the schema lock arcpy took on the data source so the next layer's
    # GDB rewrite is not blocked.
    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass