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
) -> None:
    """
    Save a GeoDataFrame to GeoPackage and generate a matching ArcGIS Pro
    layer file (.lyrx) with unique-value symbology (colour + width).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save. Should be in its original CRS (not reprojected
        for plotting).
    gpkg_path : Path
        Output path for the .gpkg file.
    lyrx_path : Path
        Output path for the .lyrx file.
    layer_name : str
        Table name inside the GeoPackage.
    labels : list[str]
        Ordered class labels matching the classified field values.
    colors : list[str]
        Hex color strings, one per label.
    width_mapping : dict[str, float]
        Line width in points per label.
    field : str
        Attribute field used for classification.
    title : str, optional
        Display name for the layer in ArcGIS. Defaults to layer_name.
    """

    # Save GeoPackage
    gdf.to_file(gpkg_path, driver="GPKG", layer=layer_name)
    print(f"Saved GeoPackage → {gpkg_path}")

    try:
        import arcpy
    except ImportError:
        print(
            "Warning: arcpy could not be imported — skipping .lyrx layer generation for ArcGIS. "
            "To enable ArcGIS layer files, run using the ArcGIS Pro Python environment: "
            r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
        )
        return

    title = title or layer_name

    # 1. Let arcpy build a valid data connection
    gpkg_layer_path = str(gpkg_path.absolute()) + f"/{layer_name}"
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