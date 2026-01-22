import arcpy
import geopandas as gpd
from pathlib import Path

arcpy.env.overwriteOutput = True

# -----------------------------------------------
# INPUT
# -----------------------------------------------
input_layer = arcpy.GetParameterAsText(0)

base_tmp = Path(r"C:\Temp\arcgis_tmp")
base_tmp.mkdir(parents=True, exist_ok=True)

gdb_path = base_tmp / "temp.gdb"
if not arcpy.Exists(str(gdb_path)):
    arcpy.management.CreateFileGDB(str(base_tmp), "temp.gdb")

out_fc_name = "roads"
out_fc = gdb_path / out_fc_name

# Copy input layer to GDB
arcpy.management.CopyFeatures(input_layer, str(out_fc))

# Read into GeoDataFrame if needed
gdf = gpd.read_file(str(gdb_path), layer=out_fc_name)
arcpy.AddMessage(f"Loaded {len(gdf)} features into GeoDataFrame")

# -----------------------------------------------
# ADD TO CURRENT MAP
# -----------------------------------------------
aprx = arcpy.mp.ArcGISProject("CURRENT")
m = aprx.activeMap

# Add the copied GDB feature class to the map
layer = m.addDataFromPath(str(out_fc))  # <- THIS IS NOW YOUR Layer object

# Optional: zoom to extent
desc = arcpy.Describe(layer)
m.defaultCamera.setExtent(desc.extent)

# -----------------------------------------------
# APPLY ROAD-TYPE COLORS AND WIDTHS
# -----------------------------------------------
# Only keep categories you want
road_categories = ['IA', 'IM', 'IB', 'IIA', 'IIB']
layer.definitionQuery = "kategorija IN ({})".format(
    ", ".join([f"'{cat}'" for cat in road_categories])
)

# Assume `layer` is already added to the map
sym = layer.symbology
sym.updateRenderer("UniqueValueRenderer")
sym.renderer.fields = ["kategorija"]

# Force ArcGIS to create the unique values from existing data
sym.renderer.generateRenderer()  # <-- IMPORTANT

# Define colors and widths
road_colors = {
    'IA': (139, 0, 0, 100),
    'IM': (30, 144, 255, 100),
    'IB': (165, 42, 42, 100),
    'IIA': (255, 165, 0, 100),
    'IIB': (34, 139, 34, 100)
}

road_widths = {
    'IA': 3.0,
    'IM': 2.5,
    'IB': 2.0,
    'IIA': 1.5,
    'IIB': 1.0
}

# Now loop over renderer classes
for class_ in sym.renderer.classes:
    road_type = class_.values[0]
    if road_type in road_colors:
        class_.symbol.color = road_colors[road_type]
        class_.symbol.width = road_widths[road_type]

# Apply symbology
layer.symbology = sym
arcpy.AddMessage("Symbology applied: colors and widths by road type")
