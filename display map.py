from pathlib import Path
import arcpy

directed_final = (
    Path("intermediate_results") / "PERS_directed_final.shp"
).resolve()

aprx = arcpy.mp.ArcGISProject("CURRENT")
m = aprx.listMaps()[0]

layer = m.addDataFromPath(str(directed_final))
arcpy.AddMessage(f"Directed graph added as layer: {layer.name}")

