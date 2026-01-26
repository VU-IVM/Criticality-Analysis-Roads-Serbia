# Standard library
import os
import re
import sys
import warnings
from pathlib import Path

# Third-party - Data and scientific computing
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# Shapely-specific imports for spatial analysis
import shapely
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, snap

# Matplotlib-specific imports for figures
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


data_path = Path('input_files') 
output_folder = Path('intermediate_results')


osm_path = data_path / "SRB.osm.pbf"


#roads_path = data_path / 'DeoniceRSDP-Jul2025..shp'
#gdf = gpd.read_file(roads_path)

import arcpy
import geopandas as gpd
from pathlib import Path

arcpy.env.overwriteOutput = True

input_layer = arcpy.GetParameterAsText(0)

base_tmp = Path(r"C:\Temp\arcgis_tmp")
base_tmp.mkdir(parents=True, exist_ok=True)

gdb_path = base_tmp / "temp.gdb"
if not arcpy.Exists(str(gdb_path)):
    arcpy.management.CreateFileGDB(str(base_tmp), "temp.gdb")

out_fc_name = "roads"
out_fc = gdb_path / out_fc_name

arcpy.management.CopyFeatures(input_layer, str(out_fc))

# ✅ Correct way
gdf = gpd.read_file(str(gdb_path), layer=out_fc_name)

arcpy.AddMessage(f"Loaded {len(gdf)} features into GeoDataFrame")


arcpy.AddMessage(f"INPUT_LAYER = '{input_layer}'")




# Filter for the road categories we want to plot
road_categories = ['IA', 'IM', 'IB', 'IIA', 'IIB']
gdf_filtered = gdf[gdf['kategorija'].isin(road_categories)].copy()

# Define colors based on the screenshot legend
road_colors = {
    'IA': '#8B0000',   # Dark red/maroon
    'IM': '#1E90FF',   # Blue
    'IB': '#A52A2A',   # Dark red/brown (slightly different from IA)
    'IIA': '#FFA500',  # Orange
    'IIB': '#228B22'   # Green
}

# Define line widths - higher tier roads are wider
road_widths = {
    'IA': 3.0,   # Highways - thickest
    'IM': 2.5,   # Main roads
    'IB': 2.0,   # Regional roads
    'IIA': 1.5,  # Secondary roads
    'IIB': 1.0   # Local roads - thinnest
}

# Create figure with high DPI for crisp visuals
fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='white')

# Convert to Web Mercator for plotting with basemap
gdf_mercator = gdf_filtered.to_crs(3857)

# Plot each road category with its specific color and width
for category in road_categories:
    if category in gdf_filtered['kategorija'].values:
        category_roads = gdf_mercator[gdf_mercator['kategorija'] == category]
        category_roads.plot(
            ax=ax,
            color=road_colors[category],
            linewidth=road_widths[category],
            alpha=0.8,
            zorder=road_widths[category]  # Higher tier roads on top
        )

# Add basemap with optimal styling for PNG

# Enhance the plot styling
ax.set_aspect('equal')
ax.axis('off')  # Remove axis for cleaner look

# Create custom legend in upper right corner matching your template style
legend_elements = []
for category in road_categories:
    if category in gdf_filtered['kategorija'].values:
        legend_elements.append(
            Patch(facecolor=road_colors[category], 
                  label=f'{category}', 
                  edgecolor='none')
        )

legend = ax.legend(handles=legend_elements, 
                  title='Road categories', 
                  loc='upper right',
                  fontsize=10,
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  framealpha=0.9,
                  facecolor='white',
                  edgecolor='#cccccc')

# Enhance overall plot appearance
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
plt.savefig(Path("figures2") / 'road_categories.png', dpi=300, bbox_inches='tight')
plt.show()


# Save plot
out_png = r"C:\Temp\arcgis_tmp\roads_plot.png"
fig.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close(fig)


aprx = arcpy.mp.ArcGISProject("CURRENT")
layouts = aprx.listLayouts()

print("Available layouts:")
for i, l in enumerate(layouts):
    print(f"{i}: {l.name}")


#if len(layouts) == 0:
#    arcpy.AddMessage("No layouts found! Add a layout in the project or skip adding plot to layout.")
#else:
#    lyt = layouts[0]
#    lyt_map = lyt.listElements("MAPFRAME_ELEMENT")[0]  # your map frame

#    # Add the PNG we just created
#    lyt_image = lyt.listElements("PICTURE_ELEMENT")
#    lyt.addPicture(out_png)