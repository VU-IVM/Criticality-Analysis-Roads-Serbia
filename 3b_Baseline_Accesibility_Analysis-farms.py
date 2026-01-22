import numpy as np
import pandas as pd
import geopandas as gpd
import igraph as ig
import shapely
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as cx
from pathlib import Path

data_path = Path('input_files')
figures = Path('figures')

# =============================================================================
# 1. Load network and create graph
# =============================================================================
base_network = gpd.read_parquet(data_path / "base_network_SRB_basins.parquet")

edges = base_network.reindex(['from_id','to_id'] + [x for x in list(base_network.columns) if x not in ['from_id','to_id']], axis=1)
graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:], directed=True)
graph = graph.connected_components().giant()
edges = edges[edges['id'].isin(graph.es['id'])]


# =============================================================================
# 2. Load agricultural data
# =============================================================================
Path_AgriFile = data_path / "1_agriculture_2023_serbia_NEW_FINAL_26092025.xlsm"
DataFrame_Agri = pd.read_excel(Path_AgriFile)

Clean_DataFrame_Agri = DataFrame_Agri.dropna(subset=["latitude", "longitude", "Utilized agricultural land (UAL)"])

geometry = [Point(xy) for xy in zip(Clean_DataFrame_Agri["longitude"], Clean_DataFrame_Agri["latitude"])]

df_agri = gpd.GeoDataFrame(
    Clean_DataFrame_Agri[["Utilized agricultural land (UAL)"]].rename(columns={"Utilized agricultural land (UAL)": "UAL"}),
    geometry=geometry,
    crs="EPSG:4326"
)

# =============================================================================
# 3. Create nodes from edges for spatial matching
# =============================================================================
vertex_lookup = dict(zip(pd.DataFrame(graph.vs['name'])[0], pd.DataFrame(graph.vs['name']).index))

tqdm.pandas()
from_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[0]))
to_id_geom = edges.geometry.progress_apply(lambda x: shapely.Point(x.coords[-1]))

from_dict = dict(zip(edges['from_id'], from_id_geom))
to_dict = dict(zip(edges['to_id'], to_id_geom))

nodes = pd.concat([
    pd.DataFrame.from_dict(to_dict, orient='index', columns=['geometry']),
    pd.DataFrame.from_dict(from_dict, orient='index', columns=['geometry'])
]).drop_duplicates()

nodes['vertex_id'] = nodes.apply(lambda x: vertex_lookup[x.name], axis=1)
nodes = nodes.reset_index()

nodes_sindex = shapely.STRtree(nodes.geometry)

# =============================================================================
# 4. Map agricultural locations to nearest network nodes
# =============================================================================
df_agri['vertex_id'] = df_agri.geometry.progress_apply(
    lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id
).values

# =============================================================================
# 5. Load sinks (borders, ports, rail)
# =============================================================================
path_to_Sinks =data_path / "Borders_Ports_Rail_geocoded.xlsx"

Sinks = pd.read_excel(path_to_Sinks)
Sinks = Sinks.rename(columns={"LON": "Longitude", "LAT": "Latitude", "TYPE OF\nTRAFFIC": "type"})
Sinks['geometry'] = Sinks.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
Sinks['vertex_id'] = Sinks.geometry.apply(lambda x: nodes.iloc[nodes_sindex.nearest(x)].vertex_id).values

# Split by type
Sinks_road = Sinks[Sinks['type'] == 'road']
Sinks_port = Sinks[Sinks['type'] == 'port']
Sinks_rail = Sinks[Sinks['type'] == 'rail']

print(f"Road border crossings: {len(Sinks_road)}")
print(f"Ports: {len(Sinks_port)}")
print(f"Rail terminals: {len(Sinks_rail)}")

# =============================================================================
# 6. Calculate OD matrices by sink type
# =============================================================================
agri_vertices = df_agri['vertex_id'].unique()

def calculate_access_times(graph, origin_vertices, sink_df, sink_name):
    """Calculate average access time from origins to sinks"""
    sink_vertices = sink_df['vertex_id'].unique()
    
    OD_matrix = np.array(graph.distances(
        source=origin_vertices,
        target=sink_vertices,
        weights='fft'
    ))
    OD_matrix[np.isinf(OD_matrix)] = 12
    
    avg_time = np.mean(OD_matrix, axis=1)
    vertex_to_avg = dict(zip(origin_vertices, avg_time))
    
    print(f"\n{sink_name}: {len(sink_vertices)} destinations, global avg = {np.mean(OD_matrix):.2f} hours")
    
    return vertex_to_avg, OD_matrix

# Calculate for each sink type
road_access, OD_road = calculate_access_times(graph, agri_vertices, Sinks_road, "Road borders")
port_access, OD_port = calculate_access_times(graph, agri_vertices, Sinks_port, "Ports")
rail_access, OD_rail = calculate_access_times(graph, agri_vertices, Sinks_rail, "Rail terminals")

# Also calculate combined (all sinks)
all_access, OD_all = calculate_access_times(graph, agri_vertices, Sinks, "All sinks combined")

# Map back to df_agri
df_agri['avg_access_road'] = df_agri['vertex_id'].map(road_access)
df_agri['avg_access_port'] = df_agri['vertex_id'].map(port_access)
df_agri['avg_access_rail'] = df_agri['vertex_id'].map(rail_access)
df_agri['avg_access_all'] = df_agri['vertex_id'].map(all_access)

# =============================================================================
# 7. Summary statistics
# =============================================================================
print("\n" + "="*60)
print("AGRICULTURAL ACCESSIBILITY SUMMARY")
print("="*60)

print(f"\nNumber of agricultural locations: {len(df_agri)}")
print(f"Total UAL (ha): {df_agri['UAL'].sum():,.0f}")

for col, label in [('avg_access_road', 'Road Borders'), 
                   ('avg_access_port', 'Ports'), 
                   ('avg_access_rail', 'Rail Terminals'),
                   ('avg_access_all', 'All Combined')]:
    print(f"\n--- {label} ---")
    print(f"  Mean:   {df_agri[col].mean():.2f} hours")
    print(f"  Median: {df_agri[col].median():.2f} hours")
    print(f"  Std:    {df_agri[col].std():.2f} hours")
    print(f"  Min:    {df_agri[col].min():.2f} hours")
    print(f"  Max:    {df_agri[col].max():.2f} hours")
    print(f"  P10:    {df_agri[col].quantile(0.10):.2f} hours")
    print(f"  P90:    {df_agri[col].quantile(0.90):.2f} hours")

# Category distribution for each type
bins = [0, 0.5, 1, 2, 5, float('inf')]
labels_cat = ['0-0.5h', '0.5-1h', '1-2h', '2-5h', '5h+']

print("\n--- Distribution by Access Time Category ---")
for col, label in [('avg_access_road', 'Road'), 
                   ('avg_access_port', 'Port'), 
                   ('avg_access_rail', 'Rail')]:
    print(f"\n{label}:")
    cat = pd.cut(df_agri[col], bins=bins, labels=labels_cat, right=False)
    for c in labels_cat:
        count = (cat == c).sum()
        pct = count / len(df_agri) * 100
        print(f"  {c}: {count} ({pct:.1f}%)")


# =============================================================================
# 8. Visualization (3 panel map)
# =============================================================================
df_agri_plot = df_agri.to_crs(3857)
Sinks_plot = gpd.GeoDataFrame(Sinks, geometry='geometry', crs="EPSG:4326").to_crs(3857)

bins = [1, 2, 3, 4, 5, float('inf')]
labels_cat = ['1-2', '2-3', '3-4', '4-5', '5+']
colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
color_map = dict(zip(labels_cat, colors))

fig, axes = plt.subplots(1, 3, figsize=(16, 8))

for ax, col, title in zip(axes, 
                          ['avg_access_road', 'avg_access_port', 'avg_access_rail'],
                          ["A","B","C"]):
                          #['Road Border Crossings', 'Ports', 'Rail Terminals']):
    
    df_plot = df_agri_plot.copy()
    df_plot['category'] = pd.cut(df_plot[col], bins=bins, labels=labels_cat, right=False)
    df_plot['category'] = df_plot['category'].astype('object')
    
    for category, color in color_map.items():
        data = df_plot[df_plot['category'] == category]
        if not data.empty:
            data.plot(ax=ax, color=color, legend=False, linewidth=0.1, edgecolor='grey', markersize=50)
    
    # Plot relevant sinks
    if 'road' in col:
        sink_subset = Sinks_plot[Sinks_plot['type'] == 'road']
        marker = '^'
    elif 'port' in col:
        sink_subset = Sinks_plot[Sinks_plot['type'] == 'port']
        marker = 's'
    else:
        sink_subset = Sinks_plot[Sinks_plot['type'] == 'rail']
        marker = 'o'
    
    sink_subset.plot(ax=ax, color='black', markersize=100, marker=marker)
    
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    # Add letter label
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=20, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.axis('off')
    #ax.set_title(title, fontsize=18, fontweight='bold')

# Shared legend
legend_patches = [mpatches.Patch(color=color, label=f'{label} hours') 
                  for label, color in zip(labels_cat, colors)]
legend_patches.extend([
    Line2D([0], [0], marker='^', color='black', lw=0, label='Road Borders', markersize=12),
    Line2D([0], [0], marker='s', color='black', lw=0, label='Ports', markersize=12),
    Line2D([0], [0], marker='o', color='black', lw=0, label='Rail Terminals', markersize=12),
])

fig.legend(handles=legend_patches, loc='lower center', ncol=8, fontsize=12, 
           title='Average Access Time', title_fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig(figures / 'agriculture_access_by_type.png', dpi=200, bbox_inches='tight')
plt.show()