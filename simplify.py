"""Network representation and utilities

Ben Dickens, Elco Koks & Tom Russell
"""
import os,sys
import re
import shapely

import numpy as np
import pandas as pd
import geopandas as gpd

import pyproj

from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from itertools import compress

# path to python scripts
sys.path.append(os.path.join('..','src','trails'))

import osm_flex.download as dl
import osm_flex.extract as ex

pd.options.mode.chained_assignment = None  

#data_path = os.path.join('..','data')
data_path = (Path(__file__).parent.absolute().parent.absolute().parent.absolute())

road_types = ['primary','trunk','motorway','motorway_link','trunk_link','primary_link','secondary','secondary_link','tertiary','tertiary_link']

class Network():
    """A Network is composed of nodes (points in space) and edges (lines)

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame, optional
    edges : geopandas.GeoDataFrame, optional

    Attributes
    ----------
    nodes : geopandas.GeoDataFrame
    edges : geopandas.GeoDataFrame

    """

    def __init__(self, nodes=None, edges=None):
        """
        Parameters
        ----------
        nodes : geopandas.GeoDataFrame, optional
        edges : geopandas.GeoDataFrame, optional

        """
        if nodes is None:
            nodes = gpd.GeoDataFrame()
        self.nodes = nodes

        if edges is None:
            edges = gpd.GeoDataFrame()
        self.edges = edges


def add_ids(network, id_col='id'):
    """Add or replace an id column with ascending ids.

    The ids are represented as int64s for easier conversion to numpy arrays.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).
        id_col (str, optional): The name of the id column. Defaults to 'id'.

    Returns:
        Network: A network with added or replaced id column.
    """
    nodes = network.nodes.copy()
    if not nodes.empty:
        nodes = nodes.reset_index(drop=True)

    edges = network.edges.copy()
    if not edges.empty:
        edges = edges.reset_index(drop=True)

    nodes[id_col] = range(len(nodes))
    edges[id_col] = range(len(edges))

    return Network(
        nodes=nodes,
        edges=edges
    )

def add_topology(network, id_col='id'):
    """Add or replace from_id, to_id to edges.

    This function adds or replaces the 'from_id' and 'to_id' columns in the edges of the network
    based on the spatial proximity of edge endpoints to nodes.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).
        id_col (str, optional): The name of the id column. Defaults to 'id'.

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
            with 'from_id' and 'to_id' columns added or replaced.
    """    

    from_ids = []
    to_ids = []
    bugs = []
    
    # Create a spatial index for efficient node lookup
    sindex = shapely.STRtree(network.nodes.geometry)
    
    # Iterate through edges to determine 'from_id' and 'to_id'
    for edge in tqdm(network.edges.itertuples(), desc="topology", total=len(network.edges)):
        start, end = line_endpoints(edge.geometry)

        try: 
            # Find the nearest node to the start point of the edge
            start_node = nearest_node(start, network.nodes, sindex)
            from_ids.append(start_node[id_col])
        except:
            bugs.append(edge.id)
            from_ids.append(-1)
        
        try:
            # Find the nearest node to the end point of the edge
            end_node = nearest_node(end, network.nodes, sindex)
            to_ids.append(end_node[id_col])
        except:
            bugs.append(edge.id)
            to_ids.append(-1)

    # Copy the edges and add 'from_id' and 'to_id' columns
    edges = network.edges.copy()
    edges['from_id'] = from_ids
    edges['to_id'] = to_ids

    # Remove edges associated with bugs (errors during node lookup)
    edges = edges.loc[~(edges.id.isin(list(bugs)))].reset_index(drop=True)

    return Network(
        nodes=network.nodes,
        edges=edges
    )

def get_endpoints(network):
    """Get nodes for each edge endpoint.

    This function extracts nodes representing each endpoint of edges in the network.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        DataFrame: A DataFrame containing nodes representing each edge endpoint.
    """    
    endpoints = []
    
    # Iterate through edges to collect endpoints
    for edge in tqdm(network.edges.itertuples(), desc="endpoints", total=len(network.edges)):
        if edge.geometry is None:
            continue
        
        # Check if the geometry is a MULTILINESTRING
        if shapely.get_type_id(edge.geometry) == 5:
            for line in edge.geometry.geoms:
                start, end = line_endpoints(line)
                endpoints.append(start)
                endpoints.append(end)
        else:
            start, end = line_endpoints(edge.geometry)
            endpoints.append(start)
            endpoints.append(end)

    # Create a DataFrame to match the nodes' geometry column name
    return matching_df_from_geoms(network.nodes, endpoints)

def add_endpoints(network):
    """Add nodes at line endpoints.

    This function adds nodes at the endpoints of each edge in the network.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
            with added nodes at line endpoints.
    """

    # Get nodes representing each edge endpoint
    endpoints = get_endpoints(network)
    
    # Concatenate and deduplicate nodes
    nodes = concat_dedup([network.nodes, endpoints])

    return Network(
        nodes=nodes,
        edges=network.edges
    )


def split_multilinestrings(network):
    """Create multiple edges from any MultiLineString edge.

    This function ensures that edge geometries are all LineStrings, duplicating attributes over any
    created multi-edges.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
            with multiple edges created from any MultiLineString edge.
    """

    simple_edge_attrs = []
    simple_edge_geoms = []
    edges = network.edges
    
    # Iterate through edges to split MultiLineString edges
    for edge in tqdm(edges.itertuples(index=False), desc="split_multi", total=len(edges)):
        if shapely.get_type_id(edge.geometry) == 5:
            # Extract individual LineStrings from MultiLineString
            edge_parts = [x for x in shapely.geometry.get_geometry(edge, shapely.geometry.get_num_geometries(edge))]
        else:
            edge_parts = [edge.geometry]

        for part in edge_parts:
            simple_edge_geoms.append(part)

        # Duplicate attributes for each new edge
        attrs = gpd.GeoDataFrame([edge] * len(edge_parts))
        simple_edge_attrs.append(attrs)

    simple_edge_geoms = gpd.GeoDataFrame(simple_edge_geoms, columns=['geometry'])
    edges = pd.concat(simple_edge_attrs, axis=0).reset_index(drop=True).drop('geometry', axis=1)
    edges = pd.concat([edges, simple_edge_geoms], axis=1)

    return Network(
        nodes=network.nodes,
        edges=edges
    )


def merge_multilinestrings(network):
    """Try to merge all multilinestring geometries into linestring geometries.

    This function attempts to merge all Multilinestring geometries within the edges of the network into
    Linestring geometries.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
            with attempted merging of Multilinestring geometries into Linestring geometries.
    """    
    edges = network.edges.copy()
    
    # Apply the merge_multilinestring function to each geometry in the edges
    edges['geometry'] = edges.geometry.apply(lambda x: merge_multilinestring(x))
    
    return Network(
        edges=edges,
        nodes=network.nodes
    )


def merge_multilinestring(geom):
    """Merge a MultiLineString to LineString.

    This function attempts to merge a MultiLineString geometry into a LineString geometry.

    Args:
        geom (shapely.geometry): A shapely geometry, most likely a linestring or a multilinestring.

    Returns:
        geom (shapely.geometry): A shapely linestring geometry if merge was successful.
        If not, it returns the input geometry.
    """        
    # Check if the geometry is a MULTILINESTRING
    if shapely.get_type_id(geom) == 5:
        # Attempt to merge the Multilinestring
        geom_inb = shapely.line_merge(geom)
        
        # Check if the merged geometry is a ring
        if geom_inb.is_ring:
            # Still something to fix if desired
            return geom_inb
        else:
            return geom_inb
    else:
        return geom


def snap_nodes(network, threshold=None):
    """Move nodes (within threshold) to edges.

    This function moves nodes within a specified threshold distance to the nearest edge in the network.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).
        threshold ([type], optional): The maximum distance within which nodes will be snapped to edges.
            Defaults to None.

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
            with nodes potentially moved to edges within the specified threshold.
    """    
    def snap_node(node):
        # Find the nearest point on edges for each node
        snap = nearest_point_on_edges(node.geometry, network.edges)
        distance = snap.distance(node.geometry)
        
        # Check if the distance exceeds the threshold
        if threshold is not None and distance > threshold:
            snap = node.geometry
        return snap

    # Snap nodes to edges
    snapped_geoms = network.nodes.apply(snap_node, axis=1)
    geom_col = geometry_column_name(network.nodes)
    
    # Create a new nodes DataFrame with updated geometries
    nodes = pd.concat([
        network.nodes.drop(geom_col, axis=1),
        gpd.GeoDataFrame(snapped_geoms, columns=[geom_col])
    ], axis=1)

    return Network(
        nodes=nodes,
        edges=network.edges
    )

def link_nodes_to_edges_within(network, distance, condition=None, tolerance=1e-9):
    """Link nodes to all edges within some distance.

    This function creates new nodes at points on edges within a specified distance from existing nodes
    and adds connecting edges between the existing nodes and the new nodes.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).
        distance ([type]): The maximum distance within which edges are considered for linking nodes.
        condition ([type], optional): A function that takes a node and an edge and returns True
            if the node should be linked to the edge, otherwise False. Defaults to None.
        tolerance ([type], optional): A small distance threshold for splitting edges at new node locations.
            Defaults to 1e-9.

    Returns:
        [type]: A network composed of nodes (points in space) and edges (lines)
            with new nodes linked to edges within the specified distance.
    """       
    new_node_geoms = []
    new_edge_geoms = []
    
    # Iterate over existing nodes
    for node in tqdm(network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)):
        # Find edges within the specified distance from the current node
        edges = edges_within(node.geometry, network.edges, distance)
        
        # Iterate over edges within the distance
        for edge in edges.itertuples():
            # Check the condition function, if provided
            if condition is not None and not condition(node, edge):
                continue
            
            # Add new node at the point on the edge nearest to the current node
            point = nearest_point_on_line(node.geometry, edge.geometry)
            if point != node.geometry:
                new_node_geoms.append(point)
                
                # Add new edge linking the existing node and the new node
                line = shapely.LineString([node.geometry, point])
                new_edge_geoms.append(line)

    # Create DataFrames from new node and edge geometries
    new_nodes = matching_df_from_geoms(network.nodes, new_node_geoms)
    all_nodes = concat_dedup([network.nodes, new_nodes])

    new_edges = matching_df_from_geoms(network.edges, new_edge_geoms)
    all_edges = concat_dedup([network.edges, new_edges])

    # Split edges as necessary after new node creation
    unsplit = Network(
        nodes=all_nodes,
        edges=all_edges
    )
    return split_edges_at_nodes(unsplit, tolerance)

def link_nodes_to_nearest_edge(network, condition=None):
    """Link nodes to all edges within some distance.

    Args:
        network (Network): A network composed of nodes (points in space) and edges (lines).
        condition (function, optional): A condition function that takes a node and an edge and returns a boolean.
            If specified, only links satisfying this condition will be created. Defaults to None.

    Returns:
        Network: A network composed of nodes (points in space) and edges (lines) with additional links.
    """    
    new_node_geoms = []
    new_edge_geoms = []

    # Iterate through nodes in the network
    for node in tqdm(network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)):
        # For each node, find edges within
        edge = nearest_edge(node.geometry, network.edges)
        
        # Check the condition, if specified, and continue if it's not satisfied
        if condition is not None and not condition(node, edge):
            continue
        
        # Add nodes at points-nearest
        point = nearest_point_on_line(node.geometry, edge.geometry)
        
        # Check if the point is different from the original node's geometry
        if point != node.geometry:
            new_node_geoms.append(point)
            
            # Add edges linking
            line = shapely.LineString([node.geometry, point])
            new_edge_geoms.append(line)

    # Create DataFrames from the new node and edge geometries
    new_nodes = matching_df_from_geoms(network.nodes, new_node_geoms)
    all_nodes = concat_dedup([network.nodes, new_nodes])

    new_edges = matching_df_from_geoms(network.edges, new_edge_geoms)
    all_edges = concat_dedup([network.edges, new_edges])

    # Combine all nodes and edges into an unsplit network
    unsplit = Network(
        nodes=all_nodes,
        edges=all_edges
    )

    # Split edges as necessary after new node creation
    return split_edges_at_nodes(unsplit)


def find_roundabouts(network):
    """Methods to find roundabouts

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines)

    Returns:
        roundabouts (list): Returns the edges that can be identified as roundabouts
    """    
    roundabouts = []
    for edge in network.edges.itertuples():
        if shapely.predicates.is_ring(edge.geometry): roundabouts.append(edge)
    return roundabouts

def clean_roundabouts(network):
    """Methods to clean roundabouts and junctions should be done before
        splitting edges at nodes to avoid logic conflicts

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines)

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines)
    """    

    # Create a spatial index for efficient spatial queries
    sindex = shapely.STRtree(network.edges['geometry'])
    
    # Copy the edges for processing
    edges = network.edges
    new_geom = network.edges
    new_edge = []
    remove_edge = []
    new_edge_id = []
    
    # Extract attributes excluding 'geometry' and 'osm_id'
    attributes = [x for x in network.edges.columns if x not in ['geometry', 'osm_id']]

    # Find roundabouts in the network
    roundabouts = find_roundabouts(network)
    
    # Iterate through each roundabout and process edges
    for roundabout in roundabouts:

        # Find the centroid of the roundabout
        round_centroid = shapely.constructive.centroid(roundabout.geometry)
        
        # Mark the roundabout for removal
        remove_edge.append(roundabout.Index)

        # Find edges that intersect with the roundabout
        edges_intersect = _intersects(roundabout.geometry, network.edges['geometry'], sindex)
        
        # Drop the roundabout from series to prevent snapping on it
        edges_intersect.drop(roundabout.Index, inplace=True)
        
        # Process each edge that intersects with the roundabout
        for e in edges_intersect.items():
            edge = edges.iloc[e[0]]
            start = shapely.get_point(e[1], 0)
            end = shapely.get_point(e[1], -1)
            
            # Check which end of the edge is closer to the roundabout centroid
            first_co_is_closer = shapely.measurement.distance(end, round_centroid) > shapely.measurement.distance(start, round_centroid) 
            
            # Get coordinates of the edge and roundabout centroid
            co_ords = shapely.coordinates.get_coordinates(edge.geometry)
            centroid_co = shapely.coordinates.get_coordinates(round_centroid)
            
            # Create a new line by snapping the edge to the roundabout centroid
            if first_co_is_closer: 
                new_co = np.concatenate((centroid_co, co_ords))
            else:
                new_co = np.concatenate((co_ords, centroid_co))
            snap_line = shapely.linestrings(new_co)

            # Check if the edge already exists in the new_edge list
            if edge.osm_id in new_edge_id:
                a = []
                counter = 0
                
                # Find the existing edge in the list
                for x in new_edge:
                    if x[0] == edge.osm_id:
                        a = counter
                        break
                    counter += 1
                
                # Remove the existing edge from the list
                double_edge = new_edge.pop(a)
                start = shapely.get_point(double_edge[-1], 0)
                end = shapely.get_point(double_edge[-1], -1)
                first_co_is_closer = shapely.measurement.distance(end, round_centroid) > shapely.measurement.distance(start, round_centroid) 
                co_ords = shapely.coordinates.get_coordinates(double_edge[-1])
                
                # Create a new line for the existing edge with updated coordinates
                if first_co_is_closer: 
                    new_co = np.concatenate((centroid_co, co_ords))
                else:
                    new_co = np.concatenate((co_ords, centroid_co))
                snap_line = shapely.linestrings(new_co)
                
                # Add the updated edge back to the list
                new_edge.append([edge.osm_id] + list(edge[list(attributes)]) + [snap_line])

            else:
                # Add the new edge to the list
                new_edge.append([edge.osm_id] + list(edge[list(attributes)]) + [snap_line])
                new_edge_id.append(edge.osm_id)
            
            # Mark the original edge for removal
            remove_edge.append(e[0])

    # Create a new DataFrame with the processed edges
    new = gpd.GeoDataFrame(new_edge, columns=['osm_id'] + attributes + ['geometry'])
    
    # Remove the marked edges from the original DataFrame
    dg = network.edges.loc[~network.edges.index.isin(remove_edge)]
    
    # Concatenate the original and new DataFrames to create the final network
    ges = pd.concat([dg, new]).reset_index(drop=True)

    # Return the updated network object
    return Network(edges=ges, nodes=network.nodes)


def find_hanging_nodes(network):
    """Finds and returns a DataFrame of nodes with degree 1.
    
    Note that not all nodes with degree 1 are necessarily "hanging" in a network context.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        DataFrame: A DataFrame containing nodes with degree 1.
    """
    # Find indices of nodes with degree 1
    hang_index = np.where(network.nodes['degree'] == 1)
    
    # Return a DataFrame of hanging nodes
    return network.nodes.iloc[hang_index]


def add_distances(network):
    """This method adds a distance column to edges in the network using Shapely and PyProj.
    The distance is calculated in meters and is based on the geometry of the edges.

    Args:
        network (Network): A network composed of nodes (points in space) and edges (lines).

    Returns:
        Network: A new network object with the added distance column in the edges.
    """    

    # Find the coordinate reference system (CRS) of the current DataFrame and an arbitrary point (lat, lon) for the new CRS
    current_crs = "epsg:4326"
    
    # The following commented-out approach does not work in all cases
    # current_crs = str(network.edges.crs.values[0])
    
    lat = shapely.get_y(network.nodes['geometry'].iloc[0])
    lon = shapely.get_x(network.nodes['geometry'].iloc[0])
    
    # Calculate an approximate CRS using a formula based on the first node's latitude and longitude
    approximate_crs = "epsg:" + str(int(32700 - np.round((45 + lat) / 90, 0) * 100 + np.round((183 + lon) / 6, 0)))

    # Convert coordinates to the new CRS using PyProj
    geometries = network.edges['geometry']
    coords = shapely.get_coordinates(geometries)
    transformer = pyproj.Transformer.from_crs(current_crs, approximate_crs, always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    
    # Set the new coordinates to the geometry of the edges
    result = shapely.set_coordinates(geometries.copy(), np.array(new_coords).T)
    
    # Calculate the length (distance) of each edge using Shapely
    dist = shapely.length(result)
    
    # Copy the edges DataFrame and add the distance column
    edges = network.edges.copy()
    edges['distance'] = dist
    
    # Create and return a new Network object with the updated edges
    return Network(nodes=network.nodes, edges=edges)

def add_travel_time(network):
    """Add a travel time column to network edges. Time is measured in hours.

    If the 'distance' column is not already present in the network nodes, it calculates distances and adds them.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines) with added travel time information.
    """    
    # Check if the 'distance' column is already present in network nodes, and add it if not
    if 'distance' not in network.nodes.columns:
        network = add_distances(network)

    # Dictionary of speed limits for different road types in meters per hour
    speed_d = {
        'motorway': 80000,
        'motorway_link': 65000,
        'trunk': 60000,
        'trunk_link': 50000,
        'primary': 50000,  # meters per hour
        'primary_link': 40000,
        'secondary': 40000,  # meters per hour
        'secondary_link': 30000,
        'tertiary': 30000,
        'tertiary_link': 20000,
        'unclassified': 20000,
        'service': 20000,
        'residential': 20000,  # meters per hour
    }

    def calculate_time(edge):
        """Calculate travel time for a given edge.

        Args:
            edge (pd.Series): A Pandas Series representing an edge in the network.

        Returns:
            float: Travel time in hours for the given edge.
        """
        try:
            return edge['distance'] / (edge['maxspeed'] * 1000)  # meters per hour
        except:
            return edge['distance'] / speed_d.get('unclassified')

    # Apply the calculate_time function to each row in the edges DataFrame and create a new 'time' column
    network.edges['time'] = network.edges.apply(calculate_time, axis=1)
    
    # Return the updated network object with added travel time information
    return network


def calculate_degree(network):
    """Calculates the connectivity degree of nodes based on 'from_id' and 'to_id' indices in the network.

    It is not recommended to call this method after removing nodes or edges without first resetting the indices.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        np.ndarray: An array representing the connectivity degree of nodes in the network.
    """    
    # The number of nodes (from index) to use as the number of bins
    ndC = len(network.nodes.index)
    
    # Check if the network structure seems irregular, and print a warning if necessary
    if ndC - 1 > max(network.edges.from_id) and ndC - 1 > max(network.edges.to_id): 
        print("Warning: calculate_degree may encounter issues with the network structure.")
    
    # Calculate the connectivity degree using numpy bincount
    return np.bincount(network.edges['from_id'], minlength=ndC) + np.bincount(network.edges['to_id'], minlength=ndC)


def add_degree(network):
    """Calculates and adds a 'degree' column to the node dataframe.

    The degree of a node is the number of edges connected to it.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network with the 'degree' column added to the node dataframe.
    """
    # Calculate the degree of each node
    degree = calculate_degree(network)

    # Add the 'degree' column to the node dataframe
    network.nodes['degree'] = degree

    return network


def drop_hanging_nodes(network, tolerance=0.005):
    """Drops single-degree nodes and their associated edges based on a distance threshold.
    
    This primarily occurs when a road is connected to residential areas, often involving link roads that no longer serve a purpose.

    Args:
        network (Network): A network composed of nodes (points in space) and edges (lines).
        tolerance (float, optional): The maximum allowed distance from hanging nodes to the network. Defaults to 0.005.

    Returns:
        Network: A modified network with dropped single-degree nodes and associated edges.
    """    
    # Calculate degrees if not already present in the network nodes
    if 'degree' not in network.nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = network.nodes['degree'].to_numpy()

    # Find nodes with degree 1
    hangNodes = np.where(deg == 1)

    # Copy edges for processing
    ed = network.edges.copy()
    
    # Extract 'to_id' and 'from_id' arrays
    to_ids = ed['to_id'].to_numpy()
    from_ids = ed['from_id'].to_numpy()

    # Check if each edge connects to a degree 1 node
    hangTo = np.isin(to_ids, hangNodes)
    hangFrom = np.isin(from_ids, hangNodes)

    # Find indices of edges connecting degree 1 nodes
    eInd = np.hstack((np.nonzero(hangTo), np.nonzero(hangFrom)))

    # Extract edges that connect degree 1 nodes
    degEd = ed.iloc[np.sort(eInd[0])]

    # List to store edge IDs to be dropped
    edge_id_drop = []

    # Iterate through edges connecting degree 1 nodes
    for d in degEd.itertuples():
        dist = shapely.measurement.length(d.geometry)
        
        # If the edge is shorter than the tolerance, add its ID to the drop list and update involved node degrees
        if dist < tolerance:
            edge_id_drop.append(d.id)
            deg[d.from_id] -= 1
            deg[d.to_id] -= 1
        
        # Drop disconnected edges; some may still persist since we have not merged yet
        if deg[d.from_id] == 1 and deg[d.to_id] == 1: 
            edge_id_drop.append(d.id)
            deg[d.from_id] -= 1
            deg[d.to_id] -= 1

    # Remove dropped edges from the edges DataFrame
    edg = ed.loc[~(ed.id.isin(edge_id_drop))].reset_index(drop=True)

    # Extract and drop 'id' column from the temporary DataFrame aa
    aa = ed.loc[ed.id.isin(edge_id_drop)]
    aa.drop(labels=['id'], axis=1, inplace=True)

    # Add a new 'id' column to the remaining edges DataFrame
    edg['id'] = range(len(edg))

    # Copy the nodes DataFrame for processing
    n = network.nodes.copy()

    # Update the 'degree' column in the nodes DataFrame
    n['degree'] = deg

    # Degree 0 Nodes are cleaned in the merge_2 method
    # x = n.loc[n.degree==0]
    # nod = n.loc[n.degree > 0].reset_index(drop=True)
    
    # Return the modified network object
    return Network(nodes=n, edges=edg)


def merge_edges(network, print_err=False):
    """Removes all degree 2 nodes and merges their associated edges.

    The method traverses edges and nodes in both directions until a node of 
    degree !=2 is found, stopping in that direction. It resets the geometry and 
    from/to ids for the merged edge, deletes the nodes and edges traversed.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).
        print_err (bool, optional): If True, print an error message for edges that fail to merge. Defaults to False.

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines) after merging.
    """    
    net = network
    nod = net.nodes.copy()
    edg = net.edges.copy()
    optional_cols = edg.columns.difference(['osm_id','geometry','from_id','to_id','id'])
    edg_sindex = shapely.STRtree(network.edges.geometry)
    if 'degree' not in network.nodes.columns:
        deg = calculate_degree(network)
    else: deg = nod['degree'].to_numpy()

    #For the 0.002s speed up, alternatively do a straightforward loc[degree==2]
    degree2 = np.where(deg==2)

    #n2: is the set of all node IDs that are degree 2
    n2 = set((nod['id'].iloc[degree2]))

    #nodes
    nodGeom = nod['geometry']
    eIDtoRemove =[]

    pbar = tqdm(total=len(n2))
    while n2:   
        newEdge = []
        info_first_edge = []
        possibly_delete = []
        pos_0_deg = []
        nodeID = n2.pop()
        pos_0_deg.append(nodeID)
        #Co-ordinates of current node
        node_geometry = nodGeom[nodeID]
        eID = set(edg_sindex.query(node_geometry,predicate='intersects'))
        #Find the nearest 2 edges, unless there is an error in the dataframe
        #this will return the connected edges using spatial indexing
        if len(eID) > 2: edgePath1, edgePath2 = find_closest_2_edges(eID,edg,node_geometry)
        elif len(eID) < 2: 
            continue
        else: 
            edgePath1 = edg.iloc[eID.pop()]
            edgePath2 = edg.iloc[eID.pop()] 
        #For the two edges found, identify the next 2 nodes in either direction    
        nextNode1 = edgePath1.to_id if edgePath1.from_id==nodeID else edgePath1.from_id
        nextNode2 = edgePath2.to_id if edgePath2.from_id==nodeID else edgePath2.from_id
        if nextNode1==nextNode2: continue
        possibly_delete.append(edgePath2.id)
        #At the moment the first edge information is used for the merged edge
        info_first_edge = edgePath1.id
        newEdge.append(edgePath1.geometry)
        newEdge.append(edgePath2.geometry)
        #While the next node along the path is degree 2 keep traversing
        while deg[nextNode1] == 2:
            if nextNode1 in pos_0_deg: break
            nextNode1Geom = nodGeom[nextNode1]
            eID = set(edg_sindex.query(nextNode1Geom,predicate='intersects'))
            eID.discard(edgePath1.id)
            try:
                edgePath1 = min([edg.iloc[match_idx] for match_idx in eID],
                key= lambda match: shapely.distance(nextNode1Geom,(match.geometry)))
            except: 
                continue
            pos_0_deg.append(nextNode1)
            n2.discard(nextNode1)
            nextNode1 = edgePath1.to_id if edgePath1.from_id==nextNode1 else edgePath1.from_id
            newEdge.append(edgePath1.geometry)
            possibly_delete.append(edgePath1.id)

        while deg[nextNode2] == 2:
            if nextNode2 in pos_0_deg: break
            nextNode2Geom = nodGeom[nextNode2]
            eID = set(edg_sindex.query(nextNode2Geom,predicate='intersects'))
            eID.discard(edgePath2.id)
            try:
                edgePath2 = min([edg.iloc[match_idx] for match_idx in eID],
                key= lambda match: shapely.distance(nextNode2Geom,(match.geometry)))
            except: continue
            pos_0_deg.append(nextNode2)
            n2.discard(nextNode2)
            nextNode2 = edgePath2.to_id if edgePath2.from_id==nextNode2 else edgePath2.from_id
            newEdge.append(edgePath2.geometry)
            possibly_delete.append(edgePath2.id)
        #Update the information of the first edge
        new_merged_geom = shapely.line_merge(shapely.multilinestrings([x for x in newEdge]))
        if shapely.get_type_id(new_merged_geom) == 1: 
            edg.at[info_first_edge,'geometry'] = new_merged_geom
            if nodGeom[nextNode1]==shapely.get_point(new_merged_geom,0):
                edg.at[info_first_edge,'from_id'] = nextNode1
                edg.at[info_first_edge,'to_id'] = nextNode2
            else: 
                edg.at[info_first_edge,'from_id'] = nextNode2
                edg.at[info_first_edge,'to_id'] = nextNode1
            eIDtoRemove += possibly_delete
            possibly_delete.append(info_first_edge)
            for x in pos_0_deg:
                deg[x] = 0
            mode_edges = edg.loc[edg.id.isin(possibly_delete)]
            edg.loc[info_first_edge,optional_cols] = mode_edges[optional_cols].mode().iloc[0].values
        else:
            if print_err: print("Line", info_first_edge, "failed to merge, has shapely type ", shapely.get_type_id(edg.at[info_first_edge,'geometry']))

        pbar.update(1)
    
    pbar.close()
    edg = edg.loc[~(edg.id.isin(eIDtoRemove))].reset_index(drop=True)

    #We remove all degree 0 nodes, including those found in dropHanging
    n = nod.loc[nod.degree > 0].reset_index(drop=True)
    return Network(nodes=n,edges=edg)


def find_closest_2_edges(edgeIDs, edges, nodGeometry):
    """Returns the 2 edges connected to the current node

    Args:
        edgeIDs ([type]): [description]
        nodeID ([type]): [description]
        edges ([type]): [description]
        nodGeometry ([type]): [description]

    Returns:
        [type]: [description]
    """    
    edgePath1 = min([edges.iloc[match_idx] for match_idx in edgeIDs],
            key=lambda match: shapely.distance(nodGeometry,match.geometry))
    edgeIDs.remove(edgePath1.id)
    edgePath2 = min([edges.iloc[match_idx] for match_idx in edgeIDs],
            key=lambda match:  shapely.distance(nodGeometry,match.geometry))
    return edgePath1, edgePath2

def geometry_column_name(df):
    """Get geometry column name, fall back to 'geometry'

    Args:
        df (pandas.DataFrame): [description]

    Returns:
        geom_col (string): [description]
    """    
    try:
        geom_col = df.geometry.name
    except AttributeError:
        geom_col = 'geometry'
    return geom_col

def matching_df_from_geoms(df, geoms):
    """Create a geometry-only DataFrame with column name to match an existing DataFrame

    Args:
        df (pandas.DataFrame): [description]
        geoms (numpy.array): numpy array with shapely geometries

    Returns:
        [type]: [description]
    """    
    geom_col = geometry_column_name(df)
    return gpd.GeoDataFrame(geoms, columns=[geom_col])

def concat_dedup(dfs):
    """Concatenate a list of GeoDataFrames, dropping duplicate geometries
    - note: repeatedly drops indexes for deduplication to work

    Args:
        dfs ([type]): [description]

    Returns:
        [type]: [description]
    """    
    cat = pd.concat(dfs, axis=0, sort=False)
    cat.reset_index(drop=True, inplace=True)
    cat_dedup = drop_duplicate_geometries(cat)
    cat_dedup.reset_index(drop=True, inplace=True)
    return cat_dedup

def node_connectivity_degree(node, network):
    """Get the degree of connectivity for a node.

    Args:
        node ([type]): [description]
        network (class): A network composed of nodes (points in space) and edges (lines)

    Returns:
        [type]: [description]
    """    
    return len(
            network.edges[
                (network.edges.from_id == node) | (network.edges.to_id == node)
            ]
    )

def drop_duplicate_geometries(gdf):
    """Drop duplicate geometries from a dataframe

    Convert to wkb so drop_duplicates will work as discussed 
    in https://github.com/geopandas/geopandas/issues/521

    Args:
        df (pandas.DataFrame): [description]
        keep (str, optional): [description]. Defaults to 'first'.

    Returns:
        [type]: [description]
    """    

    gdf = gdf.reset_index(drop=True)

    geom_wkb = gdf["geometry"].apply(lambda geom: geom.wkb)

    return gdf.loc[geom_wkb.drop_duplicates().index].reset_index(drop=True)

def nearest_point_on_edges(point, edges):
    """Find nearest point on edges to a point

    Args:
        point (shapely.geometry): [description]
        edges (network.edges): [description]

    Returns:
        [type]: [description]
    """    
    edge = nearest_edge(point, edges)
    snap = nearest_point_on_line(point, edge.geometry)
    return snap

def nearest_node(point, nodes,sindex):
    """Find nearest node to a point

    Args:
        point *shapely.geometry): [description]
        nodes (network.nodes): [description]
        sindex ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return nearest(point, nodes,sindex)

def nearest_edge(point, edges,sindex):
    """Find nearest edge to a point

    Args:
        point (shapely.geometry): [description]
        edges (network.edges): [description]
        sindex ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return nearest(point, edges,sindex)

def nearest(geom, df,sindex):
    """Find the element of a DataFrame nearest a geometry

    Args:
        geom (shapely.geometry): [description]
        df (pandas.DataFrame): [description]
        sindex ([type]): [description]

    Returns:
        [type]: [description]
    """    
    matches_idx = sindex.query(geom)
    nearest_geom = min(
        [df.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: shapely.measurement.distance(match.geometry,geom)
    )
    return nearest_geom

def edges_within(point, edges, distance):
    """Find edges within a distance of point

    Args:
        point (shapely.geometry): [description]
        edges (network.edges): [description]
        distance ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return d_within(point, edges, distance)

def d_within(geom, df, distance):
    """Find the subset of a DataFrame within some distance of a shapely geometry

    Args:
        geom (shapely.geometry): [description]
        df (pandas.DataFrame): [description]
        distance ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return _intersects(geom, df, distance)

def _intersects(geom, df, sindex,tolerance=1e-9):
    """[summary]

    Args:
        geom (shapely.geometry): [description]
        df ([type]): [description]
        sindex ([type]): [description]
        tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns:
        [type]: [description]
    """    
    buffer = shapely.buffer(geom,tolerance)
    if shapely.is_empty(buffer):
        # can have an empty buffer with too small a tolerance, fallback to original geom
        buffer = geom
    try:
        return _intersects_df(buffer, df,sindex)
    except: 
        # can exceptionally buffer to an invalid geometry, so try re-buffering
        buffer = shapely.buffer(geom,0)
        return _intersects_df(buffer, df,sindex)
  
def _intersects_df(geom, df,sindex):
    """[summary]

    Args:
        geom ([type]): [description]
        df ([type]): [description]
        sindex ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return df[sindex.query(geom,'intersects')]

def intersects(geom, df, sindex, tolerance=1e-9):
    """Find the subset of a GeoDataFrame intersecting with a shapely geometry

    Args:
        geom ([type]): [description]
        df ([type]): [description]
        sindex ([type]): [description]
        tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns:
        [type]: [description]
    """    
    return _intersects(geom, df, sindex, tolerance)

def nodes_intersecting(line,nodes,sindex,tolerance=1e-9):
    """Find nodes intersecting line

    Args:
        line ([type]): [description]
        nodes ([type]): [description]
        sindex ([type]): [description]
        tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns:
        [type]: [description]
    """    
    return intersects(line, nodes,sindex, tolerance)

def line_endpoints(line):
    """Return points at first and last vertex of a line

    Args:
        line ([type]): [description]

    Returns:
        [type]: [description]
    """    
    start = shapely.get_point(line,0)
    end = shapely.get_point(line,-1)
    return start, end
    
def snap_line(line, points, tolerance=1e-9):
    """Snap a line to points within tolerance, inserting vertices as necessary

    Args:
        line (shapely.geometry): [description]
        points (shapely.geometry): [description]
        tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns:
        [type]: [description]
    """    
    if  shapely.get_type_id(line.geometry) == 0:
        if shapely.distance(point,line) < tolerance:
            line = shapely.snap(line, points, tolerance=1e-9)
    elif shapely.get_type_id(line.geometry) == 4:
        points = [point for point in points if shapely.distance(point,line) < tolerance]
        for point in points:
            line = shapely.snap(line, points, tolerance=1e-9)
    return line

def nearest_point_on_line(point, line):
    """Return the nearest point on a line

    Args:
        point (shapely.geometry): [description]
        line (shapely.geometry): [description]

    Returns:
        [type]: [description]
    """    
    return line.interpolate(line.project(point))

def reset_ids(network):
    """Reset the ids of the nodes and edges, updating references in the edge table.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines) with updated ids.
    """    
    nodes = network.nodes.copy()
    edges = network.edges.copy()
    to_ids = edges['to_id'].to_numpy()
    from_ids = edges['from_id'].to_numpy()
    new_node_ids = range(len(nodes))
    
    # Create a dictionary of the node ids and their corresponding indices
    id_dict = dict(zip(nodes.id, new_node_ids))
    
    nt = np.copy(to_ids)
    nf = np.copy(from_ids) 

    # Update all from and to ids using the created dictionary
    for k, v in id_dict.items():
        nt[to_ids == k] = v
        nf[from_ids == k] = v

    # Update the edge table with the new from and to ids
    edges.drop(labels=['to_id', 'from_id'], axis=1, inplace=True)
    edges['from_id'] = nf
    edges['to_id'] = nt

    # Drop the old node ids and set the new ids
    nodes.drop(labels=['id'], axis=1, inplace=True)
    nodes['id'] = new_node_ids

    # Reset the indices of edges and nodes
    edges['id'] = range(len(edges))
    edges.reset_index(drop=True, inplace=True)
    nodes.reset_index(drop=True, inplace=True)

    return Network(edges=edges, nodes=nodes)


def nearest_network_node_list(gdf_admin,gdf_nodes,sg):
    """[summary]
    Args:
        gdf_admin ([type]): [description]
        gdf_nodes ([type]): [description]
        sg ([type]): [description]
    Returns:
        [type]: [description]
    """    
    gdf_nodes = gdf_nodes.loc[gdf_nodes.id.isin(sg.vs['name'])]
    gdf_nodes.reset_index(drop=True,inplace=True)
    nodes = {}
    for admin_ in gdf_admin.itertuples():
        # if (shapely.distance((admin_.geometry),gdf_nodes.geometry).min()) > 0.005:
        #     continue
        nodes[admin_.id] = gdf_nodes.iloc[shapely.distance((admin_.geometry),gdf_nodes.geometry).idxmin()].id        
    return nodes

def split_edges_at_nodes(network, tolerance=1e-9,attributes=[]):
    """Split network edges where they intersect node geometries

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines)
        tolerance (float, optional): [description]. Defaults to 1e-9.

    Returns:            
        [type]: [description]
         
    """

    #create a spatial index for the nodes
    sindex_nodes = shapely.STRtree(network.nodes['geometry'])
    sindex_edges = shapely.STRtree(network.edges['geometry'])

    node_dict = network.nodes.geometry.to_dict()
    edges_dict = network.edges.geometry.to_dict()

    grab_all_edges = []

    #iterate over all the edges
    for iter_,edge in tqdm(network.edges.iterrows(),total=len(network.edges)):

        edge_geom = edge.geometry
        
        #### find all the nodes that intersect the edge and return intersecting points
        hits_nodes = list(itemgetter(*list(sindex_nodes.query(edge_geom)))(node_dict))    
        
        #### find all the edges that intersect the edge 
        hits_edges = [itemgetter(*list(sindex_edges.query(edge_geom)))(edges_dict)][0]

        # only keep points that truly overlap and return intersecting points
        if isinstance(hits_edges,shapely.MultiLineString) | isinstance(hits_edges,shapely.LineString):
            hits_edges = []
        else: 
            overlapping_points = [x for x in edge.geometry.intersection(hits_edges) if isinstance(x,shapely.Point)]

        #### combine the overlapping points and keep only the unique ones
        all_points_combined = shapely.extract_unique_points(shapely.MultiPoint(overlapping_points+hits_nodes))

        #### split edges based on that list of points
        new_edges = shapely.get_parts(shapely.ops.split(edge_geom,all_points_combined))

        #### create a new dataframe
        grab_all_edges.append(pd.DataFrame([list(edge[attributes].values)+[x] for x in new_edges],columns=network.edges.columns))

    # return new network with split edges
    return Network(
        nodes=network.nodes,
        edges=gpd.GeoDataFrame(pd.concat(grab_all_edges))
    )
   
def get_max(x):
    return max([int(y) for y in x])

def fill_attributes(network):
    """Fill missing speed and lanes attributes in the edges dataframe.

    Args:
        network (class): A network composed of nodes (points in space) and edges (lines).

    Returns:
        network (class): A network composed of nodes (points in space) and edges (lines) with filled attributes.
    """    
    # Dictionary mapping highway types to default speed values
    speed_d = {
        'motorway': '80',
        'motorway_link': '65',
        'trunk': '60',
        'trunk_link': '50',
        'primary': '50',  # meters per hour
        'primary_link': '40',
        'secondary': '40',  # meters per hour
        'secondary_link': '30',
        'tertiary': '30',
        'tertiary_link': '20',
        'unclassified': '20',
        'service': '20',
        'residential': '20',  # meters per hour
    }

    # Dictionary mapping highway types to default lanes values
    lanes_d = {
        'motorway': '4',
        'motorway_link': '2',
        'trunk': '4',
        'trunk_link': '2',
        'primary': '2',  
        'primary_link': '1',
        'secondary': '2',  
        'secondary_link': '1',
        'tertiary': '2',
        'tertiary_link': '1',
        'unclassified': '2',
        'service': '1',
        'residential': '1',  
    }

    # Create dataframes from the speed and lanes dictionaries
    df_speed = gpd.GeoDataFrame.from_dict(speed_d, orient='index', columns=['maxspeed'])
    df_lanes = gpd.GeoDataFrame.from_dict(lanes_d, orient='index', columns=['lanes'])

    # Function to turn maxspeed values to integers
    def turn_to_int(x):
        if isinstance(x.maxspeed, str):
            if len(re.findall(r'\d+', x.maxspeed)) > 0:
                return re.findall(r'\d+', x.maxspeed)[0]
            else:
                return speed_d[x.highway]
        else:
            return x.maxspeed

    # Apply the turn_to_int function to the maxspeed column
    network.edges.maxspeed = network.edges.apply(turn_to_int, axis=1)

    try:
        # Try to get mode values for lanes and maxspeed based on highway type
        vals_to_assign = network.edges.groupby('highway')[['lanes', 'maxspeed']].agg(pd.Series.mode)
    except:
        # If there is an error, fall back on default values
        vals_to_assign = df_lanes.join(df_speed)

    try:
        # Check if there are any lane values to assign
        vals_to_assign.lanes.iloc[0]
    except:
        print('NOTE: No lanes values available, falling back on default')
        vals_to_assign = vals_to_assign.join(df_lanes)

    try:
        # Check if there are any maxspeed values to assign
        vals_to_assign.maxspeed.iloc[0]
    except:
        print('NOTE: No maxspeed values available, falling back on default')
        vals_to_assign = vals_to_assign.join(df_speed)

    # Function to fill empty maxspeed cells
    def fill_empty_maxspeed(x):
        if len(list(x.maxspeed)) == 0:
            return speed_d[x.name]
        else:
            return x.maxspeed

    # Function to fill empty lanes cells
    def fill_empty_lanes(x):
        if len(list(x.lanes)) == 0:
            return lanes_d[x.name]
        else:
            return x.lanes

    # Function to get the maximum value from a list
    def get_max_in_vals_to_assign(x):
        if isinstance(x, list):
            return max([(y) for y in x])
        else:
            try:
                return re.findall(r'\d+', x)[0]
            except:
                return x

    # Fill empty cells in the lanes column
    vals_to_assign.lanes = vals_to_assign.apply(lambda x: fill_empty_lanes(x), axis=1)

    # Fill empty cells in the maxspeed column
    vals_to_assign.maxspeed = vals_to_assign.apply(lambda x: fill_empty_maxspeed(x), axis=1)

    # Get the maximum value from the maxspeed column
    vals_to_assign.maxspeed = vals_to_assign.maxspeed.apply(lambda x: get_max_in_vals_to_assign(x))

    def fill_oneway(x):
        if isinstance(x.oneway,str):
            return x.oneway
        else:
            return 'no'

    def fill_lanes(x):
        if isinstance(x.lanes,str):
            try:
                return int(x.lanes)  
            except:
                try:
                    return int(get_max(re.findall(r'\d+', x.lanes)))
                except:
                    return int(vals_to_assign.loc[x.highway].lanes)                  
        elif x.lanes is None:
            if isinstance(vals_to_assign.loc[x.highway].lanes,np.ndarray):
                return int(get_max(vals_to_assign.loc[x.highway.split('_')[0]].lanes))
            else:           
                return int(vals_to_assign.loc[x.highway].lanes)
        elif np.isnan(x.lanes):
            if isinstance(vals_to_assign.loc[x.highway].lanes,np.ndarray):
                return int(get_max(vals_to_assign.loc[x.highway.split('_')[0]].lanes))
            else:           
                return int(vals_to_assign.loc[x.highway].lanes)
            
    def fill_maxspeed(x):
        if isinstance(x.maxspeed,str):
            try:
                return [int(s) for s in x.maxspeed.split() if s.isdigit()][0]
            except:
                try:
                    return int(get_max(vals_to_assign.loc[x.highway.split('_')[0]].maxspeed))
                except:
                    try:
                        return int(get_max(re.findall(r'\d+', x.maxspeed)))
                    except:
                        return int(vals_to_assign.loc[x.highway].maxspeed)             
        elif x.maxspeed is None:
            if isinstance(vals_to_assign.loc[x.highway].maxspeed,np.ndarray):
                return int(get_max(vals_to_assign.loc[x.highway.split('_')[0]].maxspeed))
            else:           
                try:
                    return int(get_max(re.findall(r'\d+', x.maxspeed)))
                except:
                    return int(vals_to_assign.loc[x.highway].maxspeed)  
                    
        elif np.isnan(x.maxspeed):
            if isinstance(vals_to_assign.loc[x.highway].maxspeed,np.ndarray):
                try:
                    return int(get_max(vals_to_assign.loc[x.highway.split('_')[0]].maxspeed))
                except:
                    print(vals_to_assign.loc[x.highway].maxspeed)
                    return int(vals_to_assign.loc[x.highway].maxspeed)              
            else:           
                return int(vals_to_assign.loc[x.highway].maxspeed)  

    # Fill oneway, lanes, and maxspeed columns in the edges dataframe
    #network.edges['oneway'] = network.edges.apply(lambda x: fill_oneway(x), axis=1)
    network.edges['lanes'] = network.edges.apply(lambda x: fill_lanes(x), axis=1)
    network.edges['maxspeed'] = network.edges.apply(lambda x: fill_maxspeed(x), axis=1)
    
    return network


def simplified_network(df, drop_hanging_nodes_run=False, fill_attributes_run=True):
    """Returns a geopandas dataframe of a simplified network.

    Args:
        df (DataFrame): A pandas DataFrame containing network edge data.
        drop_hanging_nodes_run (bool, optional): If True, drops hanging nodes in the network. Defaults to True.
        fill_attributes_run (bool, optional): If True, fills missing attributes in the network edges. Defaults to True.

    Returns:
        net (class): A network composed of nodes (points in space) and edges (lines).
    """    
    # Create a Network object from the input DataFrame
    net = Network(edges=df)
    
    # Perform various simplification and cleaning steps
    net = clean_roundabouts(net)
    net = add_endpoints(net)
    net = split_edges_at_nodes(net)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)    
    
    # Drop hanging nodes if specified, otherwise just calculate degrees
    if drop_hanging_nodes_run:
        net = drop_hanging_nodes(net)    
    else:
        net.nodes['degree'] = calculate_degree(net)
            
    net = merge_edges(net)
    
    # Drop duplicate geometries while keeping the first occurrence
    net.edges = drop_duplicate_geometries(net.edges) 
    
    # Reset node and edge IDs
    net = reset_ids(net) 
    
    # Add distances between nodes
    #net = add_distances(net)
    
    # Merge MultiLineStrings into LineStrings
    net = merge_multilinestrings(net)
    
    # # Fill missing attributes in the network edges
    # if fill_attributes_run:
    #     net = fill_attributes(net)
    
    # # Add travel time based on attributes
    # net = add_travel_time(net)   
    return net


def ferry_connected_network(country,data_path,tqdm_on=True):
    """
    connect ferries to main network (and connect smaller sub networks automatically)

    Args:
        country ([type]): [description]
        data_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not tqdm_on:
        from utils import tqdm_standin as tqdm

    # load full network
    full_network = roads(str(data_path.joinpath('country_osm','{}.osm.pbf'.format(country))))
    main_road_network = full_network.loc[full_network.highway.isin(road_types)].reset_index(drop=True)
    
    # load ferries
    ferry_network = ferries(str(data_path.joinpath('country_osm','{}.osm.pbf'.format(country))))
    
    # create a main network where hanging nodes are not removed
    network_with_hanging = simplified_network(main_road_network,drop_hanging_nodes_run=False)
    nodes,edges = network_with_hanging.nodes.copy(),network_with_hanging.edges.copy()
    
    # create connections between ferry network and the main network
    connectors = connect_ferries(country,full_network,ferry_network)
   

    # loop through ferry connectors to add to edges of main network
    for link in connectors.itertuples():
        start = shapely.get_point(link.geometry,0)
        end = shapely.get_point(link.geometry,-1)
        from_id = nodes.id.loc[nodes.geometry==start]
        to_id = nodes.id.loc[nodes.geometry==end]
        edges = edges.append({  'osm_id':   np.random.random_integers(1e7,1e8),
                                'geometry': link.geometry,
                                'highway':  'ferry_connector', 
                                'maxspeed': 10,
                                'oneway':   'no', 
                                'lanes':    2},
                                ignore_index=True)

    # loop through ferry network to add to edges of main network
    for iter_,ferry in ferry_network.iterrows():
        start = shapely.get_point(ferry.geometry,0)
        end = shapely.get_point(ferry.geometry,-1)
        from_id = nodes.id.loc[nodes.geometry==start]
        to_id = nodes.id.loc[nodes.geometry==end]
        #if not from_id.empty and not to_id.empty: 

        edges = edges.append({  'osm_id':   ferry.osm_id,
                                    'geometry': ferry.geometry,
                                    'highway':  'ferry', 
                                    'maxspeed': 20,
                                    'oneway':   'no', 
                                    'lanes':    2},
                                    ignore_index=True)

    # ensure the newly created edge network has the same order compared to the original one
    new_edges = edges.iloc[:,:6]
    new_edges = new_edges[[x for x in new_edges.columns if x != 'geometry']+['geometry']]            
            
    # create new network with ferry connections
    net_final = simplified_network(new_edges,fill_attributes_run=False)

    net_final.edges.osm_id = net_final.edges.osm_id.astype(int)
    net_final.edges.geometry = shapely.to_wkb(net_final.edges.geometry)
    net_final.nodes.geometry = shapely.to_wkb(net_final.nodes.geometry)

    
    net_final.edges.to_feather(data_path.joinpath('road_ferry_networks','{}-edges.feather'.format(country)))
    net_final.nodes.to_feather(data_path.joinpath('road_ferry_networks','{}-nodes.feather'.format(country)))    
    
    return net_final   

    
def connect_ferries(country,full_network,ferry_network):
    """[summary]

    Args:
        country ([type]): [description]
        full_network ([type]): [description]
        ferry_network ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # list in which we will connect new ferry connections

    collect_connectors = []

    # loop through all ferries
    for iter_,ferry in (ferry_network.iterrows()):

        # create buffer around ferry to get the full network around the ferry ends
        ferry_buffer = shapely.buffer(ferry.geometry,0.05)

        # collect the road network around the ferry
        sub_full_network = full_network.loc[shapely.intersects(full_network.geometry,ferry_buffer)].reset_index(drop=True)
        sub_main_network_nodes = [[shapely.points(shapely.get_coordinates(x)[0]),shapely.points(shapely.get_coordinates(x)[1])] for x in sub_full_network.loc[sub_full_network.highway.isin(road_types)].geometry]
        sub_main_network_nodes =  [item for sublist in sub_main_network_nodes for item in sublist]
        sub_main_network_nodes = gpd.GeoDataFrame(sub_main_network_nodes,columns=['geometry'])
        sub_main_network_nodes['id'] = [x+1 for x in range(len(sub_main_network_nodes))]

        # create a dataframe of the ferry nodes
        ferry_nodes = gpd.GeoDataFrame([shapely.points(shapely.get_coordinates(ferry.geometry)[0]),shapely.points(shapely.get_coordinates(ferry.geometry)[-1])],columns=['geometry'])
        ferry_nodes['id'] = [1,2]

        # create mini simplified network and graph of network around ferry
        net = Network(edges=sub_full_network)
        net = add_endpoints(net)
        net = split_edges_at_nodes(net)
        net = add_endpoints(net)
        net = add_ids(net)
        net = add_topology(net)    
        net = add_distances(net)

        edges = net.edges.reindex(['from_id','to_id'] + [x for x in list(net.edges.columns) if x not in ['from_id','to_id']],axis=1)
        graph= ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)

        sg = graph.copy()

        # collect nearest nodes on network in graph
        nearest_node_main = nearest_network_node_list(sub_main_network_nodes,net.nodes,sg)
        nearest_node_ferry = nearest_network_node_list(ferry_nodes,net.nodes,sg)
        dest_nodes = [sg.vs['name'].index(nearest_node_main[x]) for x in list(nearest_node_main.keys())]
        ferry_nodes_graph = [sg.vs['name'].index(nearest_node_ferry[x]) for x in list(nearest_node_ferry.keys())]

        # collect paths on both sides of the ferry, if both sides have an actual network nearby
        if len(ferry_nodes_graph) == 2:
            start_node,end_node = ferry_nodes_graph

            # collect all shortest path from one side of the ferry to main network nodes
            collect_start_paths = {}
            for dest_node in dest_nodes:
                paths = sg.get_shortest_paths(sg.vs[start_node],sg.vs[dest_node],weights='distance',output="epath")
                if len(paths[0]) != 0:
                    collect_start_paths[dest_node] = sg.es[paths[0]]['id'],np.sum(sg.es[paths[0]]['distance'])

            start_coords = ferry_nodes.geometry[ferry_nodes.id=={v: k for k, v in nearest_node_ferry.items()}[sg.vs[start_node]['name']]].values

            # if there are paths, connect them up!
            if len(collect_start_paths) != 0:
                if len(gpd.GeoDataFrame.from_dict(collect_start_paths).T.min()) != 0:
                    path_1 = gpd.GeoDataFrame.from_dict(collect_start_paths).T.min()[0]
                    p_1 = []
                    for p in path_1: 
                        high_type = net.edges.highway.loc[net.edges.id==p].values
                        if np.isin(high_type,road_types): 
                            break
                        else: 
                            p_1.append(p)
                    path_1 = net.edges.loc[net.edges.id.isin(p_1)]

                    # check if they are really connected, if not, we need to create a little linestring to connect the new connector path and the ferry
                    if len(p_1) > 0: 
                        linestring = shapely.linear.line_merge(shapely.multilinestrings(path_1['geometry'].values))

                        endpoint1 = shapely.points(shapely.coordinates.get_coordinates(linestring))[0]
                        endpoint2 = shapely.points(shapely.coordinates.get_coordinates(linestring))[-1]

                        endpoint1_distance = shapely.distance(start_coords,endpoint1)
                        endpoint2_distance = shapely.distance(start_coords,endpoint2)

                        if (endpoint1_distance == 0)  | (endpoint2_distance == 0):
                            collect_connectors.append(linestring)
                        elif endpoint1_distance < endpoint2_distance:
                            collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(start_coords),shapely.coordinates.get_coordinates(linestring)),axis=0)))
                        else:
                            collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(linestring),shapely.coordinates.get_coordinates(start_coords)),axis=0))) 
                    else:
                        local_network = net.edges.loc[net.edges.id.isin(gpd.GeoDataFrame.from_dict(collect_start_paths).T.min()[0])]
                        sub_local_network = [[shapely.points(shapely.get_coordinates(x)[0]),shapely.points(shapely.get_coordinates(x)[-1])] for x in local_network.loc[local_network.highway.isin(road_types)].geometry]
                        sub_local_network =  [item for sublist in sub_local_network for item in sublist]
                        location_closest_point = np.where(shapely.distance(start_coords[0],sub_local_network) == np.amin(shapely.distance(start_coords[0],sub_local_network)))[0][0]
                        collect_connectors.append(shapely.linestrings(np.concatenate((shapely.get_coordinates(start_coords),shapely.get_coordinates(sub_local_network[location_closest_point])),axis=0)))
            
            # if there are no paths, but if the ferry node is still very close to the main network, we create a new linestring to connect them up (sometimes the ferry dock has no road)
            elif shapely.distance(sub_main_network_nodes.geometry,start_coords).min() < 0.01:
                get_new_end_point = shapely.coordinates.get_coordinates(sub_main_network_nodes.iloc[shapely.distance(sub_main_network_nodes.geometry,start_coords).idxmin()].geometry)
                collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(start_coords),get_new_end_point),axis=0)))

            # collect all shortest path from one side of the ferry to main network nodes
            collect_end_paths = {}
            for dest_node in dest_nodes:
                paths = sg.get_shortest_paths(sg.vs[end_node],sg.vs[dest_node],weights='distance',output="epath")
                if len(paths[0]) != 0:
                    collect_end_paths[dest_node] = sg.es[paths[0]]['id'],np.sum(sg.es[paths[0]]['distance'])

            end_coords = ferry_nodes.geometry[ferry_nodes.id=={v: k for k, v in nearest_node_ferry.items()}[sg.vs[end_node]['name']]].values

            # if there are paths, connect them up!
            if len(collect_end_paths) != 0:
                if len(gpd.GeoDataFrame.from_dict(collect_end_paths).T.min()) != 0:    
                    path_2 = gpd.GeoDataFrame.from_dict(collect_end_paths).T.min()[0]
                    p_2 = []
                    for p in path_2: 
                        high_type = net.edges.highway.loc[net.edges.id==p].values
                        if np.isin(high_type,road_types): 
                            break
                        else: 
                            p_2.append(p)

                    # check if they are really connected, if not, we need to create a little linestring to connect the new connector path and the ferry
                    path_2 = net.edges.loc[net.edges.id.isin(p_2)]
                    if len(p_2) > 0: 
                        linestring = shapely.linear.line_merge(shapely.multilinestrings(path_2['geometry'].values))

                        endpoint1 = shapely.points(shapely.coordinates.get_coordinates(linestring))[0]
                        endpoint2 = shapely.points(shapely.coordinates.get_coordinates(linestring))[-1]

                        endpoint1_distance = shapely.distance(end_coords,endpoint1)
                        endpoint2_distance = shapely.distance(end_coords,endpoint2)

                        if (endpoint1_distance == 0) | (endpoint2_distance == 0):
                            collect_connectors.append(linestring)
                        elif endpoint1_distance < endpoint2_distance:
                            collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(end_coords),shapely.coordinates.get_coordinates(linestring)),axis=0)))
                        else:
                            collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(linestring),shapely.coordinates.get_coordinates(end_coords)),axis=0)))                    
                    else:
                        local_network = net.edges.loc[net.edges.id.isin(gpd.GeoDataFrame.from_dict(collect_end_paths).T.min()[0])]
                        sub_local_network = [[shapely.points(shapely.get_coordinates(x)[0]),shapely.points(shapely.get_coordinates(x)[-1])] for x in local_network.loc[local_network.highway.isin(road_types)].geometry]
                        sub_local_network =  [item for sublist in sub_local_network for item in sublist]
                        location_closest_point = np.where(shapely.distance(end_coords[0],sub_local_network) == np.amin(shapely.distance(end_coords[0],sub_local_network)))[0][0]
                        collect_connectors.append(shapely.linestrings(np.concatenate((shapely.get_coordinates(end_coords),shapely.get_coordinates(sub_local_network[location_closest_point])),axis=0)))

            # if there are no paths, but if the ferry node is still very close to the main network, we create a new linestring to connect them up (sometimes the ferry dock has no road)
            elif shapely.distance(sub_main_network_nodes.geometry,end_coords).min() < 0.01:
                get_new_end_point = shapely.coordinates.get_coordinates(sub_main_network_nodes.iloc[shapely.distance(sub_main_network_nodes.geometry,end_coords).idxmin()].geometry)
                collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(end_coords),get_new_end_point),axis=0)))

        # ferry is stand-alone, so we continue because there is nothing to connect
        elif len(ferry_nodes_graph) == 0:
            continue

        # collect paths on one side  of the ferry, as other side does not have a network nearby
        else:
            start_node = ferry_nodes_graph[0] 
            start_coords = ferry_nodes.geometry[ferry_nodes.id=={v: k for k, v in nearest_node_ferry.items()}[sg.vs[start_node]['name']]].values

            # collect all shortest path from one side of the ferry to main network nodes
            collect_start_paths = {}
            for dest_node in dest_nodes:
                paths = sg.get_shortest_paths(sg.vs[start_node],sg.vs[dest_node],weights='distance',output="epath")
                if len(paths[0]) != 0:
                    collect_start_paths[dest_node] = sg.es[paths[0]]['id'],np.sum(sg.es[paths[0]]['distance'])
 
            # if there are paths, connect them up!
            if len(collect_start_paths) != 0:
                path_1 = gpd.GeoDataFrame.from_dict(collect_start_paths).T.min()[0]
                p_1 = []
                for p in path_1: 
                    high_type = net.edges.highway.loc[net.edges.id==p].values
                    if np.isin(high_type,road_types): break
                    else: p_1.append(p)
                path_1 = net.edges.loc[net.edges.id.isin(p_1)]

                # check if they are really connected, if not, we need to create a little linestring to connect the new connector path and the ferry
                if len(p_1) > 0: 
                    linestring = shapely.linear.line_merge(shapely.multilinestrings(path_1['geometry'].values))

                    endpoint1 = shapely.points(shapely.coordinates.get_coordinates(linestring))[0]
                    endpoint2 = shapely.points(shapely.coordinates.get_coordinates(linestring))[-1]

                    endpoint1_distance = shapely.distance(start_coords,endpoint1)
                    endpoint2_distance = shapely.distance(start_coords,endpoint2)

                    if (endpoint1_distance == 0) | (endpoint2_distance == 0):
                        collect_connectors.append(linestring)
                    elif endpoint1_distance < endpoint2_distance:
                        collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(start_coords),shapely.coordinates.get_coordinates(linestring)),axis=0)))
                    else:
                        collect_connectors.append(shapely.linestrings(np.concatenate((shapely.coordinates.get_coordinates(linestring),shapely.coordinates.get_coordinates(start_coords)),axis=0)))            
                else:
                    local_network = net.edges.loc[net.edges.id.isin(gpd.GeoDataFrame.from_dict(collect_start_paths).T.min()[0])]
                    sub_local_network = [[shapely.points(shapely.get_coordinates(x)[0]),shapely.points(shapely.get_coordinates(x)[-1])] for x in local_network.loc[local_network.highway.isin(road_types)].geometry]
                    sub_local_network =  [item for sublist in sub_local_network for item in sublist]
                    location_closest_point = np.where(shapely.distance(start_coords[0],sub_local_network) == np.amin(shapely.distance(start_coords[0],sub_local_network)))[0][0]
                    collect_connectors.append(shapely.linestrings(np.concatenate((shapely.get_coordinates(start_coords),shapely.get_coordinates(sub_local_network[location_closest_point])),axis=0)))

            # if there are no paths, but if the ferry node is still very close to the main network, we create a new linestring to connect them up (sometimes the ferry dock has no road)
            elif shapely.distance(sub_main_network_nodes.geometry,start_coords).min() < 0.01:
                get_new_end_point = shapely.get_coordinates(sub_main_network_nodes.iloc[shapely.distance(sub_main_network_nodes.geometry,start_coords).idxmin()].geometry)
                collect_connectors.append(shapely.linestrings(np.concatenate((shapely.get_coordinates(start_coords),get_new_end_point),axis=0)))
    
    return gpd.GeoDataFrame(collect_connectors,columns=['geometry'])

# def add_modal(network,alter_transport,threshold=0.02):
#     """
#     Designed with the addition of ferries in mind, to snap eligible routes onto existing network
#     with special logic for loading unloading, left after other methods to protect from merge
#     splitting and dropping logic. keeps these edges seperate from road simplification. only issue
#     is the snapping threshold needs to be more forgiving as often nearest nodes have been merged away
#     worth looking at edge finding in some cases. also seems to be a good idea to 
#     ferries will have their own time calculation method 

#     Args:
#         network (class): A network composed of nodes (points in space) and edges (lines)
#         alter_transport ([type]): [description]
#         threshold (float, optional): [description]. Defaults to 0.02.

#     Returns:
#         network (class): A network composed of nodes (points in space) and edges (lines)

#     """    
#     nodes = network.nodes.copy() 
#     edges = network.edges.copy()  
#     node_degree = nodes.degree.to_numpy()
#     sindex_nodes = shapely.STRtree(nodes['geometry'])
#     sindex_edges = shapely.STRtree(edges['geometry'])
#     new_edges = []
#     edge_id_counter = len(edges)
#     counter = 0
#     for route in alter_transport.itertuples():
#         route_geom = route.geometry
#         start = pygeom.get_point(route_geom,0)
#         end = pygeom.get_point(route_geom,-1)

#         near_start = _intersects(start,edges['geometry'],sindex_edges, tolerance=threshold)
#         near_end = _intersects(end,edges['geometry'],sindex_edges, tolerance=threshold)
#         near_start = near_start.index.values
#         near_end = near_end.index.values
#         print(near_end)
#         print(near_start)
#         if len(near_start) < 1 or len(near_end) < 1: continue

#         if len(near_start) > 1: 
#             near_start = min([edges.iloc[match_idx] for match_idx in near_start],
#                 key=lambda match: shapely.distance(start,match.geometry))
#             near_start = near_start.id

#         else: near_start = edges.id.iloc[near_start[0]]
#         if len(near_end) > 1: 
#             near_end = min([edges.iloc[match_idx] for match_idx in near_end],
#                 key=lambda match: shapely.distance(end,match.geometry))
#             near_end=near_end.id
#         else: near_end = edges.id.iloc[near_end[0]]
#         if near_end==near_start: 
#             print("for counter ", counter, "we skipped")
#             continue
#         #pick nodes to create edge
#         near_start = edges.iloc[near_start]
#         near_end = edges.iloc[near_end]
#         new_line_start = shapely.coordinates.get_coordinates(route_geom)

#         from_is_closer = shapely.measurement.distance(start, nodes.iloc[near_start.from_id].geometry) < shapely.measurement.distance(start, nodes.iloc[near_start.to_id].geometry)
#         if from_is_closer:
#             start_id = near_start.from_id
#         else:
#             start_id = near_start.to_id
#         node_degree[start_id] += 1
#         new_line = np.concatenate((shapely.coordinates.get_coordinates(nodes.iloc[start_id].geometry),new_line_start))
#         from_is_closer = shapely.measurement.distance(end, nodes.iloc[near_end.from_id].geometry) < shapely.measurement.distance(end, nodes.iloc[near_end.to_id].geometry)
#         if from_is_closer:
#             end_id = near_end.from_id
#         else:
#             end_id = near_end.to_id
#         node_degree[end_id] += 1
#         new_line = np.concatenate((new_line,shapely.coordinates.get_coordinates(nodes.iloc[end_id].geometry)))
#         new_edges.append({'osm_id':route.osm_id,'geometry': shapely.linestrings(new_line),'highway':route.highway,'id':edge_id_counter,'from_id':start_id,'to_id':end_id,'distance':999,'time':999})

#         counter+=1
        
#     edges = edges.append(new_edges,ignore_index=True)
#     edges.reset_index(inplace=True)
#     return Network(edges = edges, nodes=nodes)