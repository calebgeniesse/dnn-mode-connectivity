###############################################################################
# imports
###############################################################################

import os
import sys
from typing import Dict, List, Optional, Union
import subprocess
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import numpy as np
from pyevtk.hl import imageToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkVertex, VtkLine, VtkTriangle, VtkPolyLine
from itertools import combinations as combos
from itertools import product

# code for graph constructon
import networkx as nx
# from libpysal.weights import Gabriel
from scipy.spatial import Delaunay
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

###############################################################################
# configurations
###############################################################################

# TODO: probably a better way to define these paths
if 'PVPYTHON' not in os.environ:
    os.environ['PVPYTHON'] = "/Applications/ParaView-5.11.1.app/Contents/bin/pvpython"

if 'TTK_PLUGIN' not in os.environ:
    os.environ['TTK_PLUGIN'] = "/Applications/ParaView-5.11.1.app/Contents/Plugins/TopologyToolKit.so"

# export PVPYTHON=/Applications/ParaView-5.11.1.app/Contents/bin/pvpython
# export TTK_PLUGIN=/Applications/ParaView-5.12.0.app/Contents/Plugins/TopologyToolKit.so


###############################################################################
# graph construction methods
###############################################################################

def compute_delaunay(loss_coords=None, return_graph=True, verbose=1):
    """ Compute Delaunay triangulation and construct graph. """

    if verbose > 0:
        print(f"\n... Computing Delaunay triangulation")
        
    # compute Delaunay triangulation
    tri = Delaunay(loss_coords)

    # convert to Graph
    # TODO: this way maybe be very slow for larger graphs and may be over connected
    # TODO: see https://groups.google.com/g/networkx-discuss/c/D7fMmuzVBAw?pli=1
    # NOTE: fixed by adding each uniqure edge of each triangle
    G = nx.Graph()
    for path in tri.simplices:

        # define unique edges of the triangle
        e1 = {path[0], path[1]}
        e2 = {path[0], path[2]}
        e3 = {path[1], path[2]}

        # add edges to the graph
        G.add_edges_from([e1, e2, e3])


    # compute adjacency matrix
    A = nx.adjacency_matrix(G)   

    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A

def compute_gabriel(loss_coords=None, return_graph=True, verbose=1):
    """ Compute Gabriel graph. """
    
    if verbose > 0:
        print(f"\n... Computing Gabriel graph")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from libpysal.weights import Gabriel

        # compute Gabriel graph
        gab = Gabriel(loss_coords)
    

    # convert to Graph
    G = nx.Graph()
    for node in range(len(loss_coords)): 
        G.add_node(node)
        nbrs = gab.neighbors.get(node, [])
        for nbr in nbrs: 
            G.add_edge(node, nbr)
                  
    # compute adjacency matrix
    A = nx.adjacency_matrix(G)
  
    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A

def compute_aknn(loss_coords=None, n_neighbors=None, metric="euclidean", force_symmetric=False, 
                 return_graph=True, random_state=0, verbose=1):

    """ Compute () Approximate kNN graph. """

    n_neighbors = n_neighbors or (4 * loss_coords.shape[1])
    
    if verbose > 0:
        print(f"\n... Computing Approximate k Nearest Neighbors graph (n_neighbors={n_neighbors}, force_symmetric={force_symmetric})")

        
    # build (approximate) kNN index
    aknn = NNDescent(
        loss_coords, 
        metric=metric, 
        n_neighbors=n_neighbors, 
        n_jobs=-1, 
        random_state=random_state,
        verbose=False
    )
    
    # get neighbors (not including self)
    nbrs = aknn.neighbor_graph[0][:,1:]

    
    # construct (reciprocal) ANN Graph
    G = nx.Graph()
    
    if (verbose > 1):
        print("    ", end="") 
        
    for node in range(loss_coords.shape[0]): 
        
        # display progress
        if (verbose > 1) and ((node % 1000) == 0):
            print(f"{node}.", end="")

        # add node to the graph
        G.add_node(node)

        # add edges between reciprocal neighbors     
        for nbr in nbrs[node]:    
            # require node to be a nbr of its nbrs?
            if force_symmetric and (node not in nbrs[nbr]):
                continue
            G.add_edge(node, nbr)
        
    if (verbose > 1):
        print("") 
        
        
    # compute adjacency matrix
    A = nx.adjacency_matrix(G)
  
    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A



def compute_rknn(loss_coords=None, n_neighbors=None, metric="euclidean", 
                 return_graph=True, random_state=0, verbose=1, n_jobs=1):

    """ Compute () Approximate kNN graph. """

    n_neighbors = n_neighbors or (4 * loss_coords.shape[1])
    
    if verbose > 0:
        print(f"\n... Computing Reciprocal k Nearest Neighbors graph (n_neighbors={n_neighbors}, force_symmetric={force_symmetric})")

    # build Reciprocal Adjacency matrix
    from reciprocal_isomap import ReciprocalIsomap
    r_isomap = ReciprocalIsomap(
        n_neighbors=n_neighbors,
        neighbors_mode="connectivity",
        metric=metric,
        n_jobs=n_jobs, 
    )
    embedding = r_isomap.fit_transform(loss_coords)

    # compute adjacency matrix
    A = (r_isomap.dist_matrix_.A == 1)

    # construct (reciprocal) kNN Graph
    G = nx.Graph(A)
  
    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A


###############################################################################
# process ttk inputs
###############################################################################

def loss_landscape_to_vti(
    loss_landscape: List[List[float]] = None,
    # loss_coords: List[List[float]] = None,
    # loss_values: List[float] = None,
    output_path: str = "", 
    loss_steps_dim1: int = None, 
    loss_steps_dim2: int = None
    ) -> str:
    
    # TODO: should we do this outside the function?
    output_path = output_path + '_ImageData'

    # check output folder
    output_folder = os.path.dirname(output_path)
    if len(output_folder) and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # convert to an array and check loss_steps
    loss_landscape = np.array(loss_landscape)

    # check if we have a square matrix
    if loss_steps_dim1 == loss_steps_dim2:
        loss_steps = len(loss_landscape)
        # make sure we have a square matrix, convert if not
        if np.shape(loss_landscape)[-1] == 1:
            loss_steps = int(np.sqrt(loss_steps))
            loss_landscape = loss_landscape.reshape(loss_steps, loss_steps)
        # prepare the data to store in .vti files for ttk input
        loss_landscape_3d = loss_landscape.reshape(loss_steps, loss_steps, 1)
    else:
        # prepare the data to store in .vti files for ttk input
        loss_landscape_3d = loss_landscape.reshape(loss_steps_dim1, loss_steps_dim2, 1)

    # store the loss landscape results into binary files used for ttk
    imageToVTK(output_path, pointData={"Loss": loss_landscape_3d})

    # configure filename
    output_file = output_path + '.vti'

    return output_file



def loss_landscape_to_vtu(
    loss_landscape: List[List[float]] = None,
    loss_coords = None,
    loss_values = None,
    loss_graph = None,
    embedding = None,
    output_path: str = "", 
    graph_kwargs="aknn",
    n_neighbors=None,
    # loss_steps_dim1: int, 
    # loss_steps_dim2: int
    ) -> str:

    """

    TODO:
    - should we separate this into "loss_landscape_to_aknn", "aknn_to_vtu"
    - should we make graph_type an option "aknn", "gabriel", etc.
    - if the latter, maybe add graph_kwargs

    """

    # TODO: should we do this outside the function?
    output_path = output_path + '_UnstructuredGrid' + '_' + graph_kwargs

    # check output folder
    output_folder = os.path.dirname(output_path)
    if len(output_folder) and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ### process landscape
    if loss_graph is not None:

        G = loss_graph 
        loss_steps = 1 # this is used to normalize loss values

        # TODO: assumes embedding is passed, but we need some coordinates
        # if(embedding is None) and (loss_coords is None):
        #     loss_coords = np.c_[
        #         np.random.randint(100, size=100),
        #         np.random.randint(100, size=100)
        #     ]

        # TODO: or let's just use a dummy embedding
        if (embedding is None):
            embedding = nx.circular_layout(G)
            embedding = np.vstack([embedding[_] for _ in G.nodes])

    else:
        
        # convert to an array and check loss_steps
        loss_steps = None 
        loss_dims = None
        if loss_coords is None:
            loss_landscape = np.array(loss_landscape)
            loss_steps = len(loss_landscape)

            # make sure we have a square matrix, convert if not
            # TODO: this assumes 2d, not sure we can figure out if higher d (?)
            if np.shape(loss_landscape)[-1] == 1:
                loss_steps = int(np.sqrt(loss_steps))
                loss_landscape = loss_landscape.reshape(loss_steps, loss_steps)

            # define loss coordinates
            loss_dims = np.ndim(loss_landscape)
            loss_coords = np.asarray(list(product(*[
                np.arange(loss_steps) for _ in range(loss_dims)
            ])))

        if loss_values is None:
            # TODO: extract loss values
            # TODO: will this match the coordinates (???)
            loss_values = loss_landscape.ravel()

        if loss_steps is None:
            loss_steps = np.ravel(loss_coords).max()

        if loss_dims is None:
            loss_dims = np.shape(loss_coords)[-1]

        ### construct graph 

        # TODO: make these options more flexible
        if n_neighbors is None:
            n_neighbors = 4*loss_dims

        # TODO: let user define the method
        # TODO: accept user defined kwargs
        # TODO: e.g., graph_kwargs=dict(kind="aknn", force_symmetric=True)
        if graph_kwargs == "aknn":
            _, G = compute_aknn(loss_coords=loss_coords, n_neighbors=n_neighbors,
                            metric="euclidean", force_symmetric=False, 
                            return_graph=True, random_state=0, verbose=0)
        elif graph_kwargs == "raknn":
            _, G = compute_aknn(loss_coords=loss_coords, n_neighbors=n_neighbors,
                            metric="euclidean", force_symmetric=True, 
                            return_graph=True, random_state=0, verbose=0)
        elif graph_kwargs == "rknn":
            _, G = compute_rknn(loss_coords=loss_coords, n_neighbors=n_neighbors,
                            metric="euclidean", return_graph=True, n_jobs=-1,
                            verbose=0)
        elif graph_kwargs == "gabriel":
            _, G = compute_gabriel(loss_coords=loss_coords, return_graph=True, verbose=0)
        elif graph_kwargs == "delaunay":
            _, G = compute_delaunay(loss_coords=loss_coords, return_graph=True, verbose=0)
        else:
            print(f"Graph type {graph_kwargs} not recognized, using aknn")
            _, G = compute_aknn(loss_coords=loss_coords, n_neighbors=n_neighbors,
                            metric="euclidean", force_symmetric=True, 
                            return_graph=True, random_state=0, verbose=0)
        
    ### process unstructured grid for VTK

    # TODO: should we use list or array?
    # extract the undirected edges as an array
    lines_unique = list(G.edges())
        
    # count the number of unique lines
    n_lines = len(lines_unique)

    # define points that belong in each line
    conn = np.ravel(lines_unique)

    # define offset of last vertex of each element
    offsets = (np.arange(n_lines) + 1) * 2

    # define array that defines the cell type of each element in the grid
    cell_types = np.repeat(VtkLine.tid, n_lines)

    # define dictionary with variables associated to each vertex.
    point_data = dict()
    point_data['Loss'] = np.ascontiguousarray(loss_values.ravel()).astype(float)
    print(f"{point_data=}")

    # define dictionary with variables associated to each cell.
    cell_data = None
    # cell_data = dict()
    # cell_data['Loss (mean)'] = np.mean(loss_values[lines_unique], axis=1).ravel()
    # cell_data['Loss (min)'] = np.min(loss_values[lines_unique], axis=1).ravel()
    # cell_data['Loss (max)'] = np.max(loss_values[lines_unique], axis=1).ravel()

    # TODO: compute PCA embedding of the coordinates (?)
    # TODO: accept a user defined embedding (?)
    if embedding is None:  
        # embedding = PCA(3).fit_transform(loss_coords)
        embedding = PCA(n_components=2).fit_transform(loss_coords)

    # TODO: scale the loss and combine with PCA coordinates (?)
    # TODO: this assumes step size of 40 was used
    loss_values_scaled = MinMaxScaler((0, loss_steps)).fit_transform(loss_values[:, np.newaxis].reshape(-1,1))


    # TODO: scale embedding
    # from sklearn.preprocessing import MinMaxScaler
    # embedding = MinMaxScaler((0, loss_steps)).fit_transform(embedding)


    # combine first two PCs with scaled loss
    if embedding.shape[-1] < 3:
        embedding = np.c_[embedding[:,:2], loss_values_scaled]

    # define x,y,z (ascontiguousarray to avoid VTK errors)
    x = np.ascontiguousarray(embedding[:,0]).astype(float)
    y = np.ascontiguousarray(embedding[:,1]).astype(float)
    z = np.ascontiguousarray(embedding[:,2]).astype(float)

    # save as a VTK unstructured grid
    fn_loss_vtu = unstructuredGridToVTK(
        output_path, 
        x=x, y=y, z=z,
        connectivity=conn, 
        offsets=offsets, 
        cell_types=cell_types, 
        cellData=cell_data, 
        pointData=point_data, 
        fieldData=None
    )

    # configure filename
    output_file = output_path + '.vtu'

    return output_file






###############################################################################
# process ttk outputs
###############################################################################

def process_persistence_barcode(input_file: str) -> list:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

        "ttkVertexScalarField","CriticalType","Coordinates:0","Coordinates:1","Coordinates:2","Points:0","Points:1","Points:2"
        896,0,16,22,0,0.017,0.017,0
        1599,3,39,39,0,0.017,1.6e+03,0
        1056,0,16,26,0,0.028,0.028,0

    """

    # process persistence_barcode object here
    points_0 = []
    points_1 = []
    points_2 = []
    nodeID = []
    
    # load coordinates from csv file
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            points_0.append(float(row["Points:0"]))
            points_1.append(float(row["Points:1"]))
            points_2.append(float(row["Points:2"]))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row["Point ID"]))
            nodeID.append(int(row["ttkVertexScalarField"]))


    # convert to representation for the database
    # TODO: not sure what x, y0, y1 correspond to
    persistence_barcode = [
        {"x": points_0[i], "y0": points_1[i], "y1": points_2[i]}
        for i in range(len(nodeID))
    ]

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return persistence_barcode


def process_merge_tree_side(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace('.csv', '_edge.csv')
    segmentation_file = input_file.replace('.csv', '_segmentation.csv')
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            
            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)


    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "z":pointsz[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsy[i], pointsz[i]]

    edges = []

    with open(edge_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append({
                "sourceX": node_dict[int(row['downNodeId'])][0],
                "sourceY": node_dict[int(row['downNodeId'])][1],
                "targetX": node_dict[int(row['upNodeId'])][0],
                "targetY": node_dict[int(row['upNodeId'])][1],
            })

    segmentations = []
    with open(segmentation_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append({
                    "x": float(row['points:0']),
                    "y": float(row['points:1']),
                    "z": float(row['points:2']),
                    "loss": float(row['Loss']),
                })

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree


def process_merge_tree_3d(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace('.csv', '_edge.csv')
    segmentation_file = input_file.replace('.csv', '_segmentation.csv')
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            
            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)


    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "z":pointsz[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i], pointsz[i]]

    edges = []

    with open(edge_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append({
                "sourceX": node_dict[int(row['downNodeId'])][0],
                "sourceY": node_dict[int(row['downNodeId'])][1],
                "sourceZ": node_dict[int(row['downNodeId'])][2],
                "targetX": node_dict[int(row['upNodeId'])][0],
                "targetY": node_dict[int(row['upNodeId'])][1],
                "targetZ": node_dict[int(row['upNodeId'])][2],
            })

    segmentations = []
    with open(segmentation_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append({
                    "x": float(row['points:0']),
                    "y": float(row['points:1']),
                    "z": float(row['points:2']),
                    "loss": float(row['Loss']),
                })
            elif "Loss" in row:
                segmentations.append({
                    "loss": float(row['Loss']),
                })

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree


def process_merge_tree_3d_vti(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    loss = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace('.csv', '_edge.csv')
    segmentation_file = input_file.replace('.csv', '_segmentation.csv')
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            loss.append(float(row['Scalar']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            
            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)


    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "z":pointsz[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i], pointsz[i], loss[i] ]

    edges = []

    with open(edge_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append({
                "sourceX": node_dict[int(row['downNodeId'])][0],
                "sourceY": node_dict[int(row['downNodeId'])][1],
                "sourceZ": node_dict[int(row['downNodeId'])][3],
                "targetX": node_dict[int(row['upNodeId'])][0],
                "targetY": node_dict[int(row['upNodeId'])][1],
                "targetZ": node_dict[int(row['upNodeId'])][3],
            })

    segmentations = []
    with open(segmentation_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append({
                    "x": float(row['points:0']),
                    "y": float(row['points:1']),
                    "z": float(row['points:2']),
                    "loss": float(row['Loss']),
                })
            elif "Loss" in row:
                segmentations.append({
                    "loss": float(row['Loss']),
                })

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree

def process_merge_tree_front(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace('.csv', '_edge.csv')
    segmentation_file = input_file.replace('.csv', '_segmentation.csv')
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            
            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)


    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "z":pointsz[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsz[i]]

    edges = []

    with open(edge_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append({
                "sourceX": node_dict[int(row['downNodeId'])][0],
                "sourceY": node_dict[int(row['downNodeId'])][1],
                "targetX": node_dict[int(row['upNodeId'])][0],
                "targetY": node_dict[int(row['upNodeId'])][1],
            })

    segmentations = []
    with open(segmentation_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append({
                    "x": float(row['points:0']),
                    "y": float(row['points:1']),
                    "z": float(row['points:2']),
                    "loss": float(row['Loss']),
                })

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree

def process_merge_tree(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []
    persistences = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace('.csv', '_edge.csv')
    segmentation_file = input_file.replace('.csv', '_segmentation.csv')
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            

            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)


    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i]]

    edges = []

    with open(edge_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append({
                "sourceX": node_dict[int(row['downNodeId'])][0],
                "sourceY": node_dict[int(row['downNodeId'])][1],
                "targetX": node_dict[int(row['upNodeId'])][0],
                "targetY": node_dict[int(row['upNodeId'])][1],
            })

    segmentations = []
    with open(segmentation_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append({
                    "x": float(row['points:0']),
                    "y": float(row['points:1']),
                    "z": float(row['points:2']),
                    "loss": float(row['Loss']),
                    "SegmentationId": int(row['SegmentationId']),
                })

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree

def process_merge_tree_planar(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

        "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Persistence","ClusterID","TreeID","isDummyNode","TrueNodeId","isImportantPair","isMultiPersPairNode","BranchNodeID","Points:0","Points:1","Points:2"
        0,1638,1599,3,483,38,1638,0,0,0,15,1,0,0,741.02,1638,0
        2,477.94,71,1,483,38,22.023,0,0,0,14,0,0,1,641.31,477.94,0
        2,477.94,71,1,483,38,22.023,0,0,1,14,0,0,1,741.02,477.94,0

    """
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []
    persistences = []

    # initialize some global variables
    root_x = 0

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            nodeID.append(int(row['NodeId']))
            branchID.append(int(row['BranchNodeID']))
            persistences.append(float(row["Persistence"]))

            # find the start point of each branch
            if int(row['BranchNodeID']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1

    # verify that the start and end points are correct
    temp_structure = []
    for i in range(len(start)):
        t = {'start': start[i], 'end': end[i], 'x': pointsx[i], 'y': pointsy[i], 'z': pointsz[i], 'nodeID': nodeID[i], 'branchID': branchID[i], "Persistence": persistences[i]}
        temp_structure.append(t)

    nodes = []
    for item in temp_structure:
        nodes.append({
            "id": item["nodeID"],
            "x": item["x"],
            "y": item["y"],
            "Persistence": item["Persistence"]

        })

    edges = []
    branch = {}

    for i in range(len(temp_structure)):
        item_id = temp_structure[i]["branchID"]
        if item_id not in branch:
            branch[item_id] = []
        branch[item_id].append(temp_structure[i])

    for key in branch:
        nodes = branch[key]
        for i in range(len(nodes) - 1):
            for j in range(i + 1, len(nodes) - 1):
                if i != j and (nodes[i]['x'] == nodes[j]['x'] or nodes[i]['y'] == nodes[j]['y']):
                    edges.append({
                        "sourceX": nodes[i]['x'],
                        "sourceY": nodes[i]['y'],
                        "targetX": nodes[j]['x'],
                        "targetY": nodes[j]['y'],
                    })

    res =  {
        "nodes": nodes,
        "edges": edges,
    }

    # print(res)
    return res

###############################################################################
# main functions
###############################################################################

def compute_persistence_barcode(loss_landscape: List[List[float]] = None, 
                                loss_coords: List[List[float]] = None, 
                                loss_values: List[float] = None, 
                                loss_graph: nx.Graph = None,
                                embedding: List[List[float]] = None, 
                              # loss_steps_dim1: int,
                              # loss_steps_dim2: int,
                              output_path: str = "",
                              vtk_format: str = "vtu",
                              graph_kwargs: str = "aknn",
                              n_neighbors=None,
                              persistence_threshold: float = 0.0, 
                              threshold_is_absolute: bool = False,
                              ) -> str:
    
    # convert loss_landscape into a vtk format
    output_file_vtk = None 
    if vtk_format.lower() == "vti":

        if loss_graph is not None:
            raise Exception("vtk_format == 'vti' but loss_graph != None")
        
        # convert loss_landscape into a (.vti) image data format
        # output_file_vtk = loss_landscape_to_vti(loss_landscape, output_path, loss_steps_dim1, loss_steps_dim2)
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values, 
            embedding=embedding,
            output_path=output_path
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values,
            loss_graph=loss_graph,
            embedding=embedding,
            output_path=output_path, 
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors, 
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute persistence_barcode
    output_file_csv = compute_persistence_barcode_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold, 
        threshold_is_absolute=threshold_is_absolute
    )

    # extract .csv and return persistence_barcode object
    persistence_barcode = process_persistence_barcode(output_file_csv)

    return persistence_barcode 

def compute_merge_tree(loss_landscape: List[List[float]] = None, 
                        loss_coords: List[List[float]] = None, 
                        loss_values: List[float] = None, 
                        loss_graph: nx.Graph = None,
                        embedding: List[List[float]] = None, 
                        # loss_steps_dim1: int,
                        # loss_steps_dim2: int,
                              output_path: str = "",
                              vtk_format: str = "vtu",
                              graph_kwargs: str = "aknn",
                              n_neighbors=None,
                              persistence_threshold: float = 0.0, 
                              threshold_is_absolute: bool = False,
                              ) -> str:
    
    # convert loss_landscape into a vtk format
    output_file_vtk = None 
    if vtk_format.lower() == "vti":

        if loss_graph is not None:
            raise Exception("vtk_format == 'vti' but loss_graph != None")
        
        # convert loss_landscape into a (.vti) image data format
        # output_file_vtk = loss_landscape_to_vti(loss_landscape, output_path, loss_steps_dim1, loss_steps_dim2)
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values, 
            embedding=embedding,
            output_path=output_path
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values,
            loss_graph=loss_graph,
            embedding=embedding, 
            output_path=output_path, 
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors, 

        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold, 
        threshold_is_absolute=threshold_is_absolute
    )

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree(output_file_csv)

    return merge_tree

def compute_merge_tree_planar(loss_landscape: List[List[float]] = None, 
                        loss_coords: List[List[float]] = None, 
                        loss_values: List[float] = None, 
                        loss_graph: nx.Graph = None,
                        embedding: List[List[float]] = None, 

                        # loss_steps_dim1: int,
                        # loss_steps_dim2: int,
                              output_path: str = "",
                              vtk_format: str = "vtu",
                              graph_kwargs: str = "aknn",
                              n_neighbors=None,
                              persistence_threshold: float = 0.0, 
                              threshold_is_absolute: bool = False,
                              ) -> str:
    
    ### TODO: maybe just make planar=False an argument of compute_merge_tree

    # convert loss_landscape into a vtk format
    output_file_vtk = None 
    if vtk_format.lower() == "vti":

        if loss_graph is not None:
            raise Exception("vtk_format == 'vti' but loss_graph != None")

        # convert loss_landscape into a (.vti) image data format
        # output_file_vtk = loss_landscape_to_vti(loss_landscape, output_path, loss_steps_dim1, loss_steps_dim2)
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values,
            embedding=embedding,  
            output_path=output_path
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape, 
            loss_coords=loss_coords, 
            loss_values=loss_values,
            loss_graph=loss_graph, 
            embedding=embedding, 
            output_path=output_path, 
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_planar_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold, 
        threshold_is_absolute=threshold_is_absolute
    )

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree_planar(output_file_csv)

    return merge_tree


###############################################################################
# paraview functions
###############################################################################

def compute_persistence_barcode_paraview(input_file, 
                                         output_folder=None,
                                         persistence_threshold: float = 0.0, 
                                         threshold_is_absolute: bool = False,
                                         ) -> str:
    """ Run calculate_ttk_persistence_diagram.py using pvpython. """

    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = input_file.split('.vt')[0] + simplification_str + '_PersistenceDiagram.csv'
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # format the command
        _command = [os.environ['PVPYTHON'], 
             f"{os.path.dirname(__file__)}/calculate_ttk_persistence_diagram.py",
             f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
             f"--input-file={input_file}",
             f"--output-file={output_file}",
             f"--persistence-threshold={persistence_threshold}",
             f"--threshold-is-absolute" if threshold_is_absolute else ""
        ]
        _command = list(filter(None, _command))
        # print(" ".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file

def compute_merge_tree_paraview(input_file, 
                                output_folder=None,
                                persistence_threshold: float = 0.0, 
                                threshold_is_absolute: bool = False,
                               ) -> str:
    """ Run calculate_ttk_merge_tree.py using pvpython. """

    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = input_file.split('.vt')[0] + simplification_str + '_MergeTree.csv'
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # format the command
        _command = [os.environ['PVPYTHON'],
             f"{os.path.dirname(__file__)}/calculate_ttk_merge_tree.py",
             f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
             f"--input-file={input_file}",
             f"--output-file={output_file}",
             f"--persistence-threshold={persistence_threshold}",
             f"--threshold-is-absolute" if threshold_is_absolute else ""
        ]
        _command = list(filter(None, _command))
        # print("_".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

    return output_file

def compute_merge_tree_planar_paraview(input_file, 
                                       output_folder=None,
                                       persistence_threshold: float = 0.0, 
                                       threshold_is_absolute: bool = False,
                                       ) -> str:
    """ Run calculate_ttk_merge_tree_planar.py using pvpython. """


    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = input_file.split('.vt')[0] + simplification_str + '_MergeTreePlanar.csv'
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # format the command
        _command = [os.environ['PVPYTHON'], 
             f"{os.path.dirname(__file__)}/calculate_ttk_merge_tree_planar.py",
             f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
             f"--input-file={input_file}",
             f"--output-file={output_file}",
             f"--persistence-threshold={persistence_threshold}",
             f"--threshold-is-absolute" if threshold_is_absolute else ""
        ]
        _command = list(filter(None, _command))
        # print("_".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file

def visualize_persistence_diagrams(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)


    for index, file_name in enumerate(file_names):
        if file_name.endswith(".csv"):
            if file_name.endswith("PersistenceDiagram.csv"):
                file_path = current_dir + folder_name + file_name
                pd = process_persistence_barcode(file_path)
                fig, ax = plt.subplots(figsize=plt.figaspect(1))
                max_value = 0
                min_value = float("inf")
                for i in range(len(pd)):
                    point1 = []
                    point2 = []
                    max_value = max(max_value, pd[i]["x"])
                    max_value = max(max_value, pd[i]["y0"])
                    min_value = min(min_value, pd[i]["x"])
                    min_value = min(min_value, pd[i]["y0"])
                    point1.append(pd[i]["x"])
                    point1.append(pd[i]["x"])
                    point2.append(pd[i]["x"])
                    point2.append(pd[i]["y0"])
                    ax.plot(point1, point2, color='blue')


                point1 = []
                point2 = []
                point1.append(min_value)
                point1.append(max_value)
                point2.append(min_value)
                point2.append(max_value)
                ax.plot(point1, point2, color="blue")

                print(file_name[:-4])
                save_folder_path = current_dir + "/figures/PD/"
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
            else:
                continue


def visualize_merge_tree_planar(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)

    for index, file_name in enumerate(file_names):
        if file_name.endswith(".csv"):
            if file_name.endswith("MergeTreePlanar.csv"):
                file_path = current_dir + folder_name + file_name
                # print(file_name)
                mtp = process_merge_tree_planar(file_path)
                fig, ax = plt.subplots(figsize=plt.figaspect(1))
                # print(len(mtp["edges"]))
                edges = mtp["edges"]
                for i in range(len(edges)):
                    point1 = []
                    point2 = []
                    point1.append(edges[i]["sourceX"])
                    point1.append(edges[i]["targetX"])
                    point2.append(edges[i]["sourceY"])
                    point2.append(edges[i]["targetY"])
                    ax.plot(point1, point2, color='blue')
                # plt.show()
                save_folder_path = current_dir + "/figures/MTP/"
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
            else:
                continue


def visualize_merge_tree(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)
    cmap = plt.get_cmap('plasma')

    for index, file_name in enumerate(file_names):
        if file_name.endswith(".csv"):
            if file_name.endswith("MergeTree.csv"):
                file_path = current_dir + folder_name + file_name
                # print(file_name)
                mtp = process_merge_tree(file_path)
                # print(mtp)
                fig, ax = plt.subplots(figsize=plt.figaspect(1))
                stroke_width = 0.5

                # drawing the segmentations
                segmentations = mtp["segmentations"]
                loss = [seg["loss"] for seg in segmentations]
                if len(loss) == 0:
                    pass
                    # print("No segmentations")
                else:
                    norm = mcolors.Normalize(vmin=min(loss), vmax=max(loss))
                    colors = [cmap(norm(l)) for l in loss]

                    for i in range(len(segmentations)):
                        # print(colors[i])
                        c = colors[i][0:3] + (0.8,)
                        # print(c)
                        # circle = patches.Circle((segmentations[i]["x"], segmentations[i]["y"]), 0.5, facecolor=c)
                        # ax.add_patch(circle)
                        square = patches.Rectangle((segmentations[i]["x"] - 0.5, segmentations[i]["y"] - 0.5), 1, 1, facecolor=c)
                        ax.add_patch(square)

                # drawing the edges
                edges = mtp["edges"]
                for i in range(len(edges)):
                    point1 = []
                    point2 = []
                    point1.append(edges[i]["sourceX"])
                    point1.append(edges[i]["targetX"])
                    point2.append(edges[i]["sourceY"])
                    point2.append(edges[i]["targetY"])
                    ax.plot(point1, point2, color='black', linewidth=stroke_width)

                # drawing the nodes
                for i in range(len(mtp["nodes"])):
                    if mtp["nodes"][i]["criticalType"] == 0:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='blue', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    elif mtp["nodes"][i]["criticalType"] == 1:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='red', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    else:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='black', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                # plt.show()
                save_folder_path = current_dir + "/figures/MT/"
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
            else:
                continue

def visualize_merge_tree_front(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)
    cmap = plt.get_cmap('plasma')

    for index, file_name in enumerate(file_names):
        if file_name.endswith(".csv"):
            if file_name.endswith("MergeTree.csv"):
                file_path = current_dir + folder_name + file_name
                mtp = process_merge_tree_front(file_path)
                fig, ax = plt.subplots(figsize=plt.figaspect(1))
                stroke_width = 0.5

                # drawing the edges
                edges = mtp["edges"]
                for i in range(len(edges)):
                    point1 = []
                    point2 = []
                    point1.append(edges[i]["sourceX"])
                    point1.append(edges[i]["targetX"])
                    point2.append(edges[i]["sourceY"])
                    point2.append(edges[i]["targetY"])
                    ax.plot(point1, point2, color='black', linewidth=stroke_width)

                # drawing the nodes
                for i in range(len(mtp["nodes"])):
                    if mtp["nodes"][i]["criticalType"] == 0:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["z"]), 0.4, facecolor='blue', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    elif mtp["nodes"][i]["criticalType"] == 1:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["z"]), 0.4, facecolor='red', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    else:
                        circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["z"]), 0.4, facecolor='black', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                # plt.show()
                save_folder_path = current_dir + "/figures/MTF/"
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
            else:
                continue

def visualize_merge_tree_side(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)
    cmap = plt.get_cmap('plasma')

    for index, file_name in enumerate(file_names):
        if file_name.endswith(".csv"):
            if file_name.endswith("MergeTree.csv"):
                file_path = current_dir + folder_name + file_name
                mtp = process_merge_tree_side(file_path)
                fig, ax = plt.subplots(figsize=plt.figaspect(1))
                stroke_width = 0.5


                # drawing the edges
                edges = mtp["edges"]
                for i in range(len(edges)):
                    point1 = []
                    point2 = []
                    point1.append(edges[i]["sourceX"])
                    point1.append(edges[i]["targetX"])
                    point2.append(edges[i]["sourceY"])
                    point2.append(edges[i]["targetY"])
                    ax.plot(point1, point2, color='black', linewidth=stroke_width)

                # drawing the nodes
                for i in range(len(mtp["nodes"])):
                    if mtp["nodes"][i]["criticalType"] == 0:
                        circle = patches.Circle((mtp["nodes"][i]["y"], mtp["nodes"][i]["z"]), 0.4, facecolor='blue', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    elif mtp["nodes"][i]["criticalType"] == 1:
                        circle = patches.Circle((mtp["nodes"][i]["y"], mtp["nodes"][i]["z"]), 0.4, facecolor='red', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                    else:
                        circle = patches.Circle((mtp["nodes"][i]["y"], mtp["nodes"][i]["z"]), 0.4, facecolor='black', edgecolor='black', linewidth=stroke_width)
                        ax.add_patch(circle)
                # plt.show()
                save_folder_path = current_dir + "/figures/MTS/"
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
            else:
                continue

def visualize_contour(start=-0.05, end=0.05, steps=40):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + "/loss_landscape_files/")
    for index, file_name in enumerate(file_names):
        if file_name.endswith(".npy"):
            file_path = current_dir + "/loss_landscape_files/" + file_name
            data_matrix = np.load(file_path)
            X, Y = np.meshgrid(np.linspace(start, end, steps), np.linspace(start, end, steps))
            Z = data_matrix.reshape(steps, steps)
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z, levels=80)
            save_folder_path = current_dir + "/figures/Contour/"
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            plt.savefig(save_folder_path +file_name[:-4] + ".png", dpi=300, format='png')


def quantify_merge_tree(folder_name, **kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)
    cmap = plt.get_cmap('plasma')

    for index, file_name in enumerate(file_names):
        # print(index, file_name)
        if not file_name.endswith("MergeTree.csv"):
            continue
        file_path = current_dir + folder_name + file_name

        # print(file_name)
        mtp = process_merge_tree(file_path)
        
        # print(mtp)
        # fig, ax = plt.subplots(figsize=plt.figaspect(1))
        # stroke_width = 0.5

        # drawing the segmentations
        segmentations = mtp["segmentations"]
        segmentationID = [seg["SegmentationId"] for seg in segmentations]
        loss = [seg["loss"] for seg in segmentations]

        nodes = mtp["nodes"]
        minima = [node for node in nodes if node['criticalType'] == 0]
        saddles = [node for node in nodes if node['criticalType'] == 1]

        minima_xy = [(node['x'], node['y']) for node in minima]
        saddles_xy = [(node['x'], node['y']) for node in saddles]


        # print(file_name)
        # print(f"")
        # # print(f"    len(loss) = {len(loss)}")
        # print(f"    len(nodes) = {len(nodes)}")
        # print(f"    len(minima) = {len(minima)}")
        # print(f"    len(saddles) = {len(saddles)}")
        # print(f"")
        # print(f"")

        print(f"{file_name}\t\t{len(nodes)}\t{len(minima)}\t{len(saddles)}")

        ###  TODO (urgent)
        # - save or print these quantities
        # - add table of these values for B={1:1:10} (1 for random, 1 for hessian)
        # - add plots of these values vs. acc        (3 for random, 3 for hessian)
        # - add plots of these values vs. Vol        (3 for random, 3 for hessian)


        ### TODO
        # - plot # points along a branch? 
        # - pick threshold, count remaining branches?
        # - calc volume w/ each branch
        # - for each volume, how many branches above?
        # 


        # if len(loss) > 0:
        
        #     norm = mcolors.Normalize(vmin=min(loss), vmax=max(loss))
        #     colors = [cmap(norm(l)) for l in loss]

        #     for i in range(len(segmentations)):
        #         # print(colors[i])
        #         c = colors[i][0:3] + (0.8,)
        #         # print(c)
        #         # circle = patches.Circle((segmentations[i]["x"], segmentations[i]["y"]), 0.5, facecolor=c)
        #         # ax.add_patch(circle)
        #         square = patches.Rectangle((segmentations[i]["x"] - 0.5, segmentations[i]["y"] - 0.5), 1, 1, facecolor=c)
        #         ax.add_patch(square)

        # # drawing the edges
        # edges = mtp["edges"]
        # for i in range(len(edges)):
        #     point1 = []
        #     point2 = []
        #     point1.append(edges[i]["sourceX"])
        #     point1.append(edges[i]["targetX"])
        #     point2.append(edges[i]["sourceY"])
        #     point2.append(edges[i]["targetY"])
        #     ax.plot(point1, point2, color='black', linewidth=stroke_width)

        # drawing the nodes
        # for i in range(len(mtp["nodes"])):
        #     if mtp["nodes"][i]["criticalType"] == 0:
        #         # minimum
        #         circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='blue', edgecolor='black', linewidth=stroke_width)
        #         ax.add_patch(circle)
        #     elif mtp["nodes"][i]["criticalType"] == 1:
        #         # saddle point
        #         circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='red', edgecolor='black', linewidth=stroke_width)
        #         ax.add_patch(circle)
        #     else:
        #         # other ?
        #         circle = patches.Circle((mtp["nodes"][i]["x"], mtp["nodes"][i]["y"]), 0.4, facecolor='black', edgecolor='black', linewidth=stroke_width)
        #         ax.add_patch(circle)
        # # plt.show()
        # save_folder_path = current_dir + "/figures/MT/"
        # if not os.path.exists(save_folder_path):
        #     os.makedirs(save_folder_path)
        # plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
    
    return None





def quantify_persistence_diagrams(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)


    for index, file_name in enumerate(file_names):
        # print(index, file_name)
        if not file_name.endswith("PersistenceDiagram.csv"):
            continue
        file_path = current_dir + folder_name + file_name


        # process PD
        pd = process_persistence_barcode(file_path)


        # TODO
        # - compute total/average distance between each pair and the diagonal

        # initialize array of distances
        d_arr = np.zeros((len(pd),))

        # loop over persistence pairs
        for i,pair in enumerate(pd):

            # extract coordinates
            x = pair["x"]
            y = pair["y0"]
            
            # distance from diagonal
            d_pair = np.abs(y - x) / np.sqrt(2) 

            # store distance
            d_arr[i] = d_pair


        # compute statistics
        d_max = np.nanmax(d_arr)
        d_min = np.nanmin(d_arr)
        d_med = np.nanmedian(d_arr)

        d_tot = np.nansum(d_arr)
        d_ave = d_tot / len(d_arr)


        # show statistics
        file_key = file_name.replace("_PersistenceDiagram.csv", "").replace("low_dim_","")

        if "pinn" in file_key:
            file_key = file_key.replace("_pretrained_convection_u0sin(x)_nu0.0_", "_")
            file_key = file_key.replace("_rho0.0_Nf100_50,50,50,50,1_L1.0_source0_", "_")
            file_key = file_key.replace("_dim2_points1600_", "_")
        elif "resnet" in file_key:
            file_key = file_key.replace("_resnet_loss_landscape_cifar10_", "_") 
            file_key = file_key.replace("_UnstructuredGrid_", "_") 

        # print(f"{file_key:70}\t\t{len(pd):5d}\t{d_tot:12.3f}\t{d_ave:12.3f}\t{d_med:12.3f}\t{d_min:12.3f}\t{d_max:12.3f}")
        print(f"{file_key:70}\t\t{len(pd):5d}\t{d_tot:1.3e}\t{d_ave:1.3e}\t{d_med:1.3e}\t{d_min:1.3e}\t{d_max:1.3e}")

        # fig, ax = plt.subplots(figsize=plt.figaspect(1))
        # max_value = 0
        # min_value = float("inf")
        # for i in range(len(pd)):
        #     point1 = []
        #     point2 = []
        #     max_value = max(max_value, pd[i]["x"])
        #     max_value = max(max_value, pd[i]["y0"])
        #     min_value = min(min_value, pd[i]["x"])
        #     min_value = min(min_value, pd[i]["y0"])
        #     point1.append(pd[i]["x"])
        #     point1.append(pd[i]["x"])
        #     point2.append(pd[i]["x"])
        #     point2.append(pd[i]["y0"])
        #     ax.plot(point1, point2, color='blue')


        # point1 = []
        # point2 = []
        # point1.append(min_value)
        # point1.append(max_value)
        # point2.append(min_value)
        # point2.append(max_value)
        # ax.plot(point1, point2, color="blue")

        # print(file_name[:-4])
        # save_folder_path = current_dir + "/figures/PD/"
        # if not os.path.exists(save_folder_path):
        #     os.makedirs(save_folder_path)
        # plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
    
    return None 



def quantify_merge_tree_planar(folder_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + folder_name)


    for index, file_name in enumerate(file_names):
        # print(index, file_name)
        # if not file_name.endswith("PersistenceDiagram.csv"):
        if not file_name.endswith("MergeTreePlanar.csv"):
            continue
        file_path = current_dir + folder_name + file_name


        # process PD
        mtp = process_merge_tree_planar(file_path)
        pd = mtp
        # print(mtp)
        # TODO
        # - compute total/average distance between each pair and the diagonal

        # initialize array of distances
        d_arr = np.zeros((len(mtp['nodes']),))

        # loop over persistence pairs
        for i,pair in enumerate(mtp['nodes']):

            # print(i, pair)

            d_pair = pair['Persistence']
            
            # # extract coordinates
            # x = pair["x"]
            # y = pair["y0"]
            
            # # distance from diagonal
            # d_pair = np.abs(y - x) / np.sqrt(2) 

            # store distance
            d_arr[i] = d_pair


        # compute statistics
        d_max = np.nanmax(d_arr)
        d_min = np.nanmin(d_arr)
        d_med = np.nanmedian(d_arr)

        d_tot = np.nansum(d_arr)
        d_ave = d_tot / len(d_arr)


        # show statistics
        #file_key = file_name.replace("_PersistenceDiagram.csv", "").replace("low_dim_","")
        file_key = file_name.replace("_MergeTreePlanar.csv", "").replace("low_dim_","")

        if "pinn" in file_key:
            file_key = file_key.replace("_pretrained_convection_u0sin(x)_nu0.0_", "_")
            file_key = file_key.replace("_rho0.0_Nf100_50,50,50,50,1_L1.0_source0_", "_")
            file_key = file_key.replace("_dim2_points1600_", "_")
        elif "resnet" in file_key:
            file_key = file_key.replace("_resnet_loss_landscape_cifar10_", "_") 

        # print(f"{file_key:70}\t\t{len(pd):5d}\t{d_tot:12.3f}\t{d_ave:12.3f}\t{d_med:12.3f}\t{d_min:12.3f}\t{d_max:12.3f}")
        print(f"{file_key:70}\t\t{len(mtp['nodes']):5d}\t{d_tot:12.3f}\t{d_ave:12.3f}\t{d_med:12.3f}\t{d_min:12.3f}\t{d_max:12.3f}")
        # print(f"{file_key:70}\t\t{len(mtp['nodes']):5d}\t{d_tot:1.3e}\t{d_ave:1.3e}\t{d_med:1.3e}\t{d_min:1.3e}\t{d_max:1.3e}")

        # fig, ax = plt.subplots(figsize=plt.figaspect(1))
        # max_value = 0
        # min_value = float("inf")
        # for i in range(len(pd)):
        #     point1 = []
        #     point2 = []
        #     max_value = max(max_value, pd[i]["x"])
        #     max_value = max(max_value, pd[i]["y0"])
        #     min_value = min(min_value, pd[i]["x"])
        #     min_value = min(min_value, pd[i]["y0"])
        #     point1.append(pd[i]["x"])
        #     point1.append(pd[i]["x"])
        #     point2.append(pd[i]["x"])
        #     point2.append(pd[i]["y0"])
        #     ax.plot(point1, point2, color='blue')


        # point1 = []
        # point2 = []
        # point1.append(min_value)
        # point1.append(max_value)
        # point2.append(min_value)
        # point2.append(max_value)
        # ax.plot(point1, point2, color="blue")

        # print(file_name[:-4])
        # save_folder_path = current_dir + "/figures/PD/"
        # if not os.path.exists(save_folder_path):
        #     os.makedirs(save_folder_path)
        # plt.savefig(save_folder_path +file_name[:-3] + ".png", dpi=300, format='png')
    
    return None 



# def calculate_grid_points(folder_name):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_names = os.listdir(current_dir + folder_name)
#     cmap = plt.get_cmap('plasma')

#     for index, file_name in enumerate(file_names):
#         if file_name.endswith(".csv"):
#             if file_name.endswith("MergeTree.csv"):
#                 file_path = current_dir + folder_name + file_name
#                 # print(file_name)
#                 mtp = process_merge_tree(file_path)
#                 # print(mtp)

#                 # obtain the segmentations
#                 segmentations = mtp["segmentations"]
#                 # obtain the nodes
#                 nodes = mtp["nodes"]

#                 # print(len(segmentations))
#                 print(len(nodes))

#                 loss = [seg["loss"] for seg in segmentations]
#                 segmentationID = [seg["SegmentationId"] for seg in segmentations]

#                 # print(len(loss))
#                 # print(len(segmentationID))
