# Notes:
# - accept loss landscape as a list of lists
# - return csv files provided by paraview

# load the parameters from the command line
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--loss-landscape-file", default=None, help="input npy file")
parser.add_argument("--loss-coords-file", default=None, help="input npy file")
parser.add_argument("--loss-values-file", default=None, help="input npy file")
parser.add_argument("--output-path", default=None, help="output file name (no extension)")
parser.add_argument("--output-folder", default=None, help="output folder name (use instead of --output-path)")
parser.add_argument("--vtk-format", default="vtu", help="output file format (vti or vtu)")
parser.add_argument("--graph-kwargs", default="aknn", help="algorithm for constructing graph")
parser.add_argument("--persistence-threshold", type=float, default=0, help="Threshold for simplification by persistence (use --threshold-is-absolute if passing a scalar value.")
parser.add_argument("--threshold-is-absolute", action="store_true", help="Is the threshold an absolute scalar value or a fraction (0 - 1) of the function range.")
args = parser.parse_args()

# check output path
import os
if args.output_path is None:

	if args.output_folder is None:
		args.output_path = args.loss_coords_file.replace('.npy','')
		args.output_path = args.output_path.replace('loss_landscape_files','paraview_files')
	else:
		args.output_path = os.path.basename(args.loss_coords_file.replace('.npy',''))
		args.output_path = f"{args.output_folder}/{args.output_path}"

elif args.output_path.endswith('.npy'):
	args.output_path = args.output_path.replace('.npy','')



### load loss_landscape from a file
import numpy as np
loss_coords = np.load(args.loss_coords_file)
loss_values = np.load(args.loss_values_file)


### functions taking matrix as input
from ttk_functions import compute_persistence_barcode, compute_merge_tree, compute_merge_tree_planar



### Use VTK Unstructured Grid 

# compute persistence barcode
persistence_barcode = compute_persistence_barcode(
	loss_landscape=None, loss_coords=loss_coords, loss_values=loss_values,
	output_path=args.output_path, vtk_format='vtu', graph_kwargs=args.graph_kwargs,
	persistence_threshold=args.persistence_threshold, threshold_is_absolute=args.threshold_is_absolute)

# compute merge tree
merge_tree = compute_merge_tree(
	loss_landscape=None, loss_coords=loss_coords, loss_values=loss_values,
	output_path=args.output_path, vtk_format='vtu', graph_kwargs=args.graph_kwargs,
	persistence_threshold=args.persistence_threshold, threshold_is_absolute=args.threshold_is_absolute)

# compute merge tree (planar version)
# merge_tree = compute_merge_tree_planar(
#	loss_landscape=None, loss_coords=loss_coords, loss_values=loss_values,
# 	output_path=args.output_path, vtk_format='vtu', graph_kwargs=args.graph_kwargs,
# 	persistence_threshold=args.persistence_threshold, threshold_is_absolute=args.threshold_is_absolute)

