import os
import pandas as pd
from collections import deque, defaultdict
import json
import numpy as np
import math
import re

# load the parameters from the command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="/paraview_files_tktest/",
    help="input paraview file folder",
)
parser.add_argument(
    "--output",
    default="/basin_points_tktest/",
    help="output file folder",
)

parser.add_argument(
    "--scatterDir",
    default="/loss_landscape_global_random_sampling_files/",
    help="scatter plot files to be merged to the data",
)

args = parser.parse_args()

folder_name = args.input
output_folder_name = args.output
scatter_folder_name = args.scatterDir


class TreeNode:
    def __init__(self, id, loss, parent=None) -> None:
        self.node_id = id
        self.loss = loss
        self.children = {}
        self.parent = parent
        self.off_set_points_left = []
        self.off_set_points_right = []
        self.acc_seg_point_ids = set()


def dfs(root):
    if not root:
        return

    for ck in root.children.keys():
        child = root.children[ck]
        child.parent = root
        dfs(child)


def process_csv_to_basin_points():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_names = os.listdir(parent_dir + folder_name)
    total = 0
    skiped = 0
    model_in_basin_coordinates = defaultdict(list)
    number_of_models = 3  # change this later

    for _, file_name in enumerate(file_names):
        if file_name.endswith("MergeTree.csv"):
            print(file_name)
            total += 1
            # try:
            file_prefix = parent_dir + folder_name + file_name
            df = pd.read_csv(file_prefix)
            # Scalar, CriticalType,
            # Check if the HessianEigenvalues column exists
            nodes = {}
            if "NodeId" in df.columns:
                for _, row in df.iterrows():
                    nodes[row["NodeId"]] = [row["Scalar"], row["CriticalType"]]
            else:
                for idx, row in df.iterrows():
                    nodes[idx] = [row["Scalar"], row["CriticalType"]]
            if "HessianEigenvalues" in df.columns:
                if "NodeId" in df.columns:
                    for _, row in df.iterrows():
                        nodes[row["NodeId"]].append(row["HessianEigenvalues"])
                else:
                    for idx, row in df.iterrows():
                        nodes[idx].append(row["HessianEigenvalues"])
            else:
                print("HessianEigenvalues column not found in the CSV file.")

            file_prefix = file_prefix.replace("MergeTree", "MergeTree_edge")
            df2 = pd.read_csv(file_prefix)
            # edges = df2[["upNodeId", "downNodeId", "SegmentationId"]].values.tolist()
            edges = df2[["upNodeIdDataId", "downNodeIdDataId", "SegmentationId"]].values.tolist()

            file_prefix = file_prefix.replace(
                "MergeTree_edge", "MergeTree_segmentation"
            )
            df3 = pd.read_csv(file_prefix)
            # segmentations = df3[["Loss", "SegmentationId", "DataID"]].values.tolist()
            print(df3)
            segmentations = df3[["Loss", "SegmentationId"]].values.tolist()
            print("###########################################################")
            print(segmentations[0])
            print(segmentations[1])
            print(segmentations[2])

            # node_id -> TreeNode
            node_dict = {}
            # target_node_id -> segmentation_id
            edge_dict = {}
            # segmentation_id -> list of loss values
            grouped_segmentation = defaultdict(list)

            node_metric_dict = {}
            for e in edges:
                source = e[0]
                target = e[1]
                segId = e[2]
                edge_dict[target] = segId
                if source not in node_dict:
                    # try:
                    node_dict[source] = TreeNode(source, nodes[source][0])
                    # except:
                    #    print(file_name)
                    #    exit()
                    if source not in node_metric_dict:
                        node_metric_dict[source] = {}
                    if "HessianEigenvalues" in df.columns:
                        node_metric_dict[source]["HessianEigenvalues"] = nodes[source][
                            2
                        ]

                if target not in node_dict:
                    node_dict[target] = TreeNode(
                        target, nodes[target][0], node_dict[source]
                    )
                    if target not in node_metric_dict:
                        node_metric_dict[target] = {}
                    if "HessianEigenvalues" in df.columns:
                        node_metric_dict[target]["HessianEigenvalues"] = nodes[target][
                            2
                        ]

                node_dict[source].children[target] = node_dict[target]

            for i, s in enumerate(segmentations):
                loss = s[0]
                segId = int(s[1])
                # dataID = int(s[2])
                grouped_segmentation[segId].append([loss, i])

            root = None
            for node in node_dict.values():
                if not node.parent:
                    root = node
                    root.total_width = len(segmentations)

            dfs(root)

            # put model id in the basket so that the the basins can get it.
            for i in range(number_of_models):
                segmentation = segmentations[i]
                print(segmentation)
                segId = int(segmentation[1])
                loss = segmentation[0]
                model_in_basin_coordinates[segId].append({"id": i, "y": loss})

            print(model_in_basin_coordinates)

            def build_basin(root: TreeNode):
                acc_number = 2
                child_width = 0
                for child in root.children.values():
                    child_width += build_basin(child)
                root.child_width = child_width
                if not root.parent:
                    return
                segmentations = grouped_segmentation[edge_dict[root.node_id]]
                segmentations.sort()
                off_set_points_left = deque([])
                off_set_points_right = deque([])

                off_set_points_right.append(
                    {"x": 0, "y": root.loss, "node_id": root.node_id}
                )

                off_set_points_left.appendleft(
                    {"x": 0, "y": root.loss, "node_id": root.node_id}
                )

                for s in segmentations:
                    acc_number += 1
                    off_set_points_right.append(
                        {
                            "x": math.sqrt(acc_number) / 2 + child_width / 2,
                            "y": s[0],
                            "node_id": s[1],
                        }
                    )
                    off_set_points_left.appendleft(
                        {
                            "x": -math.sqrt(acc_number) / 2 - child_width / 2,
                            "y": s[0],
                            "node_id": s[1],
                        }
                    )
                    root.acc_seg_point_ids.add(s[1])

                if root.parent:
                    off_set_points_right.append(
                        {
                            "x": math.sqrt(acc_number) / 2 + child_width / 2,
                            "y": root.parent.loss,
                            "node_id": root.parent.node_id,
                        }
                    )
                    off_set_points_left.appendleft(
                        {
                            "x": -math.sqrt(acc_number) / 2 - child_width / 2,
                            "y": root.parent.loss,
                            "node_id": root.parent.node_id,
                        }
                    )

                root.off_set_points_left = off_set_points_left
                root.off_set_points_right = off_set_points_right
                root.total_width = math.sqrt(acc_number) + child_width

                return math.sqrt(acc_number) + child_width

            build_basin(root)

            models_in_basin = {}

            def assign_center(root: TreeNode, start: int, end: int):
                if not root:
                    return

                if root.node_id in edge_dict:
                    segId = edge_dict[root.node_id]
                    if segId in model_in_basin_coordinates:
                        model_coordinates = model_in_basin_coordinates[segId]
                        for mc in model_coordinates:
                            models_in_basin[mc["id"]] = {
                                "x": (start + end) / 2,
                                "model_id": mc["id"],
                                "y": mc["y"],
                            }

                root.center = (start + end) / 2
                if len(root.children.values()) == 0:
                    return

                left = start + (end - start) / 2 - root.child_width / 2
                childrens = root.children.values()
                childrens = sorted(
                    childrens, key=lambda item: item.total_width, reverse=True
                )
                for child in childrens:
                    proportion = child.total_width / root.child_width
                    partial_length = root.child_width * proportion
                    sub_start = left
                    sub_end = left + partial_length
                    assign_center(child, sub_start, sub_end)
                    left += partial_length

            assign_center(root, 0, root.total_width)
            merge_tree_nodes = []

            def collect_merge_tree_nodes(root):
                if not root:
                    return

                merge_tree_nodes.append(
                    {
                        "id": root.node_id,
                        "y": root.loss,
                        "x": root.center,
                        "nodeType": nodes[root.node_id][1],
                    }
                )

                for child in root.children.values():
                    collect_merge_tree_nodes(child)

            collect_merge_tree_nodes(root)

            merge_tree_edges = []

            def collect_merge_tree_edges(root):
                if not root:
                    return

                if root.parent:
                    merge_tree_edges.append(
                        {
                            "id": str(root.parent.node_id) + "_" + str(root.node_id),
                            "sourceX": root.center,
                            "sourceY": root.parent.loss,
                            "targetX": root.center,
                            "targetY": root.loss,
                        }
                    )
                childrens = list(root.children.values())
                if len(childrens) == 0:
                    return
                if len(childrens) == 1:
                    merge_tree_edges.append(
                        {
                            "id": str(root.node_id) + "_" + str(childrens[0].node_id),
                            "sourceX": root.center,
                            "sourceY": root.loss,
                            "targetX": childrens[0].center,
                            "targetY": root.loss,
                        }
                    )
                else:
                    sorted_childrens = sorted(childrens, key=lambda item: item.center)
                    merge_tree_edges.append(
                        {
                            "id": str(sorted_childrens[0].node_id)
                            + "_"
                            + str(sorted_childrens[-1].node_id),
                            "sourceX": sorted_childrens[0].center,
                            "sourceY": root.loss,
                            "targetX": sorted_childrens[-1].center,
                            "targetY": root.loss,
                        }
                    )

                for child in root.children.values():
                    collect_merge_tree_edges(child)

            collect_merge_tree_edges(root)

            def collect_seg_point_ids(root: TreeNode) -> set:
                curr = set()
                for child in root.children.values():
                    curr = curr.union(collect_seg_point_ids(child))
                root.acc_seg_point_ids = curr.union(root.acc_seg_point_ids)
                return root.acc_seg_point_ids

            collect_seg_point_ids(root)

            print("Finish processing " + file_name)

            res = []
            basin_upper_points = []
            basin_upper_points_id = []
            saddles = []
            saddle_id = []

            def collect_individual_basins(root: TreeNode):
                for child in root.children.values():
                    collect_individual_basins(child)

                left = [
                    [ori["x"] + root.center, ori["y"]]
                    for ori in root.off_set_points_left
                ]
                left_id = [ori["node_id"] for ori in root.off_set_points_left]
                right = [
                    [ori["x"] + root.center, ori["y"]]
                    for ori in root.off_set_points_right
                ]
                right_id = [ori["node_id"] for ori in root.off_set_points_right]

                res.append(
                    {
                        "area": left + right,
                        "isBasin": True if len(root.children.values()) == 0 else False,
                        "segID": (
                            edge_dict[root.node_id] if root.node_id in edge_dict else -1
                        ),
                        "segPointIds": list(root.acc_seg_point_ids),
                        "hessianEigenvalues": (
                            node_metric_dict[root.node_id]["HessianEigenvalues"]
                            if root.node_id in node_metric_dict
                            and "HessianEigenvalues" in node_metric_dict[root.node_id]
                            else []
                        ),
                    }
                )

                # calculate points for the upper line of each basin
                total_individual = left + right
                total_individual_id = left_id + right_id
                max_x_upper_id = 0
                min_x_upper_id = 0
                if len(total_individual) != 0:
                    total_individual_arr = np.array(total_individual)
                    max_y = max(total_individual_arr, key=lambda x: x[1])[1]
                    min_x = min(total_individual_arr, key=lambda x: x[0])[0]
                    max_x = max(total_individual_arr, key=lambda x: x[0])[0]
                    max_x_upper = min_x
                    min_x_upper = max_x
                    for i in range(len(total_individual_arr)):
                        if total_individual_arr[i][1] == max_y:
                            if total_individual_arr[i][0] > max_x_upper:
                                max_x_upper = total_individual_arr[i][0]
                                max_x_upper_id = total_individual_id[i]
                            if total_individual_arr[i][0] < min_x_upper:
                                min_x_upper = total_individual_arr[i][0]
                                min_x_upper_id = total_individual_id[i]
                    basin_upper_points.append([min_x_upper, max_y])
                    basin_upper_points_id.append(min_x_upper_id)
                    basin_upper_points.append([max_x_upper, max_y])
                    basin_upper_points_id.append(max_x_upper_id)

                    basin_upper_points.sort(key=lambda x: (x[1], x[0]))

                    for i in range(len(basin_upper_points)):
                        if basin_upper_points[i][0] in [
                            x[0] for x in basin_upper_points[i + 1 :]
                        ] and basin_upper_points[i][1] in [
                            x[1] for x in basin_upper_points[i + 1 :]
                        ]:
                            if basin_upper_points[i] not in saddles:
                                saddles.append(basin_upper_points[i])
                                saddle_id.append(basin_upper_points_id[i])

            collect_individual_basins(root)

            # save the basin points
            save_folder_path = parent_dir + output_folder_name
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            # save an array as .json file
            save_file_path = save_folder_path + file_name.replace(
                "_MergeTree.csv", ".json"
            )
            print("saving to " + save_file_path)
            json.dump(
                {
                    "points": res,
                    "basin_upper_points": basin_upper_points,
                    "saddles": saddles,
                    "basin_upper_points_id": basin_upper_points_id,
                    "saddle_id": saddle_id,
                    "mergeTreeNodes": merge_tree_nodes,
                    "mergeTreeEdges": merge_tree_edges,
                    "groupedSegmentation": grouped_segmentation,
                    "modelsInBasin": list(models_in_basin.values()),
                },
                open(save_file_path, "w"),
            )
            model_in_basin_coordinates = defaultdict(list)

            # except Exception as e:
            #     print("#######skiped" + str(e) )
            #     skiped += 1
            #     continue
        else:
            continue

    print("total: " + str(total))
    print("skiped: " + str(skiped))


def merge_scatter_plot_data():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = parent_dir + output_folder_name
    tmp_directory = parent_dir + scatter_folder_name
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):

            # Construct the corresponding loss_value file name

            pattern = r"_UnstructuredGrid_aknn_PersistenceThreshold_[^_]+_ThresholdIsAbsolute_0"

            loss_filename = re.sub(pattern, "", filename)
            print(f"loss_filename is {loss_filename}")

            basin_point_filepath = os.path.join(directory, filename)
            loss_value_filepath = os.path.join(tmp_directory, loss_filename)
            loss_value_filepath = loss_value_filepath.replace(
                "Nf100_L1",
                "Nf100_50,50,50,50,1_L1",
            )

            # Check if the corresponding loss_value file exists
            if os.path.exists(loss_value_filepath):
                # Read basin_point JSON file
                with open(basin_point_filepath, "r") as basin_file:
                    basin_data = json.load(basin_file)

                # Read loss_value JSON file
                with open(loss_value_filepath, "r") as loss_file:
                    loss_data = json.load(loss_file)

                # Merge loss_values into basin_data
                basin_data["loss_values"] = loss_data["loss_values"]
                basin_data["loss_coordinates"] = loss_data["loss_coordinates"]

                # Write the merged data back to the basin_point JSON file
                with open(basin_point_filepath, "w") as basin_file:
                    json.dump(basin_data, basin_file, indent=4)

                print(f"Merged {loss_filename} into {filename}")
            else:
                print(f"No corresponding loss_value file found for {filename}")


if __name__ == "__main__":
    process_csv_to_basin_points()
    merge_scatter_plot_data()
