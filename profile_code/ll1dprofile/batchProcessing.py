import os
import pandas as pd
from collections import deque, defaultdict
import json
import numpy as np

# load the parameters from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="/paraview_files_dim3/", help="input paraview file folder")
parser.add_argument("--output", default="/1dprofiletool/api/mongodb_util/basin_points/", help="output file folder")
args = parser.parse_args()

folder_name = args.input
merge_tree_folder_name = "/paraview_coordinates/"
output_folder_name = args.output

class TreeNode:
    def __init__(self, id, loss, parent=None) -> None:
        self.node_id = id
        self.loss = loss
        self.children = {}
        self.parent = parent
        self.off_set_points_left = []
        self.off_set_points_right = []


def dfs(root):
    if not root:
        return

    for ck in root.children.keys():
        child = root.children[ck]
        child.parent = root
        dfs(child)


def process_csv_to_basin_points():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grand_parent_dir = os.path.dirname(parent_dir)
    file_names = os.listdir(parent_dir + folder_name)
    total = 0
    skiped = 0
    skiped_resnet = 0
    skiped_pinn = 0
    skiped_unet = 0

    for _, file_name in enumerate(file_names):
        if file_name.endswith("MergeTree.csv"):
            total += 1
            try:
                merge_tree_file_prefix = parent_dir + merge_tree_folder_name + file_name
                merge_tree_file_prefix = merge_tree_file_prefix.replace("MergeTree.csv", "MergeTree_coordinates.csv")
                file_prefix = parent_dir + folder_name + file_name

                # print(file_prefix)
                # print("2. processing " + merge_tree_file_prefix)

                df = pd.read_csv(file_prefix)
                # Scalar, CriticalType, 
                nodes = df[["Scalar", "CriticalType"]].values.tolist()
                # nodes = [element for sublist in nodes for element in sublist]

                # node metrics
                mdf = pd.read_csv(merge_tree_file_prefix)
                node_metrics = mdf[["HessianEigenvalues"]].values.tolist()

                file_prefix = file_prefix.replace("MergeTree", "MergeTree_edge")
                df2 = pd.read_csv(file_prefix)
                edges = df2[["upNodeId", "downNodeId", "SegmentationId"]].values.tolist()
                
                file_prefix = file_prefix.replace("MergeTree_edge", "MergeTree_segmentation")
                df3 = pd.read_csv(file_prefix)
                segmentations = df3[["Loss", "SegmentationId"]].values.tolist()

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
                        node_dict[source] = TreeNode(source, nodes[source][0])
                        if source not in node_metric_dict:
                            node_metric_dict[source] = {}

                        # HERE, you can add more metrics
                        node_metric_dict[source] = {
                            "HessianEigenvalues": node_metrics[source][0]
                        }

                    if target not in node_dict:
                        node_dict[target] = TreeNode(target, nodes[target][0], node_dict[source])
                        if target not in node_metric_dict:
                            node_metric_dict[target] = {}

                        # HERE, you can add more metrics
                        node_metric_dict[target] = {
                            "HessianEigenvalues": node_metrics[target][0]
                        }
                    node_dict[source].children[target] = node_dict[target]

                for i, s in enumerate(segmentations):
                    loss = s[0]
                    segId = int(s[1])
                    grouped_segmentation[segId].append([loss, i])


                root = None
                for node in node_dict.values():
                    if not node.parent:
                        root = node
                        root.total_width = len(segmentations)

                dfs(root)

                def build_basin(root: TreeNode):
                    if not root:
                        return
                    acc_number = 0
                    for child in root.children.values():
                        acc_number += build_basin(child)
                    root.child_width = acc_number
                    if not root.parent:
                        return
                    # print("get a list of segmentations of " + str(root.node_id))
                    # get a list of segmentations of loss values
                    segmentations = grouped_segmentation[edge_dict[root.node_id]]
                    segmentations.sort()
                    off_set_points_left = deque([])
                    off_set_points_right = deque([])

                    off_set_points_right.append({
                        "x": 0,
                        "y": root.loss,
                        "node_id": root.node_id
                    })
                    off_set_points_left.appendleft({
                        "x": 0,
                        "y": root.loss,
                        "node_id": root.node_id
                    })
                    for s in segmentations:
                        acc_number += 1
                        off_set_points_right.append({
                            "x": acc_number / 2,
                            "y": s[0],
                            "node_id": s[1]
                        })
                        off_set_points_left.appendleft({
                            "x": -acc_number / 2,
                            "y": s[0],
                            "node_id": s[1]
                        })

                    if root.parent:
                        off_set_points_right.append({
                            "x": acc_number / 2,
                            "y": root.parent.loss,
                            "node_id": root.parent.node_id
                        })
                        off_set_points_left.appendleft({
                            "x": - acc_number / 2,
                            "y": root.parent.loss,
                            "node_id": root.parent.node_id
                        })

                    root.off_set_points_left = off_set_points_left
                    root.off_set_points_right = off_set_points_right
                    root.total_width = acc_number

                    return acc_number

                build_basin(root)

                def assign_center(root: TreeNode, start: int, end: int):
                    if not root:
                        return

                    root.center = (start + end) / 2
                    if len(root.children.values()) == 0:
                        return


                    left = start+ (end - start) / 2 - root.child_width / 2
                    childrens = root.children.values()
                    childrens = sorted(childrens, key=lambda item: item.total_width, reverse=True);
                    for child in childrens:
                        proportion = child.total_width / root.child_width
                        partial_length = root.child_width * proportion
                        sub_start = left
                        sub_end = left + partial_length
                        assign_center(child, sub_start, sub_end)
                        left += partial_length


                assign_center(root, 0, len(segmentations) )


                merge_tree_nodes = []
                def collect_merge_tree_nodes(root):
                    if not root:
                        return

                    merge_tree_nodes.append({
                        "id": root.node_id,
                        "y": root.loss,
                        "x": root.center,
                        "nodeType": nodes[root.node_id][1],
                        "info": {
                            "HessianEigenvalues": node_metric_dict[root.node_id]["HessianEigenvalues"]
                        }
                    })

                    for child in root.children.values():
                        collect_merge_tree_nodes(child)


                collect_merge_tree_nodes(root)
                # print(merge_tree_nodes)


                merge_tree_edges = []
                # for e in edges:
                #     source = e[0]
                #     target = e[1]
                #     # vertical lines
                #     merge_tree_edges.append({
                #         "id": str(source) + "_" + str(target),
                #         "sourceX": node_dict[source].center,
                #         "sourceY": node_dict[source].loss,
                #         "targetX": node_dict[target].center,
                #         "targetY": node_dict[target].loss,
                #     })

                def collect_merge_tree_edges(root):
                    if not root:
                        return

                    if root.parent:
                        merge_tree_edges.append({
                            "id": str(root.parent.node_id) + "_" + str(root.node_id),
                            "sourceX": root.center,
                            "sourceY": root.parent.loss,
                            "targetX": root.center,
                            "targetY": root.loss
                        })
                    childrens = list(root.children.values())
                    if len(childrens) == 0:
                        return
                    if len(childrens) == 1:
                        merge_tree_edges.append({
                            "id": str(root.node_id) + "_" + str(childrens[0].node_id),
                            "sourceX": root.center,
                            "sourceY": root.loss,
                            "targetX": childrens[0].center,
                            "targetY": root.loss
                        })
                    else:
                        sorted_childrens = sorted(childrens, key=lambda item: item.center)
                        merge_tree_edges.append({
                            "id": str(sorted_childrens[0].node_id) + "_" + str(sorted_childrens[-1].node_id),
                            "sourceX": sorted_childrens[0].center,
                            "sourceY": root.loss,
                            "targetX": sorted_childrens[-1].center,
                            "targetY": root.loss
                        })


                    for child in root.children.values():
                        collect_merge_tree_edges(child)

                collect_merge_tree_edges(root)

                # print(merge_tree_edges)
                # print("Finish processing " + file_name)

                res = []
                basin_upper_points = []
                basin_upper_points_id = []
                saddles = []
                saddle_id = []

                def collect_individual_basins(root):
                    if not root:
                        return

                    for child in root.children.values():
                        collect_individual_basins(child)

                    left = [[ori["x"] + root.center, ori["y"]] for ori in root.off_set_points_left]
                    left_id = [ori["node_id"] for ori in root.off_set_points_left]
                    right = [[ori["x"] + root.center, ori["y"]] for ori in root.off_set_points_right]
                    right_id = [ori["node_id"] for ori in root.off_set_points_right]
                    res.append({
                        "area": left + right,
                        "isBasin": True if len(root.children.values()) == 0 else False
                    })

                    # calculate points for the upper line of each basin
                    total_individual = left + right
                    total_individual_id = left_id + right_id
                    max_x_upper_id = 0
                    min_x_upper_id = 0
                    if len(total_individual) != 0:
                        total_individual_arr = np.array(total_individual)
                        max_y = max(total_individual_arr, key = lambda x: x[1])[1]
                        min_x = min(total_individual_arr, key = lambda x: x[0])[0]
                        max_x = max(total_individual_arr, key = lambda x: x[0])[0]
                        max_x_upper = min_x
                        min_x_upper = max_x
                        for i in range(len(total_individual_arr)):
                            if total_individual_arr[i][1] == max_y:
                                # basin_upper_points.append(total_individual_arr[i])
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

                        # df_basin_upper_points = pd.DataFrame(basin_upper_points, columns=['x','y'])
                        # df_basin_upper_points_sorted = df_basin_upper_points.sort_values(by=['y', 'x'])
                        basin_upper_points.sort(key=lambda x: (x[1], x[0]))
                        # print("basin_upper_points", basin_upper_points)
                        # print("df_basin_upper_points_sorted", df_basin_upper_points_sorted)

                        # collect the saddle points
                        # saddles = {x for x in basin_upper_points if basin_upper_points.count(x) > 1}
                        # for element in basin_upper_points:
                        #     if basin_upper_points.count(element) > 1:
                        #         if element not in saddles:
                        #             saddles.append(element)
                        for i in range(len(basin_upper_points)):
                            if basin_upper_points[i][0] in [x[0] for x in basin_upper_points[i+1:]] and basin_upper_points[i][1] in [x[1] for x in basin_upper_points[i+1:]]:
                                if basin_upper_points[i] not in saddles:
                                    saddles.append(basin_upper_points[i])
                                    saddle_id.append(basin_upper_points_id[i])
                        
                        # df_saddles = pd.DataFrame(saddles, columns=['x','y'])
                        # df_saddles_sorted = df_saddles.sort_values(by=['y', 'x'])
                        # print("saddles", saddles)
                
                collect_individual_basins(root)
                
                # save the basin points
                save_folder_path = parent_dir + output_folder_name
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                
                # save an array as .json file
                save_file_path = save_folder_path + file_name.replace("_MergeTree.csv", ".json") 
                print("saving to " + save_file_path)
                json.dump({"points": res, "basin_upper_points": basin_upper_points, "saddles": saddles, "basin_upper_points_id": basin_upper_points_id, "saddle_id": saddle_id, "mergeTreeNodes": merge_tree_nodes, "mergeTreeEdges": merge_tree_edges}, open(save_file_path, "w"))

            except Exception as e:
                print("#######skiped" + str(e) )
                skiped += 1
                if "resnet" in file_name:
                    skiped_resnet += 1
                elif "pinn" in file_name:
                    skiped_pinn += 1
                elif "iter" in file_name:
                    skiped_unet += 1
                continue

    print("total: " + str(total))
    print("skiped: " + str(skiped))
    print("skiped_resnet: " + str(skiped_resnet))
    print("skiped_pinn: " + str(skiped_pinn))
    print("skiped_unet: " + str(skiped_unet))


if __name__ == "__main__":
    process_csv_to_basin_points()

