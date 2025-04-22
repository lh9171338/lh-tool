# -*- encoding: utf-8 -*-
"""
@File    :   pcd2ply.py
@Time    :   2024/09/16 20:28:50
@Author  :   lh9171338
@Version :   1.0
@Contact :   2909171338@qq.com
"""


import os
import glob
import cv2
import numpy as np
import open3d as o3d
from pypcd import pypcd
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def get_class_colors(num_classes):
    """
    Get class colors

    Args:
        num_classes (int): number of classes

    Returns:
        colors (list): list of class colors
    """
    colors = np.ones((1, num_classes, 3), dtype="float32")
    colors[:, :, 0] = np.linspace(0, 360, num_classes + 1)[:-1]
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2RGB)
    colors = colors[0]

    return colors


def pcd2ply(
    pcd_file,
    ply_file,
    with_label=False,
):
    """
    Convert pcd file to ply file

    Args:
        pcd_file (str): pcd file path
        ply_file (str): ply file path
        with_label (bool, optional): Whether to include label. Defaults to False

    Returns:
        None
    """
    pc = pypcd.PointCloud.from_path(pcd_file)
    pc_data = pc.pc_data
    points = np.stack(
        [
            pc_data["x"],
            pc_data["y"],
            pc_data["z"],
        ],
        axis=-1,
    )
    if with_label and "label" in pc.fields:
        labels = pc_data["label"]
        unique_labels, labels = np.unique(labels, return_inverse=True)
        num_classes = len(unique_labels)
        class_colors = get_class_colors(num_classes)
        colors = class_colors[labels]
    else:
        colors = np.ones_like(points)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_file, point_cloud)


def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="path of input pcd files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output ply files. The default is the same as the " "path of input pcd files",
    )
    parser.add_argument(
        "-l",
        "--label",
        dest="with_label",
        action="store_true",
        help="Whether to include label",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="convert pcd to ply files recursively",
    )
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        input_path = opts.input
        output_path = opts.output
        with_label = opts.with_label
        recursive = opts.recursive
        nprocs = opts.nprocs
        if os.path.isfile(input_path):
            nprocs = 1
            input_file_list = [input_path]
            if output_path is not None:
                output_file_list = [output_path]
            else:
                output_file_list = [os.path.splitext(input_path)[0] + ".ply"]
        elif os.path.isdir(input_path):
            assert recursive, "Please use -r to convert pcd to ply files recursively."
            input_file_list = glob.glob(os.path.join(input_path, "*.pcd"))
            output_file_list = []
            for input_file in input_file_list:
                if output_path is not None:
                    filename = os.path.basename(input_file)
                    filename = os.path.splitext(filename)[0] + ".ply"
                    output_file = os.path.join(output_path, filename)
                else:
                    output_file = os.path.splitext(input_file)[0] + ".ply"
                output_file_list.append(output_file)
        else:
            raise ValueError("Input path must be a file or directory.")

        if nprocs > 1:
            iterator = MultiProcess(pcd2ply, nprocs=nprocs)
        else:
            iterator = SingleProcess(pcd2ply)
        iterator.run(input_file_list, output_file_list, with_label=with_label)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
