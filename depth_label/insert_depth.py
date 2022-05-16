import csv
import json
import numpy as np
from glob import glob
from os import path as op

import config as cfg


def insert_depth():
    dir_lists = load_dirlist()
    for directory in dir_lists:
        read_label(directory)


def load_dirlist():
    dirlist = glob(op.join(cfg.data_path, '*'))
    dirlist = [directory for directory in dirlist if op.isdir(op.join(directory, "image"))]
    dirlist.sort()
    return dirlist


def read_label(directory):
    label_lists = glob(op.join(directory, "label", "*.txt"))
    label_lists.sort()
    for label_file in label_lists:
        lidar_file = label_file.replace("label", "lidar").replace(".txt", ".npz")
        lidar = np.load(lidar_file)
        depth = lidar["depth"]
        bbox_data = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            split_line = lines.index("---\n")
            bbox_lines = lines[:split_line]
            for line in bbox_lines:
                bbox_data.append(extract_depth(line, depth))
        write_label(label_file, bbox_data, lines[split_line:])


def extract_depth(line, depth):
    raw_label = line.strip("\n").split(",")
    ctgr, y1, x1, h, w, dist = raw_label
    y2 = int(y1) + int(h)
    x2 = int(x1) + int(w)
    box_depth = depth[int(y1):y2, int(x1):x2]
    dist = imple_depth(box_depth)
    return ctgr, y1, x1, h, w, dist


def imple_depth(data):
    value = data[np.where(data > 0)]
    if value.size == 0:
        return 0
    return np.quantile(value, 0.2)


def write_label(label_file, new_box_data, lane_data):
    with open(label_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(new_box_data)
        f.writelines(lane_data)
        # json.dump(lane_data, f, ensure_ascii=False)







if __name__ == "__main__":
    insert_depth()


