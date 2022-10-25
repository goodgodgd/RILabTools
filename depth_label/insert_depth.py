import csv
import config as cfg
import numpy as np
import cv2
import os.path as op
from glob import glob


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
        # image_file = label_file.replace("label", "image").replace(".txt", ".jpg")
        lidar_file = label_file.replace("label", "lidar").replace(".txt", ".npz")
        # image = cv2.imread(image_file)
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


def extract_depth(line, depth, image=None):
    raw_label = line.strip("\n").split(",")
    ctgr, y1, x1, h, w, dist = raw_label
    h = int(float(h))
    w = int(float(w))
    y1 = int(float(y1))
    x1 = int(float(x1))
    y2 = int(float(y1)) + int(float(h))
    x2 = int(float(x1)) + int(float(w))
    reduce_h = h * cfg.box_ratio
    reduce_w = w * cfg.box_ratio

    if image is not None:
        image = cv2.circle(image, (int(x1), int(y1)), 5, (0, 255, 255), 5)
        image = cv2.rectangle(image, (int(x1+reduce_w), int(y1+reduce_h)), (int(x2-reduce_w), int(y2-reduce_h)), (255, 0, 0), 5)
        cv2.imshow("test", image)
        cv2.waitKey()
    box_depth = depth[int(y1):y2, int(x1):x2]
    dist = imple_depth(box_depth)
    return ctgr, y1, x1, h, w, f" {dist}"


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


if __name__ == "__main__":
    insert_depth()


