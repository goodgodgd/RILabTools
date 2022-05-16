import cv2
import csv
from glob import glob
from os import path as op

import config as cfg


class DistModi:
    def __init__(self):
        dirlist = glob(op.join(cfg.data_path, '*'))
        self.dirlists = [directory for directory in dirlist if op.isdir(op.join(directory, "image"))]
        self.dirlists.sort()
        self.image = None
        self.bbox_data = None
        self.lane_data = None

    def __call__(self):
        for directory in self.dirlists:
            image_lists = self.load_files(directory)
            self.pairmatch_files(image_lists)

    def load_files(self, directory):
        image_lists = glob(op.join(directory, "image", "*.jpg"))
        image_lists.sort()
        return image_lists

    def pairmatch_files(self, image_files):
        for image_file in image_files:
            label_file = image_file.replace("image", "label").replace("jpg", "txt")
            self.bbox_data, self.lane_data = self.split_label(label_file)
            self.draw_bbox(image_file)

            cv2.namedWindow('image')
            cv2.setMouseCallback("image", self.onMouse, image_file)
            while True:
                cv2.imshow("image", self.image)
                k = cv2.waitKey(30)
                if k == ord("s"):
                    self.write_label(label_file)
                    break

    def split_label(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            split_line = lines.index("---\n")
            bbox_data = lines[:split_line]
            lane_data = lines[split_line:]
        return bbox_data, lane_data

    def draw_bbox(self, image_file):
        image = cv2.imread(image_file)
        for line in self.bbox_data:
            raw_label = line.strip("\n").split(",")
            ctgr, y1, x1, h, w, dist = raw_label
            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y1) + int(h)
            x2 = int(x1) + int(w)
            dist = float(dist)

            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            self.image = cv2.putText(image, f"{dist:.2f}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    def onMouse(self, event, x, y, flags, params):
        new_bbox = []
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.image = cv2.imread(params)
            for line in self.bbox_data:
                raw_label = line.strip("\n").split(",")
                ctgr, y1, x1, h, w, dist = raw_label
                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y1) + int(h)
                x2 = int(x1) + int(w)
                dist = float(dist)

                if (x1 < x <= x2) and (y1 < y <= y2):
                    dist = 0
                new_bbox.append(f"{ctgr}, {y1}, {x1}, {h}, {w}, {dist}")
            self.bbox_data = new_bbox
            self.draw_bbox(params)

    def write_label(self, label_file):
        with open(label_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows([bbox.strip("\n").split(",") for bbox in self.bbox_data])
            f.writelines(self.lane_data)


if __name__ == "__main__":
    modifier = DistModi()
    modifier()
