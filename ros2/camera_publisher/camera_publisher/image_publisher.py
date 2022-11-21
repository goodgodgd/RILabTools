#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from my_msg.msg import Mystamp

import camera_publisher.config as cfg


class CameraPub(Node):
    def __init__(self):
        super().__init__('camera_pub')
        self.publisher_ = self.create_publisher(Image, 'image_raw', qos_profile_sensor_data)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.pub_callback)
        self.subscriber = self.create_subscription(Mystamp, 'box_pub', self.data_callback, qos_profile_sensor_data)

        self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.stamp_image = Image()
        self.stamps = []
        self.img_dict = dict()
        self.boxes_dict = dict()
        self.lanes_dict = dict()
        self.count = 0
        self.ime_check_total = list()

        self.crop_tlbr = self.find_crop_range((1080, 1920))
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Test.CATEGORY_NAMES["major"])}
        self.lane_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Test.CATEGORY_NAMES["lane"])}

    def find_crop_range(self, src_hw):  # example:
        src_hw = np.array(src_hw, dtype=np.float32)  # [220, 540]
        offset = np.array([300, 0, 0, 0], dtype=np.int32)  # [10, 20, 10, 20]
        src_crop_hw = src_hw - (offset[:2] + offset[2:])  # [200, 500]
        src_hw_ratio = src_crop_hw[1] / src_crop_hw[0]  # 2.5
        dst_hw_ratio = 2.5  # 2
        if dst_hw_ratio < src_hw_ratio:  # crop x-axis, dst_hw=[200, 400]
            dst_hw = np.array([src_crop_hw[0], src_crop_hw[0] * dst_hw_ratio], dtype=np.int32)
        else:
            dst_hw = np.array([src_crop_hw[1] / dst_hw_ratio, src_crop_hw[1]], dtype=np.int32)
        # crop with fixed center, ([200, 500]-[200, 400])/2 = [0, 50]
        addi_crop_yx = ((src_crop_hw - dst_hw) // 2).astype(np.int32)
        # crop top left bottom right, [10, 20, 10, 20] + [0, 50, 0, 50] = [10, 70, 10, 70]
        crop_tlbr = offset + np.concatenate([addi_crop_yx, addi_crop_yx], axis=0)
        # cropped image range, [10, 70, [220, 540]-[10, 70]] = [10, 70, 210, 470]
        crop_tlbr = np.concatenate([crop_tlbr[:2], src_hw - crop_tlbr[2:]])
        return crop_tlbr

    def pub_callback(self):
        ret, frame = self.cap.read()
        if ret:

            enc_img, frame = self.preprocess(frame)
            cv2.imshow("sss", frame)
            cv2.waitKey(10)

            img_msg = self.br.cv2_to_imgmsg(enc_img)
            self.stamp_image = img_msg
            self.stamp_image.header.stamp = self.get_clock().now().to_msg()
            self.img_dict.update({str(self.stamp_image.header.stamp): frame})
            self.stamps.append(str(self.stamp_image.header.stamp))

            self.publisher_.publish(self.stamp_image)
            if len(self.img_dict.keys()) > 150:
                del (self.img_dict[self.stamps[0]])
                self.stamps.remove(self.stamps[0])

        self.get_logger().info('Publishing frame')

    def preprocess(self, frame):
        frame = self.crop_image(frame, self.crop_tlbr)
        frame = self.resize(frame)
        encode_param= [int(cv2.IMWRITE_JPEG_QUALITY),50]
        result, encimg = cv2.imencode('.jpg',frame, encode_param)
        return encimg, frame

    def crop_image(self, image, crop_tlbr):
        image = image[int(crop_tlbr[0]):int(crop_tlbr[2]), int(crop_tlbr[1]):int(crop_tlbr[3]), :]
        return image

    def resize(self, image):
        target_hw = np.array([512, 1280], dtype=np.float32)
        source_hw = np.array(image.shape[:2], dtype=np.float32)
        assert np.isclose(target_hw[0] / source_hw[0], target_hw[1] / source_hw[1], atol=0.001)
        # resize image
        image = cv2.resize(image,
                           (target_hw[1].astype(np.int32), target_hw[0].astype(np.int32)))  # (256, 832)

        return image

    def data_callback(self, msg):
        msg_stamp = msg.pre_header.stamp
        boxes = np.array(msg.boxes).reshape((msg.box_num, 4))
        boxes_ctgr = np.array(msg.boxes_ctgr).reshape((msg.box_num, 1))
        boxes_dist = np.array(msg.boxes_dist).reshape((msg.box_num, 1))
        lanes = np.array(msg.lanes).reshape((msg.lane_num, 10))
        lanes_ctgr = np.array(msg.lanes_ctgr).reshape((msg.lane_num, 1))
        print("box", boxes.shape)
        print("lanes", lanes.shape)
        pred_image = self.img_dict[str(msg_stamp)]
        print(pred_image.shape)

        pred_image = self.visual_box(pred_image, boxes, boxes_ctgr, boxes_dist, self.categories,
                                     cfg.Test.COLOR)

        pred_image = self.visual_lane(pred_image, lanes, lanes_ctgr, self.lane_ctgr, (0, 255, 0))
        cv2.imshow("detect_tmg", pred_image)
        cv2.waitKey(10)

    def visual_lane(self, image, lanes, lane_ctgr, lane_ctgrs, color):
        if np.size(lanes) == 0:
            return image
        height, width = image.shape[:2]
        fpoints = lanes  # (N, 10)
        category = lane_ctgr.astype(np.int32)

        for n in range(fpoints.shape[0]):
            point = (fpoints[n].reshape(-1, 2) * np.array([height, width])).astype(np.int32)
            annotation = f"{lane_ctgrs[category[n, 0]]}"
            for i in range(point.shape[0]):
                cv2.circle(image, (point[i, 1], point[i, 0]), 1, color, 6)
            cv2.putText(image, annotation, (point[2, 1], point[2, 0]), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def visual_box(self, image, bboxes, bboxes_ctgr, boxes_dist, categories, color):
        if np.size(bboxes) == 0:
            return image
        h, w,_ =image.shape
        print(h,w)
        box_yxhw = bboxes
        print("boxbox", box_yxhw.shape)
        category = bboxes_ctgr.astype(np.int32)

        box_tlbr = convert_box_format_yxhw_to_tlbr(box_yxhw) * np.array([h,w,h,w])  # (N', 4)
        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            print("boxbox222",  y1, x1, y2, x2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color[category[i, 0]], 2)
            annotation = "dontcare" if category[i] < 0 else f"{categories[category[i, 0]]}"
            annotation += f",{boxes_dist[i, 0]:.2f}"
            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color[category[i, 0]], 2)

        return image


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: type=tf.Tensor or np.array, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_tl = boxes_yxhw[..., 0:2] - (boxes_yxhw[..., 2:4] / 2)  # y1,x1 = cy,cx + h/2,w/2
    boxes_br = boxes_tl + boxes_yxhw[..., 2:4]  # y2,x2 = y1,x1 + h,w
    output = [boxes_tl, boxes_br]
    output = concat_box_output(output, boxes_yxhw)
    return output


def concat_box_output(output, boxes):
    num, dim = boxes.shape[-2:]
    # if there is more than bounding box, append it  e.g. category, distance
    if dim > 4:
        auxi_data = boxes[..., 4:]
        output.append(auxi_data)
    output = np.concatenate(output, axis=-1)
    output = output.astype(boxes.dtype)
    return output


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_publisher = CameraPub()
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        image_publisher.get_logger().info("keyboard Interrupt (SIGINT)")
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
