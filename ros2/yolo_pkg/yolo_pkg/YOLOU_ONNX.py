#!/home/falcon/.pyenv/versions/framework/bin/python3
import sys

sys.path.insert(0, "/home/falcon/.pyenv/versions/framework/lib/python3.8/site-packages")
print(sys.path)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32
from sensor_msgs.msg import Image

import yolo_pkg.settings
import cv2
import numpy as np
import os
import config as cfg
import utils.util_function as uf
import config_dir.util_config as uc
import model.model_util as mu
import tf2onnx
import onnxruntime as rt


class CameraSub(Node):
    def __init__(self):
        super().__init__("camera_sub")
        qos_profile = QoSProfile(depth=1)
        self.subscriber = self.create_subscription(Image, 'image_raw', self.listener_callback, qos_profile)
        self.br = CvBridge()
        self.box_comps = uc.get_channel_composition(False)
        self.lane_comps = uc.get_lane_channel_composition(False)
        self.nms_box = mu.NonMaximumSuppressionBox()
        self.nms_lane = mu.NonMaximumSuppressionLane()
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["major"])}
        self.lane_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["lane"])}

        self.log_keys = cfg.Train.LOG_KEYS
        self.log_color = cfg.Test.COLOR
        self.crop_tlbr = self.find_crop_range((1080, 1920))
        self.onnx_model = self.onnx()
        self.output_name = self.get_output_name()
        self.output_features = {'feat_box': {key: [] for key in self.box_comps.keys()},
                                'feat_lane': {key: [] for key in self.lane_comps.keys()}}

    def listener_callback(self, data):
        self.get_logger().info('Receiving frame')
        frame = self.br.imgmsg_to_cv2(data)
        frame = self.preprocess(frame)
        pred = self.onnx_model.run(self.output_name, {"input_1": frame})
        box = pred[:24]
        lane = pred[24:-1]
        for i, box_key in enumerate(self.box_comps):
            self.output_features['feat_box'][box_key] = box[3 * i: 3 * (i + 1)]
        for i, lane_key in enumerate(self.lane_comps):
            self.output_features['feat_lane'][lane_key] = lane[i:i + 1]
        nms_boxes = self.nms_box(self.output_features["feat_box"])
        self.output_features["inst_box"] = uf.convert_tensor_to_numpy(
            uf.slice_feature(nms_boxes, uc.get_bbox_composition(False)))
        if cfg.ModelOutput.LANE_DET:
            lane_hw = pred[-1].shape[1:3]
            nms_lanes = self.nms_lane(self.output_features["feat_lane"], lane_hw)
            self.output_features["inst_lane"] = uf.convert_tensor_to_numpy(
                uf.slice_feature(nms_lanes, uc.get_lane_composition(False)))

        # TODO: valid_...를 검출 결과로 사용하면 됩니다.
        valid_box = self.extract_valid_data(self.output_features["inst_box"], "object")
        valid_lane = self.extract_valid_data(self.output_features["inst_lane"], "lane_centerness")

        image = self.visual_img(frame[0], self.output_features["inst_box"], self.categories, self.log_keys,
                                self.log_color)
        image = self.visual_lane(image, self.output_features["inst_lane"], self.lane_ctgr, (0, 255, 0))

        cv2.imshow("detect_result", image)
        cv2.waitKey(10)

    def preprocess(self, frame):
        frame = self.crop_image(frame, self.crop_tlbr)
        frame = self.resize(frame)
        frame = np.array(uf.to_float_image(frame))[np.newaxis, ...]
        return frame

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

    def visual_img(self, image, bboxes, categories, log_keys, color):
        image = uf.to_uint8_image(image).numpy()
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][0]
        category = bboxes["category"][0]
        valid_mask = box_yxhw[:, 2] > 0  # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)  # (N', 4)
        category = category[valid_mask, 0].astype(np.int32)  # (N',)

        valid_boxes = {}
        for key in log_keys:
            scale = 1 if key == "distance" else 100
            feature = (bboxes[key][0] * scale)
            feature = feature.astype(np.int32) if key != "distance" else feature
            valid_boxes[key] = feature[valid_mask, 0]

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color[category[i]], 2)
            annotation = "dontcare" if category[i] < 0 else f"{categories[category[i]]}"
            for key in log_keys:
                annotation += f",{valid_boxes[key][i]:02d}" if key != "distance" else f",{valid_boxes[key][i]:.2f}"
            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color[category[i]], 2)

        return image

    def extract_valid_data(self, inst_data, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][0] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[0][valid_mask]
        return valid_data

    def visual_lane(self, image, lanes, lane_ctgr, color):
        image = image.copy()
        height, width = image.shape[:2]
        fpoints = lanes["lane_fpoints"][0]  # (N, 10)
        category = lanes["lane_category"][0]

        valid_mask = fpoints[:, 4] > 0
        fpoints = fpoints[valid_mask, :]
        category = category[valid_mask, 0]

        for n in range(fpoints.shape[0]):
            point = (fpoints[n].reshape(-1, 2) * np.array([height, width])).astype(np.int32)
            annotation = f"{lane_ctgr[category[n]]}"
            for i in range(point.shape[0]):
                cv2.circle(image, (point[i, 1], point[i, 0]), 1, color, 6)
            cv2.putText(image, annotation, (point[2, 1], point[2, 0]), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def onnx(self):
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        m = rt.InferenceSession(cfg.onnx_model_file, providers=providers)
        return m

    def get_output_name(self):
        output_name = []
        for i in range(9, 33):
            output_name.append(f'tf.reshape_{i}')
        for i in range(37, 41):
            output_name.append(f'tf.reshape_{i}')
        output_name.append('tf.concat_20')
        return output_name

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


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = CameraSub()
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info("keyboard Interrupt (SIGINT)")
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
