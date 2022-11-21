import numpy as np


class Paths:
    RESULT_ROOT = "/home/falcon/ros2_ws/src/YOLOU/YOLOU"
    DATAPATH = "/home/falcon/ros2_ws/src/YOLOU/YOLOU/tfrecord"
    CHECK_POINT = "/home/falcon/ros2_ws/src/YOLOU/YOLOU/ckpt"
    CONFIG_FILENAME = "/home/falcon/ros2_ws/src/YOLOU/YOLOU/config.py"
    META_CFG_FILENAME = "/home/falcon/ros2_ws/src/YOLOU/YOLOU/config_dir/meta_config.py"


class Datasets:

    class Kitti:
        NAME = "kitti"
        PATH = "/home/rilab-01/workspace/detlec/kitti"
        CATEGORIES_TO_USE = ['Pedestrian', 'Car', 'Van', 'Truck', 'Cyclist']
        CATEGORY_REMAP = {"Pedestrian": "Person", "Cyclist": "Bicycle", 
                          }
        INPUT_RESOLUTION = (256, 832)
        INCLUDE_LANE = False
        CROP_TLBR = [0, 0, 0, 0]

    class Uplus:
        NAME = "uplus"
        PATH = "/home/dolphin/kim_workspace/uplus22"
        CATEGORIES_TO_USE = ['보행자', '승용차', '트럭', '버스', '이륜차', '신호등', '자전거', '삼각콘', '차선규제봉', '과속방지턱', '포트홀', 'TS이륜차금지', 'TS우회전금지', 'TS좌회전금지', 'TS유턴금지', 'TS주정차금지', 'TS유턴', 'TS어린이보호', 'TS횡단보도', 'TS좌회전', 'TS속도제한_기타', 'TS속도제한_30', 'TS속도제한_50', 'TS속도제한_80', 'RM우회전금지', 'RM좌회전금지', 'RM직진금지', 'RM우회전', 'RM좌회전', 'RM직진', 'RM유턴', 'RM횡단예고', 'RM횡단보도', 'RM속도제한_기타', 'RM속도제한_30', 'RM속도제한_50', 'RM속도제한_80', "don't care"]
        CATEGORY_REMAP = {"보행자": "Pedestrian", "승용차": "Car", "트럭": "Truck", 
                          "버스": "Bus", "이륜차": "Motorcycle", "신호등": "Traffic light", 
                          "자전거": "Bicycle", "삼각콘": "Cone", "차선규제봉": "Lane_stick", 
                          "과속방지턱": "Bump", "포트홀": "Pothole", "don't care": "Don't Care", 
                          "lane don't care": "Lane Don't Care", "TS이륜차금지": "TS_NO_TW", "TS우회전금지": "TS_NO_RIGHT", 
                          "TS좌회전금지": "TS_NO_LEFT", "TS유턴금지": "TS_NO_TURN", "TS주정차금지": "TS_NO_STOP", 
                          "TS유턴": "TS_U_TURN", "TS어린이보호": "TS_CHILDREN", "TS횡단보도": "TS_CROSSWK", 
                          "TS좌회전": "TS_GO_LEFT", "TS속도제한_기타": "TS_SPEED_LIMIT_ETC", "TS속도제한_30": "TS_SPEED_LIMIT_30", 
                          "TS속도제한_50": "TS_SPEED_LIMIT_50", "TS속도제한_80": "TS_SPEED_LIMIT_80", "RM우회전금지": "RM_NO_RIGHT", 
                          "RM좌회전금지": "RM_NO_LEFT", "RM직진금지": "RM_NO_STR", "RM우회전": "RM_GO_RIGHT", 
                          "RM좌회전": "RM_GO_LEFT", "RM직진": "RM_GO_STR", "RM유턴": "RM_U_TURN", 
                          "RM횡단예고": "RM_ANN_CWK", "RM횡단보도": "RM_CROSSWK", "RM속도제한_기타": "RM_SPEED_LIMIT_ETC", 
                          "RM속도제한_30": "RM_SPEED_LIMIT_30", "RM속도제한_50": "RM_SPEED_LIMIT_50", "RM속도제한_80": "RM_SPEED_LIMIT_80", 
                          
                          }
        LANE_TYPES = ['차선1', '차선2', '차선3', '차선4', 'RM정지선']
        LANE_REMAP = {"차선1": "Lane", "차선2": "Lane", "차선3": "Lane", 
                      "차선4": "Lane", "RM정지선": "Stop_Line", 
                      }
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [300, 0, 0, 0]
        INCLUDE_LANE = True

    class Uplus21:
        NAME = "uplus21"
        PATH = "/home/dolphin/kim_workspace/uplus21"
        CATEGORIES_TO_USE = ['사람', '차', '차량/트럭', '차량/버스', '차량/오토바이', '신호등', '차량/자전거', '삼각콘', '표지판/이륜 통행 금지', '표지판/유턴 금지', '표지판/주정차 금지', '표지판/어린이 보호', '표지판/횡단보도', '노면표시/직진', '노면표시/횡단보도 예고', '노면표시/직진금지', '노면표시/횡단보도', "don't care"]
        CATEGORY_REMAP = {"사람": "Pedestrian", "차량/자전거": "Bicycle", "차": "Car", 
                          "차량/트럭": "Truck", "차량/버스": "Bus", "신호등": "Traffic light", 
                          "삼각콘": "Cone", "don't care": "Don't Care", "차량/오토바이": "Motorcycle", 
                          "표지판/유턴 금지": "TS_NO_TURN", "표지판/이륜 통행 금지": "TS_NO_TW", "표지판/주정차 금지": "TS_NO_STOP", 
                          "표지판/어린이 보호": "TS_CHILDREN", "표지판/횡단보도": "TS_CROSSWK", "노면표시/직진": "RM_GO_STR", 
                          "노면표시/횡단보도 예고": "RM_ANN_CWK", "노면표시/직진금지": "RM_NO_STR", "노면표시/횡단보도": "RM_CROSSWK", 
                          
                          }
        LANE_TYPES = ['차선/황색 단선 실선', '차선/백색 단선 실선', '차선/황색 단선 점선', '차선/백색 단선 점선', '차선/황색 겹선 실선']
        LANE_REMAP = {"차선/황색 단선 실선": "YSL", "차선/백색 단선 실선": "WSL", "차선/황색 단선 점선": "YSDL", 
                      "차선/백색 단선 점선": "WSDL", "차선/황색 겹선 실선": "YDL", 
                      }
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = True

    class City:
        NAME = "city"
        PATH = "/media/cheetah/IntHDD/datasets/city"
        CATEGORIES_TO_USE = ['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        CATEGORY_REMAP = {"person": "Person", "rider": "Person", "car": "Car", 
                          "bus": "Bus", "truck": "Truck", "motorcycle": "Motorcycle", 
                          "bicycle": "Bicycle", 
                          }
        INPUT_RESOLUTION = (1024, 2048)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = False
        DIST_QUANTILE = 0.2

    class A2d2:
        NAME = "a2d2"
        PATH = "/media/cheetah/IntHDD/datasets/a2d2/camera_lidar_semantic_bboxes.zip"
        CATEGORIES_TO_USE = ['Car 1', 'Car 2', 'Car 3', 'Car 4', 'Bicycle 1', 'Bicycle 2', 'Bicycle 3', 'Bicycle 4', 'Pedestrian 1', 'Pedestrian 2', 'Pedestrian 3', 'Truck 1', 'Truck 2', 'Truck 3', 'Small vehicles 1', 'Small vehicles 2', 'Small vehicles 3']
        CATEGORY_REMAP = {"Car 1": "Car", "Car 2": "Car", "Car 3": "Car", 
                          "Car 4": "Car", "Bicycle 1": "Bicycle", "Bicycle 2": "Bicycle", 
                          "Bicycle 3": "Bicycle", "Bicycle 4": "Bicycle", "Pedestrian 1": "Person", 
                          "Pedestrian 2": "Person", "Pedestrian 3": "Person", "Truck 1": "Truck", 
                          "Truck 2": "Truck", "Truck 3": "Truck", "Small vehicles 1": "Motorcycle", 
                          "Small vehicles 2": "Motorcycle", "Small vehicles 3": "Motorcycle", 
                          }
        MAX_LANE_PARAM = 50
        CATEGORY_NAMES = ['Bgd', 'Pedestrian', 'Car']
        SHARD_SIZE = 2000
        INCLUDE_LANE = False
        SEGMAP_SCALE = 4
        PIXEL_LIMIT = 50
        DIST_QUANTILE = 0.2
    DATASET_CONFIG = Uplus
    TARGET_DATASET = "uplus"
    MAX_FRAMES = 100000


class Dataloader:
    DATASETS_FOR_TFRECORD = {"uplus": ('train', 'val'), 
                             }
    MAX_BBOX_PER_IMAGE = 100
    MAX_DONT_PER_IMAGE = 100
    MAX_LANE_PER_IMAGE = 30
    MAX_POINTS_PER_LANE = 50
    CATEGORY_NAMES = {"major": ['Bgd', 'Pedestrian', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Traffic light', 'Traffic sign', 'Road mark', 'Road mark dont', 'Road mark do', 'Bicycle', 'Cone', 'Lane_stick', 'Bump', 'Pothole'], "sign": ['TS_NO_TW', 'TS_NO_RIGHT', 'TS_NO_LEFT', 'TS_NO_TURN', 'TS_NO_STOP', 'TS_U_TURN', 'TS_CHILDREN', 'TS_CROSSWK', 'TS_GO_LEFT', 'TS_SPEED_LIMIT'], "mark": ['RM_U_TURN', 'RM_ANN_CWK', 'RM_CROSSWK', 'RM_SPEED_LIMIT'], 
                      "mark_dont": ['RM_NO_RIGHT', 'RM_NO_LEFT', 'RM_NO_STR'], "mark_do": ['RM_GO_RIGHT', 'RM_GO_LEFT', 'RM_GO_STR'], "sign_speed": ['TS_SPEED_LIMIT_30', 'TS_SPEED_LIMIT_50', 'TS_SPEED_LIMIT_80', 'TS_SPEED_LIMIT_ETC'], 
                      "mark_speed": ['RM_SPEED_LIMIT_30', 'RM_SPEED_LIMIT_50', 'RM_SPEED_LIMIT_80', 'RM_SPEED_LIMIT_ETC'], "dont": ["Don't Care"], "lane": ['Bgd', 'Lane', 'Stop_Line'], 
                      "dont_lane": ["Lane Don't Care"], 
                      }
    SHARD_SIZE = 2000
    MIN_PIX = {"train": {'Bgd': 0, 'Pedestrian': 68, 'Car': 87, 'Truck': 98, 'Bus': 150, 'Motorcycle': 38, 'Traffic light': 41, 'Traffic sign': 26, 'Road mark': 20, 'Road mark dont': 20, 'Road mark do': 20, 'Bicycle': 38, 'Cone': 34, 'Lane_stick': 30, 'Bump': 75, 'Pothole': 0}, "val": {'Bgd': 0, 'Pedestrian': 76, 'Car': 98, 'Truck': 110, 'Bus': 168, 'Motorcycle': 42, 'Traffic light': 46, 'Traffic sign': 29, 'Road mark': 20, 'Road mark dont': 20, 'Road mark do': 20, 'Bicycle': 42, 'Cone': 38, 'Lane_stick': 34, 'Bump': 85, 'Pothole': 0}, 
               }
    LANE_MIN_PIX = {"train": {'Bgd': 0, 'Lane': 55, 'Stop_Line': 55}, "val": {'Bgd': 0, 'Lane': 50, 'Stop_Line': 50}, 
                    }


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    LANE_DET = True
    MINOR_CTGR = True
    SPEED_LIMIT = False
    FEAT_RAW = False
    IOU_AWARE = False
    NUM_ANCHORS_PER_SCALE = 3
    GRTR_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                             "anchor_ind": 1, 
                             }
    PRED_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "distance": 1, 
                             "category": 16, "sign_ctgr": 10, "mark_ctgr": 4, 
                             "mark_dont": 3, "mark_do": 3, 
                             }
    HEAD_COMPOSITION = {"reg": 6, "cls": 36, 
                        }
    GRTR_INST_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                             
                             }
    PRED_INST_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                             "ctgr_prob": 1, "score": 1, "anchor_ind": 1, 
                             
                             }
    NUM_MAIN_CHANNELS = 42
    NUM_LANE_ANCHORS_PER_SCALE = 1
    GRTR_FMAP_LANE_COMPOSITION = {"laneness": 1, "lane_fpoints": 10, "lane_centerness": 1, 
                                  "lane_category": 1, 
                                  }
    PRED_FMAP_LANE_COMPOSITION = {"laneness": 1, "lane_fpoints": 10, "lane_centerness": 1, 
                                  "lane_category": 3, 
                                  }
    GRTR_INST_LANE_COMPOSITION = {"lane_fpoints": 10, "lane_centerness": 1, "lane_category": 1, 
                                  
                                  }
    PRED_INST_LANE_COMPOSITION = {"lane_fpoints": 10, "lane_centerness": 1, "lane_category": 1, 
                                  
                                  }
    NUM_LANE_CHANNELS = 15


class Architecture:
    BACKBONE = "CSPDarknet53"
    NECK = "PAN"
    HEAD = "Single"
    BACKBONE_CONV_ARGS = {"activation": "mish", "scope": "back", 
                          }
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck", 
                      }
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head", 
                      }
    USE_SPP = True
    COORD_CONV = True
    SIGMOID_DELTA = 0.2

    class Resnet:
        LAYER = ('BottleneckBlock', (3, 4, 6, 3))
        CHENNELS = [64, 128, 256, 512, 1024, 2048]

    class Efficientnet:
        NAME = "EfficientNetB2"
        Channels = (112, 5, 3)
        Separable = False


class Train:
    CKPT_NAME = "split_arrow_base"
    MODE = "graph"
    AUGMENT_PROBS = None
    DATA_BATCH_SIZE = 1
    BATCH_SIZE = 1
    DATSET_SIZE = 20
    TRAINING_PLAN = [
                     ('uplus', 1, 1e-07, {'iou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([2.0, 2.0, 2.0], 'BoxObjectnessLoss', 3.0, 1.0), 'category': ([2, 2, 3.0], 'MajorCategoryLoss'), 'distance': ([1.0, 1.0, 1.0], 'DistanceLoss'), 'sign_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'sign_ctgr'), 'mark_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_ctgr'), 'mark_dont': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_dont'), 'mark_do': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_do'), 'laneness': (3.0, 'LanenessLoss', 1, 1), 'lane_fpoints': (10000.0, 'FpointLoss'), 'lane_centerness': (5.0, 'CenternessLoss', 3, 1), 'lane_category': (10.0, 'LaneCategLoss')}, True),
                     ('uplus', 20, 0.0001, {'iou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([2.0, 2.0, 2.0], 'BoxObjectnessLoss', 3.0, 1.0), 'category': ([2, 2, 3.0], 'MajorCategoryLoss'), 'distance': ([1.0, 1.0, 1.0], 'DistanceLoss'), 'sign_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'sign_ctgr'), 'mark_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_ctgr'), 'mark_dont': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_dont'), 'mark_do': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_do'), 'laneness': (3.0, 'LanenessLoss', 1, 1), 'lane_fpoints': (10000.0, 'FpointLoss'), 'lane_centerness': (5.0, 'CenternessLoss', 3, 1), 'lane_category': (10.0, 'LaneCategLoss')}, True),
                     ('uplus', 20, 1e-05, {'iou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([2.0, 2.0, 2.0], 'BoxObjectnessLoss', 3.0, 1.0), 'category': ([2, 2, 3.0], 'MajorCategoryLoss'), 'distance': ([1.0, 1.0, 1.0], 'DistanceLoss'), 'sign_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'sign_ctgr'), 'mark_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_ctgr'), 'mark_dont': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_dont'), 'mark_do': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_do'), 'laneness': (3.0, 'LanenessLoss', 1, 1), 'lane_fpoints': (10000.0, 'FpointLoss'), 'lane_centerness': (5.0, 'CenternessLoss', 3, 1), 'lane_category': (10.0, 'LaneCategLoss')}, True),
                     ('uplus', 20, 1e-06, {'iou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([2.0, 2.0, 2.0], 'BoxObjectnessLoss', 3.0, 1.0), 'category': ([2, 2, 3.0], 'MajorCategoryLoss'), 'distance': ([1.0, 1.0, 1.0], 'DistanceLoss'), 'sign_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'sign_ctgr'), 'mark_ctgr': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_ctgr'), 'mark_dont': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_dont'), 'mark_do': ([2.0, 1.0, 1.0], 'MinorCategoryLoss', 'mark_do'), 'laneness': (3.0, 'LanenessLoss', 1, 1), 'lane_fpoints': (10000.0, 'FpointLoss'), 'lane_centerness': (5.0, 'CenternessLoss', 3, 1), 'lane_category': (10.0, 'LaneCategLoss')}, True),
                     ]
    DETAIL_LOG_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    IGNORE_MASK = True
    LOG_KEYS = ['distance']
    USE_EMA = True
    EMA_DECAY = 0.9998
    INTRINSIC = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


class Scheduler:
    MIN_LR = 1e-10
    CYCLE_STEPS = 10000
    WARMUP_EPOCH = 0
    LOG = True


class FeatureDistribPolicy:
    POLICY_NAME = "SinglePositivePolicy"
    IOU_THRESH = [0.5, 0.3]
    CENTER_RADIUS = 2.5
    BOX_SIZE_STANDARD = np.array([128, 256])
    MULTI_POSITIVE_WIEGHT = 0.8


class AnchorGeneration:
    ANCHOR_STYLE = "YoloAnchor"
    ANCHORS = np.array([[[26, 293], [70, 110], [110, 69]], [[65, 734], [175, 275], [276, 173]], [[104, 1175], [280, 440], [442, 278]]])
    MUL_SCALES = [1.0, 2.5, 4.0]

    class YoloAnchor:
        BASE_ANCHOR = [70.0, 110.0]
        ASPECT_RATIO = [0.14, 1.0, 2.5]
        SCALES = [1]

    class RetinaNetAnchor:
        BASE_ANCHOR = [20, 20]
        ASPECT_RATIO = [0.5, 1, 2]
        SCALES = [1, 1.2599210498948732, 1.5874010519681994]

    class YoloxAnchor:
        BASE_ANCHOR = [8, 8]
        ASPECT_RATIO = [1]
        SCALES = [1]


class NmsInfer:
    MAX_OUT = [0, 5, 6, 5, 5, 5, 5, 5, 6, 5, 5, 5, 7, 5, 5, 5]
    IOU_THRESH = [0, 0.26, 0.2, 0.1, 0.1, 0.14, 0.1, 0.1, 0.18, 0.1, 0.32, 0.3, 0.26, 0.12, 0.18, 0.1]
    SCORE_THRESH = [1, 0.38, 0.38, 0.38, 0.36, 0.38, 0.34, 0.38, 0.38, 0.28, 0.28, 0.36, 0.36, 0.38, 0.38, 0.24]
    LANE_MAX_OUT = [0, 5, 2]
    LANE_OVERLAP_THRESH = [0, 0.78, 0.82]
    LANE_SCORE_THRESH = [1, 0.03, 0.03]


class NmsOptim:
    IOU_CANDIDATES = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4])
    SCORE_CANDIDATES = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38])
    MAX_OUT_CANDIDATES = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    LANE_IOU_CANDIDATES = np.array([0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9])
    LANE_SCORE_CANDIDATES = np.array([0.02, 0.03])
    LANE_MAX_OUT_CANDIDATES = np.array([2, 3, 4, 5])


class Validation:
    TP_IOU_THRESH = [1, 0.4, 0.5, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.2, 0.3, 0.3]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]
    MAX_BOX = 200
    LANE_TP_IOU_THRESH = [0.33]


class Log:
    VISUAL_HEATMAP = True

    class HistoryLog:
        SUMMARY = ['pos_obj', 'neg_obj', 'pos_lane', 'neg_lane', 'pos_center', 'neg_center']

    class ExhaustiveLog:
        DETAIL = ['pos_obj', 'neg_obj', 'iou_mean', 'box_yx', 'box_hw', 'true_class', 'false_class']
        LANE_DETAIL = ['pos_lane', 'neg_lane', 'pos_lanecenter', 'neg_lanecenter', 'lane_true_class', 'lane_false_class']
        COLUMNS_TO_MEAN = ['anchor', 'ctgr', 'iou', 'object', 'category', 'distance', 'pos_obj', 'neg_obj', 'iou_mean', 'box_hw', 'box_yx', 'true_class', 'false_class', 'sign_ctgr', 'mark_ctgr']
        COLUMNS_TO_LANE_MEAN = ['ctgr', 'laneness', 'lane_fpoints', 'lane_category', 'lane_centerness', 'pos_lane', 'neg_lane', 'pos_lanecenter', 'neg_lanecenter', 'lane_true_class', 'lane_false_class']
        COLUMNS_TO_SUM = ['anchor', 'ctgr', 'trpo', 'grtr', 'pred']
        COLUMNS_TO_LANE_SUM = ['ctgr', 'trpo_lane', 'grtr_lane', 'pred_lane']


class Test:
    COLOR = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 124, 255), (124, 0, 255), (0, 255, 124), (124, 255, 0), (255, 0, 124), (255, 124, 0), (255, 153, 51), (100, 100, 100), (50, 150, 200), (255, 255, 255)]

onnx_model_file = "/home/falcon/kim_workspace/model.onnx"