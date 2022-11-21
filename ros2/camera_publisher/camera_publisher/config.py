import numpy as np

class Test:
    CATEGORY_NAMES = {"major": ['Bgd', 'Pedestrian', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Traffic light', 'Traffic sign', 'Road mark', 'Road mark dont', 'Road mark do', 'Bicycle', 'Cone', 'Lane_stick', 'Bump', 'Pothole'], "sign": ['TS_NO_TW', 'TS_NO_RIGHT', 'TS_NO_LEFT', 'TS_NO_TURN', 'TS_NO_STOP', 'TS_U_TURN', 'TS_CHILDREN', 'TS_CROSSWK', 'TS_GO_LEFT', 'TS_SPEED_LIMIT'], "mark": ['RM_U_TURN', 'RM_ANN_CWK', 'RM_CROSSWK', 'RM_SPEED_LIMIT'],
                      "mark_dont": ['RM_NO_RIGHT', 'RM_NO_LEFT', 'RM_NO_STR'], "mark_do": ['RM_GO_RIGHT', 'RM_GO_LEFT', 'RM_GO_STR'], "sign_speed": ['TS_SPEED_LIMIT_30', 'TS_SPEED_LIMIT_50', 'TS_SPEED_LIMIT_80', 'TS_SPEED_LIMIT_ETC'],
                      "mark_speed": ['RM_SPEED_LIMIT_30', 'RM_SPEED_LIMIT_50', 'RM_SPEED_LIMIT_80', 'RM_SPEED_LIMIT_ETC'], "dont": ["Don't Care"], "lane": ['Bgd', 'Lane', 'Stop_Line'],
                      "dont_lane": ["Lane Don't Care"], }
    COLOR = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 124, 255), (124, 0, 255), (0, 255, 124), (124, 255, 0), (255, 0, 124), (255, 124, 0), (255, 153, 51), (100, 100, 100), (50, 150, 200), (255, 255, 255)]

onnx_model_file = "/home/samsung5g/ros2_ws/src/YOLOU/model.onnx"
