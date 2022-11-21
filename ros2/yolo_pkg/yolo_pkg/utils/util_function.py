import tensorflow as tf
import numpy as np

def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, dims)
    :param channel_composition:
    :return: sliced feature maps
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = tf.split(feature, channels, axis=-1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices

def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32)

def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.uint8)

def convert_tensor_to_numpy(feature):
    if isinstance(feature, dict):
        dict_feat = dict()
        for key, value in feature.items():
            dict_feat[key] = convert_tensor_to_numpy(value)
        return dict_feat
    elif isinstance(feature, list):
        list_feat = []
        for value in feature:
            list_feat.append(convert_tensor_to_numpy(value))
        return list_feat
    else:
        return feature.numpy()

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

    if tf.is_tensor(boxes):
        output = tf.concat(output, axis=-1)
        output = tf.cast(output, boxes.dtype)
    else:
        output = np.concatenate(output, axis=-1)
        output = output.astype(boxes.dtype)
    return output