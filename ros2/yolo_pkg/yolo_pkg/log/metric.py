import numpy as np
import utils.framework.util_function as uf
import config as cfg


def count_true_positives(grtr, pred, grtr_dontcare, num_ctgr, iou_thresh=cfg.Validation.TP_IOU_THRESH, per_class=False,
                         num_minor_ctgr=None, target_class=None):
    """
    :param grtr: slices of features["bboxes"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param grtr_dontcare: slices of features["dontcare"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :param num_minor_ctgr
    :return:
    """
    splits = split_true_false(grtr, pred, grtr_dontcare, iou_thresh)
    # ========== use split instead grtr, pred
    grtr_valid_tp = splits["grtr_tp"]["yxhw"][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"]["yxhw"][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"]["yxhw"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["yxhw"][..., 2:3] > 0
    if cfg.ModelOutput.MINOR_CTGR:
        grtr_valid_minor_tp = np.where(((grtr_valid_tp > 0) * (splits["grtr_tp"]["category"][..., :1] == target_class)),
                                      splits["grtr_tp"]["minor_ctgr"][..., :1], -1) >= 0

        grtr_valid_minor_fn = np.where(((grtr_valid_fn > 0) * (splits["grtr_fn"]["category"][..., :1] == target_class)),
                                      splits["grtr_fn"]["minor_ctgr"][..., :1], -1) >= 0

        pred_valid_minor_tp = np.where(((pred_valid_tp > 0) * (splits["pred_tp"]["category"][..., :1] == target_class)),
                                      splits["pred_tp"]["minor_ctgr"][..., :1], -1) >= 0

        pred_valid_minor_fp = np.where(((pred_valid_fp > 0) * (splits["pred_fp"]["category"][..., :1] == target_class)),
                                      splits["pred_fp"]["minor_ctgr"][..., :1], -1) >= 0
        if cfg.ModelOutput.SPEED_LIMIT:
            grtr_valid_speed_tp = np.where(
                ((grtr_valid_tp > 0) * (splits["grtr_tp"]["category"][..., :1] == target_class)),
                splits["grtr_tp"]["minor_ctgr"][..., :1], -1) >= 0

            grtr_valid_speed_fn = np.where(
                ((grtr_valid_fn > 0) * (splits["grtr_fn"]["category"][..., :1] == target_class)),
                splits["grtr_fn"]["minor_ctgr"][..., :1], -1) >= 0

            pred_valid_speed_tp = np.where(
                ((pred_valid_tp > 0) * (splits["pred_tp"]["category"][..., :1] == target_class)),
                splits["pred_tp"]["minor_ctgr"][..., :1], -1) >= 0

            pred_valid_speed_fp = np.where(
                ((pred_valid_fp > 0) * (splits["pred_fp"]["category"][..., :1] == target_class)),
                splits["pred_fp"]["minor_ctgr"][..., :1], -1) >= 0

    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_class(splits["pred_fp"], pred_valid_fp, num_ctgr)
        if cfg.ModelOutput.MINOR_CTGR and num_minor_ctgr is not None:
            grtr_minor_tp_count = count_per_minor_class(splits["grtr_tp"], grtr_valid_minor_tp, num_minor_ctgr)
            grtr_minor_fn_count = count_per_minor_class(splits["grtr_fn"], grtr_valid_minor_fn, num_minor_ctgr)
            pred_minor_tp_count = count_per_minor_class(splits["pred_tp"], pred_valid_minor_tp, num_minor_ctgr)
            pred_minor_fp_count = count_per_minor_class(splits["pred_fp"], pred_valid_minor_fp, num_minor_ctgr)
            return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                    "pred": (pred_tp_count + pred_fp_count), "minor_trpo": pred_minor_tp_count,
                    "minor_grtr": (grtr_minor_tp_count + grtr_minor_fn_count),
                    "minor_pred": (pred_minor_tp_count + pred_minor_fp_count),
                    }
        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def split_true_false(grtr, pred, grtr_dc, iou_thresh):
    pred_valid, pred_far = split_pred_far(pred)
    grtr_far, grtr_valid = split_grtr_far(pred_far, grtr, iou_thresh)
    splits = split_tp_fp_fn(pred_valid, grtr_valid, iou_thresh)
    fp_pred, dc_pred = split_dontcare_pred(splits["pred_fp"], grtr_dc)
    splits["pred_fp"] = fp_pred
    splits["pred_dc"] = dc_pred
    splits["grtr_dc"] = grtr_dc
    splits["pred_far"] = pred_far
    splits["grtr_far"] = grtr_far
    return splits


def split_pred_far(pred):
    pred_far_mask = pred["distance"] > cfg.Validation.DISTANCE_LIMIT
    pred_valid = {key: (val * (1 - pred_far_mask).astype(np.float32)) for key, val in pred.items()}
    pred_far = {key: (val * pred_far_mask).astype(np.float32) for key, val in pred.items()}
    return pred_valid, pred_far


def split_grtr_far(pred_far, grtr, iou_thresh):
    iou_far = uf.compute_iou_general(grtr["yxhw"], pred_far["yxhw"]).numpy()
    best_iou_far = np.max(iou_far, axis=-1)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = np.expand_dims(best_iou_far > iou_thresh, axis=-1)
    grtr_valid = {key: (val * (1 - iou_match)).astype(np.float32) for key, val in grtr.items()}
    grtr_far = {key: (val * iou_match).astype(np.float32) for key, val in grtr.items()}
    grtr_far["iou"] = best_iou_far
    return grtr_far, grtr_valid


def split_tp_fp_fn(pred, grtr, iou_thresh):
    batch, M, _ = pred["category"].shape
    valid_mask = grtr["object"]
    iou = uf.compute_iou_general(grtr["yxhw"], pred["yxhw"]).numpy()  # (batch, N, M)

    # ctgr_mask = grtr["category"][:, :, np.newaxis, 0] == pred["category"][:, np.newaxis, :, 0]
    # iou_ctgr = iou * ctgr_mask
    best_iou = np.max(iou, axis=-1)  # (batch, N)

    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = numpy_gather(pred["category"], best_idx, 1)  # (batch, N)

    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)

    if cfg.ModelOutput.MINOR_CTGR:
        minor_mask = (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Traffic sign")) | \
                     (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark")) | \
                     (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark dont")) | \
                     (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark do"))
        pred_minor_ctgr_aligned = numpy_gather(pred["minor_ctgr"], best_idx, 1)
        minor_ctgr_match = grtr["minor_ctgr"][..., 0] == pred_minor_ctgr_aligned
        grtr_tp_mask = np.expand_dims((1 - minor_mask) * grtr_tp_mask[..., 0]
                                      + minor_mask * minor_ctgr_match, axis=-1).astype(bool)

        if cfg.ModelOutput.SPEED_LIMIT:
            speed_mask = (grtr["minor_ctgr"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["sign"].index("TS_SPEED_LIMIT")) | \
                         (grtr["minor_ctgr"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["mark"].index("RM_SPEED_LIMIT"))
            pred_speed_ctgr_aligned = numpy_gather(pred["speed_ctgr"], best_idx, 1)
            speed_ctgr_match = grtr["speed_ctgr"][..., 0] == pred_speed_ctgr_aligned
            grtr_tp_mask = np.expand_dims((1 - speed_mask) * grtr_tp_mask[..., 0]
                                          + speed_mask * speed_ctgr_match, axis=-1).astype(bool)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_dontcare_pred(pred_fp, grtr_dc):
    B, M, _ = pred_fp["category"].shape
    iou_dc = uf.compute_iou_general(grtr_dc["yxhw"], pred_fp["yxhw"])
    best_iou_dc = np.max(iou_dc, axis=-1)  # (batch, D)
    grtr_dc["iou"] = best_iou_dc
    dc_match = np.expand_dims(best_iou_dc > 0.5, axis=-1)  # (batch, D)
    best_idx_dc = np.argmax(iou_dc, axis=-1)
    pred_dc_mask = indices_to_binary_mask(best_idx_dc, dc_match, M)  # (batch, M, 1)
    dc_pred = {key: val * pred_dc_mask for key, val in pred_fp.items()}
    fp_pred = {key: val * (1 - pred_dc_mask) for key, val in pred_fp.items()}
    return fp_pred, dc_pred


def indices_to_binary_mask(best_idx, valid_mask, depth):
    best_idx_onehot = one_hot(best_idx, depth) * valid_mask
    binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
    return binary_mask.astype(np.float32)


def get_iou_thresh_per_class(grtr_ctgr, tp_iou_thresh):
    ctgr_idx = grtr_ctgr.astype(np.int32)
    tp_iou_thresh = np.asarray(tp_iou_thresh, np.float32)
    iou_thresh = numpy_gather(tp_iou_thresh, ctgr_idx)
    return iou_thresh[..., 0]


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count

def count_per_diff_class(category, mask, num_ctgr, dist):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = category.astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
    dist_onehot = dist * boxes_onehot
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    dist_onehot = np.sum(dist_onehot, axis=(0, 1))
    return boxes_count, dist_onehot

def count_per_minor_class(boxes, mask, num_minor_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """

    boxes_ctgr = boxes["minor_ctgr"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_minor_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count

def count_per_minor_diff_class(minor_ctgr, mask, num_minor_ctgr, dist):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """

    boxes_ctgr = minor_ctgr.astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_minor_ctgr) * mask
    dist_onehot = dist * boxes_onehot
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    dist_onehot = np.sum(dist_onehot, axis=(0, 1))
    return boxes_count, dist_onehot

def count_per_lane_class(lanes, mask, num_ctgr):
    """
    :param lanes: slices of object info {'lanes': (batch, N, 10), 'lane_category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    lanes_ctgr = lanes["lane_category"][..., 0].astype(np.int32)  # (batch, N')
    lanes_onehot = one_hot(lanes_ctgr, num_ctgr) * mask
    lanes_count = np.sum(lanes_onehot, axis=(0, 1))
    return lanes_count


def one_hot(grtr_category, category_shape):
    test = np.eye(category_shape)
    one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
    return one_hot_data


def numpy_gather(params, index, dim=0):
    if dim is 1:
        batch_list = []
        for i in range(params.shape[0]):
            batch_param = params[i]
            batch_index = index[i]
            batch_gather = np.take(batch_param, batch_index)
            batch_list.append(batch_gather)
        gathar_param = np.stack(batch_list)
    else:
        gathar_param = np.take(params, index)
    return gathar_param


def count_true_positives_lane(grtr, pred, num_ctgr, img_shape, iou_thresh=cfg.Validation.LANE_TP_IOU_THRESH,
                              per_class=False, is_train=True):
    splits = split_lane_true_false(grtr, pred, iou_thresh, img_shape, is_train)

    grtr_valid_tp = splits["grtr_tp"]["lane_fpoints"][..., 1:2] > 0
    grtr_valid_fn = splits["grtr_fn"]["lane_fpoints"][..., 1:2] > 0
    pred_valid_tp = splits["pred_tp"]["lane_fpoints"][..., 1:2] > 0
    pred_valid_fp = splits["pred_fp"]["lane_fpoints"][..., 1:2] > 0
    if per_class:
        grtr_tp_count = count_per_lane_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_lane_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_lane_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_lane_class(splits["pred_fp"], pred_valid_fp, num_ctgr)
        return {"trpo_lane": pred_tp_count, "grtr_lane": (grtr_tp_count + grtr_fn_count),
                "pred_lane": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo_lane": trpo_count, "grtr_lane": grtr_count, "pred_lane": pred_count}


def split_lane_true_false(grtr, pred, iou_thresh, img_shape, is_train):
    splits = split_lane_tp_fp_fn(pred, grtr, iou_thresh, img_shape, is_train)
    return splits


def split_lane_tp_fp_fn(pred, grtr, iou_thresh, img_shape, is_train):
    batch, M, _ = pred["lane_category"].shape
    valid_mask = grtr["lane_centerness"]

    if is_train:
        iou = uf.compute_lane_iou(grtr["lane_fpoints"], pred["lane_fpoints"])  # (batch, N, M)
    else:
        iou = uf.compute_lane_iou_with_cv2(grtr["lane_fpoints"], pred["lane_fpoints"], img_shape)  # (batch, N, M)
    best_iou = np.max(iou, axis=-1)  # (batch, N)
    best_idx = np.argmax(iou, axis=-1)  # (batch, N)

    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["lane_category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = numpy_gather(pred["lane_category"], best_idx, 1)  # (batch, N, 8)

    ctgr_match = grtr["lane_category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]

    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def count_true_positives_dist(grtr, pred, grtr_dontcare, num_ctgr, iou_thresh=cfg.Validation.TP_IOU_THRESH, per_class=False,
                         num_minor_ctgr=None, target_class=None):
    splits = split_true_false(grtr, pred, grtr_dontcare, iou_thresh)
    # ========== use split instead grtr, pred

    iou = uf.compute_iou_general(splits["grtr_tp"]["yxhw"], splits["pred_tp"]["yxhw"]).numpy()  # (batch, N, M)

    best_iou = np.max(iou, axis=-1)  # (batch, N)

    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(splits["grtr_tp"]["category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = numpy_gather(splits["pred_tp"]["category"], best_idx, 1)  # (batch, N, 8)
    pred_dist_aligned = numpy_gather(splits["pred_tp"]["distance"], best_idx, 1)  # (batch, N, 8)
 # (batch, N, 8)
    grtr_dist_mask = splits["grtr_tp"]["distance"][..., 0] > 0
    ctgr_match = splits["grtr_tp"]["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_dist_tp_mask = np.expand_dims(iou_match * ctgr_match * grtr_dist_mask, axis=-1)
    grtr_valid_dist = splits["grtr_tp"]["distance"] * grtr_dist_tp_mask
    pred_valid_dist = pred_dist_aligned * grtr_dist_tp_mask[...,0]

    if cfg.ModelOutput.MINOR_CTGR:
        pred_minor_dist_aligned = numpy_gather(splits["pred_tp"]["minor_ctgr"], best_idx, 1)
        grtr_valid_minor_tp = np.where(((grtr_valid_dist > 0) * (splits["grtr_tp"]["category"][..., :1] == target_class)),
                                       splits["grtr_tp"]["minor_ctgr"][..., :1], -1) >= 0

        pred_valid_minor_tp = np.where(((pred_valid_dist > 0) * (pred_ctgr_aligned == target_class)),
                                       pred_minor_dist_aligned, -1) >= 0
    if per_class:
        grtr_tp_count, gt_dist_onehot = count_per_diff_class(splits["grtr_tp"]["category"][...,0], grtr_dist_tp_mask, num_ctgr, grtr_valid_dist)
        pred_tp_count, pred_dist_onehot = count_per_diff_class(pred_ctgr_aligned, grtr_dist_tp_mask, num_ctgr, pred_valid_dist[...,np.newaxis])
        diff = np.abs(gt_dist_onehot - pred_dist_onehot)
        if cfg.ModelOutput.MINOR_CTGR and num_minor_ctgr is not None:
            pred_minor_tp_count, gt_dist_minor_onehot = count_per_minor_diff_class(
                splits["grtr_tp"]["minor_ctgr"][..., 0], grtr_valid_minor_tp, num_minor_ctgr, grtr_valid_dist)
            pred_minor_tp_count, pred_dist_minor_onehot = count_per_minor_diff_class(pred_minor_dist_aligned,
                                                                                     pred_valid_minor_tp[
                                                                                         ..., np.newaxis],
                                                                                     num_minor_ctgr,
                                                                                     pred_valid_dist[..., np.newaxis])
            diff_minor_tp_dist = np.abs(gt_dist_minor_onehot - pred_dist_minor_onehot)
            return {"trpo": pred_tp_count, "diff": diff,
                    "minor_trpo": pred_minor_tp_count, "minor_diff": diff_minor_tp_dist,
                    }

        return {"trpo": pred_tp_count, "diff": diff}
    else:
        grtr_dist = grtr_valid_dist[grtr_dist_tp_mask > 0]
        pred_dist = pred_valid_dist[grtr_dist_tp_mask[..., 0] > 0]

        distance_diff = np.abs(grtr_dist - pred_dist) / (grtr_dist + 1e-5)
        trpo_count = np.sum(grtr_dist_tp_mask)
        diff = np.sum(distance_diff)
        return {"trpo": trpo_count, "diff": diff}
