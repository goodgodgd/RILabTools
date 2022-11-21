import config as cfg


def get_channel_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_FMAP_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_FMAP_COMPOSITION


def get_bbox_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_INST_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_INST_COMPOSITION


def get_lane_channel_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_FMAP_LANE_COMPOSITION

    else:
        return cfg.ModelOutput.PRED_FMAP_LANE_COMPOSITION


def get_lane_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_INST_LANE_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_INST_LANE_COMPOSITION

