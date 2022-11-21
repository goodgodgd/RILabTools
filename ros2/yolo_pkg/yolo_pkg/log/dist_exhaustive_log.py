import numpy as np
import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import *
from log.logger_pool import *
import utils.framework.util_function as uf


class ExhaustiveDistLog:
    def __init__(self):
        self.num_anchors = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE * len(cfg.ModelOutput.FEATURE_SCALES)
        self.num_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["category"]
        if cfg.ModelOutput.MINOR_CTGR:
            self.num_sign_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["sign_ctgr"]
            self.num_mark_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["mark_ctgr"]
            self.num_mark_dont_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["mark_dont"]
            self.num_mark_do_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["mark_do"]

        self.num_anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        self.num_scales = len(cfg.ModelOutput.FEATURE_SCALES)
        self.metric_data = pd.DataFrame()
        self.sign_metric_data = pd.DataFrame()
        self.mark_metric_data = pd.DataFrame()
        self.mark_dont_metric_data = pd.DataFrame()
        self.mark_do_metric_data = pd.DataFrame()
        self.summary = dict()

    def split_columns(self, columns):
        box_loss_columns, lane_loss_columns = [], []
        for column in columns:
            if "lane" in column:
                lane_loss_columns.append(column)
            else:
                box_loss_columns.append(column)
        return box_loss_columns, lane_loss_columns

    def __call__(self, step, grtr, pred, loss, total_loss):

        metric, sign_metric, mark_metric, mark_dont_metric, mark_do_metric = self.box_category_match(grtr,
                                                                                                     pred["inst_box"],
                                                                                                     range(1,
                                                                                                           self.num_categs),
                                                                                                     step)
        self.metric_data = self.metric_data.append(metric, ignore_index=True)
        self.sign_metric_data = self.sign_metric_data.append(sign_metric, ignore_index=True)
        self.mark_metric_data = self.mark_metric_data.append(mark_metric, ignore_index=True)
        self.mark_dont_metric_data = self.mark_dont_metric_data.append(mark_dont_metric, ignore_index=True)
        self.mark_do_metric_data = self.mark_do_metric_data.append(mark_do_metric, ignore_index=True)

    def box_category_match(self, grtr, pred_bbox, categories, step):
        metric_data = [{"step": step, "ctgr": 0, "diff": 0, "trpo": 0}]
        sign_metric_data = []
        mark_metric_data = []
        mark_dont_metric_data = []
        mark_do_metric_data = []

        for category in categories:
            pred_mask = self.create_mask(pred_bbox, category, "category")
            grtr_mask = self.create_mask(grtr["inst_box"], category, "category")
            grtr_match_box = self.box_matcher(grtr["inst_box"], grtr_mask)
            pred_match_box = self.box_matcher(pred_bbox, pred_mask)
            metric = count_true_positives_dist(grtr_match_box, pred_match_box, grtr["inst_dc"], self.num_categs)
            metric["ctgr"] = category
            metric["step"] = step
            metric_data.append(metric)
            if cfg.ModelOutput.MINOR_CTGR:
                if category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Traffic sign"):
                    minor_metric = count_true_positives_dist(grtr_match_box, pred_match_box, grtr["inst_dc"],
                                                             self.num_categs,
                                                             per_class=True, num_minor_ctgr=self.num_sign_categs,
                                                             target_class=category)
                    sign_metric_data.append(minor_metric)
                elif category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark"):
                    minor_metric = count_true_positives_dist(grtr_match_box, pred_match_box, grtr["inst_dc"],
                                                             self.num_categs,
                                                             per_class=True, num_minor_ctgr=self.num_mark_categs,
                                                             target_class=category)
                    mark_metric_data.append(minor_metric)
                elif category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark dont"):
                    minor_metric = count_true_positives_dist(grtr_match_box, pred_match_box, grtr["inst_dc"],
                                                             self.num_categs,
                                                             per_class=True, num_minor_ctgr=self.num_mark_dont_categs,
                                                             target_class=category)
                    mark_dont_metric_data.append(minor_metric)
                elif category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark do"):
                    minor_metric = count_true_positives_dist(grtr_match_box, pred_match_box, grtr["inst_dc"],
                                                        self.num_categs,
                                                        per_class=True, num_minor_ctgr=self.num_mark_do_categs,
                                                        target_class=category)
                    mark_do_metric_data.append(minor_metric)

        return metric_data, sign_metric_data, mark_metric_data, mark_dont_metric_data, mark_do_metric_data

    def create_mask(self, data, index, key):
        if key:
            valid_mask = data[key] == index
        else:
            valid_mask = data == index
        return valid_mask

    def create_scale_mask(self, data, index, scale, key):
        if key:
            valid_mask = data[key][scale] == index
        else:
            valid_mask = data == index
        return valid_mask

    def box_matcher(self, bbox, mask):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key] * mask
        return match_bbox

    def box_scale_matcher(self, bbox, mask, scale):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key][scale] * mask
        return match_bbox

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        # make summary dataframe
        self.make_summary(epoch_time)
        # write total_data to file

    def make_summary(self, epoch_time):
        sum_summary, sign_summary, mark_summary, mark_dont_summary, mark_do_summary = self.compute_sum_summary()
        # if exist sum_summary
        # summary = pd.merge(left=mean_summary, right=sum_summary, how="outer", on=["anchor", "ctgr"])
        # self.summary = summary

        # summary["time_m"] = round(epoch_time, 5)

        self.summary = sum_summary
        self.mark_summary = mark_summary
        self.mark_dont_summary = mark_dont_summary
        self.mark_do_summary = mark_do_summary
        self.sign_summary = sign_summary
    #
    # def compute_mean_summary(self, epoch_time):
    #     mean_data = self.metric_data[["ctgr", "diff", "trpo"]]
    #     mean_category_data = mean_data.groupby("ctgr", as_index=False).mean()
    #     mean_epoch_data = pd.DataFrame([mean_data.mean(axis=0)])
    #     mean_epoch_data["anchor"] = -1
    #     mean_epoch_data["ctgr"] = -1
    #     mean_epoch_data["time_m"] = epoch_time
    #
    #     mean_summary = pd.concat([mean_epoch_data, mean_category_data],
    #                              join='outer', ignore_index=True)


    def compute_sum_summary(self):
        sum_data = self.metric_data[["ctgr", "diff", "trpo"]]
        sum_category_data = sum_data.groupby("ctgr", as_index=False).sum()
        sum_category_data = pd.DataFrame({
                                          "ctgr": sum_data["ctgr"][:self.num_categs],
                                          "diff": sum_category_data["diff"],
                                          "diff_mean": sum_category_data["diff"] / (sum_category_data["trpo"] + 1e-5),
                                          "trpo": sum_category_data["trpo"]})

        sum_epoch_data = pd.DataFrame([sum_data.sum(axis=0).to_dict()])
        sum_epoch_data = pd.DataFrame({"ctgr": -1,
                                       "diff": sum_epoch_data["diff"],
                                       "diff_mean": sum_epoch_data["diff"] / (sum_epoch_data["trpo"] + 1e-5),
                                       "trpo": sum_epoch_data["trpo"]})

        sum_summary = pd.concat([sum_epoch_data, sum_category_data], join='outer', ignore_index=True)
        if cfg.ModelOutput.MINOR_CTGR:
            sign_data = self.sign_metric_data
            sign_trpo = np.sum(np.stack(sign_data["minor_trpo"].tolist(), axis=0), axis=0)
            sign_diff = np.sum(np.stack(sign_data["minor_diff"].tolist(), axis=0), axis=0)
            sign_summary = pd.DataFrame({"diff_mean": sign_diff / (sign_trpo + 1e-5),
                                         "trpo": sign_trpo})
            mark_data = self.mark_metric_data
            mark_trpo = np.sum(np.stack(mark_data["minor_trpo"].tolist(), axis=0), axis=0)
            mark_diff = np.sum(np.stack(mark_data["minor_diff"].tolist(), axis=0), axis=0)
            mark_summary = pd.DataFrame({"diff_mean": mark_diff / (mark_trpo + 1e-5),
                                         "trpo": mark_trpo})
            mark_dont_data = self.mark_dont_metric_data
            mark_dont_trpo = np.sum(np.stack(mark_dont_data["minor_trpo"].tolist(), axis=0), axis=0)
            mark_dont_diff = np.sum(np.stack(mark_dont_data["minor_diff"].tolist(), axis=0), axis=0)
            mark_dont_summary = pd.DataFrame({"diff_mean": mark_dont_diff / (mark_dont_trpo + 1e-5),
                                              "trpo": mark_dont_trpo})
            mark_do_data = self.mark_do_metric_data
            mark_do_trpo = np.sum(np.stack(mark_do_data["minor_trpo"].tolist(), axis=0), axis=0)
            mark_do_diff = np.sum(np.stack(mark_do_data["minor_diff"].tolist(), axis=0), axis=0)
            mark_do_summary = pd.DataFrame({"recall": mark_do_diff / (mark_do_trpo + 1e-5),
                                            "trpo": mark_do_trpo})
            return sum_summary, sign_summary, mark_summary, mark_dont_summary, mark_do_summary
        return sum_summary, None, None, None, None

    def get_summary(self):
        return self.summary

    def get_minor_summary(self):
        return self.sign_summary, self.mark_summary, self.mark_dont_summary, self.mark_do_summary


def summarize_data():
    raw_data = pd.read_csv(
        "/home/falcon/kim_workspace/ckpt/scaled_weight_inverse+lane_bgd/exhaust_log/exhaust_box_val_total.csv")
    log_data = raw_data.copy()
    log_data = log_data.fillna(0)
    print(log_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN].head())
    print(log_data[["grtr"]].head())
    for key in cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN[2:]:
        log_data[key] = log_data[key] * log_data["grtr"]
    print(log_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN].head())

    ctgr_data = log_data.groupby("ctgr", as_index=False).sum()
    ctgr_data["anchor"] = -1
    anch_data = log_data.groupby("anchor", as_index=False).sum()
    anch_data["ctgr"] = -1
    single_summary = pd.DataFrame(np.sum(log_data.values, axis=0, keepdims=True), columns=list(log_data))
    single_summary[["anchor"]] = -1
    single_summary[["ctgr"]] = -1
    print(single_summary)

    summary_data = pd.concat([single_summary, ctgr_data, anch_data], axis=0)
    print(summary_data)
    for key in cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN[2:]:
        summary_data[key] = summary_data[key] / (summary_data["grtr"] + 1e-5)
    print(summary_data)

    total_data = pd.concat([summary_data, raw_data])
    print(total_data)
    total_data.to_csv(
        "/home/falcon/kim_workspace/ckpt/scaled_weight_inverse+lane_bgd/exhaust_log/exhaust_box_val_new.csv",
        encoding='utf-8', index=False, float_format='%.4f')


if __name__ == "__main__":
    summarize_data()
