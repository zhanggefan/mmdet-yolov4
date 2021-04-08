# from collections import OrderedDict

import numpy as np
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.progressbar import track_iter_progress

from mmdet.ops.eval_utils.iou import iou_coco
from mmdet.ops.eval_utils.match import match_coco
from .mean_ap import average_precision

EVAL_BREAKDOWN = Registry('2D Evaluation Breakdown')


class NoBreakdown:

    def __init__(self, classes, apply_to=None):
        if apply_to is None:
            apply_to = classes
        self.classes = classes
        self.apply_to = apply_to
        self.names = ['All']

    def breakdown_flags(self, boxes, attrs=None):
        num_boxes = len(boxes)
        flags = np.ones((1, num_boxes), dtype=np.bool)
        if attrs is not None and 'ignore' in attrs:
            flags[:, attrs['ignore']] = False
        return flags

    def breakdown(self, boxes, label, attrs=None):
        flags = self.breakdown_flags(boxes, attrs)
        if self.classes[label] in self.apply_to:
            return flags
        else:
            return flags[:0]

    def breakdown_names(self, label):
        if self.classes[label] in self.apply_to:
            return [f'{n}' for n in self.names]
        else:
            return []


@EVAL_BREAKDOWN.register_module()
class ScaleBreakdown(NoBreakdown):

    def __init__(self, scale_ranges, classes, apply_to=None):
        super(ScaleBreakdown, self).__init__(classes, apply_to)
        self.names = []
        self.area_ranges = []
        for k in scale_ranges:
            self.names.append(k)
            smin, smax = scale_ranges[k]
            self.area_ranges.append((smin * smin, smax * smax))

    def breakdown_flags(self, boxes, attrs=None):
        num_ranges = len(self.area_ranges)
        num_boxes = len(boxes)
        if attrs is not None and 'area' in attrs:
            area = attrs['area']
        else:
            wh = boxes[:, 2:] - boxes[:, :2]
            area = wh[:, 0] * wh[:, 1]
        area_flags = np.zeros((num_ranges, num_boxes), dtype=np.bool)
        for dist_idx, (min_area, max_area) in enumerate(self.area_ranges):
            area_flags[dist_idx][(area >= min_area) & (area < max_area)] = True
        if attrs is not None and 'ignore' in attrs:
            area_flags[:, attrs['ignore']] = False
        return area_flags


def statistics_single(det, anno, iou_thrs, breakdown=[]):
    """Check if detected bboxes are true positive or false positive."""
    tp_score_info = []

    num_cls = len(det)
    num_iou_thrs = len(iou_thrs)

    gt_bboxes = anno['gt_bboxes']
    gt_labels = anno['gt_labels']
    gt_attrs = anno['gt_attrs']

    for cls in range(num_cls):
        # prepare detections
        cls_tp_score_info = []

        cls_det_bboxes = det[cls][:, :4]
        cls_det_scores = det[cls][:, 4]
        sort_ind = cls_det_scores.argsort()[::-1]
        cls_det_bboxes = cls_det_bboxes[sort_ind]
        cls_det_scores = cls_det_scores[sort_ind]
        cls_num_dets = cls_det_scores.shape[0]

        # prepare ground-truths
        cls_gt_msk = gt_labels == cls
        cls_gt_bboxes = gt_bboxes[cls_gt_msk]
        cls_gt_attrs = {k: v[cls_gt_msk] for k, v in gt_attrs.items()}
        cls_gt_ignore_msk = cls_gt_attrs['ignore']
        cls_gt_crowd_msk = cls_gt_attrs['iscrowd']
        cls_num_ignore_gts = np.count_nonzero(cls_gt_ignore_msk)
        cls_num_gts = len(cls_gt_bboxes) - cls_num_ignore_gts

        # prepare breakdown masks
        cls_det_bkd = []
        cls_gt_bkd = []
        cls_bkd_names = []
        for fun in breakdown:
            cls_det_bkd.append(fun.breakdown(cls_det_bboxes, cls))
            cls_gt_bkd.append(fun.breakdown(cls_gt_bboxes, cls, cls_gt_attrs))
            cls_bkd_names += fun.breakdown_names(cls)
        cls_det_bkd = np.concatenate(cls_det_bkd, axis=0)
        cls_gt_bkd = np.concatenate(cls_gt_bkd, axis=0)
        num_bkd = cls_gt_bkd.shape[0]

        # all detections are false positive by default
        cls_tp = np.zeros((num_iou_thrs, cls_num_dets), dtype=np.bool)

        # calculate num gt (not considering ignored gt boxes)
        cls_gt_count = []
        for bkd_idx in range(num_bkd):
            cls_gt_count.append(np.count_nonzero(cls_gt_bkd[bkd_idx]))

        # handling empty det or empty gt
        if (cls_num_gts + cls_num_ignore_gts) == 0 or cls_num_dets == 0:
            for bkd_idx in range(num_bkd):
                cls_tp_score_info.append(
                    (cls_bkd_names[bkd_idx], cls_gt_count[bkd_idx],
                     cls_det_scores, cls_tp,
                     cls_det_bkd[bkd_idx:bkd_idx + 1].repeat(
                         num_iou_thrs, axis=0)))
        else:
            ious = iou_coco(cls_det_bboxes, cls_gt_bboxes, cls_gt_crowd_msk)

            for bkd_idx in range(num_bkd):
                cls_gt_bkd_msk = ((~cls_gt_ignore_msk) & (cls_gt_bkd[bkd_idx]))
                matched_gt_idx = match_coco(
                    ious, np.array(iou_thrs, dtype=np.float32),
                    (~cls_gt_bkd_msk), cls_gt_crowd_msk.astype(np.bool))

                cls_tp[...] = False
                cls_tp[matched_gt_idx > -1] = True

                _msk_fp = (
                    cls_det_bkd[bkd_idx:bkd_idx + 1] & (matched_gt_idx == -1))
                _msk_tp = ((cls_gt_bkd_msk[matched_gt_idx]) &
                           (matched_gt_idx > -1))
                cls_tp_score_info.append(
                    (cls_bkd_names[bkd_idx], cls_gt_count[bkd_idx],
                     cls_det_scores, cls_tp, (_msk_fp | _msk_tp)))

        tp_score_info.append(cls_tp_score_info)

    return tp_score_info


def eval_map_flexible(det_results,
                      annotations,
                      iou_thrs=[0.5],
                      breakdown=[],
                      classes=None,
                      logger=None,
                      nproc=4):
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    # num_classes = len(det_results[0])
    # num_ious = len(iou_thrs)

    for bkd_idx in range(len(breakdown)):
        breakdown[bkd_idx] = build_from_cfg(
            breakdown[bkd_idx],
            EVAL_BREAKDOWN,
            default_args=dict(classes=classes))
    breakdown.insert(0, NoBreakdown(classes))
    tp_score_infos = []

    for det_result, annotation in zip(
            track_iter_progress(det_results), annotations):
        tp_score_infos.append(
            statistics_single(det_result, annotation, iou_thrs, breakdown))

    for img in range(1, num_imgs):
        for cls, cls_0_tp_score_infos in enumerate(tp_score_infos[0]):
            for bkd, bdk_cls_0_tp_score_infos in enumerate(
                    cls_0_tp_score_infos):
                (_name, _num_gt, _score, _tp, _bkd_msk) = \
                    tp_score_infos[img][cls][bkd]
                if isinstance(bdk_cls_0_tp_score_infos, tuple):
                    bdk_cls_0_tp_score_infos = list(bdk_cls_0_tp_score_infos)
                    bdk_cls_0_tp_score_infos[2] = [bdk_cls_0_tp_score_infos[2]]
                    bdk_cls_0_tp_score_infos[3] = [bdk_cls_0_tp_score_infos[3]]
                    bdk_cls_0_tp_score_infos[4] = [bdk_cls_0_tp_score_infos[4]]
                    tp_score_infos[0][cls][bkd] = bdk_cls_0_tp_score_infos
                assert bdk_cls_0_tp_score_infos[0] == _name
                bdk_cls_0_tp_score_infos[1] += _num_gt
                bdk_cls_0_tp_score_infos[2].append(_score)
                bdk_cls_0_tp_score_infos[3].append(_tp)
                bdk_cls_0_tp_score_infos[4].append(_bkd_msk)

    tp_score_infos = tp_score_infos[0]

    map50, map75, map5095, smap, mmap, lmap = [], [], [], [], [], []

    eval_result_list = []

    for cls, cls_tp_score_infos in enumerate(tp_score_infos):
        for bkd, bdk_cls_tp_score_infos in enumerate(cls_tp_score_infos):
            cls_name = classes[cls]
            bkd_name = bdk_cls_tp_score_infos[0]
            num_gt = bdk_cls_tp_score_infos[1]
            scores = np.concatenate(bdk_cls_tp_score_infos[2], axis=0)
            tp = np.concatenate(bdk_cls_tp_score_infos[3], axis=1)
            bkd_msk = np.concatenate(bdk_cls_tp_score_infos[4], axis=1)
            rank = scores.argsort()[::-1]
            scores = scores[rank]
            tp = tp[:, rank]
            bkd_msk = bkd_msk[:, rank]
            for iou_thr_idx, iou_thr in enumerate(iou_thrs):
                tpcumsum = tp[iou_thr_idx, bkd_msk[iou_thr_idx]].cumsum()
                num_dets = len(tpcumsum)
                recall = tpcumsum / max(num_gt, 1e-7)
                precision = tpcumsum / np.arange(1, num_dets + 1)
                m_ap = average_precision(recall, precision)
                max_recall = recall.max() if len(recall) > 0 else 0
                eval_result_list.append((cls_name, bkd_name, iou_thr, num_dets,
                                         num_gt, max_recall, m_ap))
                if num_gt > 0:
                    if bkd_name == 'All':
                        if iou_thr == 0.5:
                            map50.append(m_ap)
                        elif iou_thr == 0.75:
                            map75.append(m_ap)
                        map5095.append(m_ap)
                    elif bkd_name == 'Scale_S':
                        smap.append(m_ap)
                    elif bkd_name == 'Scale_M':
                        mmap.append(m_ap)
                    elif bkd_name == 'Scale_L':
                        lmap.append(m_ap)

    print('map50:', sum(map50) / len(map50))
    print('map75:', sum(map75) / len(map75))
    print('map5095:', sum(map5095) / len(map5095))
    print('smap:', sum(smap) / len(smap))
    print('mmap:', sum(mmap) / len(mmap))
    print('lmap:', sum(lmap) / len(lmap))

    # print_map_summary(mean_ap, eval_results, dataset, area_ranges,
    #                   logger=logger)

    # return mean_ap, eval_results
