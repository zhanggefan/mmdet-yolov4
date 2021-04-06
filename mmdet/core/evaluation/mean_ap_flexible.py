from collections import OrderedDict

import numpy as np
import tqdm
from mmcv.utils import Registry

from mmdet.ops.eval_utils.match import match_coco
from .bbox_overlaps import bbox_overlaps
from .mean_ap import average_precision

EVAL_BREAKDOWN = Registry('2D Evaluation Breakdown')


class NoBreakDown:

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
class ScaleBreakDown(NoBreakDown):

    def __init__(self, scale_ranges, classes, apply_to=None):
        super(ScaleBreakDown, self).__init__(classes, apply_to)
        self.names = [f'S{smin}-{smax}' for smin, smax in scale_ranges]
        self.area_ranges = [(smin * smin, smax * smax)
                            for smin, smax in scale_ranges]

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
    tp_score = []
    breakdown_info = []

    num_cls = len(det)

    gt_bboxes = anno['gt_bboxes']
    gt_labels = anno['gt_labels']
    gt_attrs = anno['gt_attrs']

    for cls in range(num_cls):
        # prepare detections
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
        cls_tp_score = np.zeros((cls_num_dets, 2), dtype=np.float32)
        cls_tp_score[:, -1] = cls_det_scores

        # calculate num gt (not considering ignored gt boxes)
        cls_gt_count = []
        for bkd_idx in range(num_bkd):
            cls_gt_count.append(np.count_nonzero(cls_gt_bkd[bkd_idx]))

        # handling empty det or empty gt
        if (cls_num_gts + cls_num_ignore_gts) == 0 or cls_num_dets == 0:
            tp_score += [cls_tp_score] * len(iou_thrs)
            breakdown_info += [[(cls_bkd_names[bkd_idx], cls_det_bkd[bkd_idx],
                                 cls_gt_count[bkd_idx])
                                for bkd_idx in range(num_bkd)]] * len(iou_thrs)
            continue

        ious = np.empty((cls_num_dets, cls_num_gts + cls_num_ignore_gts),
                        dtype=np.float32)
        ious[:,
             cls_gt_crowd_msk] = bbox_overlaps(cls_det_bboxes,
                                               cls_gt_bboxes[cls_gt_crowd_msk],
                                               'iof')
        ious[:, ~cls_gt_crowd_msk] = bbox_overlaps(
            cls_det_bboxes, cls_gt_bboxes[~cls_gt_crowd_msk])

        matched_gt_idx = match_coco(ious, np.array(iou_thrs, dtype=np.float32),
                                    cls_gt_ignore_msk.astype(np.bool),
                                    cls_gt_crowd_msk.astype(np.bool))

        for iou_matched_gt_idx in matched_gt_idx:
            cls_tp_score[:, 0] = 0
            cls_tp_score[iou_matched_gt_idx > -1, 0] = 1
            tp_score.append(cls_tp_score.copy())

            breakdown_info.append([])
            for bkd_idx in range(num_bkd):
                _msk_fp = (cls_det_bkd[bkd_idx] & (iou_matched_gt_idx == -1))
                _msk_tp = (
                    cls_gt_bkd[bkd_idx][iou_matched_gt_idx] &
                    (iou_matched_gt_idx > -1))
                breakdown_info[-1].append(
                    (cls_bkd_names[bkd_idx], (_msk_fp | _msk_tp),
                     cls_gt_count[bkd_idx]))

    return tp_score, breakdown_info


def eval_map_flexible(det_results,
                      annotations,
                      scale_ranges=None,
                      iou_thrs=[0.5],
                      dataset=None,
                      logger=None,
                      nproc=4):
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num
    num_ious = len(iou_thrs)

    breakdown = [NoBreakDown(dataset)]
    if scale_ranges is not None:
        breakdown.append(ScaleBreakDown(scale_ranges, dataset))

    tp_scores, breakdown_infos = [], []

    for det_result, annotation in zip(tqdm.tqdm(det_results), annotations):
        statistics = statistics_single(det_result, annotation, iou_thrs,
                                       breakdown)
        tp_scores.append(statistics[0])
        breakdown_infos.append(statistics[1])

    tp_scores = [np.concatenate(x, axis=0) for x in zip(*tp_scores)]

    breakdown_dicts = []

    for cls_iou in range(num_classes * num_ious):
        breakdown_dict = OrderedDict()
        for img in range(num_imgs):
            for name, mask, gt_count in breakdown_infos[img][cls_iou]:
                if name not in breakdown_dict:
                    breakdown_dict[name] = [[mask], gt_count]
                else:
                    breakdown_dict[name][0].append(mask)
                    breakdown_dict[name][1] += gt_count
        for name in breakdown_dict:
            breakdown_dict[name][0] = np.concatenate(
                breakdown_dict[name][0], axis=0)
        breakdown_dicts.append(breakdown_dict)

    map50, map75, map5095, smap, mmap, lmap = [], [], [], [], [], []
    for cls_iou, (tp_score, breakdown_dict) in enumerate(
            zip(tp_scores, breakdown_dicts)):
        rank = tp_score[:, 1].argsort()[::-1]
        tp = tp_score[:, 0][rank]
        # classname = dataset[cls_iou // num_ious]
        iou_thr = iou_thrs[cls_iou % num_ious]
        for bkd_name, (mask, num_gt) in breakdown_dict.items():
            tpcumsum = tp[mask[rank]].cumsum()
            num_dets = len(tpcumsum)
            recall = tpcumsum / max(num_gt, 1e-7)
            precision = tpcumsum / np.arange(1, num_dets + 1)
            m_ap = average_precision(recall, precision)
            # max_recall = recall.max() if len(recall) > 0 else 0

            # print(classname, iou_thr, bkd_name, num_dets, num_gt, max_recall,
            #       m_ap)
            if num_gt > 0:
                if bkd_name == 'All':
                    if iou_thr == 0.5:
                        map50.append(m_ap)
                    elif iou_thr == 0.75:
                        map75.append(m_ap)
                    map5095.append(m_ap)
                elif bkd_name == 'S0-32':
                    smap.append(m_ap)
                elif bkd_name == 'S32-96':
                    mmap.append(m_ap)
                elif bkd_name == 'S96-10000':
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
