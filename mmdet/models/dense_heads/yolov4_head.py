# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from mmdet.models.backbones.darknetcsp import Mish
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from ..losses.iou_loss import ciou_loss, giou_loss
import math


@HEADS.register_module()
class YOLOV4Head(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='YOLOV4AnchorGenerator',
                     base_sizes=[[(12, 16), (19, 36), (40, 28)],  # P3/8
                                 [(36, 75), (76, 55), (72, 146)],  # P4/16
                                 [(142, 110), (192, 243), (459, 401)]],  # P5/32
                     strides=[8, 16, 32]),
                 bbox_coder=dict(type='YOLOV4BBoxCoder'),
                 featmap_strides=[8, 16, 32],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.5),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='GIoULoss',
                     loss_weight=0.05),
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV4Head, self).__init__()
        # Check params
        assert (len(in_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = None
        self.shape_match_thres = 4.
        self.conf_iou_loss_ratio = 1.
        if self.train_cfg:
            if hasattr(self.train_cfg, 'assigner'):
                self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            if hasattr(self.train_cfg, 'conf_iou_loss_ratio'):
                self.conf_iou_loss_ratio = self.train_cfg.conf_iou_loss_ratio
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.shape_match_thres = self.train_cfg.get('shape_match_thres', 4.)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_bbox = build_loss(loss_bbox)

        self.num_anchors = self.anchor_generator.num_base_anchors
        self._init_layers()
        self.fp16_enabled = False

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_anchors[i] * self.num_attrib, 1)
            self.convs_pred.append(conv_pred)

    def init_weights(self, class_frequency=None):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)
        for m, stride in zip(self.convs_pred, self.featmap_strides):
            b = m.bias.view(-1, self.num_attrib)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if class_frequency is None else torch.log(
                class_frequency / class_frequency.sum())  # cls
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            pred_map = self.convs_pred[i](feats[i])
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map = pred_map.sigmoid()
            pred_map[..., :2] = pred_map[..., :2] * 2. - 0.5
            pred_map[..., 2:4] = (pred_map[..., 2:4] * 2) ** 2
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = pred_map[..., 4].view(-1)
            cls_pred = pred_map[..., 5:].view(-1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and (multi_lvl_conf_scores.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0,))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0],
                                                 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding],
                                         dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores,
                    multi_lvl_conf_scores)

    # @force_fp32(apply_to=('pred_maps',))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]

        responsible_indices = []
        for img_id in range(len(img_metas)):
            responsible_indices.append(
                self.anchor_generator.responsible_indices(
                    featmap_sizes,
                    gt_bboxes[img_id],
                    neighbor=2 if self.assigner is None else 3,
                    shape_match_thres=self.shape_match_thres,
                    device=device))

        if self.assigner is None:
            results = self.get_targets_no_assigner(responsible_indices, gt_bboxes, gt_labels)

            (mlvl_pos_indices, mlvl_gt_bboxes_targets, mlvl_gt_labels_targets) = results

            mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)

            losses_cls, losses_conf, losses_bbox = multi_apply(
                self.loss_single_no_assigner,
                pred_maps,
                mlvl_anchors,
                self.featmap_strides,
                mlvl_pos_indices,
                mlvl_gt_bboxes_targets,
                mlvl_gt_labels_targets
            )
        else:
            raise NotImplementedError

        balance_conf = [4.0, 1.0, 0.4, 0.1, 0.1]
        losses_conf = [loss_conf * balance for loss_conf, balance in zip(losses_conf, balance_conf)]

        return dict(
            loss_cls=losses_cls * len(img_metas),
            loss_conf=losses_conf * len(img_metas),
            loss_bbox=losses_bbox * len(img_metas))

    def loss_single_no_assigner(self,
                                pred_map,
                                anchors,
                                stride,
                                pos_indices,
                                target_bboxes,
                                target_labels):
        """Compute loss of a single level.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            anchors (Tensor): anchors for a single level.
                has shape (num_anchors, 4)
            pos_indices (tuple[Tensor]): positive sample indices.
                (img_idx, anchor_idx), each indices tensor has shape
                (k,), which stands for k positive samples
            pos_bboxes (Tensor): target tensor for positive samples.
                has shape (k, 4)
            pos_labels (Tensor): target tensor for positive samples.
                has shape (k, self.num_classes)

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)

        img_ind, anchor_ind = pos_indices

        pred_conf = pred_map[..., 4]
        target_conf = torch.zeros_like(pred_conf, requires_grad=False)

        loss_bbox = pred_map.new_zeros((1,))
        loss_cls = pred_map.new_zeros((1,))

        if anchor_ind.numel():
            pred_map_pos = pred_map[img_ind, anchor_ind]
            anchor_pos = anchors[anchor_ind]

            # apply transforms on bbox prediction
            pred_bbox = pred_map_pos[..., :4].sigmoid()
            pred_bbox_xy = pred_bbox[..., :2] * 2. - 0.5
            pred_bbox_wh = (pred_bbox[..., 2:] * 2) ** 2
            pred_bbox = self.bbox_coder.decode(anchor_pos, torch.cat((pred_bbox_xy, pred_bbox_wh), dim=-1), stride)

            loss_bbox += self.loss_bbox(pred_bbox, target_bboxes)

            pred_cls = pred_map_pos[..., 5:]
            target_cls = target_labels

            loss_cls += self.loss_cls(pred_cls, target_cls)

            target_conf[pos_indices] = (1 - self.conf_iou_loss_ratio) + self.conf_iou_loss_ratio * (
                    1 - giou_loss(pred_bbox, target_bboxes, reduction='none')).detach().clamp(0).type(target_conf.dtype)

        loss_conf = self.loss_conf(pred_conf, target_conf)

        return loss_cls, loss_conf, loss_bbox

    def get_targets_no_assigner(self,
                                responsible_indices_list,
                                gt_bboxes_list,
                                gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            responsible_indices_list (list[list[Tensor]]): Multi level responsible
                indices of each image. Each element is a tensor of shape
                (m, 2), where m stands for the number of matches
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - mlvl_pos_indices (tuple[Tensor, Tensor]): Postive image/anchor
                    indices of each level.
                - mlvl_gt_bboxes_targets (list[Tensor]): Gt bbox target corresponds
                    to the aforementioned indices.
                - mlvl_gt_labels_targets (list[Tensor]): Gt label corresponds
                    to the aforementioned indices.
        """
        num_imgs = len(gt_bboxes_list)
        mlvl_pos_indices = [None for _ in range(self.num_levels)]
        mlvl_gt_bboxes_targets = [None for _ in range(self.num_levels)]
        mlvl_gt_labels_targets = [None for _ in range(self.num_levels)]

        for lvl in range(self.num_levels):
            img_indices = []
            anchor_indices = []
            gt_labels_targets = []
            gt_bboxes_targets = []
            for img_ind in range(num_imgs):
                anchor_ind, gt_ind = responsible_indices_list[img_ind][lvl]
                anchor_indices.append(anchor_ind)
                img_indices.append(anchor_ind.new_full(anchor_ind.shape, img_ind))
                gt_bboxes_targets.append(gt_bboxes_list[img_ind][gt_ind])
                gt_labels_targets.append(gt_labels_list[img_ind][gt_ind])

            mlvl_pos_indices[lvl] = (torch.cat(img_indices, dim=0),
                                     torch.cat(anchor_indices, dim=0))

            mlvl_gt_bboxes_targets[lvl] = torch.cat(gt_bboxes_targets, dim=0)
            gt_labels_targets = torch.cat(gt_labels_targets, dim=0)

            gt_labels_targets = F.one_hot(
                gt_labels_targets, num_classes=self.num_classes).float()
            if self.one_hot_smoother != 0:  # label smooth
                gt_labels_targets = gt_labels_targets * (
                        1 - self.one_hot_smoother
                ) + self.one_hot_smoother / self.num_classes
            mlvl_gt_labels_targets[lvl] = gt_labels_targets

        return mlvl_pos_indices, mlvl_gt_bboxes_targets, mlvl_gt_labels_targets

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
