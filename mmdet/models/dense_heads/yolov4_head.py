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
from ..losses import reduce_loss
import math
from torch.cuda.amp import autocast


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
                 norm_cfg=dict(type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
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
        self.conf_level_balance_weight = [4.0, 1.0, 0.4, 0.1, 0.1]
        if self.train_cfg:
            if hasattr(self.train_cfg, 'assigner'):
                self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            if hasattr(self.train_cfg, 'conf_iou_loss_ratio'):
                self.conf_iou_loss_ratio = self.train_cfg.conf_iou_loss_ratio
            if hasattr(self.train_cfg, 'conf_level_balance_weight'):
                self.conf_level_balance_weight = self.train_cfg.conf_level_balance_weight
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
        self.loss_bbox_weight = self.loss_bbox.loss_weight
        self.loss_bbox.loss_weight = 1.

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

    # @force_fp32(apply_to=('pred_maps',))
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
        with autocast(enabled=False):
            num_levels = len(pred_maps)
            num_image = len(img_metas)

            featmap_sizes = [pred_maps[i].shape[-2:] for i in range(num_levels)]
            mlvl_anchors = self.anchor_generator.grid_anchors(
                featmap_sizes, pred_maps[0].device)

            mlvl_bbox_pred = []
            mlvl_conf_pred = []
            mlvl_score_pred = []

            for lvl in range(num_levels):
                lvl_pred_maps = pred_maps[lvl].permute(0, 2, 3, 1).reshape(
                    (num_image, -1, self.num_attrib))
                # activation
                lvl_pred_maps = lvl_pred_maps.sigmoid()
                # class score
                mlvl_score_pred.append(lvl_pred_maps[:, :, 5:])
                # conf score
                mlvl_conf_pred.append(lvl_pred_maps[:, :, 4])
                # bbox transform
                lvl_pred_maps[:, :, :2] = lvl_pred_maps[:, :, :2] * 2. - 0.5
                lvl_pred_maps[:, :, 2:4] = (lvl_pred_maps[:, :, 2:4] * 2) ** 2
                lvl_bbox_pred = lvl_pred_maps[:, :, :4].reshape(-1, 4)
                lvl_anchors = mlvl_anchors[lvl][None, ...].repeat((num_image, 1, 1))

                lvl_bbox_pred = self.bbox_coder.decode(bboxes=lvl_anchors.reshape(-1, 4),
                                                       pred_bboxes=lvl_bbox_pred.reshape(-1, 4),
                                                       stride=self.featmap_strides[lvl])
                lvl_bbox_pred = lvl_bbox_pred.reshape((num_image, -1, 4))
                mlvl_bbox_pred.append(lvl_bbox_pred)

            mimg_score_pred = [score for score in torch.cat(mlvl_score_pred, dim=1)]
            mimg_conf_pred = [conf for conf in torch.cat(mlvl_conf_pred, dim=1)]
            mimg_bbox_pred = [bbox for bbox in torch.cat(mlvl_bbox_pred, dim=1)]

            result_list = []

            for img_id in range(len(img_metas)):
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self._get_bboxes_single(mimg_score_pred[img_id], mimg_conf_pred[img_id],
                                                    mimg_bbox_pred[img_id], scale_factor, cfg, rescale, with_nms)
                result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_pred,
                           conf_pred,
                           bbox_pred,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_pred (Tensor): Score of each predicted bbox of a single image.
            conf_pred (Tensor): Confidence of each predicted bbox of a single image.
            bbox_pred (Tensor): Predicted bbox of a single image.
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

        # Filtering out all predictions with conf < conf_thr
        conf_thr = cfg.get('conf_thr', -1)
        conf_inds = conf_pred.ge(conf_thr)
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

        if with_nms and (cls_pred.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0,))

        if rescale:
            bbox_pred /= bbox_pred.new_tensor(scale_factor)

        if with_nms:
            # In mmdet 2.x, the class_id for background is num_classes.
            # i.e., the last column.
            padding = cls_pred.new_zeros(cls_pred.shape[0], 1)
            cls_pred = torch.cat([cls_pred, padding], dim=1)

            det_bboxes, det_labels = multiclass_nms(
                bbox_pred,
                cls_pred,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=conf_pred)
            return det_bboxes, det_labels
        else:
            cls_pred = cls_pred * conf_pred[:, None]
            class_score, class_id = cls_pred.max(dim=-1)
            return torch.cat((bbox_pred, class_score[:, None]), dim=-1), class_id

    @force_fp32(apply_to=('pred_maps',))
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
        with autocast(enabled=False):

            pred_maps = [p.float() for p in pred_maps]

            device = pred_maps[0][0].device

            featmap_sizes = [
                pred_maps[i].shape[-2:] for i in range(self.num_levels)
            ]

            responsible_indices = self.anchor_generator.responsible_indices(
                featmap_sizes,
                gt_bboxes,
                neighbor=2 if self.assigner is None else 3,
                shape_match_thres=self.shape_match_thres,
                device=device)

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

            losses_conf = [loss_conf * balance for loss_conf, balance in
                           zip(losses_conf, self.conf_level_balance_weight)]

            # lcls, lbox, lobj = compute_loss(pred_maps, img_metas, gt_bboxes, gt_labels)

        return dict(
            loss_cls=losses_cls,
            # l_cls=lcls * len(img_metas),
            loss_conf=losses_conf,
            # l_conf=lobj * len(img_metas),
            loss_bbox=losses_bbox,
            # l_bbox=lbox * len(img_metas),
        )

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

            giou_loss = self.loss_bbox(pred_bbox, target_bboxes, reduction_override='none')

            loss_bbox += reduce_loss(giou_loss, reduction=self.loss_bbox.reduction)

            pred_cls = pred_map_pos[..., 5:]
            target_cls = target_labels

            loss_cls += self.loss_cls(pred_cls, target_cls)

            target_conf[pos_indices] = (1 - self.conf_iou_loss_ratio) + self.conf_iou_loss_ratio * (
                    1 - giou_loss).detach().clamp(0.0, 1.0).type(target_conf.dtype)

        loss_conf = self.loss_conf(pred_conf, target_conf)

        return loss_cls * num_imgs, loss_conf * num_imgs, loss_bbox * self.loss_bbox_weight * num_imgs

    def get_targets_no_assigner(self,
                                responsible_indices_list,
                                gt_bboxes_list,
                                gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            responsible_indices_list ([list[tuple[Tensor]]]): Multi level responsible
                indices. Each element is a tuple of 3 tensors of shape (m,),
                    (img_id, anchor_id, img_gt_id)
                where m stands for the number of matches
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

        mlvl_pos_indices = [None for _ in range(self.num_levels)]
        mlvl_gt_bboxes_targets = [None for _ in range(self.num_levels)]
        mlvl_gt_labels_targets = [None for _ in range(self.num_levels)]

        gt_bboxes = torch.cat(gt_bboxes_list, dim=0)
        gt_labels = torch.cat(gt_labels_list, dim=0)

        for lvl in range(self.num_levels):
            img_ind, anchor_ind, gt_ind = responsible_indices_list[lvl]

            mlvl_pos_indices[lvl] = (img_ind, anchor_ind)
            mlvl_gt_bboxes_targets[lvl] = gt_bboxes[gt_ind]
            gt_labels_targets = gt_labels[gt_ind]

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


# # following code are moved here for comparison only


# def compute_loss(p, img_metas, gt_bboxes_list, gt_labels_list):  # predictions, targets, model

#     tgt = []
#     for i in range(len(gt_bboxes_list)):
#         h, w = p[0].shape[2:]
#         h*=8
#         w*=8
#         num_gt = gt_bboxes_list[i].shape[0]
#         gt_bboxes_list[i] /= torch.tensor([w, h, w, h]).cuda()
#         gt_xy = (gt_bboxes_list[i][:, :2] + gt_bboxes_list[i][:, 2:]) / 2
#         gt_wh = gt_bboxes_list[i][:, 2:] - gt_bboxes_list[i][:, :2]
#         img_id = gt_wh.new_full((num_gt, 1), i)
#         tgt.append(torch.cat([img_id, gt_labels_list[i][:, None], gt_xy, gt_wh], dim=-1))
#     targets = torch.cat(tgt, dim=0)

#     device = targets.device
#     ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
#     lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
#     tcls, tbox, indices, anchors = build_targets(p, targets)  # targets

#     red = 'mean'  # Loss reduction (sum or mean)

#     # Define criteria
#     BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction=red).to(device)
#     BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction=red).to(device)

#     # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#     cp, cn = 1.0, 0.0

#     # per output
#     nt = 0  # number of targets
#     np = len(p)  # number of outputs
#     balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
#     for i, pi in enumerate(p):  # layer index, layer predictions
#         bs, _, h, w = pi.shape
#         pi = pi.reshape(bs, 3, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
#         b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#         tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

#         nb = b.shape[0]  # number of targets
#         if nb:
#             nt += nb  # cumulative targets
#             ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

#             # GIoU
#             pxy = ps[:, :2].sigmoid() * 2. - 0.5
#             pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#             pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
#             giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
#             lbox += (1.0 - giou).mean()  # giou loss

#             # Obj
#             tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

#             t = torch.full_like(ps[:, 5:], cn).to(device)  # targets
#             t[range(nb), tcls[i]] = cp
#             lcls += BCEcls(ps[:, 5:], t)  # BCE

#             # Append targets to text file
#             # with open('targets.txt', 'a') as file:
#             #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

#         lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

#     s = 3 / np  # output count scaling
#     lbox *= 0.05 * s
#     lobj *= 1.0 * s * (1.4 if np == 4 else 1.)
#     lcls *= 0.5 * s

#     return lcls, lbox, lobj


# def build_targets(p, targets):
#     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

#     na, nt = 3, targets.shape[0]  # number of anchors, targets
#     tcls, tbox, indices, anch = [], [], [], []
#     gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
#     off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
#     at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

#     g = 0.5  # offset
#     style = 'rect4'

#     det_anchors = torch.tensor([[[1.50000, 2.00000],
#                                  [2.37500, 4.50000],
#                                  [5.00000, 3.50000]],

#                                 [[2.25000, 4.68750],
#                                  [4.75000, 3.43750],
#                                  [4.50000, 9.12500]],

#                                 [[4.43750, 3.43750],
#                                  [6.00000, 7.59375],
#                                  [14.34375, 12.53125]]], device=targets.device)

#     for i in range(3):
#         anchors = det_anchors[i]
#         gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#         # Match targets to anchors
#         a, t, offsets = [], targets * gain, 0
#         if nt:
#             r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
#             j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare
#             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
#             a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

#             # overlaps
#             gxy = t[:, 2:4]  # grid xy
#             z = torch.zeros_like(gxy)
#             if style == 'rect2':
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
#                 offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g
#             elif style == 'rect4':
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
#                 a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
#                 offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

#         # Define
#         b, c = t[:, :2].long().T  # image, class
#         gxy = t[:, 2:4]  # grid xy
#         gwh = t[:, 4:6]  # grid wh
#         gij = (gxy - offsets).long()
#         gi, gj = gij.T  # grid xy indices

#         # Append
#         indices.append((b, a, gj, gi))  # image, anchor, grid indices
#         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#         anch.append(anchors[a])  # anchors
#         tcls.append(c)  # class

#     return tcls, tbox, indices, anch


# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
#     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
#     box2 = box2.t()

#     # Get the coordinates of bounding boxes
#     if x1y1x2y2:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#     else:  # transform from xywh to xyxy
#         b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

#     # Intersection area
#     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
#     union = (w1 * h1 + 1e-16) + w2 * h2 - inter

#     iou = inter / union  # iou
#     if GIoU or DIoU or CIoU:
#         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
#         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#         if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
#             c_area = cw * ch + 1e-16  # convex area
#             return iou - (c_area - union) / c_area  # GIoU
#         if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             # convex diagonal squared
#             c2 = cw ** 2 + ch ** 2 + 1e-16
#             # centerpoint distance squared
#             rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
#             if DIoU:
#                 return iou - rho2 / c2  # DIoU
#             elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#                 with torch.no_grad():
#                     alpha = v / (1 - iou + v)
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU

#     return iou
