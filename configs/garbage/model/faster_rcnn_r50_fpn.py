_base_ = '../../_base_/models/faster_rcnn_r50_fpn.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=True),
    roi_head=dict(bbox_head=dict(num_classes=3)))
