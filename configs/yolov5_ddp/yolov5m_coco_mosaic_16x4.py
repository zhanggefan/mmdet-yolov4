_base_ = '../yolov5/yolov5m_coco_mosaic.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
    neck=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
    bbox_head=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
)

optimizer = dict(lr=0.64)

data = dict(samples_per_gpu=16, workers_per_gpu=2)
