_base_ = '../yolov4/yolov4l_coco_mosaic.py'

model = dict(
    backbone=dict(scale='v5l5p', out_indices=[2, 3, 4]),
    neck=dict(type='YOLOV5Neck', in_channels=[256, 512, 1024]),
)
