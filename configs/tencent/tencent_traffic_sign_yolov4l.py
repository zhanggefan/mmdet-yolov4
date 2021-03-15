_base_ = '../yolov4/yolov4l_coco_mosaic.py'

model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='DarknetCSP',
        scale=[['conv', 'bottleneck', 'csp', 'csp', 'csp', 'sppv4'],
               [None, 1, 2, 8, 8, 4], [16, 32, 64, 128, 256, 256]],
        out_indices=[2, 3, 4, 5]),
    neck=dict(
        type='YOLOV4Neck',
        in_channels=[64, 128, 256, 256],
        out_channels=[64, 128, 256, 512],
        csp_repetition=2),
    bbox_head=dict(
        type='YOLOCSPHead',
        anchor_generator=dict(
            type='YOLOV4AnchorGenerator',
            base_sizes=[
                [(8, 8)],  # P2/4
                [(16, 16)],  # P3/8
                [(32, 32)],  # P4/16
                [(64, 64)]
            ],  # P5/32
            strides=[4, 8, 16, 32]),
        featmap_strides=[4, 8, 16, 32],
        num_classes=1,
        in_channels=[64, 128, 256, 512],
        class_agnostic=True,
        loss_conf=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        num_obj_per_image=3, conf_level_balance_weight=[4.0, 4.0, 1.0, 0.4]),
    test_cfg=dict(
        min_bbox_size=0,
        nms_pre=-1,
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=300))

dataset_type = 'TrafficSignDataset'
data_root = 'data/tencent/det/'
img_norm_cfg = dict(mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(
        type='MosaicPipeline',
        individual_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True)
        ],
        pad_val=114),
    dict(
        type='Albu',
        update_pad_shape=True,
        skip_img_without_anno=False,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            min_area=4,
            min_visibility=0.1,
            label_fields=['gt_labels'],
            check_each_transform=False),
        transforms=[
            dict(
                type='PadIfNeeded',
                min_height=1920,
                min_width=1920,
                border_mode=0,
                value=(114, 114, 114),
                always_apply=True),
            dict(
                type='RandomCrop', width=1280, height=1280, always_apply=True),
            dict(
                type='RandomScale',
                scale_limit=(-0.5, 0),
                interpolation=1,
                always_apply=True),
            dict(type='CenterCrop', width=640, height=640, always_apply=True),
            dict(type='HorizontalFlip', p=0.5)
        ]),
    dict(
        type='HueSaturationValueJitter',
        hue_ratio=0.015,
        saturation_ratio=0.7,
        value_ratio=0.4),
    dict(type='GtBBoxesFilter', min_size=2, max_aspect_ratio=20),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1440, 816),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainsplit/label/',
        img_prefix=data_root + 'trainsplit/img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/label/',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/label/',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline))

checkpoint_config = dict(interval=2, max_keep_ckpts=40)

evaluation = dict(interval=2, metric='mAP')

total_epochs = 100
