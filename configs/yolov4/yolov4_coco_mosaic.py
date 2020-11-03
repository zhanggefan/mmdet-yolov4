_base_ = ['../_base_/default_runtime.py']

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='MosaicPipeline',
         individual_pipeline=[
             dict(type='LoadImageFromFile'),
             dict(type='LoadAnnotations', with_bbox=True),
             dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
             dict(type='RandomFlip', flip_ratio=0.5)
         ],
         pad_val=114),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu',
         update_pad_shape=True,
         skip_img_without_anno=False,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             min_area=4,
             min_visibility=0.2,
             label_fields=['gt_labels']
         ),
         transforms=[
             dict(
                 type='ShiftScaleRotate',
                 shift_limit=(-0.25, 0.25),
                 scale_limit=0,
                 rotate_limit=0,
                 interpolation=1,
                 border_mode=0,
                 value=(114, 114, 114),
                 always_apply=True),
             dict(
                 type='RandomScale',
                 scale_limit=(0.5, -0.5),
                 interpolation=1,
                 always_apply=True),
             dict(
                 type='CenterCrop',
                 width=640,
                 height=640,
                 always_apply=True),
             dict(
                 type='HueSaturationValue',
                 hue_shift_limit=4,
                 sat_shift_limit=30,
                 val_shift_limit=20,
                 always_apply=True),
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RoundPad', size_divisor=32, pad_val=114 / 255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

test_cfg = dict(
    min_bbox_size=0,
    nms_pre=-1,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=-1)

# optimizer
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0005,
                 nesterov=True,
                 paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
optimizer_config = dict(
    type='AMPGradAccumulateOptimizerHook',
    accumulation=2,
    # grad_clip=dict(max_norm=35, norm_type=2),
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.2,
)
# runtime settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    dict(
        type='YoloV4WarmUpHook',
        warmup_iters=10000,
        lr_weight_warmup=0.,
        lr_bias_warmup=0.1,
        momentum_warmup=0.9,
        priority='NORMAL'
    ),
    dict(
        type='YOLOV4EMAHook',
        momentum=0.9999,
        interval=2,
        warm_up=10000,
        resume_from=None,
        priority='HIGH'
    )
]

total_epochs = 300
# fp16 = dict(loss_scale=512.)

checkpoint_config = dict(interval=10)
