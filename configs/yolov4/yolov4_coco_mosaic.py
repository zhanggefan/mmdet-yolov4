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
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
    workers_per_gpu=4,
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
    nms_pre=1000,
    score_thr=0.001,
    conf_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.937, weight_decay=0.0005,
                 nesterov=True,
                 paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
optimizer_config = dict(
    type='AMPGradAccumulateOptimizerHook',
    accumulation=2,
    # grad_clip=dict(max_norm=35, norm_type=2),
)

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.2,
    # warmup='linear',
    # warmup_iters=2000,  # same as burn-in in darknet
    # warmup_ratio=0.001,
)
# runtime settings
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    # dict(
    #     type='LrBiasPreHeatHook',
    #     preheat_iters=2000,
    #     preheat_ratio=10.,
    #     priority='NORMAL'
    # ),
    dict(
        type='YOLOV4EMAHook',
        momentum=0.9999,
        interval=2,
        warm_up=4000,
        resume_from=None,
        priority='HIGH'
    )
]

total_epochs = 300
evaluation = dict(interval=1, metric=['bbox'])
# fp16 = dict(loss_scale=512.)
