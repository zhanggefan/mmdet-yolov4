_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/coco.py']

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.93, weight_decay=0.0005,
                 nesterov=True,
                 paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.2,
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.001)
# runtime settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
total_epochs = 300
evaluation = dict(interval=1, metric=['bbox'])
fp16 = dict(loss_scale=512.)