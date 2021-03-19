_base_ = [
    '../dataset/garbage_cropaug.py', '../model/faster_rcnn_r50_fpn.py',
    '../schedule/lr0.02_1x.py', '../../_base_/default_runtime.py'
]

optimizer = dict(lr=0.05)

custom_hooks = [
    dict(
        type='StateEMAHook',
        momentum=0.9995,
        nominal_batch_size=32,
        warm_up=500,
        resume_from=None,
        priority='HIGH')
]
