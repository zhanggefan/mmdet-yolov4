_base_ = [
    '../dataset/garbage_cropaug.py', '../model/faster_rcnn_r50_fpn.py',
    '../schedule/lr0.02_1x.py', '../../_base_/default_runtime.py'
]

optimizer = dict(lr=0.05)
