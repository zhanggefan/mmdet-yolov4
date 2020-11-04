import torch
import re
import sys
import copy

sys.path.append('../PyTorch_YOLOv4')
from models import *

mmdet = torch.load('work_dirs/yolov4/yolov4_20201101/epoch_40.pth', map_location='cpu')['state_dict']
yolo = torch.load('../PyTorch_YOLOv4/weights/epoch_300_yolo.pt', map_location='cpu')

mmdet_keys=[*filter(lambda x: not x.startswith('ema'), mmdet.keys())]
for _,mdl in yolo['model'].named_modules():
    mdl._non_persistent_buffers_set=set()
yolo_keys=[n for n in yolo['model'].state_dict()]

rules={
    'model.0.':'backbone.conv0.',
    'model.1.':'backbone.bottleneck1.conv_downscale.',
    'model.2.':'backbone.bottleneck1.conv_bottleneck.',
    'model.3.':'backbone.csp2.conv_downscale.',
    'model.4.':'backbone.csp2.conv_csp.',
    'model.5.':'backbone.csp3.conv_downscale.',
    'model.6.':'backbone.csp3.conv_csp.',
    'model.7.':'backbone.csp4.conv_downscale.',
    'model.8.':'backbone.csp4.conv_csp.',
    'model.9.':'backbone.csp5.conv_downscale.',
    'model.10.':'backbone.csp5.conv_csp.',
    'model.11.':'neck.sppcsp.',
    
    'model.12.':'neck.pre_upsample_convs.1.',
    'model.17.':'neck.pre_upsample_convs.0.',
    
    'model.14.':'neck.lateral_convs.1.',
    'model.19.':'neck.lateral_convs.0.',
    
    'model.16.':'neck.post_upsample_concat_convs.1.',
    'model.21.':'neck.post_upsample_concat_convs.0.',
    
    'model.23.':'neck.downsample_convs.0.',
    'model.27.':'neck.downsample_convs.1.',
    
    'model.25.':'neck.post_downsample_concat_convs.0.',
    'model.29.':'neck.post_downsample_concat_convs.1.',
    
    'model.22.':'neck.out_convs.0.',
    'model.26.':'neck.out_convs.1.',
    'model.30.':'neck.out_convs.2.',
    
    'model.31.m.':'bbox_head.convs_pred.',
    
    '.cv.':'.conv.',
    '.cv1.':'.conv1.',
    '.cv2.':'.conv2.',
    '.cv3.':'.conv3.',
    '.cv4.':'.conv4.',
    '.cv5.':'.conv5.',
    '.cv6.':'.conv6.',
    '.cv7.':'.conv7.',
    '.m.':'.bottlenecks.'
}
yolo2mmdet = {}
mmdet2yolo = {}
for kyolo in yolo_keys:
    _kyolo = kyolo[:]
    for r in rules:
        kyolo = kyolo.replace(r, rules[r])
    if kyolo in mmdet:
        assert mmdet[kyolo].shape == yolo['model'].state_dict()[_kyolo].shape
    else: 
        print(f'{kyolo} not in mmdet')
    mmdet2yolo[kyolo] = _kyolo
    yolo2mmdet[_kyolo] = kyolo
for i in mmdet_keys:
    if i not in mmdet2yolo:
        print(i)
for i in yolo_keys:
    if i not in yolo2mmdet:
        print(i)

sd1 = copy.deepcopy(mmdet)
sd = type(sd1)()
for i in sd1:
    if i in mmdet2yolo:
        sd[mmdet2yolo[i]]=sd1[i].half()
yolo['model'].load_state_dict(sd, strict=False)
torch.save(yolo, '/home/zhanggefan/gitrepo/PyTorch_YOLOv4/weights/epoch_coco128_300.pt')
