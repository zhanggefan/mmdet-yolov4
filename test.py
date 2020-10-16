from mmdet.models.necks import PACSPFPN
from mmdet.models import DarknetCSP
from mmdet.models.dense_heads import YOLOV4Head

from mmdet.models.detectors import SingleStageDetector
from mmcv import Config

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model, init_dist, load_checkpoint
import mmcv
from mmcv.runner.hooks import HOOKS

from mmdet.apis import train_detector, single_gpu_test
import tqdm
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

cfg = Config.fromfile('configs/yolov4/yolov4_coco.py')
# cfg.data.workers_per_gpu = 0
cfg.gpu_ids = range(1)
cfg.seed = 0
cfg.work_dir = 'work_dirs/yolov4/yolov4_20201016'


class s5p(SingleStageDetector):
    def __init__(self):
        super(SingleStageDetector, self).__init__()
        self.backbone = DarknetCSP('s5p', out_indices=[3, 4, 5])
        self.neck = PACSPFPN([128, 256, 512], [128, 256, 512], csp_repetition=1)
        self.bbox_head = YOLOV4Head(num_classes=80, in_channels=[128, 256, 512], test_cfg=cfg.test_cfg)


class m5p(SingleStageDetector):
    def __init__(self):
        super(SingleStageDetector, self).__init__()
        self.backbone = DarknetCSP('m5p', out_indices=[3, 4, 5])
        self.neck = PACSPFPN([192, 384, 768], [192, 384, 768], csp_repetition=1)
        self.bbox_head = YOLOV4Head(num_classes=80, in_channels=[192, 384, 768], test_cfg=cfg.test_cfg)


class l5p(SingleStageDetector):
    def __init__(self):
        super(SingleStageDetector, self).__init__()
        self.backbone = DarknetCSP('l5p', out_indices=[3, 4, 5])
        self.neck = PACSPFPN([256, 512, 1024], [256, 512, 1024], csp_repetition=2)
        self.bbox_head = YOLOV4Head(num_classes=80, in_channels=[256, 512, 1024], test_cfg=cfg.test_cfg)


class x5p(SingleStageDetector):
    def __init__(self):
        super(SingleStageDetector, self).__init__()
        self.backbone = DarknetCSP('x5p', out_indices=[3, 4, 5])
        self.neck = PACSPFPN([320, 640, 1280], [320, 640, 1280], csp_repetition=3)
        self.bbox_head = YOLOV4Head(num_classes=80, in_channels=[320, 640, 1280], test_cfg=cfg.test_cfg)


model = s5p()
model.init_weights()
# torch.save(model.state_dict(), "/home/cowa006/jupyternotebooks/x5p.pt")

# load_checkpoint(model, 'work_dirs/yolov4/yolov4_coco/epoch_11.pth', map_location='cpu')

# model.load_state_dict(torch.load("pretrained/yolo_s5p.pt"))
# model.eval()
# model.forward_dummy(torch.randn(1,3,640,640))
# exit(0)

# model.eval()
#
# model.forward = model.forward_dummy
# torch.onnx.export(model, torch.randn(1, 3, 640, 640), 'temp.onnx', opset_version=9)
#
# import pprint
#
# pprint.pprint(model)
#
# from torchstat import stat
#
# model.forward = model.forward_dummy
# stat(model.backbone, (3, 640, 640))

# dataset = build_dataset(cfg.data.val, dict(test_mode=True))
dataset = build_dataset(cfg.data.train)
#
# # dataloader = build_dataloader(
# #     dataset,
# #     samples_per_gpu=1,
# #     workers_per_gpu=3,
# #     dist=False,
# #     shuffle=False)
#
model.CLASSES = dataset.CLASSES
# model = MMDataParallel(model.cuda())

train_detector(model, dataset, cfg, validate=True)
#
# data_loaders = build_dataloader(
#     dataset,
#     cfg.data.samples_per_gpu,
#     cfg.data.workers_per_gpu,
#     len(cfg.gpu_ids),
#     dist=False,
#     seed=cfg.seed)
#
# from mmcv.runner import build_optimizer
#
# optimizer = build_optimizer(model, cfg.optimizer)
# model.train()
#
# cfg.lr_config.type = cfg.lr_config.policy + "LrUpdaterHook"
# cfg.lr_config.pop("policy")
# lt_hook = mmcv.build_from_cfg(cfg.lr_config, HOOKS)

from collections import OrderedDict
#
# d = torch.load('io.pt')['in']
#
# ret = model.train_step(d, optimizer)
#
# pass
#
# # for d in tqdm.tqdm(data_loaders):
#     io = OrderedDict()
#     io['in'] = d
#     torch.save(io, 'io.pt')
#     ret = model.train_step(d, optimizer)
#     io = OrderedDict()
#     io['out'] = ret
#     torch.save(io, 'ret.pt')
#     optimizer.zero_grad()
#     ret['loss'].backward()
#     break

# exit(0)


# from collections import OrderedDict

# do = OrderedDict()
# for i, d in enumerate(dataloader):
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **d)
#     # do[f'{i}'] = d['img'][0]
#     if i == 0:
#         # torch.save(do, '../PyTorch_YOLOv4/input.pt')
#         break
# result = single_gpu_test(model, dataloader, show=True)
# dataset.evaluate(result)
#
# from mmdet.core import YOLOV4AnchorGenerator
#
# ya = YOLOV4AnchorGenerator(base_sizes=[[(13, 17), (22, 25), (27, 66), (55, 41)],
#                                        [(57, 88), (112, 69), (69, 177), (136, 138)],
#                                        [(136, 138), (287, 114), (134, 275), (268, 248)],
#                                        [(268, 248), (232, 504), (445, 416), (640, 640)],
#                                        [(812, 393), (477, 808), (1070, 908), (1408, 1408)]],
#                            strides=[8, 16, 32, 64, 128])
#
# import torch
# import matplotlib.pyplot as plt
#
#
# def drawrbbox(fig, bboxes, color='r', lw=0.1, fill=True):
#     ax = fig.add_subplot(111)
#     bboxes = bboxes.cpu().clone()
#     bboxes = torch.cat([0.5 * (bboxes[:, :2] + bboxes[:, 2:]), (bboxes[:, 2:] - bboxes[:, :2])], -1)
#     for bbox in bboxes:
#         xc, yc, w, h = bbox.tolist()
#         rect = plt.Rectangle((xc - w / 2, yc - h / 2),
#                              w,
#                              h,
#                              0,
#                              fill=True,
#                              alpha=0.2 if fill else 1,
#                              linewidth=lw,
#                              facecolor='none' if not fill else color,
#                              edgecolor='none' if fill else color)
#         ax.add_patch(rect)
#         ax.plot([xc], [yc], '+', c=color)
#         if not fill:
#             ax.plot([xc, xc + w / 2], [yc, yc + h / 2], c=color, lw=lw)
#             ax.plot([xc, xc - w / 2], [yc, yc + h / 2], c=color, lw=lw)
#             ax.plot([xc, xc - w / 2], [yc, yc - h / 2], c=color, lw=lw)
#             ax.plot([xc, xc + w / 2], [yc, yc - h / 2], c=color, lw=lw)
#     ax.axis('equal')
#     ax.set_xlim(0, 640)
#     ax.set_ylim(0, 640)
#     if not ax.yaxis_inverted():
#         ax.invert_yaxis()
#
#
# gt_bboxes = torch.rand((3, 2, 2)) * 600
# gt_bboxes = torch.cat([gt_bboxes.min(1).values, gt_bboxes.max(1).values], dim=-1)
#
# feat_shape = [(1536 // 2 ** x, 1536 // 2 ** x) for x in [3, 4, 5, 6, 7]]
#
# mlvl_indices = ya.responsible_indices(feat_shape, gt_bboxes, 4)
# mlvl_anchors = ya.grid_anchors(feat_shape)
#
# fig = plt.figure(dpi=200)
# drawrbbox(fig, gt_bboxes, lw=1, fill=True)
#
# colors = plt.get_cmap("jet")
#
# for i in range(ya.num_levels):
#     c = colors(i / ya.num_levels)
#     anchors = mlvl_anchors[i]
#     anchor_indices, gt_indices = mlvl_indices[i]
#     anchors = anchors[anchor_indices]
#     gts = gt_bboxes[gt_indices]
#     a_xy = 0.5 * (anchors[:, :2] + anchors[:, 2:]).cpu()
#     gt_xy = 0.5 * (gts[:, :2] + gts[:, 2:])
#     match_line = torch.stack([a_xy, gt_xy], dim=-1)
#     drawrbbox(fig, anchors, c, fill=False)
#     plt.plot(match_line[:, 0].T, match_line[:, 1].T, c=c)
# plt.show()
#
# print(1)
