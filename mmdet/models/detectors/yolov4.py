# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.runner import Hook, Fp16OptimizerHook, HOOKS, OptimizerHook
from mmcv.parallel import is_module_wrapper
import math
from torch.cuda.amp import GradScaler, autocast
from ...datasets import PIPELINES
from ...datasets.pipelines.compose import Compose
import mmcv
import numpy as np
import os.path as osp
import random
import cv2
from mmcv.runner.dist_utils import master_only


@DETECTORS.register_module()
class YOLOV4(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_amp=True):
        super(YOLOV4, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        self.use_amp = use_amp

    def forward_train(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOV4, self).forward_train(*wargs, **kwargs)
        else:
            return super(YOLOV4, self).forward_train(*wargs, **kwargs)

    def simple_test(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOV4, self).simple_test(*wargs, **kwargs)
        else:
            return super(YOLOV4, self).simple_test(*wargs, **kwargs)


@HOOKS.register_module()
class AMPGradAccumulateOptimizerHook(OptimizerHook):
    def __init__(self, *wargs, **kwargs):
        self.accumulation = kwargs.pop('accumulation', 1)
        self.scaler = GradScaler()
        super(AMPGradAccumulateOptimizerHook, self).__init__(*wargs, **kwargs)

    def before_run(self, runner):
        assert hasattr(runner.model.module,
                       'use_amp') and runner.model.module.use_amp, 'model should support AMP when using this optimizer hook!'
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

    def before_train_iter(self, runner):
        if runner.iter % self.accumulation == 0:
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        scaled_loss = self.scaler.scale(runner.outputs['loss'])
        scaled_loss.backward()

        if (runner.iter + 1) % self.accumulation == 0:
            scale = self.scaler.get_scale()
            if self.grad_clip is not None:
                self.scaler.unscale_(runner.optimizer)
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.log_buffer.update({'grad_scale': float(scale)},
                                     runner.outputs['num_samples'])
            self.scaler.step(runner.optimizer)
            self.scaler.update()


@HOOKS.register_module()
class Fp16GradAccumulateOptimizerHook(Fp16OptimizerHook):
    def __init__(self, *wargs, **kwargs):
        self.accumulation = kwargs.pop('accumulation', 1)
        super(Fp16GradAccumulateOptimizerHook, self).__init__(*wargs, **kwargs)

    def before_run(self, runner):
        super(Fp16GradAccumulateOptimizerHook, self).before_run(runner)
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

    def before_train_iter(self, runner):
        if runner.iter % self.accumulation == 0:
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        """
        # clear grads of last iteration
        if (runner.iter + 1) % self.accumulation == 0:
            model_zero_grad = runner.model.zero_grad
            optimizer_zero_grad = runner.optimizer.zero_grad

            def dummyfun(*args):
                pass

            runner.model.zero_grad = dummyfun
            runner.optimizer.zero_grad = dummyfun

            super(Fp16GradAccumulateOptimizerHook,
                  self).after_train_iter(runner)

            runner.model.zero_grad = model_zero_grad
            runner.optimizer.zero_grad = optimizer_zero_grad
        else:
            scaled_loss = runner.outputs['loss'] * self.loss_scale
            scaled_loss.backward()


@HOOKS.register_module()
class YoloV4WarmUpHook(Hook):
    def __init__(self,
                 warmup_iters=1000,
                 lr_weight_warmup=0.,
                 lr_bias_warmup=0.1,
                 momentum_warmup=0.9):

        self.warmup_iters = warmup_iters
        self.lr_weight_warmup = lr_weight_warmup
        self.lr_bias_warmup = lr_bias_warmup
        self.momentum_warmup = momentum_warmup

        self.bias_base_lr = {}  # initial lr for all param groups
        self.weight_base_lr = {}
        self.base_momentum = {}

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if len(runner.optimizer.param_groups) != len([*runner.model.parameters()]):
            runner.logger.warning(f"optimizer config does not support preheat because"
                                  " it is not using seperate param-group for each parameter")
            return

        for group_ind, (name, param) in enumerate(runner.model.named_parameters()):
            group = runner.optimizer.param_groups[group_ind]
            self.base_momentum[group_ind] = group['momentum']
            if name.endswith('.bias'):
                self.bias_base_lr[group_ind] = group['lr']
            elif name.endswith('.weight'):
                self.weight_base_lr[group_ind] = group['lr']

    def before_train_iter(self, runner):
        if runner.iter <= self.warmup_iters:
            prog = runner.iter / self.warmup_iters
            for group_ind, bias_base in self.bias_base_lr.items():
                bias_warmup_lr = prog * bias_base + \
                                 (1 - prog) * self.lr_bias_warmup
                runner.optimizer.param_groups[group_ind]['lr'] = bias_warmup_lr
            for group_ind, weight_base in self.weight_base_lr.items():
                weight_warmup_lr = prog * weight_base + \
                                   (1 - prog) * self.lr_weight_warmup
                runner.optimizer.param_groups[group_ind]['lr'] = weight_warmup_lr
            for group_ind, momentum_base in self.base_momentum.items():
                warmup_momentum = prog * momentum_base + \
                                  (1 - prog) * self.momentum_warmup
                runner.optimizer.param_groups[group_ind]['momentum'] = warmup_momentum


@HOOKS.register_module()
class YOLOV4EMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str): The checkpoint path. Defaults to None.
    """

    def __init__(self,
                 momentum=0.9999,
                 interval=2,
                 warm_up=2000,
                 resume_from=None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.checkpoint = resume_from

    @master_only
    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    @master_only
    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            momentum = self.momentum * \
                       (1 - math.exp(-runner.iter / self.warm_up))
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(momentum).add_(
                    1 - momentum, parameter.data)

    @master_only
    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    @master_only
    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    @master_only
    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)


@PIPELINES.register_module()
class MosaicPipeline(object):
    def __init__(self,
                 individual_pipeline,
                 pad_val=0):
        self.individual_pipeline = Compose(individual_pipeline)
        self.pad_val = pad_val

    def __call__(self, results):
        input_results = results.copy()
        mosaic_results = [results]
        dataset = results['dataset']
        # load another 3 images
        for _ in range(3):
            idx = random.randint(0, len(dataset) - 1)
            img_info = dataset.data_infos[idx]
            ann_info = dataset.get_ann_info(idx)
            _results = dict(img_info=img_info, ann_info=ann_info)
            if dataset.proposals is not None:
                _results['proposals'] = dataset.proposals[idx]
            dataset.pre_pipeline(_results)
            mosaic_results.append(_results)

        for idx in range(4):
            mosaic_results[idx] = self.individual_pipeline(mosaic_results[idx])

        shapes = [results['pad_shape'] for results in mosaic_results]
        cxy = max(shapes[0][0], shapes[1][0], shapes[0][1], shapes[2][1])
        canvas_shape = (cxy * 2, cxy * 2, shapes[0][2])

        # base image with 4 tiles
        canvas = dict()
        for key in mosaic_results[0].get('img_fields', []):
            canvas[key] = np.full(canvas_shape, self.pad_val, dtype=np.uint8)
        for i, results in enumerate(mosaic_results):
            h, w = results['pad_shape'][:2]
            # place img in img4
            if i == 0:  # top left
                x1, y1, x2, y2 = cxy - w, cxy - h, cxy, cxy
            elif i == 1:  # top right
                x1, y1, x2, y2 = cxy, cxy - h, cxy + w, cxy
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = cxy - w, cxy, cxy, cxy + h
            elif i == 3:  # bottom right
                x1, y1, x2, y2 = cxy, cxy, cxy + w, cxy + h

            for key in mosaic_results[0].get('img_fields', []):
                canvas[key][y1:y2, x1:x2] = results[key]

            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, 0::2] = bboxes[:, 0::2] + x1
                bboxes[:, 1::2] = bboxes[:, 1::2] + y1
                results[key] = bboxes

        output_results = input_results
        output_results['filename'] = None
        output_results['ori_filename'] = None
        output_results['img_fields'] = mosaic_results[0].get('img_fields', [])
        output_results['bbox_fields'] = mosaic_results[0].get(
            'bbox_fields', [])
        for key in output_results['img_fields']:
            output_results[key] = canvas[key]

        for key in output_results['bbox_fields']:
            output_results[key] = np.concatenate(
                [r[key] for r in mosaic_results], axis=0)

        output_results['gt_labels'] = np.concatenate(
            [r['gt_labels'] for r in mosaic_results], axis=0)

        output_results['img_shape'] = canvas_shape
        output_results['ori_shape'] = canvas_shape
        output_results['flip'] = False
        output_results['flip_direction'] = None

        return output_results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'individual_pipeline={self.individual_pipeline}, '
                    f'pad_val={self.pad_val})')
        return repr_str


@PIPELINES.register_module()
class HueSaturationValueJitter(object):

    def __init__(self, hue_ratio=0.5, saturation_ratio=0.5, value_ratio=0.5):
        self.h_ratio = hue_ratio
        self.s_ratio = saturation_ratio
        self.v_ratio = value_ratio

    def __call__(self, results):
        for key in results.get('img_fields', []):
            img = results[key]
            # random gains
            r = np.array([random.uniform(-1., 1.) for _ in range(3)]) * \
                [self.h_ratio, self.s_ratio, self.v_ratio] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
                sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR,
                         dst=results[key])  # no return needed
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hue_ratio={self.h_ratio}, '
                    f'saturation_ratio={self.s_ratio}, '
                    f'value_ratio={self.v_ratio})')
        return repr_str


@PIPELINES.register_module()
class GtBBoxesFilter(object):
    def __init__(self, min_size=2, max_aspect_ratio=20):
        assert max_aspect_ratio > 1
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, results):
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        valid = (w > self.min_size) & (
                h > self.min_size) & (ar < self.max_aspect_ratio)
        results['gt_bboxes'] = bboxes[valid]
        results['gt_labels'] = labels[valid]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'min_size={self.min_size}, '
                    f'max_aspect_ratio={self.max_aspect_ratio})')
        return repr_str
