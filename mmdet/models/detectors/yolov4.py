# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.runner import Hook, Fp16OptimizerHook, HOOKS
from mmcv.parallel import is_module_wrapper
from mmcv.runner.fp16_utils import allreduce_grads
import math


@DETECTORS.register_module()
class YOLOV4(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV4, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)


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

            super(Fp16GradAccumulateOptimizerHook, self).after_train_iter(runner)

            runner.model.zero_grad = model_zero_grad
            runner.optimizer.zero_grad = optimizer_zero_grad
        else:
            scaled_loss = runner.outputs['loss'] * self.loss_scale
            scaled_loss.backward()


@HOOKS.register_module()
class LrBiasPreHeatHook(Hook):
    def __init__(self,
                 preheat_iters=1000,
                 preheat_ratio=10.):

        self.preheat_iters = preheat_iters
        self.preheat_ratio = preheat_ratio

        self.bias_base_lr = {}  # initial lr for all param groups

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if len(runner.optimizer.param_groups) != len([*runner.model.parameters()]):
            runner.logger.warning(f"optimizer config does not support preheat because"
                                  " it is not using seperate param-group for each parameter")
            return
        for group_ind, (name, param) in enumerate(runner.model.named_parameters()):
            if '.bias' in name:
                group = runner.optimizer.param_groups[group_ind]
                self.bias_base_lr[group_ind] = group['lr']

    def before_train_iter(self, runner):
        if runner.iter < self.preheat_iters:
            prog = runner.iter / self.preheat_iters
            cur_ratio = (self.preheat_ratio - 1) * (1 - prog) + 1
            for group_ind, lr_init_value in self.bias_base_lr.items():
                runner.optimizer.param_groups[group_ind]['lr'] = cur_ratio * lr_init_value


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
                 interval=1,
                 warm_up=2000,
                 resume_from=None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum ** interval
        self.checkpoint = resume_from

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

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        # momentum = min(self.momentum,
        #                (1 + curr_step) / (self.warm_up + curr_step))

        momentum = self.momentum * (1 - math.exp(-curr_step / self.warm_up))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(momentum).add_(1 - momentum, parameter.data)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
