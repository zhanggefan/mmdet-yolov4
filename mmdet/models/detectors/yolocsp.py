# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Fp16OptimizerHook, Hook, OptimizerHook
from mmcv.runner.dist_utils import get_dist_info
from torch.cuda.amp import GradScaler, autocast

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLOCSP(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_amp=True):
        super(YOLOCSP, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)
        self.use_amp = use_amp

    def forward_train(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOCSP, self).forward_train(*wargs, **kwargs)
        else:
            return super(YOLOCSP, self).forward_train(*wargs, **kwargs)

    def simple_test(self, *wargs, **kwargs):
        if self.use_amp:
            with autocast():
                return super(YOLOCSP, self).simple_test(*wargs, **kwargs)
        else:
            return super(YOLOCSP, self).simple_test(*wargs, **kwargs)


@HOOKS.register_module()
class AMPGradAccumulateOptimizerHook(OptimizerHook):

    def __init__(self, *wargs, **kwargs):
        nominal_batch_size = kwargs.pop('nominal_batch_size', None)
        samples_per_gpu = kwargs.pop('samples_per_gpu', None)
        if nominal_batch_size is not None and samples_per_gpu is not None:
            _, word_size = get_dist_info()
            self.accumulation = math.ceil(nominal_batch_size /
                                          (samples_per_gpu * word_size))
        else:
            self.accumulation = 1
        self.scaler = GradScaler()
        super(AMPGradAccumulateOptimizerHook, self).__init__(*wargs, **kwargs)

    def before_run(self, runner):
        assert hasattr(
            runner.model.module, 'use_amp'
        ) and runner.model.module.use_amp, 'model should support AMP when ' \
                                           'using this optimizer hook! '
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
        nominal_batch_size = kwargs.pop('nominal_batch_size', None)
        samples_per_gpu = kwargs.pop('samples_per_gpu', None)
        if nominal_batch_size is not None and samples_per_gpu is not None:
            _, word_size = get_dist_info()
            self.accumulation = math.ceil(nominal_batch_size /
                                          (samples_per_gpu * word_size))
        else:
            self.accumulation = 1
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
        if len(runner.optimizer.param_groups) != \
                len([*runner.model.parameters()]):
            runner.logger.warning(
                'optimizer config does not support preheat because'
                ' it is not using seperate param-group for each parameter')
            return

        for group_ind, (name,
                        param) in enumerate(runner.model.named_parameters()):
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
                runner.optimizer.param_groups[group_ind][
                    'lr'] = weight_warmup_lr
            for group_ind, momentum_base in self.base_momentum.items():
                warmup_momentum = prog * momentum_base + \
                                  (1 - prog) * self.momentum_warmup
                runner.optimizer.param_groups[group_ind][
                    'momentum'] = warmup_momentum


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
                 interval=None,
                 nominal_batch_size=None,
                 samples_per_gpu=None,
                 warm_up=2000,
                 resume_from=None):
        if interval is not None:
            assert isinstance(interval, int) and interval > 0
            self.interval = interval
        else:
            if nominal_batch_size is not None and samples_per_gpu is not None:
                _, word_size = get_dist_info()
                self.interval = math.ceil(nominal_batch_size /
                                          (samples_per_gpu * word_size))
            else:
                self.interval = 1
        self.warm_up = warm_up * self.interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.checkpoint = resume_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = model.state_dict()
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
        if (runner.iter + 1) % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            momentum = self.momentum * \
                       (1 - math.exp(-runner.iter / self.warm_up))
            buffer_name = self.param_ema_buffer[name]
            if parameter.dtype.is_floating_point:
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(momentum).add_(
                    parameter.data, alpha=1 - momentum)
            else:
                self.model_buffers[buffer_name] = parameter.data

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
