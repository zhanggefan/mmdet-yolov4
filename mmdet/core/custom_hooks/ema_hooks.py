import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import get_dist_info


@HOOKS.register_module()
class StateEMAHook(Hook):
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
                 warm_up=2000,
                 resume_from=None):
        self.interval = 1
        self.nominal_batch_size = None
        self.warm_up = warm_up
        if interval is not None:
            assert isinstance(interval, int) and interval > 0
            self.interval = interval
        elif nominal_batch_size is not None:
            self.interval = None
            self.nominal_batch_size = nominal_batch_size

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
            momentum = self.momentum * (
                1 - math.exp(-runner.iter / (self.warm_up * self.interval)))
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
        if self.interval is None:
            assert self.nominal_batch_size is not None
            samples_per_gpu = runner.data_loader.sampler.samples_per_gpu
            _, word_size = get_dist_info()
            self.interval = math.ceil(self.nominal_batch_size /
                                      (samples_per_gpu * word_size))
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
