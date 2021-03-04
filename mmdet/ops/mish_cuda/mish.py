import torch  # Must import torch before C extension
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS

from .mish_cuda_ext import mish_backward, mish_forward


class MishCudaFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        if not inp.is_contiguous():
            inp = inp.contiguous()
        ctx.save_for_backward(inp)
        return mish_forward(inp)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        if not ctx.needs_input_grad[0]:
            return (None, )
        return mish_backward(grad_out, inp)


class Mish(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Mish, self).__init__()

    def forward(self, inp):
        return MishCudaFunction.apply(inp)


ACTIVATION_LAYERS.register_module(module=Mish)
