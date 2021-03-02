import torch  # Must import torch before C extension
from torch.autograd import gradcheck
from torch.nn import functional as F

from mmdet.ops.mish_cuda.mish import Mish


def test_act_mish():
    x = torch.randn(1000).double()
    x_cu = x.cuda()

    act = Mish()

    print(torch.allclose(act(x), act(x_cu).cpu()))
    print(torch.allclose((x_cu * (torch.tanh(F.softplus(x_cu)))), act(x_cu)))

    x = (torch.randn(100) * 100 - 50).double().cuda()
    x.requires_grad_()
    print(gradcheck(act, (x, )))
