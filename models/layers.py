import torch
import torch.nn as nn
import torch.nn.functional as F


class MishInline(nn.Module):
    """ https://arxiv.org/abs/1908.08681v1 """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh( F.softplus(x) )


@torch.jit.script
def mish(x):
    ''' https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py '''
    return x * torch.tanh( F.softplus(x) )

class Mish(nn.Module):
    '''https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, 
                stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0,
                            dilation=1, groups=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

