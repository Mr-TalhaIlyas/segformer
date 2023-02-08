import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_bn.nn.modules import SynchronizedBatchNorm2d
from functools import partial

norm_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # N -> HxW
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool =  True):
    
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input
    
    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise

class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise. 
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''
    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode
    
    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)
    
    def __repr__(self):
       s = f"{self.__class__.__name__}(p={self.p})"
       return s

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ConvModule(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.norm = norm_layer(outChannels)
        self.act = act_layer()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthWiseConv, self).__init__()
        self.kernel_size = kernel_size

        if self.kernel_size != 1:
            self.depthwise = ConvModule(inChannels, inChannels, kernel_size=kernel_size,
                                        stride=stride,  padding=padding, dilation=dilation, groups=inChannels)
        self.pointwise = ConvModule(inChannels, outChannels, kernel_size=1)

    def forward(self, x):
        if self.kernel_size != 1:
            x = self.depthwise(x)
        out = self.pointwise(x)
        return out