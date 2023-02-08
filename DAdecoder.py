#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_bn.nn.modules import SynchronizedBatchNorm2d
from bricks import resize, ConvModule, DepthWiseConv

import numpy as np
from functools import partial

norm_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)

class MLP(nn.Module):
    def __init__(self, inputDim=2048, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(inputDim, embed_dim)
    
    def forward(self, x):
        x = x.flatten(2).transpose(1,2)# B*C*H*W -> B*C*HW -> B*HW*C
        x = self.proj(x)
        return x
    
class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.
    """

    def __init__(self, dilations, inChannels, embed_dim):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.inChannels = inChannels
        self.embed_dim = embed_dim
        for dilation in dilations:
            self.append(
                DepthWiseConv(
                    self.inChannels,
                    self.embed_dim,
                    kernel_size=1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    bias=False))
         
    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs

class DAFormerHead(nn.Module):

    def __init__(self, inChannels=[64, 128, 320, 512], dilation_rates=[1, 6, 12, 18],
                dropout_ratio=0.1, act_layer=nn.ReLU, num_classes=19, embed_dim=256,
                align_corners=False):

        super().__init__()
        # 1st step embed each Fi to same number of channels Ce

        self.linear_c = nn.ModuleList([
                                MLP(inputDim=inChannel, embed_dim=embed_dim)
                                for inChannel in inChannels
                            ])
        # 2nd step is resizing 
        ## so in forward
        # 3rd step is applying ASPP module
        self.aspp = ASPPModule(dilation_rates, embed_dim*4, embed_dim)
        # 4ht step is concating and fusing
        self.fuse = ConvModule(embed_dim*len(dilation_rates), embed_dim, kernel_size=3, bias=False)
        # final prediction
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(embed_dim, num_classes, kernel_size=1))
        
    
    def forward(self, inputs):
        
        c1, c2, c3, c4 = inputs
        N = c4.shape[0]
        # 1st
        features = []
        for i, c in enumerate([c1, c2, c3, c4]):
            feat = self.linear_c[i](c).permute(0,2,1).reshape(N, -1, c.shape[2], c.shape[3])
            features.append(feat)
        # 2nd
        features = [resize(feature, size=features[0].shape[2:], mode='bilinear') for feature in features]
        features = torch.cat(features, dim=1)
        # 3rd
        aspp_outs = self.aspp(features)
        aspp_outs = torch.cat(aspp_outs, dim=1)
        # 4th
        fused = self.fuse(aspp_outs)
        out = self.cls_conv(fused)

        return out

da = DAFormerHead()
#%%