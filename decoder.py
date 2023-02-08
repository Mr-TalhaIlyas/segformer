import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_bn.nn.modules import SynchronizedBatchNorm2d
from bricks import resize

import numpy as np
from functools import partial

norm_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)

class MLP(nn.Module):
    def __init__(self, inputDim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(inputDim, embed_dim)
    
    def forward(self, x):
        x = x.flatten(2).transpose(1,2)# B*C*H*W -> B*C*HW -> B*HW*C
        x = self.proj(x)
        return x
    
    
class SegFormerHead(nn.Module):

    def __init__(self, inChannels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                dropout_ratio=0.1, act_layer=nn.ReLU, num_classes=20, embed_dim=768, align_corners=False):

        super().__init__()
    
        assert len(feature_strides) == len(inChannels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.inChannels = inChannels
        self.num_classes = num_classes
        embed_dim = embed_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.inChannels

        # 1st step unify the channel dimensions
        self.linear_c4 = MLP(inputDim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(inputDim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(inputDim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(inputDim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = nn.Conv2d(embed_dim*4, embed_dim, kernel_size=3) # 3 is in DAFormer confirmed88
        self.norm = norm_layer(embed_dim)
        self.act = act_layer()
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

    
    def forward(self, inputs):

        c1, c2, c3, c4 = inputs
        N = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(N, -1, c4.shape[2], c4.shape[3]) # 1st step unify the channel dimensions
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False) # 2nd step upsampling the dimensions
        
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(N, -1, c3.shape[2], c3.shape[3]) # reshaped ot B*C*H*W
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(N, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(N, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # 3rd setp adapting concatenated features
        _c = self.norm(_c)
        _c = self.act(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x) # 4th step predict classes

        return x

#%%

# from torchsummary import summary
# 
# model = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratio=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratio=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
# x = torch.randn((1,3,224,224))
# y = model.forward(x)

# dec = SegFormerHead()
# out = dec.forward(y)