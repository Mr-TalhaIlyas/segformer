#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:55:27 2023

@author: user01
"""

from backbone import MixVisionTransformer
from DAdecoder import DAFormerHead
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class DAFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratio=[4, 4, 4, 4], feature_strides=[4, 8, 16, 32], depths=[3, 6, 40, 3], sr_ratio=[8, 4, 2, 1],
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 drop_rate=0.0, drop_path_rate=0.1, num_classes=19, embed_dim=768, align_corners=False):
        super(DAFormer, self).__init__()

        self.backbone = MixVisionTransformer(img_size=img_size, patch_size=patch_size, embed_dims=embed_dims,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depths=depths, sr_ratio=sr_ratio,
                                            drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        
        self.decoder_head = DAFormerHead()
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        enc_features = self.backbone(x)
        out = self.decoder_head(enc_features)

        return out
    
from torchsummary import summary

model = DAFormer()
model = model.to('cuda')
summary(model, (3,224,224), depth=9)

# x = torch.randn((1,3,224,224))
# y = model.forward(x)

# for i in range(len(y)):
#     print(y[i].shape)

