from backbone import MixVisionTransformer
from decoder import SegFormerHead
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SegFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratio=[4, 4, 4, 4], feature_strides=[4, 8, 16, 32], depths=[3, 6, 40, 3], sr_ratio=[8, 4, 2, 1],
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 drop_rate=0.0, drop_path_rate=0.1, num_classes=19, embed_dim=768, align_corners=False):
        super(SegFormer, self).__init__()

        self.backbone = MixVisionTransformer(img_size=img_size, patch_size=patch_size, embed_dims=embed_dims,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depths=depths, sr_ratio=sr_ratio,
                                            drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        
        self.decoder_head = SegFormerHead(inChannels=embed_dims, feature_strides=feature_strides,
                                     dropout_ratio=0.1, act_layer=nn.ReLU, num_classes=num_classes,
                                     embed_dim=embed_dim, align_corners=align_corners)
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
    
# from torchsummary import summary

# model = SegFormer()

# summary(model, (3,224,224), depth=7)

# x = torch.randn((1,3,224,224))
# y = model.forward(x)

# for i in range(len(y)):
#     print(y[i].shape)

#%%
ckpt = '/home/user01/data/talha/segformer/chkpts/segformer.b5.1024x1024.city.160k.pth'

def load_pretrained_chkpt(model, pretrained_path=None):
    if pretrained_path is not None:
        chkpt = torch.load(pretrained_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # load pretrained
            pretrained_dict = chkpt['state_dict']
            # load model state dict
            state = model.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('module.') == nk.strip('module.') and pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            model.load_state_dict(state)
            print('Pre-trained state loaded successfully...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            print(f'ERROR in pretrained_dict @ {pretrained_path}')
    else:
        print('Enter pretrained_dict path.')
    return matched, unmatched

matched, unmatched = load_pretrained_chkpt(model, pretrained_path=ckpt)
print(unmatched)

'''
Only the decoder keys that don't have the exact same name are creatgin mismatch error but 
everything elese is according to the official implementation.'
'''

# chkpt = torch.load(ckpt,
#                     map_location='cuda' if torch.cuda.is_available() else 'cpu')
# pretrained_dict = chkpt['state_dict']
# state = model.state_dict()
# unmatched_keys = set(state.keys()) ^ set(pretrained_dict.keys())

# print(unmatched_keys)