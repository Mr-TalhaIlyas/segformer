#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial

from bricks import DWConv, StochasticDepth

'''
Mix Transformer Encoder
'''
class Mlp(nn.Module):
    '''
    Implements Eq 3 in SegFormer Paper
    '''
    def __init__(self, inFeats, hidFeats=None, outFeats=None, act_layer = nn.GELU, drop=0.0):
        super(Mlp, self).__init__()
        outFeats = outFeats or inFeats
        hidFeats = hidFeats or inFeats

        self.fc1 = nn.Linear(inFeats, hidFeats)
        self.dwconv = DWConv(hidFeats)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidFeats, outFeats)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
 
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim // self.num_heads
        self.scale = qkv_scale or head_dim ** -0.5 # as in Eq 1 -> (1/(d_head)**1/2)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # BxNxC
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3) # BxNxhx(C/h) -> BxhxNx(C/h)

        if self.sr_ratio > 1: # reduction ratio in EQ 2.
            x_ = x.permute(0,2,1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1) # BxCxHxW -> BxCxN -> BxNxC
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
            # BxNxC -> Bx-1x2xhx(C/h) -> 2xBxhx-1x(C/h) i.e. seprating Keys and values BxhxNxC/h
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2,-1)) * self.scale # EQ 1 in paper
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self,  dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0., sr_ratio=1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.drop_path = StochasticDepth(p=drop_path, mode='batch') if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        ml_hid_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, ml_hid_dim, act_layer=act_layer, drop=drop)

    
    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, inChannels=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1] 
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(inChannels, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.ModuleDict):

    def __init__(self, img_size=224, patch_size=16, inChannels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1,2,4,8], mlp_ratio=[4,4,4,4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, depths=[3,4,6,3], sr_ratio=[8,4,2,1], norm_layer=nn.LayerNorm):
        super(MixVisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, inChannels=inChannels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, inChannels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, inChannels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, inChannels=embed_dims[2], embed_dim=embed_dims[3])

        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[0], norm_layer=norm_layer
        ) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[1], norm_layer=norm_layer
        ) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[2], norm_layer=norm_layer
        ) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[3], norm_layer=norm_layer
        ) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

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

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1 
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs
    
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

#%%

from torchsummary import summary
model = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratio=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratio=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

summary(model, (3,224,224), depth=2)
#%%
ckpt = '/home/user01/data/talha/segformer/chkpts/mit_b5.pth'

def load_pretrained_chkpt(model, pretrained_path=None):
    if pretrained_path is not None:
        chkpt = torch.load(pretrained_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # load pretrained
            pretrained_dict = chkpt#['model_state_dict']
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

load_pretrained_chkpt(model, pretrained_path=ckpt)
#%%
ckpt = '/home/user01/data/talha/segformer/chkpts/mit_b5.pth'


chkpt = torch.load(ckpt,
                    map_location='cuda' if torch.cuda.is_available() else 'cpu')

# load pretrained
pretrained_dict = chkpt#['model_state_dict']
# %%
