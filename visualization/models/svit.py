# --------------------------------------------------------
# SuperToken ViT Transformer
# Copyright (c) 2023 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
 
class Inception(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.inceptions = nn.ModuleList([nn.Identity(),
                                         nn.Conv2d(dim//4, dim//4, 3, 1, 1, groups=dim//4),
                                         nn.Conv2d(dim//4, dim//4, 5, 1, 2, groups=dim//4),
                                         nn.Conv2d(dim//4, dim//4, 7, 1, 3, groups=dim//4),
                                        ])
    def forward(self, x):
        xl = x.chunk(4, dim=1)
        yl = []
        for x, f in zip(xl, self.inceptions):
            y = f(x)
            yl.append(y)
        return torch.cat(yl, dim=1)
 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
        
 
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.rpe = rpe
        
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
        
        # self.shuffle = nn.Conv2d(num_buckets*3, num_buckets*3, 1, 1, 0, groups=3) if num_buckets > 1 else nn.Identity()
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn.transpose(-1, -2)

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, seg_dim, kernel_size, layers=2, hard_label=True, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False):
        super().__init__()
        
        self.n_iter = layers
        self.kernel_size = kernel_size
        self.hard_label = hard_label
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, rpe=rpe)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.kernel_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        # 976
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        # 955
        
        # 935
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        # 911
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                # 874
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                # 871
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                    # 777
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    # 853
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    # 777            
                    
                    # 771
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                    # 767
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        # 853
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        # 767
        
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.reshape(B, C, hh*ww).transpose(-1, -2)
                stoken_features, attn = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, hh, ww)
            else:
                stoken_features = self.stoken_refine(stoken_features)
            
        # 727
        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        # 714
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        # 687
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features, attn, affinity_matrix.reshape(B, hh, ww, h, w, 9).permute(0, 5, 1, 3, 2, 4).reshape(B, 9, H, W) 
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features, attn = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features, attn, None
        
    def forward(self, x):
        if self.kernel_size[0] > 1 or self.kernel_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        
        # self.conv_constant = nn.Parameter(torch.zeros(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.data[:,:,kernel_size//2,kernel_size//2] = 1.
        # self.conv_constant.requires_grad = False
        
    def forward(self, x):
        # if self.training:
        return self.conv(x) + x
        # else:
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim)

class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, seg_dim, seg_layers, stoken_size,  stoken_refine=True, stoken_refine_attention=True, hard_label=False,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5, rpe=False):
        super().__init__()
                        
        self.layerscale = layerscale
        
        # self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        # self.norm1 = nn.BatchNorm2d(dim)
        self.attn = StokenAttention(dim, seg_dim=seg_dim, 
                                    kernel_size=stoken_size, layers=seg_layers, 
                                    refine=stoken_refine, refine_attention=stoken_refine_attention, 
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    hard_label=hard_label,
                                    attn_drop=attn_drop, proj_drop=drop, rpe=rpe)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # self.norm2 = LayerNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        # x = x + self.pos_embed(x)
        x = self.pos_embed(x)
        xa, attn, affinity_matrix = self.attn(self.norm1(x))
        if self.layerscale:            
            x = x + self.drop_path(self.gamma_1 * xa)
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(xa)
            x = x + self.drop_path(self.mlp2(self.norm2(x))) 
       
        return x, attn, affinity_matrix

class BasicLayer(nn.Module):        
    def __init__(self, num_layers, dim, seg_dim, seg_layers, stoken_size,  stoken_refine=True, stoken_refine_attention=True, hard_label=False,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5, rpe=False,
                 downsample=False,
                 use_checkpoint=False, checkpoint_num=None):
        super().__init__()        
                
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
                
        self.blocks = nn.ModuleList([StokenAttentionLayer(
                                           dim=dim[0], seg_dim=seg_dim, seg_layers=seg_layers, stoken_size=stoken_size, 
                                           stoken_refine=stoken_refine, stoken_refine_attention=stoken_refine_attention,
                                           hard_label=hard_label,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           act_layer=act_layer, 
                                           layerscale=layerscale, init_values=init_values,
                                           rpe=rpe) for i in range(num_layers)])
                                           
                                                                           
                
        if downsample:            
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None
         
    def forward(self, x):
        ats = []
        afs = []
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x, at, af = checkpoint.checkpoint(blk, x)
            else:
                x, at, af = blk(x)
            ats.append(at)
            afs.append(af)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, ats, afs
       
class PatchEmbed(nn.Module):        
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
                        
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),            
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class SViT(nn.Module):   
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 seg_dim=[20, 20, 20, 20], seg_layers=[3, 2, 1, 0], stoken_size=[8, 4, 2, 1],
                 stoken_refine=True, stoken_refine_attention=True, hard_label=False, rpe=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 use_checkpoint=False, checkpoint_num=[0,0,0,0], 
                 layerscale=[False, False, False, False], init_values=1e-6, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim        
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        self.freeze_bn = freeze_bn

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
                
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],
                               seg_dim=seg_dim[i_layer],
                               seg_layers=seg_layers[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),
                               stoken_refine=stoken_refine,
                               stoken_refine_attention=stoken_refine_attention,hard_label=hard_label,                               
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,
                               use_checkpoint=use_checkpoint,
                               checkpoint_num=checkpoint_num[i_layer],                               
                               layerscale=layerscale[i_layer],
                               init_values=init_values,
                               rpe=rpe)
            self.layers.append(layer)

        # self.norm = LayerNorm2d(self.num_features)        
        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.projection = nn.Linear(self.num_features, projection) if projection else None
        self.head = nn.Linear(projection or self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        self.freeze_batchnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def freeze_batchnorm(self, ):        
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()                    
                    for param in m.parameters():
                        param.requires_grad = False
                        # print('Set requires_grad False')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)        
        x = self.pos_drop(x)
        
        ats = []
        afs = []
        for layer in self.layers:            
            x, at, af = layer(x)
            ats += at
            afs += af
        
        x = self.proj(x)
        x = self.norm(x)
        x = self.swish(x)
        
        x = self.avgpool(x).flatten(1)  # B C 1        
        return x, ats, afs

    def forward(self, x):
        x, ats, afs = self.forward_features(x)       
        x = self.head(x)
        return x, ats, afs

@register_model
def svit_tiny():
    model = SViT(embed_dim=[32, 64, 160, 256],
                    depths=[2, 2, 6, 2],
                    num_heads=[1, 2, 5, 8],
                    seg_dim=[16, 16, 16, 16],
                    seg_layers=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    mlp_ratio=4,
                    stoken_refine=True,
                    stoken_refine_attention=True, 
                    rpe=False,                    
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    drop_path_rate=0, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=False,
                    init_values=1e-6,)
    model.default_cfg = _cfg()
    return model    


@register_model
def svit_small():
    model = SViT(embed_dim=[64, 128, 320, 512],
                    depths=[3, 5, 9, 3],
                    num_heads=[1, 2, 5, 8],
                    seg_layers=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    projection=1024,
                    stoken_refine=True,
                    stoken_refine_attention=True, 
                    freeze_bn=False,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    drop_path_rate=0, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, False, False],
                    init_values=1e-5,)
    model.default_cfg = _cfg()
    return model    

@torch.no_grad()
def throughput(model):
    # model =  SViT(embed_dim=[64, 128, 320, 512],
                    # depths=[3, 5, 9, 3],
                    # num_heads=[1, 2, 5, 8],
                    # reduction=[8, 8, 8, 8],
                    # seg_layers=[3, 2, 1, 0], 
                    # stoken_size=[8, 4, 2, 1],
                    # stoken_refine=True, 
                    # stoken_refine_attention=True, 
                    # rpe=False,
                    # mlp_ratio=4,
                    # qkv_bias=True,
                    # qk_scale=None,
                    # use_checkpoint=False,
                    # checkpoint_num = [0, 0, 0, 0],
                    # layerscale=False,
                    # init_values=1e-6,)

    model.eval()
    
    model.cuda()
    
    images = torch.randn(64, 3, 224, 224).cuda()
    
    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    return  
    
def test():
    model =  SViT(                    
                    embed_dim=[64, 128, 320, 512],
                    depths=[3, 5, 9, 3],
                    num_heads=[1, 2, 5, 8],
                   
                    
                    seg_layers=[1, 1, 1, 1], 
                    stoken_size=[8, 4, 1, 1],
                    
                    projection=1024,
                    
                    mlp_ratio=4,
                    stoken_refine=True,
                    stoken_refine_attention=True,
                    hard_label=False,
                    rpe=False,                    
                    qkv_bias=True,
                    qk_scale=None,
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False]*4,
                    init_values=1e-6,)
    
    
    # model = svit_small(None)
                       
    print(model)
    
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    
    # print('Number of flop: %.4fG' % (model.flops() / 1e9))
    
    # model(torch.rand(1, 3, 224, 224))
    # return
    
    model.eval()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
    print(flop_count_table(flops))
    print("Number of parameter: %.4fM" % (total / 1e6))
    
    throughput(model)
    
    exit() 
    
def test2():  
    model = StokenAttention(dim=64, seg_dim=8, layers=2, kernel_size=[8,8], hard_label=False, refine=True, refine_attention=True, num_heads=8, qkv_bias=True, rpe=False)
    x = torch.randn(1, 64, 56, 56)
    
    # model = Attention(64)
    # x = torch.randn(64, 49, 64)
    
    print(model)
    
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops))
    
    # model.cuda()
    # x = x.cuda()
    with torch.no_grad():
        model(x)
    # t0 = time.time()
    # for _ in range(5):
        # y = 
        
    # print((time.time()-t0)/5)
    
    
    exit() 
    
if __name__ == '__main__':
    test()      
    # throughput()
