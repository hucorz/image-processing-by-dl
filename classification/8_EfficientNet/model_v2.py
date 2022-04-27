from numpy import block
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math
import copy
from typing import List
from functools import partial
from collections import OrderedDict

def drop_path(x, drop_prob: float = 0., training: bool = False): 
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    # 以 图像为例，shape 为 (batch_size, 1, 1, 1)，可以对每张图片经过的 residual 模块中的主分支舍弃
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvBNAct(nn.Module):
    '''
    在 v1 中此模块用的 nn.Sequential, 目前我不觉得这2者在效果上有区别, 我猜这么做是为了匹配 pth 文件中的键值对
    Conv +BN + Activation
    stride = 1, 不改变 HxW
    stride = 2, H/2 x W/2
    '''

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU):
        super().__init__()
        
        # kernel_size=1 => padding=0, kernel_size=3 => padding=1
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = norm_layer(out_channel)
        self.act = activation_layer()

    def forward(self, x):
        res = self.conv(x)
        res = self.bn(res)
        res = self.act(res)
        return res

class SqueezeExcitation(nn.Module):
    '''
    SE 注意力模块
    这里 SE 模块的 中间节点数是整个 block 的 in_channel 的 1/4, 而不是 SE 模块 in_channel 的 1/4
    '''

    def __init__(self, in_channel, expand_channel, squeeze_ratio=0.25):
        super().__init__()
        squeeze_channel = int(in_channel * squeeze_ratio)
        self.conv_reduce = nn.Conv2d(expand_channel, squeeze_channel, kernel_size=1)
        self.act1 = nn.SiLU()   # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)  # 用的是 conv，所以不需要 flatten
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return x*scale


class MBConv(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, expand_ratio, stride, se_ratio, drop_rate, norm_layer):
        super().__init__()
        assert stride in [1, 2]

        self.has_shortcut = (stride==1) and (in_channel==out_channel)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = in_channel *expand_ratio

        assert expand_ratio != 1 # v2-S 中 MBConv 的 expand_ratio 必定 in [4, 6]

        # pw conv 在 MBConv 中肯定有
        self.expand_conv = ConvBNAct(in_channel, expanded_c, 1, norm_layer=norm_layer, activation_layer=activation_layer)
        # dw Conv 
        self.dwconv = ConvBNAct(expanded_c, expanded_c, kernel_size=kernel_size, stride=stride, groups=expanded_c, norm_layer=norm_layer, activation_layer=activation_layer)
        self.se = SqueezeExcitation(in_channel, expanded_c, se_ratio) if se_ratio >0 else nn.Identity()

        # pw Conv
        self.project_conv = ConvBNAct(expanded_c, out_channel, 1, norm_layer=norm_layer, activation_layer=nn.Identity) # 这里没有激活函数

        self.drop_rate = drop_rate
        if self.drop_rate > 0 and self.has_shortcut:
            self.dropout = DropPath(drop_rate)
    def forward(self, x):
        res = self.expand_conv(x)
        res = self.dwconv(res)
        res = self.se(res)
        res = self.project_conv(res)

        if self.has_shortcut:
            if self.drop_rate > 0:
                res = self.dropout(res)
            res += x
        
        return res

class FusedMBConv(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, expand_ratio: int, stride, se_ratio, drop_rate, norm_layer):
        super().__init__()

        assert stride in [1, 2]
        assert se_ratio == 0   # FusedMBConv 中不使用 SE 模块

        self.has_shortcut = (stride==1) and (in_channel==out_channel)
        self.has_expansion = (expand_ratio != 1)

        activation_layer = nn.SiLU # alias Swish
        expanded_channel = expand_ratio *in_channel

        if self.has_expansion: # 如果 expand_ratio != 1
            self.expand_conv = ConvBNAct(in_channel, expanded_channel, kernel_size=kernel_size, stride=stride, norm_layer=norm_layer, activation_layer=activation_layer)
            self.project_conv = ConvBNAct(expanded_channel, out_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity)  # 没有激活函数
        else:
            self.project_conv = ConvBNAct(in_channel, out_channel, kernel_size=kernel_size, stride=stride, norm_layer=norm_layer, activation_layer=activation_layer) # 有激活函数
        
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        if self.has_expansion:
            res = self.expand_conv(x)
            res = self.project_conv(res)
        else:
            res = self.project_conv(x)
        
        if self.has_shortcut:
            if self.drop_rate > 0:
                res = self.dropout(res)
            res += x

        return res
        
class EfficientNetV2(nn.Module):
    def __init__(self, 
                model_cnf,
                num_classes=1000,
                num_features=1280,
                dropout_rate=0.2,
                drop_connect_rate=0.2):
        super().__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8
        
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4] # stage 1 的 in_channel

        self.stem = ConvBNAct(3, stem_filter_num, kernel_size=3, stride=2, norm_layer=norm_layer) # activation 默认 SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1], 
                                in_channel=cnf[4] if i == 0 else cnf[5],  # 第一个 block 之后的 block 的 in_channel = out_channel
                                out_channel=cnf[5],
                                expand_ratio=cnf[3],
                                stride=cnf[2] if i == 0 else 1,                      # 第一个 block 之后的 block 不改变 HxW
                                se_ratio=cnf[-1],
                                drop_rate=drop_connect_rate *block_id / total_blocks,   # drop_rate 逐渐上升
                                norm_layer=norm_layer))
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        head_in_channel = model_cnf[-1][-3]  # 预测头 in_channel
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_in_channel, num_features, kernel_size=1, norm_layer=norm_layer)}) # activation 默认 SiLU
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(dropout_rate)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model



