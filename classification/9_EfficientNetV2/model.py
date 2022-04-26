import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math
import copy
from typing import List
from functools import partial
from collections import OrderedDict


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)   # 离 ch 最近的 divisor 的倍数
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

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

class ConvBNActivation(nn.Sequential):
    '''
    Conv +BN + Activation
    stride = 1, 不改变 HxW
    stride = 2, H/2 x W/2
    '''

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None):
        # kernel_size=1 => padding=0, kernel_size=3 => padding=1
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            norm_layer(out_channel),
            activation_layer()
        )


class SqueezeExcitation(nn.Module):
    '''
    SE 注意力模块
    这里 SE 模块的 中间节点数是整个 block 的 in_channel 的 1/4, 而不是 SE 模块 in_channel 的 1/4
    '''

    def __init__(self, in_channel, expand_channel, squeeze_factor=4):
        super().__init__()
        squeeze_channel = in_channel // squeeze_factor
        self.fc1 = nn.Conv2d(expand_channel, squeeze_channel, kernel_size=1)
        self.ac1 = nn.SiLU()   # alias Swish
        self.fc2 = nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)  # 用的是 conv，所以不需要 flatten
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return x*scale


class InvertedResidualConfig:
    def __init__(self,
                 kernel_size,
                 in_channel,
                 out_channel,
                 expanded_ratio: int,    # 扩张系数，1倍或6倍
                 stride: int,    # 1 或 2
                 use_se: bool,      # Efficient Net 中总是使用 SE 模块，为 True
                 drop_rate: float,
                 index: str,    # 1a, 2a, 2b ...
                 width_multi: float):
        self.in_channel = _make_divisible(in_channel*width_multi, 8)
        self.kernel_size = kernel_size
        self.expanded_c = self.in_channel * expanded_ratio
        self.out_channel = _make_divisible(out_channel*width_multi, 8)
        self. use_se = use_se
        self.stride = stride
        self.index = index
        self.drop_rate = drop_rate

    @staticmethod   # 此静态方法后面会用到
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    '''
    MBConv 模块
    '''

    def __init__(self, cnf: InvertedResidualConfig, norm_layer):
        super().__init__()
        assert cnf.stride in [1, 2], "stride must be 1 or 2"

        self.use_res_connect = (cnf.stride == 1 and cnf.in_channel == cnf.out_channel)  # 只有输入输出完全一样才使用 shortcut

        layers = OrderedDict()
        activation_layer = nn.SiLU

        if cnf.expanded_c != cnf.in_channel:  # 如果中间 channel 不需要扩张就不需要加 1x1 conv
            layers.update({"expand_conv": ConvBNActivation(
                cnf.in_channel,
                cnf.expanded_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer)})

        layers.update({"dwconv": ConvBNActivation(
            cnf.expanded_c,
            cnf.expanded_c,
            kernel_size=cnf.kernel_size,
            stride=cnf.stride,
            groups=cnf.expanded_c,  # dw conv
            norm_layer=norm_layer,
            activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.in_channel, cnf.expanded_c)})  # se 需要 整个 block 的 in_channel

        # project，最后的 1x1 conv
        layers.update({"project_conv": ConvBNActivation(
            cnf.expanded_c,
            cnf.out_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer)})

        self.block = nn.Sequential(layers)
        # self.out_channels = cnf.out_channel
        # self.is_strided = cnf.stride > 1

        # 源码中只有用到 shortcut 并且 drop_rate >0 时才使用 dropout
        if self.use_res_connect and cnf.drop_rate > 0:  
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x) -> Tensor:
        res = self.block(x)
        res = self.dropout(res)
        if self.use_res_connect:
            res += x
        return res


class EfficientNet(nn.Module):
    def __init__(self, 
                        width_multi: float,
                        depth_multi: float, 
                        num_classes: int=1000,
                        dropout_rate: float=0.2, 
                        drop_connect_rate: float=0.2,
                        block=None, 
                        norm_layer=None):
        '''
        width_multi: 增加网络的宽度(channel)
        depth_multi: 增加网络深度
        '''
        super().__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):  # 计算增加后的深度 向上取整
            return int(math.ceil(depth_multi*repeats))

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
        bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf)) # 计算所有 block 的总数

        b = 0
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))): # block 重复次数
                if i > 0:
                    cnf[-3] = 1 # 最后的 block 数 pop 掉了，所以 -3 是 stride，每个 stage 除了第一个 block，stride 都为 1
                    cnf[1] = cnf[2] # 每个 stage 除了第一个 block，in_channel = out_channel
                
                cnf[-1] = args[-2] *b / num_blocks   # dropout_rate 逐渐增大
                index = str(stage+1) +chr(i +97)
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        layers = OrderedDict()
        
        # first conv
        layers.update({"stem_conv": ConvBNActivation(
                    in_channel=3,
                    out_channel=adjust_channels(32),
                    kernel_size=3,
                    stride=2,
                    norm_layer=norm_layer)})

        # building inverted residual blocks， stage1, stage2....
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # building top
        last_conv_in_channel = inverted_residual_setting[-1].out_channel
        last_conv_out_channel = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(last_conv_in_channel, last_conv_out_channel, 1, norm_layer=norm_layer)})
        
        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate >0:
            classifier.append(nn.Dropout(dropout_rate))
        classifier.append(nn.Linear(last_conv_out_channel, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_multi=1.0,
                        depth_multi=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_multi=1.0,
                        depth_multi=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_multi=1.1,
                        depth_multi=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_multi=1.2,
                        depth_multi=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_multi=1.4,
                        depth_multi=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_multi=1.6,
                        depth_multi=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_multi=1.8,
                        depth_multi=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_multi=2.0,
                        depth_multi=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)