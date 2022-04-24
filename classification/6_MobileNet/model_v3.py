import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from functools import partial


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


class ConvBNActivation(nn.Sequential):
    '''
    Conv +BN + Activation
    '''

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None):
        # kernel_size=1 => padding=0, kernel_size=3 => padding=1
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            norm_layer(out_channel),
            activation_layer()
        )


# SE 注意力模块
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, squeeze_factor=4):
        super().__init__()
        squeeze_channel = _make_divisible(in_channel // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channel, squeeze_channel, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channel, in_channel, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale) # 用的是 conv，所以不需要 flatten
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale)
        return x*scale


class InvertedResidualConfig:
    def __init__(self, in_channel, kernel_size, expanded_c, out_channel, use_se: bool, activation: str, stride, width_multi: float):
        '''
        width_multi: 控制模型宽度(channel)
        '''
        self.in_channel = _make_divisible(in_channel*width_multi, 8)
        self.kernel_size = kernel_size
        self.expanded_c = _make_divisible(expanded_c*width_multi, 8)
        self.out_channel = _make_divisible(out_channel*width_multi, 8)
        self. use_se = use_se
        self.use_hs = (activation == "HS")
        self.stride = stride

    @staticmethod   # 此静态方法后面会用到
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    '''
    bneck 模块
    '''
    def __init__(self, cnf: InvertedResidualConfig, norm_layer):
        super().__init__()
        assert cnf.stride in [1, 2], "stride must be 1 or 2"

        self.use_res_connect = (cnf.stride == 1 and cnf.in_channel == cnf.out_channel)  # 只有输入输出完全一样才使用 shortcut

        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_c != cnf.in_channel:  # 如果中间 channel 不需要扩张就不需要加 1x1 conv
            layers.append(ConvBNActivation(cnf.in_channel, cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))

        layers.append(ConvBNActivation(
            cnf.expanded_c,
            cnf.expanded_c,
            kernel_size=cnf.kernel_size,
            stride=cnf.stride,
            groups=cnf.expanded_c,  # dw conv
            norm_layer=norm_layer,
            activation_layer=activation_layer
        ))  

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project，最后的 1x1 conv
        layers.append(ConvBNActivation(
            cnf.expanded_c,
            cnf.out_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        ))

        self.block = nn.Sequential(*layers)
        # self.out_channels = cnf.out_channel
        # self.is_strided = cnf.stride > 1

    def forward(self, x):
        res = self.block(x)
        if self.use_res_connect:
            res += x
        return res

class MobileNetV3(nn.Module):
    '''
    MobileNet(v3)
    '''
    def __init__(self, inverted_residual_setting, last_channel, num_classes, block=None, norm_layer=None):
        '''
        args:
            - last_channel: 最后一个 fc 层的 in_channel
        '''
        super().__init__()
        assert inverted_residual_setting, "inverted_residual_setting should not be empty"
        assert isinstance(inverted_residual_setting, List) \
            and all([isinstance(cnf, InvertedResidualConfig) for cnf in inverted_residual_setting]), "The inverted_residual_setting should be List[InvertedResidualConfig]"

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        layers = []

        # building first layer
        firstconv_out_cahnnel = inverted_residual_setting[0].in_channel # setting 不包括开头的 conv
        layers.append(ConvBNActivation(
            in_channel=3,
            out_channel=firstconv_out_cahnnel,
            kernel_size=3, 
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish
        ))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        # building last several layers  
        lastconv_in_channel = inverted_residual_setting[-1].out_channel
        lastconv_out_channel = 6*lastconv_in_channel  # 最后的 conv 层的 out_channel 是 in_channel 的 6 倍
        layers.append(ConvBNActivation(
            lastconv_in_channel,
            lastconv_out_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish
        ))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_out_channel, last_channel),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenet_v3_large(num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=num_classes)

