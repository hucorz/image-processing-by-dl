import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvBNReLU(nn.Sequential):
    '''
    注意这里继承的 nn.Sequential
    Conv +BN +ReLU
    stride = 1 不改变 HxW
    stride = 2 H/2 x W/2
    '''
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size-1) // 2 # kernel_size=1 => padding=0, kernel_size=3 => padding=1
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        '''
        args:
            - expand_ratio: 中间层扩展倍数
        '''
        super().__init__()
        hidden_channel = in_channel *expand_ratio
        self.use_shortcut = (stride == 1) and (out_channel == in_channel)

        layers = []

        if expand_ratio != 1:   # 如果中间层channel不扩展，开始1x1卷积没有意义
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))  # 1x1 pw conv
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel), # 3x3 dw conv
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False), #1x1 pw conv，因为最后的 conv 使用线性激活函数，所以不用 ConvBNReLU
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32*alpha, round_nearest) # 32 是经过第一个 conv 后 第一个 block 的 in_channel
        last_channel = _make_divisible(1280*alpha, round_nearest) # 1280 是最后一个 block 的 out_channel

        inverted_residual_setting = [  
            # t, c, n, s, 
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        # feature
        layers = []
        layers.append(ConvBNReLU(3, input_channel, stride=2))

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*alpha, round_nearest)
            for i in range(n):
                stride = s if  i == 0 else 1 # 第一个block后面的block不
                layers.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        
        layers.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*layers)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
