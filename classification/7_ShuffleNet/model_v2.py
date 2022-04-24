from typing import List

import torch
from torch import Tensor
import torch.nn as nn

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, h, w = x.size()
    channels_per_group = num_channels // groups

    # [b, c, h, w] -> [b, groups, channel_per_groups, h, w]
    x = x.view(batch_size, groups, channels_per_group, h, w)

    # eg
    # 原来 x = [[1, 2], [3, 4]] 在内存中为 1, 2, 3, 4
    # transpose 后 x = [[1, 3], [2, 4]] 在内存中还是 1, 2, 3, 4
    # contiguous 后 x = [[1, 3], [2, 4]] 在内存是 1, 3, 2, 4
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batch_size, -1, h, w)

    return x

class InvertedResidual(nn.Module):
    '''
    ShuffleNet(v2) 的基本 block
    '''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride

        assert out_channel % 2 == 0
        branch_features = out_channel // 2 # 最后要 concat, 所以分成 2 半
        assert (self.stride != 1) or (in_channel == branch_features <<1) # stride 为 1 时, in_channel 为 branch_channel 的 2 倍

        if self.stride == 2:  # 做下采样
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channel, in_channel, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU()
            )
        else:   # 不做下采样
            self.branch1 = nn.Sequential()
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU()
        )
    
    @staticmethod
    def depthwise_conv(in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False) -> nn.Conv2d:
            return nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias, groups=in_channel)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, 1)
            x2 = self.branch2(x2)
            res = torch.cat([x1, x2], dim=1)
        elif self.stride == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            res = torch.cat([x1, x2], dim=1)
        
        res = channel_shuffle(res, 2)
        return res

class ShuffleNetV2(nn.Module):
    def __init__(self, 
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int=1000,
        inverted_residual=InvertedResidual):
        super().__init__()

        assert len(stages_repeats) == 3
        assert len(stages_out_channels) == 5

        # self._stage_out_channels = stages_out_channels

        in_channel = 3
        out_channel = stages_out_channels[0]

        self.conv1 = nn.Sequential( # out_size: [112, 112]
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        in_channel = out_channel

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # out_size: [56, 56]

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, out_channel in zip(stage_names, stages_repeats, stages_out_channels[1:]): # 列表长度不同时, zip默认用最短的
            layers = [inverted_residual(in_channel, out_channel, 2)] # 第一个 block 要下采样，stride=2
            for _ in range(repeats-1):
                layers.append(inverted_residual(out_channel, out_channel, 1))
            setattr(self, name, nn.Sequential(*layers))  # 属性赋值
            in_channel = out_channel
        
        out_channel = stages_out_channels[-1] # 最后一个 out_channel 上面 zip 里并没有用到
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.fc = nn.Linear(out_channel, num_classes)

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) # global pool
        x = self.fc(x)
        return x

def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model