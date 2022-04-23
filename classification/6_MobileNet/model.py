import torch
import torch.nn as nn
import torch.nn.functional as F

'''
下面注释中的 conv2.x 不包括 conv1 后面的 maxpool
'''


class BasicBlock(nn.Module):
    """
    每个 Block 有 2 个 3x3 conv, 是 Res18 / 34 的基本结构
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.downsample = downsample       # 如果残差结构的主分支的输出 HxW 和输入不同，虚线残差操作也需要做相应的变化

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    """
    每个 Bottleneck 有 1 个 1x1 conv, 1 个 3x3 conv, 1 个 1x1 conv, 是 Res50 /101 / 152 的基本结构

    原论文中 conv3.x 4.x 5.x 中的第一个 1x1conv 步距为 2, 第二个 3x3conv 步距为 1
    pytorch 官方实现: conv3.x 4.x 5.x 中的第一个 1x1conv 步距为 1, 第二个 3x3conv 步距为 2, top1 acc 可以提升 0.5%

    args:
        - out_channel: 不代表最终的 out_channel, 而是中间的 channel, 最终的 out_channel 为 out_channel * expansion
        - group: 和 ResNeXt 有关
        - width_per_group: 和 ResNeXt 有关
    
    """

    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        '''
        args:
        - out_channel: 不是最终的 out_channel, 是中间的 3x3 conv 的输出 channel,  最终的输出 channel 为 out_channel*expansion
        - stride 是 3x3 conv 的 stride
        '''
        super().__init__()
        # 普通 ResNet 情况下 groups 和 width_per_group 为默认参数，width = out_channel
        # ResNeXt-50(32x4d) 时，groups = 32, width_per_group=4，width=out_channel*2
        width = int(out_channel * (width_per_group / 64.)) * groups 

        self.conv1 = nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False)
        self. bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        '''
        args:
            block: res18 和 res34 使用 BasicBlock, res50,101,152 使用 Bottleneck
            blocks_num: list, 表示 conv2.x conv3.x 4.x 5.x 的个数
        '''
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64                    # conv2.x 开始的 in_channel

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # conv2.x 一开始的 maxpool, 使得 conv2.x 的输出长宽减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])                             # conv2.x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)        # conv3.x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)        # conv4.x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)        # conv5.x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            '''
            downsample 为 残差块中的残差操作

            res18 / 34:
                - conv2.x: stride=1, expansion=1, 不会进来
                - conv3.x / 4.x / 5.x: stride=2, 会进来, downsample改变channel改变长宽
            res50 / 101 / 152:
                - conv2.x: 会进来, downsample 只改变 channel
                - 3.x / 4.x / 5/x: stride=2, 会进来, downsample改变channel改变长宽

            
            '''
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            # conv n.x 中 只有第一个 block 会指定 stride,  后面的 block 不会改变 HxW
            # 且后面的 block 的 in_channel 和 out_channel 一定相同，这里的 block 的参数 channel 并不是最终的 out_channel，block 中的 out_channel = channel * block.expansion
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group)) 

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)   # conv2.x
        x = self.layer2(x)   # conv3.x
        x = self.layer3(x)   # conv4.x
        x = self.layer4(x)   # conv5.x

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
