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