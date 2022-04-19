import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_version, num_classes=1000):
        super().__init__()
        self.enc = get_vgg_enc(vgg_version)    # 编码器 (特征提取)
        self.head = nn.Sequential( # 预测头
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, img):
        feats = self.enc(img) # 5 个 maxpool, (3x224x224) -> (512x7x7)
        feats = torch.flatten(feats, start_dim=1)
        output = self.head(feats)
        return output

def get_vgg_enc(vgg_version):
    assert vgg_version in cfgs.keys(), "vgg version not found"

    layers = []
    in_channels = 3
    for v in cfgs[vgg_version]:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1))
            layers.append(nn.ReLU())
            in_channels = v
    return nn.Sequential(*layers)
        
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

if __name__ == "__main__":
    net = VGG(vgg_version="vgg16")
    print(net)