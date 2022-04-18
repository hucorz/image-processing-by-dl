from turtle import forward
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(  # 编码器(特征抽取)
            nn.Conv2d(1, 6, 5),       # input: 32x32, output: 28x28
            nn.Sigmoid(),
            nn.MaxPool2d(2),         # output: 14x14
            nn.Conv2d(6, 16, 5),     # output: 10x10
            nn.Sigmoid(),
            nn.MaxPool2d(2)            # output: 5x5
        )
        self.head = nn.Sequential( # 预测头
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feats = self.enc(img)
        output = self.head(feats.view(feats.shape[0], -1))
        return output

if __name__ == "__main__":
    net = LeNet()
    print(net)