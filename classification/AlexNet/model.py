import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.enc = nn.Sequential(  # 编码(器(特征抽取)
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),       # input: (3x224x224), output: (48x55x55)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                    # output: (28x27x27)
            nn.Conv2d(48, 128, kernel_size=3, padding=1),     # output: (128x27x27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # output: (128x13x13)
            nn.Conv2d(128, 192, kernel_size=3, padding=1), # output:(192x13x13)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1), # output: (192x13x13)
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # output: (128x13x13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output: (128x6x6)
        )
        self.head = nn.Sequential( # 预测头
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, img):
        feats = self.enc(img)
        feats = torch.flatten(feats, start_dim=1)
        output = self.head(feats)
        return output

if __name__ == "__main__":
    net = AlexNet(1000)
    print(net)