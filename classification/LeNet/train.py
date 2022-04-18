import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model import LeNet

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="path of image to train, if None, FashionMNIST will be download and used", default=None, required=False)
    parser.add_argument("--save_path", type=str, help="output file's saving path", default="./output", required=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001, required=False)
    parser.add_argument("--epoch", type=int, help="epoch", default=30, required=False)
    config = parser.parse_args()
    return config

def main(config):
    print(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [transforms.Resize([32, 32]),
        transforms.ToTensor()]
    )
    if config.img_path:
        pass
    else:
        # 训练集 60,000 样本
        train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=60, shuffle=True, num_workers=4)
        # 验证集 10,000 样本
        val_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transform)
        val_loader = DataLoader(val_set, batch_size=100, shuffle=False, num_workers=4)

    model = LeNet().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_loss_record = []    # train loss record each 100 steps
    train_acc_record = []     # train acc record
    val_loss_record = []    # val loss record
    val_acc_record = []     # val acc record
    best_val_acc = 0.0


    train_loss, train_acc = [], []
    for epoch in range(config.epoch):
        for step, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(dim=-1) == labels).float().mean()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

            if step % 100 == 99:   # each 100 steps do validation and recording
                model.eval()
                val_loss, val_acc = [], []
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        
                        loss = criterion(outputs, labels)
                        acc = (outputs.argmax(dim=-1) == labels).float().mean()

                        val_loss.append(loss.item())
                        val_acc.append(acc.item())
                    
                    val_loss_record.append(sum(val_loss) / len(val_loss))
                    val_acc_record.append(sum(val_acc) / len(val_acc))
                    train_loss_record.append(sum(train_loss) / len(train_loss))
                    train_acc_record.append(sum(train_acc) / len(train_acc))
                    train_loss, train_acc = [], []

                    if (val_acc_record[-1] > best_val_acc):
                        #  torch.save(model.state_dict(), os.path.join(config.save_path, f"LeNet_checkpoint_epoch{epoch}step_{step}.pth")) # 这输出的pth太多了
                        torch.save(model.state_dict(), os.path.join(config.save_path, f"LeNet_checkpoint.pth"))
                        best_val_acc = val_acc_record[-1]
                    print(f"[epoch:{epoch:03d}/{config.epoch:03d}, step:{step:04d}] train loss:{train_loss_record[-1]:.4f}, train acc:{train_acc_record[-1]:.4f} | val loss:{val_loss_record[-1]:.4f} val acc:{val_acc_record[-1]:.4f}")
                model.train()

    plt.figure()
    plt.subplot(221);
    plt.plot(train_loss_record);plt.title("train loss record");
    plt.subplot(222);
    plt.plot(train_acc_record);plt.title("train acc record");
    plt.subplot(223);
    plt.plot(val_loss_record);plt.title("val loss record");
    plt.subplot(224);
    plt.plot(val_acc_record);plt.title("val acc record");
    plt.savefig(os.path.join(config.save_path, "result.png"))

if __name__ == "__main__":
    config = get_config()
    
    if "output" not in os.listdir("./"):
        os.mkdir("./output")

    main(config)