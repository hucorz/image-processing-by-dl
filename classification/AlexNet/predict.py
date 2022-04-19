import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from model import LeNet

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", help="path of image to predict", default=img_path, required=True)
    parser.add_argument("--pth_path", help="pth file's path", default="./output/LeNet_checkpoint.pth")
    config = parser.parse_args()
    return config

def main(config):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转化为灰度图
        transforms.Resize([32, 32]),
        transforms.ToTensor()]
    )
    model = LeNet().to(device)
    model.load_state_dict(torch.load(config.pth_path))
    model.eval()
    
    im = Image.open(config.img_path)
    im = transform(im)
    im = torch.unsqueeze(im, dim=0) # [C, H, W] -> [N, C, H, W]

    with torch.no_grad():
        output = model(im.to(device))
        pred = output.argmax(dim=-1)
        pred = pred.item()
    print(f"result: {labels[pred]}")


if __name__ == "__main__":
    img_path = ""
    config = get_config()
    main(config)