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

import matplotlob.pyplot as plt

from model import LeNet

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="path of image to predict", default=img_path)
    parser.add_argument("pth_path", help="pth file's path", default="./LeNet.pth")
    config = parser.parse_args()
    return config

def main(config):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [transforms.Resize([32, 32]),
        transforms.ToTensor()]
    )
    model = LeNet().to(device)
    torch
    model.eval()
    


if __name__ == "__main__":
    img_path = ""
    config = get_config(img_path)
    main(config)