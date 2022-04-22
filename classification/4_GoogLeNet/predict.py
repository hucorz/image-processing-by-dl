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

from model import GoogLeNet

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="path of image to predict", required=True)
    parser.add_argument("--pth_path", type=str, help="pth file's path", default="./output/GoogleNet_checkpoint.pth", required=True)
    parser.add_argument("--num_classes",type=int, default=1000, required=True)
    config = parser.parse_args()
    return config

def main(config):
    labels = ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = GoogLeNet(config.num_classes, aux_logits=True).to(device)
    model.load_state_dict(torch.load(config.pth_path))
    model.eval()
    
    im = Image.open(config.img_path)
    im = transform(im)
    im = torch.unsqueeze(im, dim=0) # [C, H, W] -> [N, C, H, W]

    with torch.no_grad():
        output = model(im.to(device))
        output = torch.squeeze(output).cpu()
        # print(output.shape)
        output = torch.softmax(output, dim=0)
        pred = output.argmax(dim=-1)
        pred = pred.item()
    print(f"result: {labels[pred]}, {output[pred]}")


if __name__ == "__main__":
    img_path = ""
    config = get_config()
    main(config)