import argparse
import pdb
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, datasets, utils
import torchvision.models.resnet

import matplotlib.pyplot as plt

from model import resnet34, resnext50_32x4d



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="path of image to train, if None", default=None, required=True)
    parser.add_argument("--output_path", type=str, help="output file's saving path", default="./output", required=False)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001, required=False)
    parser.add_argument("--epoch", type=int, help="epoch", default=5, required=False)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32, required=False)
    parser.add_argument("--linear_eval", help="do linear evaluation with a pretrained model", action="store_true", required=False)
    parser.add_argument("--pretrained_path", help="pretrained model's pth file path", type=str, required=False)


    config = parser.parse_args()
    return config

def main(config):
    print(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = {
        "train":transforms.Compose([
           transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    }

    nw = min([os.cpu_count(), 8, config.batch_size if config.batch_size > 1 else 0])

    assert config.img_path, "img_path is needed"

    train_root = os.path.join(config.img_path, "train")
    val_root = os.path.join(config.img_path, "val")
    train_set = datasets.ImageFolder(root=train_root, transform=transform["train"])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=nw)
    val_set = datasets.ImageFolder(root=val_root, transform=transform["val"])
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config.batch_size, num_workers=nw)
    print(f"length of train set: {len(train_set)}")
    print(f"length of val set: {len(val_set)}")

    class2idx = train_set.class_to_idx
    print(class2idx)
    idx2class = dict((idx, cla) for cla, idx in class2idx.items())

    if config.linear_eval:
        model = resnet34()
        assert config.pretrained_path, "do linear evaluation need pretrained pth file"
        state_dict = torch.load(config.pretrained_path)
        # for key in list(state_dict.keys()):                                   # 删除预训练文件中的 fc 层
        #     if "fc" in key:
        #         state_dict.pop(key)
        model_dict = model.state_dict()
        model.load_state_dict(state_dict)
        # for name, para in model.named_parameters():  # 冻结非 fc 层的参数
        #     if "fc" not in name:
        #         para.requires_grad = False      
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, len(class2idx))
        model.to(device)     
    else:
        model = resnet34(num_classes=len(class2idx)).to(device)      
              
    # pdb.set_trace()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_loss_record = []    # train loss record each 100 steps
    train_acc_record = []     # train acc record
    val_loss_record = []    # val loss record
    val_acc_record = []     # val acc record
    best_val_acc = 0.0


    for epoch in range(config.epoch):
        running_loss, running_acc = 0.0, 0.0
        model.train()
        pbar = tqdm(train_loader)
        for data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(dim=-1) == labels).float().sum()

            running_loss += loss.item()
            running_acc += acc.item()
            # pbar.desc = f"[{epoch+1} / {config.epoch}] loss: {loss}"

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=-1) == labels).float().sum()

                val_loss += loss.item()
                val_acc += acc.item()

            
            val_loss_record.append(val_loss)
            val_acc_record.append(val_acc / len(val_set))
            train_loss_record.append(running_loss)
            train_acc_record.append(running_acc / len(train_set))

            if (val_acc_record[-1] > best_val_acc):
                torch.save(model.state_dict(), os.path.join(config.output_path, f"ResNet34_checkpoint.pth"))
                best_val_acc = val_acc_record[-1]
            print(f"[epoch:{epoch+1:03d}/{config.epoch:03d}] train loss:{train_loss_record[-1]:.4f}, train acc:{train_acc_record[-1]:.4f} | val loss:{val_loss_record[-1]:.4f} val acc:{val_acc_record[-1]:.4f}")

    np.save(os.path.join(config.output_path, "train_loss_record.npy"), train_loss_record)
    np.save(os.path.join(config.output_path, "train_acc_record.npy"), train_acc_record)
    np.save(os.path.join(config.output_path, "val_loss_record.npy"), val_loss_record)
    np.save(os.path.join(config.output_path, "val_acc_record.npy"), val_acc_record)
    plt.figure()
    plt.subplot(221);
    plt.plot(train_loss_record);plt.title("train loss record");
    plt.subplot(222);
    plt.plot(train_acc_record);plt.title("train acc record");
    plt.subplot(223);
    plt.plot(val_loss_record);plt.title("val loss record");
    plt.subplot(224);
    plt.plot(val_acc_record);plt.title("val acc record");
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, "result.png"))

if __name__ == "__main__":
    config = get_config()
    
    if config.output_path == "./output" and "output" not in os.listdir("./"):
        os.mkdir("./output")

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)

    main(config)