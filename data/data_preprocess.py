import enum
from logging import root
import os

root_path = "data/tiny-imagenet-200"
train_path = os.path.join(root_path, "train")
val_path = os.path.join(root_path, "val")

assert os.path.exists(root_path), "data root path not exists"

print(len(os.listdir(train_path)))
classes = sorted(os.listdir(train_path))
classes2idx = {name: idx for idx, name in enumerate(classes)}

with open(root_path + "/words.txt", "r") as f1:
    with open(root_path + "/classes.txt", "w") as f2:
        lines = f1.readlines()
        class2label = {}
        for line in lines:
            cls = line.split()[0]
            label = line[len(cls):]
            class2label[cls] = label 
        for c in classes:
            f2.write(class2label[c])

with open(root_path + "/train_annotations", "w") as f:
    for i, name in enumerate(classes):
        imgs = os.listdir(train_path+"/" + name + "/images")
        for img in imgs:
            img_name = img.split('.')[-2] + ".JPEG"
            f.write(img_name + f" {i}\n")

with open(val_path + "/val_annotations.txt", "r") as f1:
    anno = f1.readlines()
    with open(root_path + "/val_annotations.txt", "w") as f2:
        for line in anno:
            line = line.split()
            f2.write(line[0] + ' ' + str(classes2idx[line[1]]) + '\n')

