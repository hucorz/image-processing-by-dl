# image-processing-by-dl

Studying recording about image processing by deep learning based on Pytorch and OpenMMLab

## My Environments

```python
ubuntu: 18.04
python: 3.8.0
pytorch: 1.8.2
torchvision: 0.9.2
cuda: 10.2
mmcv-full: 1.5.0
mmclassification: 0.23.0
```

根据 mmcls 官方的 colab 教程下载版本对应的 mmcv-full 的命令：

```python
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.2/index.html 
```

## 参考

[WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing) （本项目的代码和图片基本来自这位大佬的库和博客）

[花书 pytorch 版](https://tangshusen.me/Dive-into-DL-PyTorch/#/)

[花书官网](https://zh.d2l.ai/index.html) 

[paper reading](https://github.com/mli/paper-reading) 

[OpenMMLab CodeBase](https://openmmlab.com/codebase)

## windows 需要注意的地方

代码中 num_workers 需要改为 0

```python
# 原来
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=4)
# 改为
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=0)
```

## Google Colab 的使用

如果缺乏 GPU 资源（比如我），可以使用 Google Colab

前提：会翻墙

[大致参考流程](https://github.com/hucorz/image-processing-by-dl/blob/main/others/google_colab_example.ipynb)

## 数据集

**Tiny Imagenet 200** (还是太大了，训练起来太慢就没有使用)

下载地址： http://cs231n.stanford.edu/tiny-imagenet-200.zip

**花分类数据集** (本项目都用的这个数据集做训练)

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set

预处理脚本:

- 基于 Pytorch
    - mmclassification/data/split_data.py (来自 [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set))：把数据集变成 ImageNet 格式

- 基于 OpenMMLab
    - mmclassification/data/split_data.py (来自 [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set))：把数据集变成 ImageNet 格式
    - mmclassification/data/data_process.py：处理 OpenMMLab 需要的三个文件 classes.txt / train_annotations.txt / val_annotations.txt

## 其他

部分图片不开梯子会无法显示，貌似是因为 md 中用了 相对路径

