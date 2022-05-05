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

## windows 需要注意的地方

代码中 num_workers 需要改为 0

```python
# 原来
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=4)
# 改为
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=0)
```

## Google Colab 的使用

如果缺乏 GPU 资源，且不需要跑特别复杂的网络（比如我），可以使用 Google Colab

前提：会翻墙

[大致参考流程](https://github.com/hucorz/image-processing-by-dl/blob/main/others/google_colab_example.ipynb)

## Tiny Imagenet 200 数据集

下载地址： http://cs231n.stanford.edu/tiny-imagenet-200.zip

预处理脚本：./data/data_preprocess.py

## 其他

部分图片不开梯子会无法显示，貌似是因为 md 中用了 相对路径

