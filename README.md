# image-processing-by-dl
studying recording about image processing by deep learning

## My Environments

```python
ubuntu: 18.04
python: 3.8.0
pytorch: 1.8.2
torchvision: 0.9.2
cuda: 10.2
```

## 参考

主要是跟着这位大佬的库：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

https://tangshusen.me/Dive-into-DL-PyTorch/#/

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

## 其他

部分图片不开梯子会无法显示，因为 md 中用了 相对路径

