# image-processing-by-dl
studying recording about image processing by deep learning

## Environments

```python
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
