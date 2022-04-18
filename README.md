# image-processing-by-dl
studying recording about image processing by deep learning

## windows 需要注意的地方

代码中 num_workers 需要改为 0

```python
# 原来
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=4)
# 改为
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform, num_workers=0)
```

