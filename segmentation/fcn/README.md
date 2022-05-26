# FCN(Fully Convolutional Networks for Semantic Segmentation)

代码来源：[WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation/fcn)

训练结果：因为是用的官方预训练权重且数据集相同，所以精度没什么变化

```
[epoch: 0]
train_loss: 0.7927
lr: 0.000100
global correct: 93.3
average row correct: ['97.2', '90.5', '71.7', '81.5', '77.2', '53.9', '90.1', '78.8', '93.4', '53.3', '83.6', '63.9', '80.9', '81.3', '87.7', '95.3', '61.3', '87.3', '67.6', '84.4', '79.7']
IoU: ['93.1', '84.6', '37.1', '79.0', '69.1', '51.1', '87.0', '72.5', '81.0', '40.4', '75.4', '52.8', '72.3', '76.0', '77.8', '86.7', '54.5', '75.7', '51.0', '80.8', '73.9']
mean IoU: 70.1
```

## 代码结构

- src：
    - backbone：resnet50 / resnet101 backbone
    - fcn_model: 对 backbone 重构以得到中间层的输出，然后添加 预测头 和 辅助预测头
- train_utils: 
    - train_and_eval: 定义了 train 和 eval 时会用到的一些函数
- my_dataset: 自定义数据集
- transformers.py: 自定义 transformer
- train.py: 训练文件

## 完整的结构图

图中的 layer3 和 layer4 对应 resnet 中的 conv4 和 conv5

两个 bottleneck 都引入了膨胀卷积

bottleneck1 的残差分支不进行下采样，下采样过大对语义分割的影响很大

![](../../img/segmentation/fcn.png)

