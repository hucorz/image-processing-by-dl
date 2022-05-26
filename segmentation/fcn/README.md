# FCN(Fully Convolutional Networks for Semantic Segmentation)

## 代码结构

- src：
    - backbone：resnet50 / resnet101 backbone
    - fcn_model: 对 backbone 重构以得到中间层的输出，然后添加 预测头 和 辅助预测头
- my_dataset: 自定义数据集
- train_utils: 
    - train_and_eval: 定义了 train 和 eval 时会用到的一些函数

## 完整的结构图

![](../../img/segmentation/fcn.png)

