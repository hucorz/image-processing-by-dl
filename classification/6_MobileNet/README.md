# MobileNet

## Conclusion

使用预训练的 v2 pth 文件，lr=0.0001，batch_size=32，训练 5 个 epoch 的结果：

![](output/result_v2.png)

使用预训练的 v3-large pth 文件，lr=0.0001，batch_size=32，训练 5 个 epoch 的结果：

![](output/result_v3.png)

## Model

相比于前面的模型非常轻量，可以应用于移动端或者嵌入式设备，且正确率也不错

### MobileNet(v2)

倒残差 Block，Block 的输入和输出的维度完全相同才做 shortcut 操作，即 shortcut 不做任何的变换

Residual Block: 降维再升维

Inverted Residual Block：升维再降维

![Inverted Residual Block](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v2)_1.png)

输入 k 维 输出 k' 维的 Bottleneck residual block

![](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v2)_2.png)

t 表示bottleneck block 中维度的扩张倍数

n 表示 block 个数

c 表示输出维度（n 个 block 中只有第一个 block 的输入和输出维度不同）

s 表示 stride

最后的 1x1 conv 效果和 fc 层相同

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v2)_3.png" style="zoom:67%;" />



### MobileNet(v3)

SE 注意力机制

![](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v3)_1.png)

MobileNet(v3)-Large

![](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v3)_Large.png)

MobileNet(v3)-Small

![](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/MobileNet(v3)_Small.png)
