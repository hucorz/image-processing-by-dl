# ResNet

## Conclusion

代码中使用的 resnet34，如果使用预训练的 ResNet34 来做 linear evaluation，train set 上可以训练的很好，但测试集上非常差，过拟合？但是 [WZMIAOMIAO的视频](https://www.bilibili.com/video/BV14E411H7Uw/?spm_id_from=333.788)中一个 epoch 的效果就很好，很奇怪

---

下面的总结不把 conv1.x 后的 maxpool 考虑在 conv2.x 中

ResNet18 / 34  中：

- conv2.x 中所有 block 都不改变 channel 和 HxW，实线的残差操作
- conv3.x / 4.x / 5.x 中 只有第一个 block 改变 channel 和大小（虚线残差），后面的 block in_channel = out_channel，且所有 stride 为 1（实线残差）

ResNet50 / 101 / 152  中：

- conv2.x 中 不改变 HxW，第一个 block 输出 channel 为 256，后面的 block in_channel=out_channel=256
- conv3.x / 4.x / 5.x 中 只有第一个 block 改变 channel 和大小（虚线残差），后面的 block in_channel = out_channel，且所有 stride 为 1（实线残差）


## Model

![ResNet](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/ResNet.png)

![ResNet](https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/ResNet34.png)