# Swin-Transformer

## Conclusion

## Model

参考：https://blog.csdn.net/qq_37541097/article/details/121119988

Swin Transformer 与 Vit 的对比：

-   Swin 采用了类似卷积的层次化构建方法
-   使用了 Windows Multi-Head Self-Attention（W-MSA）与 Shifted Windows Multi-Head Self-Attention（SW-MSA）；W-SMA 减少了计算量，SW-MSA 使得不同窗口间可以进行信息交换

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_1.png" style="zoom: 80%;" />

整体流程：

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_2.png" style="zoom: 80%;" />

-   Patch Partition：4x4x3 个像素为一个 Patch，然后展平，[H, W, 3] -> [H/4, W/4, 48]；用 Conv 实现
-   然后通过 4 个 stage：Stage 1 先经过一个 Linear Embedding，后面的 Stage 先经过一个 Patch Merging 进行下采样；每个 Block 的数量都是偶数，因为 W-MSA 和 SW-MSA 是成对出现的
-   最后接上 Layer Norm，全局池化和全连接层

具体配置：

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_8.png" style="zoom: 80%;" />

### Patch Merging

如图所示，结果是 [H, W, C] -> [H/2, W/2, C*2]

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_3.png" style="zoom: 80%;" />

### W-MSA

W-MSA 的引入是为了减小计算量；MSA 是直接对整个 feature map 做 SA，W-MSA 是将 feature map 分为 MxM 块后对每块单独做 SA

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_4.png" style="zoom: 80%;" />

### SW-MSA

目的是为了不同的 windows 之间有信息的交流，但是这样窗口增多了，为了计算方式和普通的 W-MSA 保持一致，作者提出 `Efficient batch computation for shifted configuration`

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_5.png" style="zoom: 80%;" />

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_6.png" style="zoom: 80%;" />

为了防止信息乱窜，使用了 `masked MSA`

以下图区域 5 和 3 为例，5 和 3 本来没有关系，可以在计算出 $\alpha$ 后对 一些表示 5 和 3 区域 的 $\alpha$ 减去 100，经过 softmax 后得到的权重基本为 0；

计算完成后数据会回来原来的位置上

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/classification/Swin_Transformer_7.png" style="zoom: 80%;" />

### Relative Position Bias

略

