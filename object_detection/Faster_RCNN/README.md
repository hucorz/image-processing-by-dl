# Faster RCNN

代码来源：[WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn)



## 模型结构图

黄色线是只有在训练时才会有的步骤

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/obj_detection/fasterRCNN_1.png"  />

## Model

### GeneralizedRCNNTransform

-   Normalize
-   Resize，这里的每张图片的尺寸都是不一样的，所以这里把每个 Batch 的图片的长宽都 Resize 到同一个范围内，Resize 的方法采用双线性插值
-   batch images，把 Resize 后的图片全都打包到同样的维度，方法是选取一个 Batch 中所有图片最大的长和宽，然后在图片的右边和下面补 0，补到最大的长和宽；如下图

<img src="https://cdn.jsdelivr.net/gh/hucorz/image-processing-by-dl/img/obj_detection/fasterRCNN_2.png" style="zoom:40%;" />

## 代码结构

### backbone



### network_file

#### GeneralizedRCNNTransfom 部分

-   image_list.py
-   transform.py: 其中有一个后面会用到的后处理方法

#### RPN 部分

- ron_function.py
- boxes

会用到 boxes 和 det_utils 里的一些类和方法

#### ROI 部分

-   faster_rcnn_framework.py 中定义的 ROIAling, TwoMLPHead, FasterRCNNPredicter, 这三个作为参数传给 ROIHeads 类 
-   ROI_head.py 