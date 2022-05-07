需要先 clone mmclassification 到此目录

再把此目录中的文件复制到对应的位置

- checkpoint 放 pth 文件
- data 中解压缩 花分类数据集，并用脚本处理
- config 中添加 flower_data 文件夹, 里面是针对花数据集的分类模型配置文件

暂时只尝试了 2 个模型

Vit-Base-P16_1xb32_flower.ipynb 是参照官方的 colab 教程做的 Vit-B16 训练过程

**花分类数据集**

只训练 5 epoch，具体看配置文件；预训练权重在 [mmclassification官方文档](https://mmclassification.readthedocs.io/zh_CN/latest/model_zoo.html)

| Model    | Top-1 | Config                                                       | Log                                                          |
| -------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet50 | 93.96 | [config](./configs/flower_data/resnet50_1xb32_flower.py)     | [log](work_dirs/resnet50_1xb32_flower/20220507_103719.log)   |
| Vit-B16  | 97.53 | [config](./configs/flower_data/vit-base-p16_1xb32_flower.py) | [log](work_dirs/vit-base-p16_1xb32_flower/20220507_032132.log) |

