# Mask RCNN

## Model

在 roi_head.py 中增加了 mask 分支相关的参数, 能够被 2 个 roi_head 的复用，并在 forward 中加一个判断是否有 mask 分支