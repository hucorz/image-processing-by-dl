import random
from torchvision.transforms import functional as F

# 自定义一些 transformer
# 因为这里的 transformer 涉及到了 翻转操作，会相应的导致 target 改变，所以全都要重写, 返回值都要是 2 个返回值 

class Compose(object):
    """组合多个transform函数
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob: # 如要要翻转
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            # 水平翻转 y 坐标的值不变，xmin = width-xmax, xmax = width-xmin
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target