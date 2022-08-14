# encoding: utf-8
"""
@author:  Haimin Luo
@email: luohm@shanghaitech.edu.cn
"""

import torchvision.transforms as T

from .random_transforms import Random_Transforms


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:

        transform = Random_Transforms((cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0]), cfg.DATASETS.SHIFT,
                                      cfg.DATASETS.MAXRATION, cfg.DATASETS.ROTATION)
    else:
        transform = Random_Transforms((cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]), 0)

    return transform
