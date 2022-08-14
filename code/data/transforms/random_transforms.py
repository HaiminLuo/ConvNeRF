# encoding: utf-8
"""
@author:  Haimin Luo
@email: luohm@shanghaitech.edu.cn
"""

import torchvision.transforms as T

from torch.utils import data
import torch

import numpy as np

import random
import PIL
from PIL import Image
import collections
import math

'''
INPUT: mask is a (h,w) numpy array 
every pixel larger than 0 will be in count
'''


def calc_center(mask):
    grid = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    grid_mask = mask[grid[0], grid[1]].astype(np.bool)
    X = grid[0, grid_mask]
    Y = grid[1, grid_mask]

    return np.mean(X), np.mean(Y)


def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class Random_Transforms(object):
    def __init__(self, size, random_range=0, random_ration=0, random_rotation=0, interpolation=Image.BICUBIC,
                 isTrain=True, is_center=False):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.random_range = random_range
        self.random_scale = random_ration
        self.isTrain = isTrain
        self.random_rotation = random_rotation
        self.is_center = is_center

    def __call__(self, img, Ks=None, Ts=None, mask=None, depth=None, alpha=None, background=None):

        K = Ks.clone()
        Tc = Ts.clone()
        img_np = np.asarray(img)

        offset = random.randint(-self.random_range, self.random_range)
        offset2 = random.randint(-self.random_range, self.random_range)

        rotation = (random.random() - 0.5) * np.deg2rad(self.random_rotation)
        ration = random.random() * self.random_scale + 1.0

        width, height = img.size

        R = torch.Tensor(rodrigues_rotation_matrix(np.array([0, 0, 1]), rotation))

        Tc[0:3, 0:3] = torch.matmul(Tc[0:3, 0:3], R)

        m_scale = height / self.size[0]

        cx, cy = 0, 0

        if mask is not None and self.isTrain:
            mask_np = np.asarray(mask)
            mask_np = mask_np[:, :, 0]
            cy, cx = calc_center(mask_np)

            cx = cx - width / 2
            cy = cy - height / 2

        translation = (offset * m_scale - cx, offset2 * m_scale - cy)

        if self.is_center:
            translation = [width / 2 - K[0, 2], height / 2 - K[1, 2]]
            translation = list(translation)
            ration = 1.05

            if (self.size[1] / 2) / (self.size[0] * ration / height) - K[0, 2] != translation[0]:
                ration = 1.2
            translation[1] = (self.size[0] / 2) / (self.size[0] * ration / height) - K[1, 2]
            translation[0] = (self.size[1] / 2) / (self.size[0] * ration / height) - K[0, 2]
            translation = tuple(translation)

        # translation = (width /2-K[0,2],height/2-K[1,2])

        img = T.functional.rotate(img, angle=np.rad2deg(rotation), resample=Image.BICUBIC, center=(K[0, 2], K[1, 2]))
        img = T.functional.affine(img, angle=0, translate=translation, scale=1, shear=0)
        img = T.functional.crop(img, 0, 0, int(height / ration), int(height * self.size[1] / ration / self.size[0]))
        img = T.functional.resize(img, self.size, self.interpolation)
        img = T.functional.to_tensor(img)

        ROI = np.ones_like(img_np) * 1.

        ROI = Image.fromarray(np.uint8(ROI))
        ROI = T.functional.rotate(ROI, angle=np.rad2deg(rotation), resample=Image.BICUBIC, center=(K[0, 2], K[1, 2]))
        ROI = T.functional.affine(ROI, angle=0, translate=translation, scale=1, shear=0)
        ROI = T.functional.crop(ROI, 0, 0, int(height / ration), int(height * self.size[1] / ration / self.size[0]))
        ROI = T.functional.resize(ROI, self.size, self.interpolation)
        ROI = T.functional.to_tensor(ROI)
        ROI = ROI[0:1, :, :]

        if mask is not None:
            mask = T.functional.rotate(mask, angle=np.rad2deg(rotation), resample=Image.BICUBIC,
                                       center=(K[0, 2], K[1, 2]))
            mask = T.functional.affine(mask, angle=0, translate=translation, scale=1, shear=0)
            mask = T.functional.crop(mask, 0, 0, int(height / ration),
                                     int(height * self.size[1] / ration / self.size[0]))
            mask = T.functional.resize(mask, self.size, self.interpolation)
            mask = T.functional.to_tensor(mask)

        if depth is not None:
            depth = T.functional.rotate(depth, angle=np.rad2deg(rotation), resample=Image.NEAREST,
                                        center=(K[0, 2], K[1, 2]))
            depth = T.functional.affine(depth, angle=0, translate=translation, scale=1, shear=0)
            depth = T.functional.crop(depth, 0, 0, int(height / ration),
                                      int(height * self.size[1] / ration / self.size[0]))
            depth = T.functional.resize(depth, self.size, Image.NEAREST)
            depth = T.functional.to_tensor(depth)

        if alpha is not None:
            alpha = T.functional.rotate(alpha, angle=np.rad2deg(rotation), resample=Image.BICUBIC,
                                        center=(K[0, 2], K[1, 2]))
            alpha = T.functional.affine(alpha, angle=0, translate=translation, scale=1, shear=0)
            alpha = T.functional.crop(alpha, 0, 0, int(height / ration),
                                      int(height * self.size[1] / ration / self.size[0]))
            alpha = T.functional.resize(alpha, self.size, Image.BICUBIC)
            alpha = T.functional.to_tensor(alpha)

        if background is not None:
            background = T.functional.rotate(background, angle=np.rad2deg(rotation), resample=Image.BICUBIC,
                                      center=(K[0, 2], K[1, 2]))
            background = T.functional.affine(background, angle=0, translate=translation, scale=1, shear=0)
            background = T.functional.crop(background, 0, 0, int(height / ration),
                                              int(height * self.size[1] / ration / self.size[0]))
            background = T.functional.resize(background, self.size, self.interpolation)
            background = T.functional.to_tensor(background)

        # K = K / m_scale
        # K[2,2] = 1

        K[0, 2] = K[0, 2] + translation[0]
        K[1, 2] = K[1, 2] + translation[1]

        s = self.size[0] * ration / height

        K = K * s

        K[2, 2] = 1
        # print(img.size(),mask.size(),ROI.size())

        return img, K, Tc, mask, ROI, depth, alpha, background

    def __repr__(self):
        return self.__class__.__name__ + '()'
