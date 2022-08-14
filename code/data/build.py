# encoding: utf-8
"""
@author:  Haimin Luo
@email: luohm@shanghaitech.edu.cn
"""

from torch.utils import data

from .datasets.ray_source import IBRay_NHR, IBRay_NHR_View
from .datasets.batch_ray_source import IBRay_NHR_Patch
from .transforms import build_transforms


def build_dataset(data_folder_path, transforms, bunch, use_mask, num_frame,
                  use_depth, data_type, no_boundary, b_width, use_alpha, patch_size, keep_bg, use_bg, synthesis, cam_num):
    datasets = IBRay_NHR_Patch(data_folder_path, transforms=transforms, use_mask=use_mask, num_frame=num_frame,
                               use_depth=use_depth, data_type=data_type, no_boundary=no_boundary,
                               boundary_width=b_width, use_alpha=use_alpha, patch_sizes=patch_size, keep_bg=keep_bg,
                               use_bg=use_bg, synthesis=synthesis, cam_num=cam_num
                               )
    return datasets


def build_dataset_view(data_folder_path, transforms, use_mask, num_frame, use_depth, data_type, use_range, use_alpha,
                       use_bg, synthesis):
    datasets = IBRay_NHR_View(data_folder_path, transforms=transforms, use_mask=use_mask, num_frame=num_frame,
                              use_depth=use_depth, data_type=data_type, use_range= use_range, use_alpha=use_alpha,
                              use_bg=use_bg, synthesis=synthesis,
                              )
    return datasets


def make_data_loader(cfg, is_train=True):
    batch_size = cfg.SOLVER.BATCH_SIZE

    if is_train:
        shuffle = True
    else:
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg.DATASETS.TRAIN,
                             transforms,
                             bunch=cfg.SOLVER.BUNCH,
                             use_mask=cfg.DATASETS.USE_MASK,
                             num_frame=cfg.DATASETS.NUM_FRAME,
                             use_depth=cfg.DATASETS.USE_DEPTH,
                             data_type=cfg.DATASETS.TYPE,
                             no_boundary=cfg.DATASETS.NO_BOUNDARY,
                             b_width=cfg.DATASETS.BOUNDARY_WIDTH,
                             use_alpha=cfg.DATASETS.USE_ALPHA,
                             patch_size=cfg.DATASETS.PATCH_SIZE,
                             keep_bg=cfg.DATASETS.KEEP_BG,
                             use_bg=cfg.DATASETS.USE_BG,
                             synthesis=cfg.DATASETS.SYNTHESIS,
                             cam_num=cfg.DATASETS.NUM_CAMERA
                             )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader, datasets


def make_data_loader_view(cfg, is_train=False):
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset_view(cfg.DATASETS.TEST,
                                  transforms,
                                  use_mask=cfg.DATASETS.USE_MASK,
                                  num_frame=cfg.DATASETS.NUM_FRAME,
                                  use_depth=cfg.DATASETS.USE_DEPTH,
                                  data_type=cfg.DATASETS.TYPE,
                                  use_range=cfg.SOLVER.UPDATE_RANGE,
                                  use_alpha=cfg.DATASETS.USE_ALPHA,
                                  use_bg=cfg.DATASETS.USE_BG,
                                  synthesis=cfg.DATASETS.SYNTHESIS,
                                  )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets
