# encoding: utf-8

from .rfrender import RFRender
from .UNet import UNet
from .model import GeneralModel, Discriminator


def build_model(cfg):
    model = GeneralModel(cfg, use_unet=True)
    return model


def build_discriminator(cfg):
    model = Discriminator(patch_size=cfg.DATASETS.PATCH_SIZE)
    return model