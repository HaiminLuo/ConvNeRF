import torch
import torch.nn as nn
from .rfrender import RFRender
from .UNet import PatchFeaUNet


class GeneralModel(nn.Module):
    def __init__(self, cfg, use_unet=True):
        super(GeneralModel, self).__init__()
        self.use_unet = use_unet
        self.rfrender = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING,
                                 cfg.MODEL.FINE_RAY_SAMPLING,
                                 boarder_weight=cfg.MODEL.BOARDER_WEIGHT,
                                 sample_method=cfg.MODEL.SAMPLE_METHOD,
                                 same_space_net=cfg.MODEL.SAME_SPACENET,
                                 TriKernel_include_input=cfg.MODEL.TKERNEL_INC_RAW,
                                 depth_field=cfg.MODEL.DEPTH_FIELD,
                                 depth_ratio=cfg.MODEL.DEPTH_RATIO,
                                 sample_inf=cfg.MODEL.SAMPLE_INF,
                                 noise_std=cfg.MODEL.NOISE_STD,
                                 use_alpha=cfg.DATASETS.USE_ALPHA,
                                 synthesis=cfg.DATASETS.SYNTHESIS,
                                 encode_pos=cfg.MODEL.ENCODE_POS_DIM,
                                 encode_dir=cfg.MODEL.ENCODE_DIR_DIM,
                                 encode_sigma=cfg.MODEL.GAUSSIAN_SIGMA,
                                 encode_kernel=cfg.MODEL.KERNEL_TYPE,
                                 n_importance=cfg.MODEL.INPORTANCE_RAY_SAMPLE,
                                 )
        if self.use_unet:
            self.unet = PatchFeaUNet(self.rfrender.spacenet_fine.head_dim, self.rfrender.density_feature_dim - 1, 3, 1)

    def forward(self, rays, bboxes=None, only_coarse=False, near_fars=None, depth=None, rs=None, rgb_mask=None):
        batch_size, h, w = rays.size(0), rays.size(1), rays.size(2)
        dirs = rays[..., 0:3]
        rays = rays.reshape(-1, rays.size(3))

        near_fars = near_fars.reshape(-1, near_fars.size(3))
        ds = depth.reshape(-1)
        rs = rs.reshape(-1)
        bboxes = bboxes[0].repeat(rays.size(0), 1, 1) if bboxes is not None else None

        stage2, stage1, ray_mask = self.rfrender(rays, bboxes, only_coarse, near_far=near_fars, depth=ds, rs=rs,
                                                 rgb_mask=rgb_mask)
        img_1, img_0 = stage2[0].reshape(batch_size, h, w, 3), stage1[0].reshape(batch_size, h, w, 3)
        alpha_1, alpha_0 = stage2[2].reshape(batch_size, h, w, 1), stage1[2].reshape(batch_size, h, w, 1)
        patch_1, patch_0 = torch.cat([img_1, alpha_1], -1).permute(0, 3, 1, 2).cuda(), \
                           torch.cat([img_0, alpha_0], -1).permute(0, 3, 1, 2).cuda()
        rgb_feature_map = stage2[3].reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        density_feature_map = stage2[4].reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        if self.use_unet:
            dirs = dirs.permute(0, 3, 1, 2)
            depth = depth.unsqueeze(1).expand(-1, 3, -1, -1)
            dirs[depth == 0] = 0
            res_1 = self.unet(rgb_feat=rgb_feature_map, alpha_feat=density_feature_map)
        else:
            res_1 = patch_1

        return stage2, stage1, ray_mask, patch_1, patch_0, res_1


class Discriminator(nn.Module):
    def __init__(self, patch_size=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 1, kernel_size=3, stride=2),
        )

        self.out = nn.Linear(int(patch_size / 2) ** 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape(batch_size, -1)
        x = self.out(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
