import torch
from torch import nn

from utils import make_encoding_kernel, make_dir_encoding_kernel


class SpaceNet(nn.Module):
    def __init__(self, c_pos=3, include_input=True, encode_pos=30,
                 encode_dir=6, encode_sigma=10., encode_kernel="POS", ret_feature=False):
        super(SpaceNet, self).__init__()

        self.tri_kernel_pos = make_encoding_kernel(in_dim=3, L=encode_pos, L1=encode_dir, sigma=encode_sigma,
                                                   method=encode_kernel, include_input=include_input)
        self.tri_kernel_dir = make_dir_encoding_kernel(in_dim=3, L=encode_dir, sigma=encode_sigma,
                                                       method=encode_kernel, include_input=include_input)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)

        self.ret_feature = ret_feature

        backbone_dim = 256
        head_dim = 128

        self.backbone_dim = backbone_dim
        self.head_dim = head_dim

        self.stage1 = nn.Sequential(
            nn.Linear(self.pos_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            nn.Linear(backbone_dim + self.pos_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, backbone_dim),
        )

        self.density_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, head_dim),
            nn.ReLU(inplace=True),
        )
        self.rgb_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim + self.dir_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
        )
        self.density_out = nn.Linear(head_dim, 1)
        self.rgb_out = nn.Linear(head_dim, 3)

    def forward(self, pos, rays=None, maxs=None, mins=None):

        # beg = time.time()
        rgbs = None
        if rays is not None:
            dirs = rays[..., 0:3]

        bins_mode = False
        if len(pos.size()) > 2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1, self.c_pos))  # (N,c_pos)
            if rays is not None:
                dirs = dirs.unsqueeze(1).repeat(1, L, 1)
                dirs = dirs.reshape((-1, self.c_pos))  # (N,3)

        if maxs is not None:
            pos = ((pos - mins) / (maxs - mins) - 0.5) * 2

        pos = self.tri_kernel_pos(pos)
        if rays is not None:
            dirs = self.tri_kernel_dir(dirs)

        x = self.stage1(pos)
        x = self.stage2(torch.cat([x, pos], dim=1))

        density_feature = self.density_net(x)
        density = self.density_out(density_feature)

        rgb_feature = None
        if rays is not None:
            rgb_feature = self.rgb_net(torch.cat([x, dirs], dim=1))
            rgbs = self.rgb_out(rgb_feature)

        if bins_mode:
            density = density.reshape((-1, L, 1))
            density_feature = density_feature.reshape(-1, L, self.head_dim)
            if rays is not None:
                rgbs = rgbs.reshape((-1, L, 3))
                rgb_feature = rgb_feature.reshape(-1, L, self.head_dim)

        if self.ret_feature:
            return rgbs, density, rgb_feature, density_feature

        return rgbs, density
