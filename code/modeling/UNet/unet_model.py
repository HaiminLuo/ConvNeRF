# full assembly of the sub-parts to form the complete net
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 24)
        self.down1 = down(24, 48)
        self.down2 = down(48, 96)
        self.down3 = down(96, 384)
        self.down4 = down(384, 384)
        self.up1 = up(768, 192)
        self.up2 = up(288, 48)
        self.up3 = up(96, 48)
        self.up4 = up(72, 30)
        self.outc = outconv(30, n_classes1)

        self.inc2 = inconv(n_channels + n_classes1, 16)
        self.down5 = down(16, 32)
        self.down6 = down(32, 64)
        self.up5 = up(144, 32)
        self.up6 = up(72, 8)
        self.outc2 = outconv(8, n_classes2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x6 = self.up2(x6, x3)
        x6 = self.up3(x6, x2)
        x6 = self.up4(x6, x1)
        x_rgb = self.outc(x6)

        x = torch.cat([x, x_rgb], dim=1)
        x1_2 = self.inc2(x)
        x2_2 = self.down5(x1_2)
        x3_2 = self.down6(x2_2)

        x6 = self.up5(x3_2, torch.cat([x2, x2_2], dim=1))
        x6 = self.up6(x6, torch.cat([x1, x1_2], dim=1))
        x_alpha = self.outc2(x6)

        x = torch.cat([x_rgb, x_alpha], dim=1)
        return x


class PatchUNet(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2):
        super(PatchUNet, self).__init__()
        self.inc1 = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.up1 = up(48, 8)
        self.outc1 = outconv(8, n_classes1)

        self.inc2 = inconv(n_channels + n_classes1, 8)
        self.down2 = down(8, 16)
        self.up2 = up(40, 8)
        self.outc2 = outconv(8, n_classes2)

    def forward(self, x):
        x1 = self.inc1(x)
        x2 = self.down1(x1)
        x3 = self.up1(x2, x1)
        x_rgb = self.outc1(x3)

        # print(x_rgb.shape, x.shape)
        x = torch.cat([x, x_rgb], dim=1)
        x4 = self.inc2(x)
        x5 = self.down2(x4)
        x6 = self.up2(x5, torch.cat([x1, x4], dim=1))
        x_alpha = self.outc2(x6)

        return torch.cat([x_rgb, x_alpha], dim=1)


class PatchFeaUNet(nn.Module):
    def __init__(self, rgb_feat_channels, alpha_feat_channels, rgb_channels, alpha_channels):
        super(PatchFeaUNet, self).__init__()
        self.inc1 = inconv(rgb_feat_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.up1 = up(96, 32)
        self.up2 = up(48, 16)
        self.outc1 = outconv(16, rgb_channels)

        self.inc2 = inconv(alpha_feat_channels + rgb_channels, 16)
        self.down3 = down(16, 32)
        self.up3 = up(64, 16)
        self.outc2 = outconv(16, alpha_channels)

    def forward(self, rgb_feat, alpha_feat):
        x1 = self.inc1(rgb_feat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x4 = self.up2(x4, x1)
        x_rgb = self.outc1(x4)

        # print(x_rgb.shape, x.shape)
        x = torch.cat([alpha_feat, x_rgb], dim=1)
        x5 = self.inc2(x)
        x6 = self.down3(x5)
        x7 = self.up3(x6, torch.cat([x1, x5], dim=1))
        x_alpha = self.outc2(x7)

        return torch.cat([x_rgb, x_alpha], dim=1)


class PatchFeaDirUNet(nn.Module):
    def __init__(self, rgb_feat_channels, alpha_feat_channels, rgb_channels, alpha_channels, dir_channels=0):
        super(PatchFeaDirUNet, self).__init__()
        self.inc1 = inconv(rgb_feat_channels + dir_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.up1 = up(96, 32)
        self.up2 = up(48, 16)
        self.outc1 = outconv(16, rgb_channels)

        self.inc2 = inconv(alpha_feat_channels + rgb_channels, 16)
        self.down3 = down(16, 32)
        self.up3 = up(64, 16)
        self.outc2 = outconv(16, alpha_channels)

    def forward(self, rgb_feat, alpha_feat, dirs=None):
        if dirs is not None:
            rgb_feat = torch.cat([rgb_feat, dirs], dim=1)
        x1 = self.inc1(rgb_feat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x4 = self.up2(x4, x1)
        x_rgb = self.outc1(x4)

        # print(x_rgb.shape, x.shape)
        x = torch.cat([alpha_feat, x_rgb], dim=1)
        x5 = self.inc2(x)
        x6 = self.down3(x5)
        x7 = self.up3(x6, torch.cat([x1, x5], dim=1))
        x_alpha = self.outc2(x7)

        return torch.cat([x_rgb, x_alpha], dim=1)