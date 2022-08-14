import torch
from torch import nn

from utils import Trigonometric_kernel, sample_pdf
from layers.RaySamplePoint import RaySamplePoint, RaySamplePoint_Near_Far, RaySamplePoint_Depth
from .spacenet import SpaceNet
from layers.render_layer import VolumeRenderer, gen_weight


class RFRender(nn.Module):
    def __init__(self, coarse_ray_sample, fine_ray_sample, boarder_weight, sample_method='NEAR_FAR',
                 same_space_net=False, TriKernel_include_input=True, depth_field=0.1, depth_ratio=0.25,
                 sample_inf=False, noise_std=0.0, use_alpha=False,
                 synthesis=False, encode_pos=30, encode_dir=6, encode_sigma=10., encode_kernel="POS", n_importance=0):
        super(RFRender, self).__init__()

        self.coarse_ray_sample = coarse_ray_sample
        self.fine_ray_sample = fine_ray_sample
        self.n_importance = n_importance
        self.density_feature_dim = fine_ray_sample + coarse_ray_sample + n_importance
        self.sample_method = sample_method
        self.depth_field = depth_field
        self.depth_ratio = depth_ratio
        self.sample_inf = sample_inf
        self.noise_std = noise_std

        if self.sample_method == 'NEAR_FAR':
            self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)  # use near far to sample points on rays
        elif self.sample_method == 'DEPTH':
            self.rsp_coarse = RaySamplePoint_Depth(sample_num=self.coarse_ray_sample,
                                                   sample_inf=self.sample_inf,
                                                   n_importance=n_importance)
        else:
            self.rsp_coarse = RaySamplePoint(
                self.coarse_ray_sample)  # use bounding box to define point sampling ranges on rays

        self.spacenet = SpaceNet(include_input=TriKernel_include_input,
                                 encode_pos=encode_pos,
                                 encode_dir=encode_dir,
                                 encode_sigma=encode_sigma,
                                 encode_kernel=encode_kernel,
                                 )

        self.spacenet_fine = self.spacenet
        if not same_space_net:
            self.spacenet_fine = SpaceNet(include_input=TriKernel_include_input,
                                          encode_pos=encode_pos,
                                          encode_dir=encode_dir,
                                          encode_sigma=encode_sigma,
                                          encode_kernel=encode_kernel,
                                          ret_feature=True,
                                          )
        sample_depth = self.sample_method == 'DEPTH'
        self.volume_render = VolumeRenderer(use_alpha=use_alpha,
                                            boarder_weight=boarder_weight,
                                            sample_inf=sample_inf and sample_depth,
                                            synthesis=synthesis,
                                            )

        self.maxs = None
        self.mins = None

    def forward(self, rays, bboxes, only_coarse=False, near_far=None, depth=None, rs=None, rgb_mask=None):
        ray_mask = None
        if self.sample_method == 'NEAR_FAR':
            assert near_far is not None, 'require near_far as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz = self.rsp_coarse.forward(rays, near_far=near_far)
            rays_t = rays
        elif self.sample_method == 'DEPTH':
            assert depth is not None, 'require depth as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = self.rsp_coarse.forward(rays,
                                                                                               depth=depth,
                                                                                               depth_field=self.depth_field,
                                                                                               ratio=self.depth_ratio,
                                                                                               rs=rs,
                                                                                               )
            sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
            rays_t = rays[ray_mask].detach()
        else:
            assert bboxes is not None, 'require bboxes as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = self.rsp_coarse.forward(rays, bboxes)
            sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
            rays_t = rays[ray_mask].detach()

        if rgb_mask is not None:
            rgb_mask_t = rgb_mask[ray_mask]

        if rays_t.size(0) > 1:

            sampled_rays_coarse_t = sampled_rays_coarse_t.detach()
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach()

            rgbs, density = self.spacenet(sampled_rays_coarse_xyz, rays_t, self.maxs, self.mins)

            if rgb_mask is not None:
                rgbs[rgb_mask_t, -1, :3] = 1.0

            density[sampled_rays_coarse_t[:, :, 0] < 0, :] = 0.0
            color_0, depth_0, acc_map_0, weights_0 = self.volume_render(sampled_rays_coarse_t,
                                                                        rgbs,
                                                                        density,
                                                                        self.noise_std
                                                                        )
            if not only_coarse:
                z_samples = sample_pdf(sampled_rays_coarse_t.squeeze().detach(),
                                        weights_0.squeeze()[..., 1:-1].detach(),
                                        N_samples=self.fine_ray_sample)
                z_samples = z_samples.detach()  # (N,L)

                z_vals_fine, _ = torch.sort(torch.cat([sampled_rays_coarse_t.squeeze(), z_samples], -1), 1)  # (N, L1+L2)
                samples_fine_xyz = z_vals_fine.unsqueeze(-1) * rays_t[:, :3].unsqueeze(1) + rays_t[:, 3:].unsqueeze(1)  # (N,L1+L2,3)

                if self.spacenet_fine.ret_feature:
                    rgbs, density, rgb_feature, density_feature = self.spacenet_fine(samples_fine_xyz,
                                                                                         rays_t,
                                                                                         self.maxs, self.mins)
                else:
                    rgbs, density = self.spacenet_fine(samples_fine_xyz, rays_t, self.maxs, self.mins)

                if rgb_mask is not None:
                    rgbs[rgb_mask_t, -1, :3] = 1.0

                color, depth, acc_map, weights = self.volume_render(z_vals_fine.unsqueeze(-1),
                                                                        rgbs,
                                                                        density,
                                                                        self.noise_std,
                                                                        )
                rgb_feature = torch.sum(rgb_feature[:, :, :] * weights[:, :, :], dim=1)
                density_feature = weights[:, :-1, :]

                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0), 3, device=rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0), 1, device=rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0), 1, device=rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0

            else:
                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0), 3, device=rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0), 1, device=rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0), 1, device=rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0
                color, depth, acc_map, weights = color_0, depth_0, acc_map_0, weights_0
            color_final, depth_final, acc_map_final = color, depth, acc_map

            if not self.sample_method == 'NEAR_FAR':
                color_final = torch.zeros(rays.size(0), 3, device=rays.device)
                color_final[ray_mask] = color
                depth_final = torch.zeros(rays.size(0), 1, device=rays.device)
                depth_final[ray_mask] = depth
                acc_map_final = torch.zeros(rays.size(0), 1, device=rays.device)
                acc_map_final[ray_mask] = acc_map
                rgb_feature_map = torch.zeros(rays.size(0), rgb_feature.size(1), device=rays.device)
                rgb_feature_map[ray_mask] = rgb_feature
                density_feature_map = torch.zeros(rays.size(0), density_feature.size(1), device=rays.device)
                density_feature_map[ray_mask] = density_feature.squeeze()
        else:
            color_final_0 = torch.zeros(rays.size(0), 3, device=rays.device).requires_grad_()
            depth_final_0 = torch.zeros(rays.size(0), 1, device=rays.device).requires_grad_()
            acc_map_final_0 = torch.zeros(rays.size(0), 1, device=rays.device).requires_grad_()
            rgb_feature_map = torch.zeros(rays.size(0), self.spacenet_fine.head_dim, device=rays.device)
            density_feature_map = torch.zeros(rays.size(0), self.density_feature_dim - 1,
                                                  device=rays.device)
            color_final, depth_final, acc_map_final = color_final_0, depth_final_0, acc_map_final_0

        return (color_final, depth_final, acc_map_final, rgb_feature_map, density_feature_map), \
               (color_final_0, depth_final_0, acc_map_final_0), ray_mask

    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins
