import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

from .ibr_dynamic import IBRDynamicDataset
from utils import ray_sampling, render_depth, depth_sampling, range_sampling, alpha_sampling, patch_sampling
from utils import SingleObjectRender, render_near_far_depth
import trimesh
import time
import pytorch3d.io


class IBRay_NHR_Patch(torch.utils.data.Dataset):
    def __init__(self, data_folder_path, transforms, use_mask=False, num_frame=1, use_depth=False,
                 data_type="NR", no_boundary=False, boundary_width=3, use_alpha=False, patch_sizes=None,
                 keep_bg=False, use_bg=False, synthesis=False, cam_num=999):
        super(IBRay_NHR_Patch, self).__init__()
        if patch_sizes is None:
            patch_sizes = [16]
        elif not isinstance(patch_sizes, list):
            patch_sizes = [patch_sizes]

        self.num_frame = num_frame
        self.data_folder_path = data_folder_path
        self.NHR_dataset = IBRDynamicDataset(data_folder_path,
                                             num_frame,
                                             use_mask,
                                             transforms,
                                             skip_step=1,
                                             random_noisy=0,
                                             holes='None',
                                             use_depth=use_depth,
                                             use_alpha=use_alpha,
                                             use_bg=use_bg,
                                             synthesis=synthesis,
                                             data_type=data_type,
                                             no_boundary=no_boundary,
                                             boundary_width=boundary_width,
                                             cam_num=cam_num
                                             )

        self.fine_obj = None
        if use_depth:
            self.coarse_obj = pytorch3d.io.load_objs_as_meshes([os.path.join(data_folder_path,
                                                                             "meshes/frame%d.obj" % num_frame)],
                                                               load_textures=False)
            if os.path.exists(os.path.join(data_folder_path, "meshes/frame%d_fine.obj" % num_frame)):
                self.fine_obj = pytorch3d.io.load_objs_as_meshes([os.path.join(data_folder_path,
                                                                               "meshes/frame%d_fine.obj" % num_frame)],
                                                                 load_textures=False)

        self.rays = []
        self.rgbs = []
        self.ds = []
        self.rs = []
        self.als = []
        self.near_fars = []
        self.frame_ids = []
        self.vs = []

        self.Ks, self.Ts, self.depths, self.ranges = [], [], [], []

        self.use_mask = use_mask
        self.use_depth = use_depth
        self.use_alpha = use_alpha

        log_suffix = 'rays_tmp'
        if keep_bg:
            log_suffix = log_suffix + '_bg'

        log_path = os.path.join(data_folder_path, log_suffix)

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        for patch_size in patch_sizes:
            if not os.path.exists(os.path.join(log_path, 'rays_%d.pt' % patch_size)):
                for i in range(len(self.NHR_dataset)):
                    img, vs, frame_id, T, K, near_far, _ = self.NHR_dataset.__getitem__(i)
                    if vs is not None:
                        self.vs.append(vs)

                    self.Ts.append(T.numpy())
                    self.Ks.append(K.numpy())

                    img_rgb = torch.cat([img[0:3, :, :], img[7:10, :, :]], dim=0)
                    range_map = torch.zeros_like(img[0, :, :])
                    self.ranges.append(range_map)
                    range_map = range_map.unsqueeze(0)
                    alpha_map = img[6, :, :]

                    fine_depth = None
                    if self.use_depth:
                        opencv_K = K.numpy()
                        cam2world_R = T[0:3, 0:3].numpy()
                        cam2world_t = T[0:3, 3].numpy()
                        render = SingleObjectRender(objs=self.coarse_obj, frame_size=(img.size(2), img.size(1)),
                                                    enable_texture=False, enable_cull=False)
                        depth, far_depth = render_near_far_depth(render, opencv_K, cam2world_R, cam2world_t,
                                                                 view_size=(img.size(2), img.size(1)))
                        render.release()
                        if self.fine_obj is not None:
                            render_fine = SingleObjectRender(objs=self.fine_obj, frame_size=(img.size(2), img.size(1)),
                                                             enable_texture=False, enable_cull=False)
                            fine_depth, _ = render_near_far_depth(render_fine, opencv_K, cam2world_R, cam2world_t,
                                                                  view_size=(img.size(2), img.size(1)))
                            fine_depth = torch.from_numpy(fine_depth)
                            fine_depth = fine_depth.unsqueeze(0)
                            render_fine.release()

                        if no_boundary:
                            depth_mask = depth.copy()
                            depth_mask[depth_mask > 0] = 1

                            kernel = np.ones((boundary_width, boundary_width), np.uint8)
                            depth_mask = cv2.erode(depth_mask, kernel, iterations=1)

                            depth = depth * depth_mask
                        self.depths.append(depth)

                        depth = torch.from_numpy(depth)
                        far_depth = torch.from_numpy(far_depth)
                        if fine_depth is None:
                            fine_depth = torch.zeros_like(far_depth)

                    else:
                        depth = img[5, :, :]
                        far_depth = depth
                        fine_depth = depth

                    if self.use_alpha:
                        img_rgb[:3, alpha_map == 0] = img_rgb[3:6, alpha_map == 0]
                    if self.use_depth:
                        img_rgb[:3, depth == 0] = img_rgb[3:6, depth == 0]
                        alpha_map[depth == 0] = 0.0

                    if self.use_mask:
                        mask = img[4, :, :] * img[3, :, :]
                        img_rgb[:, mask < 0.5] = 1.0
                        if self.use_depth:
                            depth[mask < 0.5] = 0.0
                        if self.use_alpha:
                            alpha_map[mask < 0.5] = 0.0
                        mask = mask.unsqueeze(0)
                    else:
                        mask = img[4, :, :].unsqueeze(0)

                    depth = depth.unsqueeze(0)
                    far_depth = far_depth.unsqueeze(0)
                    alpha_map = alpha_map.unsqueeze(0)

                    rays, rgbs, ds, als = patch_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1), img.size(2)),
                                                         patch_size=patch_size, images=img_rgb.unsqueeze(0),
                                                         depth=depth, far_depth=far_depth, fine_depth=fine_depth,
                                                         alpha=alpha_map, keep_bg=keep_bg)
                    fine_ds = ds[:, :, :, 2]
                    ds[:, :, :, 1][fine_ds > 0] = fine_ds[fine_ds > 0]
                    rs = ds[:, :, :, 1] - ds[:, :, :, 0]
                    ds = ds[:, :, :, 0]
                    print(rs.max(), rs.min())

                    self.als.append(als)
                    self.rays.append(rays)
                    self.rgbs.append(rgbs)
                    self.ds.append(ds)
                    self.rs.append(rs)

                    self.frame_ids.append(
                        torch.ones(rays.size(0), rays.size(1), rays.size(2), 1) * frame_id)  # (N, h, w, 1)
                    self.near_fars.append(near_far.repeat(rays.size(0), rays.size(1), rays.size(2), 1))  # (N, h, w, 2)
                    print(i, '| generate %d rays.' % (rays.size(0) * rays.size(1) * rays.size(2)))

                self.rays = torch.cat(self.rays, dim=0)
                self.rgbs = torch.cat(self.rgbs, dim=0)
                self.near_fars = torch.cat(self.near_fars, dim=0)  # (M,2)
                self.frame_ids = torch.cat(self.frame_ids, dim=0)

                torch.save(self.rays, os.path.join(log_path, 'rays_%d.pt' % patch_size))
                torch.save(self.rgbs, os.path.join(log_path, 'rgb_%d.pt' % patch_size))
                torch.save(self.near_fars,
                           os.path.join(log_path, 'near_fars_%d.pt' % patch_size))
                torch.save(self.frame_ids,
                           os.path.join(log_path, 'frameid_%d.pt' % patch_size))

                self.ds = torch.cat(self.ds, dim=0)
                torch.save(self.ds, os.path.join(log_path, 'd_%d.pt' % patch_size))

                self.als = torch.cat(self.als, dim=0)
                torch.save(self.als, os.path.join(log_path, 'a_%d.pt' % patch_size))

                self.rs = torch.cat(self.rs, dim=0)
                torch.save(self.rs, os.path.join(log_path, 'r_%d.pt' % patch_size))

                self.Ts = np.stack(self.Ts)
                np.save(os.path.join(log_path, 'Ts_%d.npy' % patch_size), self.Ts)

                self.Ks = np.stack(self.Ks)
                np.save(os.path.join(log_path, 'Ks_%d.npy' % patch_size), self.Ks)

                if data_type == "NR":
                    self.vs = torch.cat(self.vs, dim=0)
            else:
                self.rays = torch.load(os.path.join(log_path, 'rays_%d.pt' % patch_size))
                self.rgbs = torch.load(os.path.join(log_path, 'rgb_%d.pt' % patch_size))
                self.near_fars = torch.load(
                    os.path.join(log_path, 'near_fars_%d.pt' % patch_size))
                self.frame_ids = torch.load(
                    os.path.join(log_path, 'frameid_%d.pt' % patch_size))
                self.ds = torch.load(os.path.join(log_path, 'd_%d.pt' % patch_size))
                self.als = torch.load(os.path.join(log_path, 'a_%d.pt' % patch_size))
                self.rs = torch.load(os.path.join(log_path, 'r_%d.pt' % patch_size))
                self.Ts = np.load(os.path.join(log_path, 'Ts_%d.npy' % patch_size))
                self.Ks = os.path.join(log_path, 'Ks_%d.npy' % patch_size)

                img, self.vs, _, T, K, _, _ = self.NHR_dataset.__getitem__(0)
                if self.use_depth:
                    for i in range(len(self.NHR_dataset)):
                        self.ranges.append(torch.zeros_like(img[0, :, :]))
                print('load %d rays.' % (self.rays.size(0) * self.rays.size(1) * self.rays.size(2)))

        if data_type == "NR":
            max_xyz = torch.max(self.vs, dim=0)[0]
            min_xyz = torch.min(self.vs, dim=0)[0]

            tmp = (max_xyz - min_xyz) * 0.15

            max_xyz = max_xyz + tmp
            min_xyz = min_xyz - tmp

            minx, miny, minz = min_xyz[0], min_xyz[1], min_xyz[2]
            maxx, maxy, maxz = max_xyz[0], max_xyz[1], max_xyz[2]
            bbox = np.array(
                [[minx, miny, minz], [maxx, miny, minz], [maxx, maxy, minz], [minx, maxy, minz], [minx, miny, maxz],
                 [maxx, miny, maxz], [maxx, maxy, maxz], [minx, maxy, maxz]])
            self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
        elif data_type == "LLFF":
            self.bbox = torch.zeros((1, 8, 3))

    def __len__(self):
        return self.rays.size(0)

    def __getitem__(self, index):
        return self.rays[index, ...], \
               self.rgbs[index, ...], \
               self.bbox.squeeze(), \
               self.near_fars[index, ...], \
               self.frame_ids[index, ...], \
               self.ds[index], \
               self.rs[index], \
               self.als[index]
