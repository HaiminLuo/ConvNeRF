import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image, ImageFilter
import torchvision
import torch.distributions as tdist


def merge_holes(pc1, pc2):
    # change point color here

    return np.concatenate([pc1, pc2], axis=0)


class IBRDynamicDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder_path, frame_num, use_mask, transforms, skip_step, random_noisy, holes,
                 use_depth=False, use_alpha=False, use_bg=False, synthesis=False, data_type="NR", no_boundary=False,
                 boundary_width=2, cam_num=999):
        super(IBRDynamicDataset, self).__init__()

        self.frame_num = frame_num
        self.data_folder_path = data_folder_path
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.use_alpha = use_alpha
        self.use_bg = use_bg
        self.synthesis = synthesis

        self.skip_step = skip_step
        self.random_noisy = random_noisy
        self.holes = holes

        self.boundary = no_boundary
        self.boundary_width = boundary_width

        self.file_path = os.path.join(data_folder_path, 'img')
        self.data_type = data_type

        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index = []

        self.nears, self.fars = [], []

        camposes = np.loadtxt(os.path.join(data_folder_path, 'CamPose.inf'))
        self.Ts = torch.Tensor(campose_to_extrinsic(camposes))
        self.cam_num = self.Ts.size(0) if cam_num > self.Ts.size(0) else cam_num
        self.indices = np.linspace(0, self.Ts.shape[0] - 1, self.cam_num).astype(np.int)
        print('camera index: ', self.indices)

        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path, 'Intrinsic.inf')))
        print('load %d Ts, %d Ks, %d frames' % (self.Ts.size(0), self.Ks.size(0), self.frame_num))

        self.transforms = transforms

        if self.data_type == "NR":
            sum_tmp = 0
            for i in range(frame_num):
                if os.path.exists(os.path.join(data_folder_path, 'pointclouds/frame%d.npy' % (i + 1))):
                    tmp = np.load(os.path.join(data_folder_path, 'pointclouds/frame%d.npy' % (i + 1)))
                elif os.path.exists(os.path.join(data_folder_path, 'pointclouds/frame%d.obj' % (i + 1))):
                    tmp = np.loadtxt(os.path.join(data_folder_path, 'pointclouds/frame%d.obj' % (i + 1)),
                                     usecols=(0, 1, 2, 3, 4, 5))
                elif os.path.exists(os.path.join(data_folder_path, 'pointclouds/frame%d.xzy' % (i + 1))):
                    # tmp = np.loadtxt(os.path.join(data_folder_path, 'pointclouds/frame%d.xzy' % (i + 1)),
                    #                 usecols=(0, 1, 2, 3, 4, 5))
                    tmp = np.loadtxt(os.path.join(data_folder_path, 'pointclouds/frame%d.xzy' % (i + 1)),
                                     usecols=(0, 1, 2))
                elif os.path.exists(os.path.join(data_folder_path, 'pointclouds/frame%d.txt' % (i + 1))):
                    tmp = np.loadtxt(os.path.join(data_folder_path, 'pointclouds/frame%d.txt' % (i + 1)),
                                     usecols=(0, 1, 2))

                if os.path.exists(os.path.join(self.holes, 'holes/frame%d.npy' % (i + 1))):
                    tmp2 = np.load(os.path.join(self.holes, 'holes/frame%d.npy' % (i + 1)))
                    tmp = merge_holes(tmp, tmp2)
                    if i % 50 == 0:
                        print('merge holes', tmp2.shape[0])

                vs_tmp = tmp[:, 0:3]
                vs_rgb_tmp = tmp[:, 3:6]
                self.vs_index.append(sum_tmp)
                self.vs.append(torch.Tensor(vs_tmp))
                self.vs_rgb.append(torch.Tensor(vs_rgb_tmp))
                self.vs_num.append(vs_tmp.shape[0])
                sum_tmp = sum_tmp + vs_tmp.shape[0]

                if i % 50 == 0:
                    print(i, '/', frame_num)

            self.vs = torch.cat(self.vs, dim=0)
            self.vs_rgb = torch.cat(self.vs_rgb, dim=0)

            if random_noisy > 0:
                n = tdist.Normal(torch.tensor([0.0, 0.0, 0.0]),
                                 torch.tensor([random_noisy, random_noisy, random_noisy]))
                kk = torch.min((torch.max(self.vs, dim=1)[0] - torch.min(self.vs, dim=1)[0]) / 500)
                self.vs = self.vs + kk * n.sample((self.vs.size(0),))

            # self.black_list = [625,747,745,738,62,750,746,737,739,762]
            print('load %d vertices' % (self.vs.size(0)))

            self._all_imgs = None
            self._all_Ts = None
            self._all_Ks = None
            self._all_width_height = None

            inv_Ts = torch.inverse(self.Ts).unsqueeze(1)  # (M,1,4,4)
            vs = self.vs.clone().unsqueeze(-1)  # (N,3,1)
            vs = torch.cat([vs, torch.ones(vs.size(0), 1, vs.size(2))], dim=1)  # (N,4,1)

            pts = torch.matmul(inv_Ts, vs)  # (M,N,4,1)

            pts_max = torch.max(pts, dim=1)[0].squeeze()  # (M,4)
            pts_min = torch.min(pts, dim=1)[0].squeeze()  # (M,4)

            pts_max = pts_max[:, 2]  # (M)
            pts_min = pts_min[:, 2]  # (M)

            self.near = pts_min * 0.5
            self.near[self.near < (pts_max * 0.1)] = pts_max[self.near < (pts_max * 0.1)] * 0.1

            self.far = pts_max * 1.3
            print('dataset initialed. near: %f  far: %f' % (self.near.min(), self.far.max()))

        elif self.data_type == "LLFF":
            for i in range(frame_num):
                bd = np.load(os.path.join(data_folder_path, 'bd/%d/bd.npy' % i))
                self.nears.append(torch.from_numpy(bd[:, 0]).unsqueeze(0))
                self.fars.append(torch.from_numpy(bd[:, 1]).unsqueeze(0))
            self.nears = torch.cat(self.nears, dim=0).float()
            self.fars = torch.cat(self.fars, dim=0).float()

            print('dataset initialed. near: %f  far: %f' % (self.nears.min(), self.fars.max()))

    def __len__(self):
        return self.cam_num * (self.frame_num // self.skip_step)

    def __getitem__(self, index, need_transform=True):

        frame_id = ((index // self.cam_num) * self.skip_step) % self.frame_num
        # cam_id = index % self.cam_num
        cam_id = self.indices[index % self.cam_num]
        
        img = None
        # cam_id += 1
        if os.path.exists(os.path.join(self.file_path, '%d/img_%04d.jpg' % (frame_id, cam_id))):
            img = Image.open(os.path.join(self.file_path, '%d/img_%04d.jpg' % (frame_id, cam_id)))
        elif os.path.exists(os.path.join(self.file_path, '%d/img_%04d.png' % (frame_id, cam_id))):
            img = Image.open(os.path.join(self.file_path, '%d/img_%04d.png' % (frame_id, cam_id)))
        img_mask, depth, img_alpha, img_bg = None, None, None, None

        if self.use_mask:
            img_mask = Image.open(os.path.join(self.file_path, '%d/mask/img_%04d.jpg' % (frame_id, cam_id)))
            if self.boundary:
                img_mask = img_mask.filter(ImageFilter.MinFilter(self.boundary_width))

        if self.use_depth:
            if os.path.exists(os.path.join(self.file_path, '%d/depth/img_%04d.tiff' % (frame_id, cam_id))):
                depth = Image.open(os.path.join(self.file_path, '%d/depth/img_%04d.tiff' % (frame_id, cam_id)))

        if self.use_alpha:
            if os.path.exists(os.path.join(self.file_path, '%d/img_%04d_alpha.png' % (frame_id, cam_id))):
                img_alpha = Image.open(os.path.join(self.file_path, '%d/img_%04d_alpha.png' % (frame_id, cam_id)))

        if self.use_bg:
            if os.path.exists(os.path.join(self.file_path, '%d/img_%04d_bg.jpg' % (frame_id, cam_id))):
                img_bg = Image.open(os.path.join(self.file_path, '%d/img_%04d_bg.jpg' % (frame_id, cam_id)))
            if os.path.exists(os.path.join(self.file_path, '%d/img_%04d_bg.png' % (frame_id, cam_id))):
                img_bg = Image.open(os.path.join(self.file_path, '%d/img_%04d_bg.png' % (frame_id, cam_id)))

        # cam_id -= 1
        img, K, T, im_mask, ROI, depth, img_alpha, img_bg = self.transforms(img, self.Ks[cam_id], self.Ts[cam_id],
                                                                            img_mask, depth, img_alpha, img_bg)
        mask_pad = im_mask[0:1, :, :] if im_mask is not None else torch.zeros(img[0:1, :, :].size())
        depth_pad = depth[0:1, :, :] if depth is not None else torch.zeros(img[0:1, :, :].size())
        alpha_pad = img_alpha[0:1, :, :] if img_alpha is not None else torch.zeros(img[0:1, :, :].size())
        default_bg_pad = torch.ones(img[0:3, :, :].size()) if self.synthesis else torch.zeros(img[0:3, :, :].size())
        back_pad = img_bg[0:3, :, :] if img_bg is not None else default_bg_pad
        img = torch.cat([img, mask_pad, ROI, depth_pad, alpha_pad, back_pad], dim=0)

        if self.data_type == "NR":
            near_far = torch.tensor([self.near[cam_id], self.far[cam_id]]).unsqueeze(0)
            vs = self.vs[self.vs_index[frame_id]:self.vs_index[frame_id] + self.vs_num[frame_id], :]
            vs_rgb = self.vs_rgb[self.vs_index[frame_id]:self.vs_index[frame_id] + self.vs_num[frame_id], :]
        elif self.data_type == "LLFF":
            near_far = torch.tensor([self.nears[frame_id, cam_id], self.fars[frame_id, cam_id]]).unsqueeze(0)
            vs = None
            vs_rgb = None
        elif self.data_type == "BLENDER":
            pass

        return img, \
               vs, \
               frame_id, \
               T, \
               K, \
               near_far, \
               vs_rgb

    def get_vertex_num(self):
        return torch.Tensor(self.vs_num)
