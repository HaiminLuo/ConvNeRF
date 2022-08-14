import os

import torch
import sys
import numpy as np
sys.path.append('..')

import argparse
from utils import batchify_ray, ray_sampling, render_near_far_depth, SingleObjectRender
from config import cfg
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
from modeling import build_model
from imageio_ffmpeg import write_frames
import cv2


def render_image(model, glrenderer, K, T, img_size=(450, 800)):
    opencv_K = K.numpy()
    cam2world_R = T[0:3, 0:3].numpy()
    cam2world_t = T[0:3, 3].numpy()
    depth, far_depth = render_near_far_depth(glrenderer, opencv_K, cam2world_R, cam2world_t)
    depth_raw = depth.copy()
    depth = torch.from_numpy(depth).unsqueeze(0).cuda()
    far_depth = torch.from_numpy(far_depth).unsqueeze(0).cuda()

    rays, _, ds = ray_sampling(K.unsqueeze(0).cuda(), T.unsqueeze(0).cuda(), img_size, depth=depth, far_depth=far_depth)
    rs = ds[..., 1] - ds[..., 0]
    ds = ds[..., 0]
    with torch.no_grad():
        stage2, _, _ = batchify_ray(model.rfrender, rays, chuncks=1024 * 12, depth=ds, rs=rs)

    color_1 = stage2[0]
    depth_1 = stage2[1]
    acc_map_1 = stage2[2]
    rgb_features = stage2[3]
    alpha_features = stage2[4]

    color_img = color_1.reshape((img_size[0], img_size[1], 3)).permute(2, 0, 1)
    depth_img = depth_1.reshape((img_size[0], img_size[1], 1)).permute(2, 0, 1)
    acc_map = acc_map_1.reshape((img_size[0], img_size[1], 1)).permute(2, 0, 1)
    rgb_features = rgb_features.reshape((img_size[0], img_size[1], -1)).permute(2, 0, 1)
    alpha_features = alpha_features.reshape((img_size[0], img_size[1], -1)).permute(2, 0, 1)

    rgb_in = rgb_features
    alpha_in = alpha_features

    with torch.no_grad():
        unet_out = model.unet(rgb_feat=rgb_in.unsqueeze(0),
                              alpha_feat=alpha_in.unsqueeze(0),
                              ).squeeze()

    rgb = torch.sigmoid(unet_out[:3, :, :])

    alpha = torch.nn.Hardtanh()(unet_out[3:4, :, :] + acc_map)
    alpha = (alpha + 1) / 2
    alpha = torch.clamp(alpha, min=0, max=1.)

    comp_img = rgb * alpha + (1 - alpha)
    img_unet = comp_img.permute(1, 2, 0).detach().cpu().numpy()
    alpha = alpha.squeeze().detach().cpu().numpy()
    depth = depth_img.squeeze().detach().cpu().numpy()

    return img_unet, alpha, depth


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='')
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--cam_pose', type=str, default='')
parser.add_argument('--intrinsic', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--out_dir', type=str, default='')

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu_id)

cfg_path = opt.config
assert cfg_path is not '' and os.path.exists(cfg_path), 'config file does not exist.'
assert opt.out_dir is not '' and os.path.exists(opt.out_dir), 'out_dir does not exist.'

cfg.merge_from_file(cfg_path)
cfg.freeze()
img_size=cfg.INPUT.SIZE_TEST
dataset_path = cfg.DATASETS.TRAIN
dataset_val_path = cfg.DATASETS.TEST

# load proxy mesh
glrenderer = SingleObjectRender(os.path.join(dataset_path, "meshes/frame1.obj"),
                                             frame_size=img_size,
                                             enable_texture=False, enable_cull=False)
# load camera
Ts = torch.Tensor(campose_to_extrinsic(np.loadtxt(opt.cam_pose)))
Ks = torch.Tensor(read_intrinsics(opt.intrinsic))

if img_size[0] != Ks[0,0,2] * 2 or img_size[1] != Ks[0, 1, 2] * 2:
    img_size[0] = int(Ks[0, 0, 2] * 2)
    img_size[1] = int(Ks[0, 1, 2] * 2)

# load model
model = build_model(cfg).cuda()
ckpt_path = opt.ckpt
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# video

writer_raw_rgb = write_frames(os.path.join(opt.out_dir, 'video_rgb.mp4'), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
writer_raw_alpha = write_frames(os.path.join(opt.out_dir, 'video_alpha.mp4'), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
writer_raw_depth = write_frames(os.path.join(opt.out_dir, 'video_depth.mp4'), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
writer_raw_rgb.send(None)
writer_raw_alpha.send(None)
writer_raw_depth.send(None)

for i, T in enumerate(Ts):
    K = Ks[i]
    img, alpha, depth = render_image(model=model, glrenderer=glrenderer, K=K, T=T, img_size=(img_size[1], img_size[0]))
    img = img * 255
    alpha = torch.from_numpy(alpha).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255
    depth = depth / depth.max()
    depth = torch.from_numpy(depth).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255
    img = img.copy(order='C')
    writer_raw_rgb.send(img.astype(np.uint8))
    writer_raw_alpha.send(alpha.astype(np.uint8))
    writer_raw_depth.send(depth.astype(np.uint8))

    cv2.imwrite(os.path.join(opt.out_dir, 'rgb_%04d.jpg' % i),
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(opt.out_dir, 'alpha_%04d.jpg' % i), alpha)
    cv2.imwrite(os.path.join(opt.out_dir, 'depth_%04d.jpg' % i), depth)
    # print(os.path.join(opt.out_dir, 'depth_%04d.jpg' % i))

writer_raw_rgb.close()
writer_raw_alpha.close()
writer_raw_depth.close()












