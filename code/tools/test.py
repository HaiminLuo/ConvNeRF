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

from data import make_data_loader, make_data_loader_view
from tqdm import tqdm


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
# parser.add_argument('--cam_pose', type=str, default='')
# parser.add_argument('--intrinsic', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--out_dir', type=str, default='')
# parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--dataset_val_path', type=str, default=None)


opt = parser.parse_args()
torch.cuda.set_device(opt.gpu_id)

cfg_path = opt.config
assert cfg_path is not None and os.path.exists(cfg_path), 'config file does not exist.'
assert opt.out_dir is not None and os.path.exists(opt.out_dir), 'out_dir does not exist.'

cfg.merge_from_file(cfg_path)
if opt.dataset_val_path is not None:
    cfg.DATASETS.TEST = opt.dataset_val_path
    cfg.DATASETS.TRAIN = opt.dataset_val_path

# load validation dataset
val_loader, dataset_val = make_data_loader_view(cfg, is_train=False)

img_size=cfg.INPUT.SIZE_TEST
# dataset_path = cfg.DATASETS.TRAIN 
dataset_val_path = cfg.DATASETS.TEST

# load proxy mesh
glrenderer = SingleObjectRender(os.path.join(dataset_val_path, "meshes/frame1.obj"),
                                             frame_size=img_size,
                                             enable_texture=False, enable_cull=False)

# load model
model = build_model(cfg).cuda()
ckpt_path = opt.ckpt
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# metric
loss_fn = torch.nn.MSELoss()
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

psnrs_rgb, psnrs_alpha = [], []

for i in tqdm(range(len(dataset_val.NHR_dataset))):
    img_raw, _, _, T, K, _, _ = dataset_val.NHR_dataset.__getitem__(i)  
    rgb_gt = img_raw[:3, ...].permute(1, 2, 0)
    alpha_gt = img_raw[6:7, ...].permute(1, 2, 0).repeat(1, 1, 3)

    img, alpha, depth = render_image(model=model, glrenderer=glrenderer, K=K, T=T, img_size=(img_size[1], img_size[0]))
    
    mse_rgb = loss_fn(torch.from_numpy(img), rgb_gt)
    psnr_rgb = mse2psnr(mse_rgb)

    mse_alpha = loss_fn(torch.from_numpy(alpha), alpha_gt[..., 0])
    psnr_alpha = mse2psnr(mse_alpha)

    img = img * 255
    alpha = torch.from_numpy(alpha).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255
    depth = depth / depth.max()
    depth = torch.from_numpy(depth).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255
    rgb_gt = rgb_gt.numpy() * 255
    alpha_gt = alpha_gt.numpy() * 255
    
    cv2.imwrite(os.path.join(opt.out_dir, 'rgb_%04d.jpg' % i),
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(opt.out_dir, 'alpha_%04d.jpg' % i), alpha)
    cv2.imwrite(os.path.join(opt.out_dir, 'depth_%04d.jpg' % i), depth)
    cv2.imwrite(os.path.join(opt.out_dir, 'gt_rgb_%04d.jpg' % i), cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(opt.out_dir, 'gt_alpha_%04d.jpg' % i), alpha_gt)

    psnrs_rgb.append(psnr_rgb.item())
    psnrs_alpha.append(psnr_alpha.item())
    
print('PSNR-RGB:', sum(psnrs_rgb) / len(psnrs_rgb), ' ', 'PSNR-ALPHA:', sum(psnrs_alpha) / len(psnrs_alpha))