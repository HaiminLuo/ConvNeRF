import torch

'''
Sample depth from views (and images) with/without masks

--------------------------
INPUT Tensors
Ks: intrinsics of cameras (M,3,3)
Ts: extrinsic of cameras (M,4,4)
image_size: the size of image [H,W]
mask_threshold: a float threshold to mask rays
masks:(M,H,W)
-------------------
OUPUT:
Ds:  (N,1)
'''


def depth_sampling(Ks, image_size, masks=None, mask_threshold=0.5, depth=None):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)

    x = torch.linspace(0, h - 1, steps=h, device=Ks.device)
    y = torch.linspace(0, w - 1, steps=w, device=Ks.device)

    grid_x, grid_y = torch.meshgrid(x, y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M, 1, 1, 1)  # (M,2,H,W)
    coordinates = torch.cat([coordinates, torch.ones(coordinates.size(0), 1, coordinates.size(2),
                                                     coordinates.size(3), device=Ks.device)], dim=1).permute(0, 2, 3,
                                                                                                             1).unsqueeze(
        -1)

    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks, coordinates)  # (M,H,W,3,1)
    dirs = dirs / torch.norm(dirs, dim=3, keepdim=True)

    # zs to depth
    if depth is not None:
        td = dirs.view(M, h, w, -1)
        depth = depth / td[:, :, :, 2]

    ds = None

    if masks is not None:
        if depth is not None:
            ds = depth[masks > mask_threshold]
    else:
        if depth is not None:
            ds = depth.reshape((-1,))

    return ds


def range_sampling(masks=None, mask_threshold=0.5, range_map=None):

    rs = None

    if masks is not None:
        if range_map is not None:
            rs = range_map[masks > mask_threshold]
    else:
        if range_map is not None:
            rs = range_map.reshape((-1,))

    return rs


def alpha_sampling(masks=None, mask_threshold=0.5, alpha_map=None):

    als = None

    if masks is not None:
        if alpha_map is not None:
            als = alpha_map[masks > mask_threshold]
    else:
        if alpha_map is not None:
            als = alpha_map.reshape((-1,))

    return als