import torch
from .sample_pdf import sample_pdf


def vis_density(model, L=32):
    x = torch.linspace(0, 1, steps=L).cuda()
    grid_x, grid_y, grid_z = torch.meshgrid(x, x, x)
    xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (L,L,L,3)

    xyz = xyz * (model.maxs - model.mins) + model.mins

    xyz = xyz.reshape((-1, 3))  # (L*L*L,3)

    _, density = model.spacenet(xyz, None, model.maxs, model.mins)  # (L*L*L,1)

    density = torch.nn.functional.relu(density)
    density = density / density.max()
    xyz = xyz[density.squeeze() > 0.3, :]
    density = density[density.squeeze() > 0.3, :].repeat(1, 3)

    density[:, 1:3] = 0

    return xyz.unsqueeze(0), density.unsqueeze(0)


def get_density(model, rays, ds=None, bboxes=None, near_far=None, sample_num=4, scale=1, ratio=0.5):
    rays = rays.cuda()
    if bboxes is not None:
        bboxes = bboxes.cuda()
    if near_far is not None:
        near_far = near_far.cuda()
    if ds is not None:
        ds = ds.cuda()

    raw_sample_num = model.rsp_coarse.sample_num
    model.rsp_coarse.sample_num = sample_num

    if model.sample_method == 'NEAR_FAR':
        assert near_far is not None, 'require near_far as input '
        sampled_rays_coarse_t, sampled_rays_coarse_xyz = model.rsp_coarse.forward(rays, near_far=near_far)
        rays_t = rays
    elif model.sample_method == 'DEPTH':
        sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = model.rsp_coarse.forward(rays,
                                                                                            depth=ds,
                                                                                            depth_field=model.depth_field * scale,
                                                                                            ratio=ratio)
        sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
        sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
        rays_t = rays[ray_mask].detach()
    else:
        sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = model.rsp_coarse.forward(rays, bboxes)
        sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
        sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
        rays_t = rays[ray_mask].detach()

    with torch.no_grad():
        rgbs, density = model.spacenet_fine(sampled_rays_coarse_xyz, rays_t, model.maxs, model.mins)

    density = torch.nn.functional.relu(density)

    model.rsp_coarse.sample_num = raw_sample_num

    return sampled_rays_coarse_xyz.detach().cpu(), density.detach().cpu()


def get_weight(model, rays, bboxes, only_coarse=False, near_far=None, depth=None, rs=None):
    ray_mask = None
    torch.cuda.empty_cache()
    # beg = time.time()
    if model.sample_method == 'NEAR_FAR':
        assert near_far is not None, 'require near_far as input '
        sampled_rays_coarse_t, sampled_rays_coarse_xyz = model.rsp_coarse.forward(rays, near_far=near_far)
        rays_t = rays
    elif model.sample_method == 'DEPTH':
        sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = model.rsp_coarse.forward(rays,
                                                                                            depth=depth,
                                                                                            depth_field=model.depth_field,
                                                                                            ratio=model.depth_ratio,
                                                                                            rs=rs,
                                                                                            )
        sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
        sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
        rays_t = rays[ray_mask].detach()
    else:
        sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask = model.rsp_coarse.forward(rays, bboxes)
        sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
        sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
        rays_t = rays[ray_mask].detach()

    sampled_rays_coarse_t = sampled_rays_coarse_t.detach()
    sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach()

    with torch.no_grad():
        rgbs, density = model.spacenet(sampled_rays_coarse_xyz, rays_t, model.maxs, model.mins)

    density[sampled_rays_coarse_t[:, :, 0] < 0, :] = 0.0
    color_0, depth_0, acc_map_0, weights_0 = model.volume_render(sampled_rays_coarse_t, rgbs, density)

    if rays_t.size(0) > 1:

        if not only_coarse:
            z_samples = sample_pdf(sampled_rays_coarse_t.squeeze(), weights_0.squeeze()[..., 1:-1],
                                   N_samples=model.fine_ray_sample)
            z_samples = z_samples.detach()  # (N,L)

            z_vals_fine, _ = torch.sort(torch.cat([sampled_rays_coarse_t.squeeze(), z_samples], -1),
                                        -1)  # (N, L1+L2)
            samples_fine_xyz = z_vals_fine.unsqueeze(-1) * rays_t[:, :3].unsqueeze(1) + rays_t[:, 3:].unsqueeze(
                1)  # (N,L1+L2,3)

            with torch.no_grad():
                rgbs, density = model.spacenet_fine(samples_fine_xyz, rays_t, model.maxs, model.mins)
            color, depth, acc_map, weights = model.volume_render(z_vals_fine.unsqueeze(-1), rgbs, density)
        else:
            color, depth, acc_map, weights = color_0, depth_0, acc_map_0, weights_0

    else:
        weights = torch.zeros([rays_t.size(0), model.fine_ray_sample + model.coarse_ray_sample, 1], device=rays_t.device)
        weights[:, :model.fine_ray_sample, :] = weights_0

    torch.cuda.empty_cache()
    return weights, ray_mask


def get_weights(model, rays, bboxes, chuncks=1024 * 7, only_coarse=False, near_far=None, depth=None, rs=None):
    N = rays.size(0)
    torch.cuda.empty_cache()
    if N < chuncks:
        return get_weight(model, rays, bboxes, only_coarse=only_coarse, near_far=near_far, depth=depth, rs=rs)
    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None] * len(rays)

        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None] * len(rays)

        if depth is not None:
            depth = depth.split(chuncks, dim=0)
        else:
            depth = [None] * len(rays)

        if rs is not None:
            rs = rs.split(chuncks, dim=0)
        else:
            rs = [None] * len(rays)

        ray_masks = []
        weights = []

        for i in range(len(rays)):
            weight, ray_mask = get_weight(model, rays[i], bboxes[i], only_coarse=only_coarse,
                                          near_far=near_far[i], depth=depth[i], rs=rs[i])
            if ray_mask is not None:
                ray_masks.append(ray_mask)
                weights.append(weight)

        if len(ray_masks) > 0:
            ray_masks = torch.cat(ray_masks, dim=0)
            weights = torch.cat(weights, dim=0)

        torch.cuda.empty_cache()
        return weights, ray_masks
