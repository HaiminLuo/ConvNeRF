import torch


def batchify_ray(model, rays, bboxes=None, chuncks=1024 * 7, near_far=None, depth=None, rs=None):
    N = rays.size(0)
    if N < chuncks:
        return model(rays, bboxes, near_far=near_far)

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

        colors = [[], []]
        depths = [[], []]
        acc_maps = [[], []]

        rgb_features = [[], []]
        density_features =[[], []]

        ray_masks = []

        for i in range(len(rays)):
            stage2, stage1, ray_mask = model(rays[i], bboxes[i], near_far=near_far[i], depth=depth[i], rs=rs[i])
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])

            rgb_features[1].append(stage2[3])
            density_features[1].append(stage2[4])

            if ray_mask is not None:
                ray_masks.append(ray_mask)

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)

        rgb_features[1] = torch.cat(rgb_features[1], dim=0)
        density_features[1] = torch.cat(density_features[1], dim=0)

        if len(ray_masks) > 0:
            ray_masks = torch.cat(ray_masks, dim=0)

        return (colors[1], depths[1], acc_maps[1], rgb_features[1], density_features[1]), \
               (colors[0], depths[0], acc_maps[0]), ray_masks
