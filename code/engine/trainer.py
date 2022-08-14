# encoding: utf-8
"""
@author:  Haimin Luo
@contact: luohm@shanghaitech.edu.cn
"""

import logging

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

import torch

from utils import batchify_ray
from torchvision.utils import make_grid

mse2psnr = lambda x: -10. * torch.log(x + 1e-10) / torch.log(torch.tensor([10.]).cuda())


def gan_loss(pred, label_type):
    if label_type:
        labels = torch.ones(pred.shape)
    else:
        labels = torch.zeros(pred.shape)
    labels = torch.autograd.Variable(labels.cuda())
    return torch.nn.MSELoss()(pred, labels)


def create_supervised_trainer(model, optimizer, discriminator, optimizerD, loss_fn, perceptual_loss_fn,
                              use_cuda=True, coarse_stage=0, swriter=None, use_alpha=False):
    if use_cuda:
        model.cuda()

    use_unet = model.use_unet and True
    use_gan = use_unet and True

    def _update(engine, batch):

        model.train()

        rays, rgbs, bboxes, near_fars, frame_ids, ds, rs, als = batch
        backgrounds = rgbs[..., 3:]
        rgbs = rgbs[..., :3]

        depth_mask = ds.detach().clone() + als.detach().clone()
        depth_mask[depth_mask > 0] = 1
        depth_mask[depth_mask < 0] = 0
        mask = depth_mask.unsqueeze(3).repeat(1, 1, 1, 4).permute(0, 3, 1, 2).cuda()
        tgt = torch.cat([rgbs, als.unsqueeze(3)], -1).permute(0, 3, 1, 2).cuda()

        rays = rays.cuda()
        rgbs = rgbs.cuda()
        near_fars = near_fars.cuda()
        frame_ids = frame_ids.cuda()
        ds = ds.cuda()
        als = als.cuda()
        rs = rs.cuda()
        backgrounds = backgrounds.cuda()

        rgb_mask = None

        if engine.state.epoch < coarse_stage:
            stage2, stage1, ray_mask, patch_1, patch_2, res_1 = model(rays, bboxes, True, near_fars=near_fars,
                                                                      depth=ds, rs=rs, rgb_mask=rgb_mask)
        else:
            stage2, stage1, ray_mask, patch_1, patch_2, res_1 = model(rays, bboxes, False,
                                                                      near_fars=near_fars,
                                                                      depth=ds, rs=rs, rgb_mask=rgb_mask)

        if ray_mask is not None:
            alpha_1, alpha_2 = stage2[2].squeeze(), stage1[2].squeeze()
            weights_1 = 2 - 1 * als.reshape(-1)[ray_mask].detach()
            weights_2 = 2 - 1 * als.reshape(-1)[ray_mask].detach()

            bgs = backgrounds.reshape(-1, backgrounds.size(3))
            crgb_1, crgb_2 = stage2[0] + (1 - stage2[2]) * bgs, stage1[0] + (1 - stage1[2]) * bgs

            if ray_mask.sum() > 0:
                if engine.state.epoch >= 10:
                    weighted_loss1, loss1 = loss_fn(crgb_1[ray_mask], rgbs.reshape(-1, rgbs.size(3))[ray_mask],
                                                    weights_1.reshape(-1, 1).repeat(1, rgbs.size(3)))
                    weighted_loss2, loss2 = loss_fn(crgb_2[ray_mask], rgbs.reshape(-1, rgbs.size(3))[ray_mask],
                                                    weights_2.reshape(-1, 1).repeat(1, rgbs.size(3)))
                else:
                    weighted_loss1, loss1 = loss_fn(crgb_1[ray_mask], rgbs.reshape(-1, rgbs.size(3))[ray_mask])
                    weighted_loss2, loss2 = loss_fn(crgb_2[ray_mask], rgbs.reshape(-1, rgbs.size(3))[ray_mask])

                weighted_loss_alpha_1, loss_alpha_1 = loss_fn(alpha_1[ray_mask], als.reshape(-1)[ray_mask], weights_1)
                weighted_loss_alpha_2, loss_alpha_2 = loss_fn(alpha_2[ray_mask], als.reshape(-1)[ray_mask], weights_2)
            else:
                weighted_loss1, loss1 = 0, 0
                weighted_loss2, loss2 = 0, 0
                weighted_loss_alpha_1, loss_alpha_1 = 0, 0
                weighted_loss_alpha_2, loss_alpha_2 = 0, 0
        else:
            alpha_1, alpha_2 = stage2[2].squeeze(), stage1[2].squeeze()
            weights_1 = 10 - 9 * alpha_1.detach()
            weights_2 = 10 - 9 * alpha_2.detach()

            bgs = backgrounds.reshape(-1, backgrounds.size(3))
            crgb_1, crgb_2 = stage2[0] + (1 - stage2[2]) * bgs, stage1[0] + (1 - stage1[2]) * bgs

            weighted_loss1, loss1 = loss_fn(crgb_1, rgbs.reshape(-1, rgbs.size(3)),
                                            weights_1.reshape(-1, 1).repeat(1, rgbs.size(3)))
            weighted_loss2, loss2 = loss_fn(crgb_2, rgbs.reshape(-1, rgbs.size(3)),
                                            weights_2.reshape(-1, 1).repeat(1, rgbs.size(3)))

            weighted_loss_alpha_1, loss_alpha_1 = loss_fn(alpha_1, als.reshape(-1), weights_1)
            weighted_loss_alpha_2, loss_alpha_2 = loss_fn(alpha_2, als.reshape(-1), weights_2)

        if not use_alpha:
            patch_1[:, 3:4, :, :] = 0   
            patch_2[:, 3:4, :, :] = 0
        feature_loss_1, feature_loss_2 = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()

        # unet for patches
        loss_unet = 0
        discriminator_in = patch_1[:, :3, :, :]
        if use_unet:
            rgb = res_1[:, 0:3, :, :]
            rgb = torch.sigmoid(rgb)
            composed_alpha_mask = torch.nn.Hardtanh()(res_1[:, 3:4, :, :] + patch_1[:, 3:4, :, :])
            composed_alpha_mask = (composed_alpha_mask + 1) / 2

            # alpha blending
            final_rgb = rgb * composed_alpha_mask + (1 - composed_alpha_mask) * backgrounds.permute(0, 3, 1, 2)
            # final_rgb = rgb  + (1 - composed_alpha_mask)

            # use ada alpha as pseu-supervision
            loss_pseu_fg, loss_zero_alpha, loss_pseu_fg_zero, loss_pseu_alpha_all = 0, 0, 0, 0
            alpha_mask = tgt[:, 3:4, :, :]
            al_mask = alpha_mask > 0.98
            fg_mask = al_mask.repeat(1, 3, 1, 1)
            zero_mask = alpha_mask == 0
            if al_mask.sum() > 0:
                _, loss_pseu_alpha_all = loss_fn(composed_alpha_mask[al_mask], alpha_mask[al_mask].detach())
                pseu_fg = (tgt[:, :3, :, :] + alpha_mask - 1) / (alpha_mask + 1e-7)
                _, loss_pseu_fg = loss_fn(rgb[fg_mask], pseu_fg[fg_mask])

            if zero_mask.sum() > 0:
                _, loss_zero_alpha = loss_fn(composed_alpha_mask[zero_mask], tgt[:, 3:4, :, :][zero_mask])
                _, loss_pseu_fg_zero = loss_fn(rgb[zero_mask.repeat(1, 3, 1, 1)],
                                               torch.zeros_like(rgb[zero_mask.repeat(1, 3, 1, 1)]) * 0)

            loss_input = torch.cat([final_rgb, composed_alpha_mask], dim=1)
            loss_unet_vgg, loss_unet_mse, _ = perceptual_loss_fn(loss_input, tgt)
            loss_unet = loss_unet_mse + loss_unet_vgg + loss_pseu_alpha_all + loss_pseu_fg_zero

            discriminator_in = final_rgb

        iters = engine.state.iteration

        loss_ganG, loss_ganD_fake, loss_ganD_real, lossD = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), \
                                                    torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        if use_gan:
            # train discriminator
            fake_response = discriminator(discriminator_in.detach())
            real_response = discriminator(tgt[:, :3, :, :])
            loss_ganD_fake = gan_loss(pred=fake_response, label_type=False)
            loss_ganD_real = gan_loss(pred=real_response, label_type=True)
            lossD = (loss_ganD_real + loss_ganD_fake) * 0.5

            if iters % 5 == 0:
                optimizerD.zero_grad()
                lossD.backward(retain_graph=True)
                optimizerD.step()

            # train generator
            fake_response = discriminator(discriminator_in)
            loss_ganG = gan_loss(pred=fake_response, label_type=True)

        if engine.state.epoch < coarse_stage:
            loss = weighted_loss1 + feature_loss_1
            if use_alpha:
                loss = loss + weighted_loss_alpha_1
        else:
            loss = weighted_loss1 + weighted_loss2 + feature_loss_1 + feature_loss_2 + loss_unet
            if not use_alpha:
                loss = loss + weighted_loss_alpha_1 + weighted_loss_alpha_2

        lossG = loss_ganG + loss

        if iters % 1 == 0:
            optimizer.zero_grad()
            lossG.backward()
            optimizer.step()

        # visualization
        res = res_1.detach().clone()
        res = res * mask
        res[:, :3, :, :][mask[:, :3, :, :] == 0] = 1

        loss_tot = lossG + lossD

        if iters % 50 == 0:
            psnr_coarse = mse2psnr(loss2)
            psnr_fine = mse2psnr(loss1)

            swriter.add_scalar('Train/loss', loss.item(), iters)
            swriter.add_scalar('Train/vgg_loss_coarse', feature_loss_1.item(), iters)
            swriter.add_scalar('Train/vgg_loss_fine', feature_loss_2.item(), iters)
            swriter.add_scalar('Train/psnr_coarse', psnr_coarse.item(), iters)
            swriter.add_scalar('Train/psnr_fine', psnr_fine.item(), iters)

            swriter.add_scalar('Train/gan_generator_loss', lossG.item(), iters)
            swriter.add_scalar('Train/gan_discriminator_loss', lossD.item(), iters)
            swriter.add_scalar('Train/gan_generator_loss_fake', loss_ganG.item(), iters)
            swriter.add_scalar('Train/gan_discriminator_loss_real', loss_ganD_real.item(), iters)
            swriter.add_scalar('Train/gan_discriminator_loss_fake', loss_ganD_fake.item(), iters)

            if use_unet:
                psnr_unet = mse2psnr(loss_unet_mse)
                swriter.add_scalar('Train/psnr_unet', psnr_unet.item(), iters)

        if use_unet:
            if iters % 100 == 0:
                # print(res_1.shape)
                swriter.add_image('unet/input_rgb', make_grid(patch_1[:, 0:3, :, :]), iters // 100)
                swriter.add_image('unet/input_alpha', make_grid(patch_1[:, 3:4, :, :]), iters // 100)
                swriter.add_image('unet/rendered', make_grid(rgb[:, 0:3, :, :]), iters // 100)
                swriter.add_image('unet/alpha', make_grid(composed_alpha_mask), iters // 100)
                swriter.add_image('unet/rgb_gt', make_grid(tgt[:, 0:3, :, :]), iters // 100)
                swriter.add_image('unet/alpha_gt', make_grid(tgt[:, 3:4, :, :]), iters // 100)
        return loss_tot.item()

    return Engine(_update)


def evaluator(val_dataset, model, loss_fn, swriter, epoch):
    model.eval()
    noise_std = model.rfrender.noise_std
    model.rfrender.noise_std = 0.0

    boarder_weight_0 = model.rfrender.volume_render.boarder_weight
    model.rfrender.volume_render.boarder_weight = 1e-10

    use_unet = model.use_unet

    rays, rgbs, bboxes, color, mask, ROI, near_far, frame_id, ds, depth, rs, alpha = val_dataset.__getitem__(0)

    rgbs = rgbs[:3, ...]

    rays = rays.cuda()
    rgbs = rgbs.cuda()
    bboxes = bboxes.cuda()
    color_gt = color.cuda()
    backgrounds = color_gt[3:, ...]
    color_gt = color_gt[:3, ...]
    mask = mask.cuda()
    ROI = ROI.cuda()
    near_far = near_far.cuda()
    rs = rs.cuda()
    alpha = alpha.cuda()

    if ds is not None:
        ds = ds.cuda()
    if depth is not None:
        depth_gt = depth.cuda()
    else:
        depth_gt = None

    depth_mask = ds.detach().clone().reshape(color_gt.size(1), color_gt.size(2), 1) + \
                 alpha.detach().clone().reshape(color_gt.size(1), color_gt.size(2), 1)
    depth_mask[depth_mask > 0] = 1
    depth_mask[depth_mask < 0] = 0
    mask = depth_mask.repeat(1, 1, 4).permute(2, 0, 1).cuda()

    with torch.no_grad():
        stage2, stage1, _ = batchify_ray(model.rfrender, rays, bboxes, near_far=near_far, depth=ds, rs=rs)

        color_1 = stage2[0]
        depth_1 = stage2[1]
        acc_map_1 = stage2[2]
        rgb_features = stage2[3]
        alpha_features = stage2[4]

        color_0 = stage1[0]
        depth_0 = stage1[1]
        acc_map_0 = stage1[2]

        color_img = color_1.reshape((color_gt.size(1), color_gt.size(2), 3)).permute(2, 0, 1)
        depth_img = depth_1.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        acc_map = acc_map_1.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
        rgb_features = rgb_features.reshape((color_gt.size(1), color_gt.size(2), -1)).permute(2, 0, 1)
        alpha_features = alpha_features.reshape((color_gt.size(1), color_gt.size(2), -1)).permute(2, 0, 1)

        color_img_0 = color_0.reshape((color_gt.size(1), color_gt.size(2), 3)).permute(2, 0, 1)
        depth_img_0 = depth_0.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
        depth_img_0 = (depth_img_0 - depth_img_0.min()) / (depth_img_0.max() - depth_img_0.min())
        acc_map_0 = acc_map_0.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)

        color_img = color_img + (1 - acc_map) * backgrounds
        color_img_0 = color_img_0 + (1 - acc_map_0) * backgrounds

        swriter.add_image('GT/rgb', color_gt, epoch)
        if depth_gt is not None:
            depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-6)
            swriter.add_image('GT/depth', depth_gt, epoch)

        swriter.add_image('GT/alpha', alpha, epoch)

        swriter.add_image('stage2/rendered', color_img, epoch)
        swriter.add_image('stage2/depth', depth_img, epoch)
        swriter.add_image('stage2/alpha', acc_map, epoch)

        swriter.add_image('stage1/rendered', color_img_0, epoch)
        swriter.add_image('stage1/depth', depth_img_0, epoch)
        swriter.add_image('stage1/alpha', acc_map_0, epoch)

        if use_unet:
            rays = rays.reshape(color_gt.size(1), color_gt.size(2), -1).permute(2, 0, 1)
            dirs = rays[:3, :, :]
            dirs[depth.expand(3, -1, -1) == 0] = 0

            unet_out = model.unet(rgb_feat=rgb_features.unsqueeze(0),
                                  alpha_feat=alpha_features.unsqueeze(0),
                                  ).squeeze()
            rgb = torch.sigmoid(unet_out[:3, :, :]).detach()
            alpha = torch.nn.Hardtanh()(unet_out[3:4, :, :] + acc_map)
            alpha = (alpha + 1) / 2

            alpha[alpha > 1] = 1
            alpha[alpha < 0] = 0
            comp_img = rgb * alpha + (1 - alpha) * backgrounds
            # comp_img = rgb + (1 - acc_map)
            comp_img[comp_img > 1] = 1

            _, loss_unet = loss_fn(comp_img[:3, :, :], color_gt)
            psnr_unet = mse2psnr(loss_unet)

            swriter.add_scalar('Test/unet_loss', loss_unet.item(), epoch)
            swriter.add_scalar('Test/unet_psnr', psnr_unet.item(), epoch)
            swriter.add_image('unet/unet_img', rgb, epoch)
            swriter.add_image('unet/unet_alpha', alpha, epoch)
            swriter.add_image('unet/unet_compose_img', comp_img[:3, :, :], epoch)

        model.rfrender.noise_std = noise_std
        model.rfrender.volume_render.boarder_weight = boarder_weight_0

        _, loss_val = loss_fn(color_img, color_gt)
        psnr = mse2psnr(loss_val)
        swriter.add_scalar('Test/psnr', psnr.item(), epoch)

        return loss_val.item()


def global_step_from_engine(engine):
    def wrapper(_, event_name=None):
        return engine.state.iteration

    return wrapper


def do_train(
        cfg,
        model,
        discriminator,
        train_loader,
        val_loader,
        optimizer,
        optimizerD,
        scheduler,
        schedulerD,
        loss_fn,
        perceptual_loss_fn,
        swriter,
        resume_iter=0
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("RFRender.%s.train" % cfg.OUTPUT_DIR.split('/')[-1])
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, discriminator, optimizerD, loss_fn,
                                        perceptual_loss_fn,
                                        coarse_stage=cfg.SOLVER.COARSE_STAGE,
                                        swriter=swriter,
                                        use_alpha=cfg.DATASETS.USE_ALPHA)

    checkpointer = ModelCheckpoint(output_dir, 'rfnr', n_saved=1000, require_empty=False)
    checkpointer._iteration = resume_iter

    timer = Timer(average=True)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_period), checkpointer,
                              {'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                               'discriminator': discriminator, 'optimizerD': optimizerD, 'schedulerD': schedulerD}
                              )
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    def val_vis(engine):
        avg_loss = evaluator(val_loader, model, loss_fn, swriter, engine.state.epoch)
        logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_loss)
                    )
        swriter.add_scalar('Test/loss', avg_loss, engine.state.epoch)

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        if resume_iter > 0:
            engine.state.iteration = resume_iter
            engine.state.epoch = resume_iter // len(train_loader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3e} Lr: {:.2e} Speed: {:.1f}[rays/s]"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], lr,
                                float(cfg.SOLVER.BATCH_SIZE * cfg.DATASETS.PATCH_SIZE ** 2) / timer.value()))
        if iter % 500 == 1:
            val_vis(engine)

        scheduler.step()
        schedulerD.step()

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            float(cfg.SOLVER.BATCH_SIZE * cfg.DATASETS.PATCH_SIZE ** 2) / timer.value()))
        timer.reset()

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            val_vis(engine)
            pass

    trainer.run(train_loader, max_epochs=epochs)
