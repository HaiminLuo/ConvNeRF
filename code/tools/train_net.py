import os
import sys
import shutil

sys.path.append('..')

from config import cfg
from data import make_data_loader, make_data_loader_view
from engine.trainer import do_train
from modeling import build_model, build_discriminator
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler, make_grad_scaler
from layers import make_loss, make_perceptual_loss, make_laplacian_loss, make_weighted_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
import torch

torch.cuda.set_device(int(sys.argv[-1]))

iter = 0
if len(sys.argv) > 2:
    iter = int(sys.argv[3])
    cfg_root_path = sys.argv[1]
    assert os.path.exists(cfg_root_path), 'cfg_root_path does not exist.'
    cfg.merge_from_file(os.path.join(cfg_root_path, 'configs.yml'))
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)

else:
    cfg.merge_from_file('../configs/configs.yml')
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)
    shutil.copy('../configs/configs.yml', os.path.join(cfg.OUTPUT_DIR, 'configs.yml'))

writer.add_text('OUT_PATH', output_dir, 0)
logger = setup_logger("RFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

train_loader, dataset = make_data_loader(cfg, is_train=True)
val_loader, dataset_val = make_data_loader_view(cfg, is_train=False)
model = build_model(cfg).cuda()
discriminator = build_discriminator(cfg).cuda()

optimizer = make_optimizer(cfg, model)
optimizerD = make_optimizer(cfg, discriminator)
scaler = make_grad_scaler(cfg)


scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS,
                            cfg.SOLVER.LR_SCALE)
schedulerD = build_scheduler(optimizerD, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS,
                            cfg.SOLVER.LR_SCALE)

loss_fn = make_loss(cfg)
perceptual_loss_fn = make_perceptual_loss(cfg)
laplacian_loss_fn = make_laplacian_loss(cfg)
weighted_loss_fn = make_weighted_loss(cfg)

do_train(
    cfg,
    model,
    discriminator,
    train_loader,
    dataset_val,
    optimizer,
    optimizerD,
    scheduler,
    schedulerD,
    weighted_loss_fn,
    perceptual_loss_fn,
    swriter=writer,
    resume_iter=iter,
)
