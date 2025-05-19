import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
from lightning_fabric.utilities.seed import seed_everything  # atualizado

from src.Datasets.DeepPhaseDataModule import Style100DataModule
from src.Datasets.Style100Processor import StyleLoader, Swap100StyJoints
from src.Net.DeepPhaseNet import DeepPhaseNet, Application
from src.utils import BVH_mod as BVH
from src.utils.locate_model import locate_model
from src.utils.motion_process import subsample


def setup_seed(seed: int):
    seed_everything(seed, workers=True)


def test_model():
    return {
        "limit_train_batches": 1.,
        "limit_val_batches": 1.
    }

def detect_nan_par():
    return {"detect_anomaly": True}

def select_gpu_par():
    return {
        "accelerator": 'gpu',
        "devices": 1,
        "auto_select_gpus": True,
    }

def create_common_states(prefix: str):
    log_name = prefix + '/'
    parser = ArgumentParser()
    parser.add_argument("--dev_run", action="store_true")
    parser.add_argument("--version", type=str, default="-1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_phases", type=int, default=10)
    parser.add_argument("--epoch", type=str, default='')
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    ckpt_path = "results/"
    version = args.version if args.version != "-1" else None
    log_name += "dev_run" if args.dev_run else "myResults"

    tb_logger = pl.loggers.TensorBoardLogger(save_dir="tensorboard_logs/", name=log_name, version=version)
    ckpt_path = os.path.join(ckpt_path, log_name, str(tb_logger.version))

    if args.resume:
        resume_from_checkpoint = os.path.join(ckpt_path, "last.ckpt")
    else:
        resume_from_checkpoint = None

    checkpoint_callback = [
        ModelCheckpoint(dirpath=ckpt_path, save_top_k=-1, save_last=True, every_n_epochs=1, save_weights_only=True),
        ModelCheckpoint(dirpath=ckpt_path, save_top_k=1, monitor="val_loss", every_n_epochs=1)
    ]
    checkpoint_callback[0].CHECKPOINT_NAME_LAST = "last"

    profiler = SimpleProfiler()

    trainer_dict = {
        "callbacks": checkpoint_callback,
        "profiler": profiler,
        "logger": tb_logger
    }
    return args, trainer_dict, resume_from_checkpoint, ckpt_path


def read_style_bvh(style, content, clip=None):
    swap_joints = Swap100StyJoints()
    anim = BVH.read_bvh(os.path.join("MotionData/100STYLE/", style, f"{style}_{content}.bvh"), remove_joints=swap_joints)
    if clip:
        anim.quats = anim.quats[clip[0]:clip[1], ...]
        anim.hip_pos = anim.hip_pos[clip[0]:clip[1], ...]
    return subsample(anim, ratio=2)


def training_style100():
    args, trainer_dict, resume_from_checkpoint, ckpt_path = create_common_states("deephase_sty")

    frequency = 30
    window = 61
    batch_size = 4

    style_loader = StyleLoader()
    data_module = Style100DataModule(batch_size=batch_size, shuffle=True, data_loader=style_loader, window_size=window)
    model = DeepPhaseNet(args.n_phases, data_module.skeleton, window, 1.0 / frequency, batch_size=batch_size)

    if not args.test:
        if args.dev_run:
            trainer = Trainer(
                **trainer_dict,
                **test_model(),
                **select_gpu_par(),
                precision=32,
                log_every_n_steps=50,
                max_epochs=30,
                auto_lr_find=True
            )
        else:
            trainer = Trainer(
                **trainer_dict,
                max_epochs=500,
                **select_gpu_par(),
                log_every_n_steps=50,
                resume_from_checkpoint=resume_from_checkpoint
            )
        trainer.fit(model, datamodule=data_module)
    else:
        anim = read_style_bvh("WildArms", "FW", [509, 1009])
        DEVICE = torch.device("cpu")
        check_file = ckpt_path + "/"
        modelfile = locate_model(check_file, args.epoch)
        model = DeepPhaseNet.load_from_checkpoint(modelfile).to(DEVICE)

        data_module.setup()
        app = Application(model, data_module).float()
        anim = subsample(anim, 1)
        app.setAnim(anim)
        app.forward()
        BVH.save_bvh("source.bvh", anim)


if __name__ == '__main__':
    setup_seed(3407)
    training_style100()
